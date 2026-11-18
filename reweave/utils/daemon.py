# Copyright 2026 The RPent Authors.
# Modified for the ReWeave namespace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Subprocess RPC servers: spawn (parent side), death-watch (child side)."""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Callable

from reweave.utils.config import get_repo_root
from reweave.utils.logging import get_logger

logger = get_logger("daemon")

# Owned RPC children echo this per-start identity before any client registration
# or business call. It identifies a process launch, not a model/runtime version.
SERVICE_INSTANCE_ENV = "REWEAVE_SERVICE_INSTANCE_ID"
RPC_LISTEN_FD_ENV = "REWEAVE_RPC_LISTEN_FD"


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def watch_parent_death(on_death: Callable[[], None]) -> None:
    """Call ``on_death()`` once, from a background thread, when stdin hits EOF.

    Under :class:`ProcessDaemon` this fires exactly when the parent dies. If
    invoked from a terminal, ``read()`` blocks on user input and never fires;
    if stdin is redirected from ``/dev/null`` or already closed, it fires
    immediately.
    """

    def _watch() -> None:
        try:
            sys.stdin.buffer.read()
        except Exception:
            pass
        on_death()

    threading.Thread(target=_watch, daemon=True).start()


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Return an unused TCP port on ``host``.

    This does not reserve the port until a child binds it. Owned RPC readiness
    must verify the child's launch identity rather than trust the address.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


@dataclass
class PortReservation:
    """A bound TCP endpoint that can be transferred to an RPC child.

    Keeping the socket open closes the usual ``bind(port=0) -> close -> spawn``
    race. :class:`ProcessDaemon` passes the descriptor to its child and closes
    only the parent's copy after a successful spawn; :class:`RpcFacade` then
    activates that exact socket.
    """

    host: str
    port: int
    _socket: socket.socket

    @classmethod
    def acquire(cls, host: str = "127.0.0.1", port: int = 0) -> PortReservation:
        reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # The reservation is the listener transferred to the child, rather
            # than a probe socket that is closed before the child binds.  Mark
            # it reusable before bind so the same owned daemon can reacquire
            # its endpoint immediately after a clean stop, then listen now so
            # no second REUSEADDR binder can claim the unactivated endpoint.
            reserved.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            reserved.bind((host, port))
            reserved.listen()
        except BaseException:
            reserved.close()
            raise
        bound_host, bound_port = reserved.getsockname()
        return cls(
            host=str(bound_host),
            port=int(bound_port),
            _socket=reserved,
        )

    @property
    def fileno(self) -> int:
        descriptor = self._socket.fileno()
        if descriptor < 0:
            raise RuntimeError("port reservation is already closed")
        return descriptor

    def close(self) -> None:
        self._socket.close()


def reserve_free_port(host: str = "127.0.0.1") -> PortReservation:
    """Bind and retain a free TCP port for transfer to an RPC child."""

    return PortReservation.acquire(host)


class ProcessDaemon:
    """Wraps a subprocess server with ready-detection and lifecycle."""

    def __init__(
        self,
        name: str,
        cmd: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
        log_path: str | None = None,
        cwd: str | None = None,
        port_reservation: PortReservation | None = None,
        stop_timeout_s: float = 15.0,
    ) -> None:
        """Set the finite shutdown grace, including any child-owned evidence return."""
        if (
            isinstance(stop_timeout_s, bool)
            or not isinstance(stop_timeout_s, (int, float))
            or not math.isfinite(stop_timeout_s)
            or stop_timeout_s <= 0
        ):
            raise ValueError("stop_timeout_s must be finite and positive")
        self.name = name
        self.cmd = cmd
        self.subprocess_env = os.environ.copy()
        self.subprocess_env.update(env_overrides or {})
        self.log_path = log_path
        self.cwd = cwd
        self.stop_timeout_s = stop_timeout_s
        self._port_reservation = port_reservation
        self._reserved_endpoint = (
            None
            if port_reservation is None
            else (port_reservation.host, port_reservation.port)
        )
        self._proc: subprocess.Popen | None = None
        self._log_f = None
        self.instance_id: str | None = None

    def poll(self) -> int | None:
        """Return the subprocess exit code, or None if it's still running.

        None also before ``start()``. A non-None value means the child has
        exited — used by ``wait_for_ready`` to fail fast on early crashes.
        """
        return self._proc.poll() if self._proc is not None else None

    def start(self) -> None:
        """Spawn the subprocess. Returns immediately; does not wait for readiness."""
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError(f"{self.name} is already running")
        if self._port_reservation is None and self._reserved_endpoint is not None:
            self._port_reservation = PortReservation.acquire(*self._reserved_endpoint)
        self.instance_id = uuid.uuid4().hex
        self.subprocess_env[SERVICE_INSTANCE_ENV] = self.instance_id
        reservation_fd = None
        if self._port_reservation is not None:
            reservation_fd = self._port_reservation.fileno
            self.subprocess_env[RPC_LISTEN_FD_ENV] = str(reservation_fd)
        try:
            self._log_f = (
                open(self.log_path, "a") if self.log_path else open(os.devnull, "w")
            )
            # Every service retains the episode's device lease. Even abrupt
            # worker death cannot release the GPU before its services exit.
            lease = self.subprocess_env.get("REWEAVE_ROLLOUT_LEASE_FD")
            pass_fds = tuple(
                descriptor
                for descriptor in (
                    None if lease is None else int(lease),
                    reservation_fd,
                )
                if descriptor is not None
            )
            for descriptor in pass_fds:
                os.fstat(descriptor)
            self._proc = subprocess.Popen(
                self.cmd,
                # stdin pipe lets the child detect our death via EOF; see
                # watch_parent_death above.
                stdin=subprocess.PIPE,
                stdout=self._log_f,
                stderr=subprocess.STDOUT,
                env=self.subprocess_env,
                cwd=self.cwd or get_repo_root(),
                pass_fds=pass_fds,
            )
        except BaseException:
            if self._log_f is not None:
                self._log_f.close()
            self._log_f = None
            # start() is atomic: callers never have to retain a daemon whose
            # process was not created merely to release its socket/log handles.
            # _reserved_endpoint remains available for an explicit retry.
            if self._port_reservation is not None:
                self._port_reservation.close()
                self._port_reservation = None
            self.subprocess_env.pop(RPC_LISTEN_FD_ENV, None)
            raise
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None
        self.subprocess_env.pop(RPC_LISTEN_FD_ENV, None)
        logger.info(
            "%s spawned (pid=%s, instance_id=%s)",
            self.name,
            self._proc.pid,
            self.instance_id,
        )

    def stop(self, timeout: float | None = None) -> None:
        """Drain within the configured grace; an explicit timeout takes priority."""
        timeout = self.stop_timeout_s if timeout is None else timeout
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.close()
        if self._log_f is not None:
            try:
                self._log_f.close()
            except Exception:
                pass
            self._log_f = None
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None


__all__ = [
    "PortReservation",
    "ProcessDaemon",
    "RPC_LISTEN_FD_ENV",
    "SERVICE_INSTANCE_ENV",
    "pick_free_port",
    "reserve_free_port",
    "watch_parent_death",
]
