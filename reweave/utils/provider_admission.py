"""Cross-process native-model admission and a shared rate-limit circuit breaker."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import socket
import time
from contextlib import asynccontextmanager, contextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import time as _wall_time

from reweave.utils.persistence import atomic_json


class NativeQuotaExhausted(RuntimeError):
    """The subscription/account budget must be restored before another query."""


def is_quota_exhausted(value) -> bool:
    if isinstance(value, str):
        text = value.lower()
        return any(
            marker in text
            for marker in (
                "subscription sharing usage limit",
                "insufficient_quota",
                "usage_limit_reached",
                "quota_exceeded",
                "credit_balance_exhausted",
                "organization_usage_limit_exceeded",
                "hit your chatgpt usage limit",
            )
        )
    if isinstance(value, (dict, list)):
        return any(
            is_quota_exhausted(x)
            for x in (value.values() if isinstance(value, dict) else value)
        )
    return False


def is_rate_limit(value) -> bool:
    """Inspect error payloads, never ordinary account/usage notifications."""
    if not isinstance(value, (dict, list, str)) or is_quota_exhausted(value):
        return False
    if isinstance(value, str):
        return bool(
            re.search(r"\b429\b|too many requests|rate[ _-]?limit", value, re.I)
        )
    if isinstance(value, list):
        return any(is_rate_limit(item) for item in value)
    for key, item in value.items():
        if key in {
            "http_status_code",
            "httpStatusCode",
            "status_code",
            "statusCode",
        } and item in (429, "429"):
            return True
        if key in {
            "message",
            "additional_details",
            "additionalDetails",
            "detail",
            "details",
            "code",
        } and is_rate_limit(item):
            return True
        if isinstance(item, (dict, list)) and is_rate_limit(item):
            return True
    return False


def is_native_transport_failure(error):
    """Recognize explicit transport variants and observed provider error texts."""
    if not isinstance(error, dict) or is_rate_limit(error) or is_quota_exhausted(error):
        return False
    variant = error.get("codexErrorInfo", error.get("codex_error_info"))
    status = None
    if isinstance(variant, dict) and len(variant) == 1:
        kind, detail = next(iter(variant.items()))
        if isinstance(detail, dict):
            status = detail.get("httpStatusCode", detail.get("http_status_code"))
    elif isinstance(variant, str):
        kind = variant
    else:
        return False
    kind = kind.replace("_", "").lower()
    if status is not None and (
        not isinstance(status, int)
        or status not in {408, 409}
        and not 500 <= status < 600
    ):
        return False
    if kind in {
        "httpconnectionfailed",
        "responsestreamdisconnected",
        "responsestreamconnectionfailed",
        "responsetoomanyfailedattempts",
        "serveroverloaded",
        "internalservererror",
    }:
        return True
    message = error.get("message", "")
    return kind == "other" and (
        message == "request timed out"
        or str(message).startswith("stream disconnected before completion:")
    )


class ProviderAdmission:
    """One account-route budget shared by online, graph and author processes.

    OS locks release after process death. A rate limit closes admission for all
    workers; after cooling down up to four probes run before normal admission
    resumes. An old in-flight success cannot clear a newer rate-limit incident.
    """

    def __init__(
        self,
        route,
        *,
        root=None,
        capacity=None,
        cooldown_s=60.0,
        max_cooldown_s=900.0,
        start_interval_s=0.0,
        lane="online",
    ):
        base = Path(
            root
            or os.environ.get(
                "REWEAVE_PROVIDER_ADMISSION_ROOT",
                str(Path.home() / ".codex/harness-provider-admission"),
            )
        )
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", route):
            raise ValueError("Invalid provider route")
        self.root = base / route
        if lane not in {"online", "offline"}:
            raise ValueError("Invalid native admission lane")
        self.lane = lane
        self._default_capacity = capacity is None
        self.capacity = int(
            os.environ.get("REWEAVE_NATIVE_MODEL_CONCURRENCY", "32")
            if capacity is None
            else capacity
        )
        if (
            self.capacity < 1
            or cooldown_s <= 0
            or max_cooldown_s < cooldown_s
            or start_interval_s < 0
        ):
            raise ValueError("Invalid native provider admission policy")
        self.cooldown_s, self.max_cooldown_s = cooldown_s, max_cooldown_s
        self.start_interval_s = start_interval_s
        self.remote_endpoint = (
            os.environ.get("REWEAVE_PROVIDER_ADMISSION_ENDPOINT")
            if root is None
            else None
        )
        self.remote_token_file = (
            os.environ.get("REWEAVE_PROVIDER_ADMISSION_TOKEN_FILE")
            if self.remote_endpoint
            else None
        )
        if self.remote_endpoint and not self.remote_token_file:
            raise ValueError("Remote admission requires its private token file")

    def _remote_address(self):
        host, port = self.remote_endpoint.rsplit(":", 1)
        if host != "127.0.0.1":
            raise ValueError("Remote admission must use a loopback SSH forward")
        return host, int(port)

    def _remote_message(self, command, **values):
        return (
            json.dumps(
                {
                    "token": Path(self.remote_token_file).read_text().strip(),
                    "command": command,
                    **values,
                }
            ).encode()
            + b"\n"
        )

    def _remote_note(self, command, error):
        with socket.create_connection(self._remote_address(), timeout=10) as stream:
            stream.sendall(self._remote_message(command, error=error))
            with stream.makefile("rb") as response:
                value = json.loads(response.readline())
        if value.get("status") != "ok":
            raise RuntimeError("Shared admission failed to record provider protection")
        return value["recorded"]

    @contextmanager
    def _state(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.root / "control.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / "state.json"
            value = json.loads(path.read_text()) if path.exists() else {}
            try:
                yield value
            finally:
                atomic_json(path, value)

    @staticmethod
    def _try_lock(path):
        stream = path.open("a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return stream
        except BlockingIOError:
            stream.close()
            return None

    def note_rate_limit(self, error):
        if not is_rate_limit(error):
            return False
        if self.remote_endpoint:
            return self._remote_note("rate_limit", error)
        with self._state() as state:
            now = _wall_time()
            strikes = state.get("strikes", 0)
            if now - state.get("last_rate_limit_at", 0) > 5:
                strikes += 1
            wait = min(
                self.cooldown_s * 2 ** min(max(strikes - 1, 0), 10), self.max_cooldown_s
            )

            def retry_after(value):
                if not isinstance(value, dict):
                    return 0.0
                delay = 0.0
                for key, item in value.items():
                    if key.lower().replace("_", "-") in {"retry-after", "retryafter"}:
                        try:
                            delay = max(delay, float(item))
                        except (TypeError, ValueError):
                            try:
                                delay = max(
                                    delay, parsedate_to_datetime(item).timestamp() - now
                                )
                            except (TypeError, ValueError, OverflowError):
                                pass
                    elif isinstance(item, dict):
                        delay = max(delay, retry_after(item))
                return delay

            wait = max(wait, retry_after(error))
            state.update(
                strikes=strikes,
                last_rate_limit_at=now,
                cooldown_until=max(state.get("cooldown_until", 0), now + wait),
                probe_required=True,
                generation=state.get("generation", 0) + 1,
            )
        return True

    def note_quota_exhausted(self, error):
        if not is_quota_exhausted(error):
            return False
        if self.remote_endpoint:
            return self._remote_note("quota_exhausted", error)
        with self._state() as state:
            state.update(
                quota_blocked=True,
                quota_exhausted_at=_wall_time(),
                quota_reason="subscription_or_account_usage_limit",
                probe_required=True,
                generation=state.get("generation", 0) + 1,
            )
        return True

    def note_transport_failure(self, error):
        """Record individual transport faults without stopping unrelated requests.

        The caller retains its bounded same-session recovery. A stream failure
        alone is not evidence that the shared account is rate limited.
        """
        if not is_native_transport_failure(error):
            return False
        if self.remote_endpoint:
            # Keep the admission wire kind stable across SDK variants and live
            # gate versions. The exact SDK payload remains in the caller audit.
            return self._remote_note(
                "transport_failure",
                {
                    "codex_error_info": "response_stream_disconnected",
                    "message": error.get("message"),
                    "source_error": error,
                },
            )
        with self._state() as state:
            now = _wall_time()
            strikes = state.get("transport_strikes", 0)
            if now - state.get("last_transport_failure_at", 0) > 5:
                strikes += 1
            state.update(
                transport_strikes=strikes,
                last_transport_failure_at=now,
            )
        return True

    def _acquire(self):
        with self._state() as state:
            now = _wall_time()
            capacity = state.setdefault("capacity", self.capacity)
            effective = state.get("effective_capacity", capacity)
            if self._default_capacity:
                effective = max(effective, self.capacity)
                state["effective_capacity"] = effective
            if (
                not self._default_capacity
                and capacity != self.capacity
                and effective != self.capacity
            ):
                raise ValueError(
                    "Provider concurrency differs from the shared account policy"
                )
            if type(effective) is not int or effective < 1:
                raise ValueError("Invalid shared native admission capacity")
            self.effective_capacity = effective
            if state.get("quota_blocked"):
                return None, 1.0
            wait = (
                max(state.get("cooldown_until", 0), state.get("next_start_at", 0)) - now
            )
            if wait > 0:
                return None, min(wait, 1.0)
            probe = None
            if state.get("probe_required"):
                for index in range(min(4, effective)):
                    # Keep the original probe lock as the first recovery slot.
                    name = "probe.lock" if index == 0 else f"probe-{index:02d}.lock"
                    probe = self._try_lock(self.root / name)
                    if probe is not None:
                        break
                if probe is None:
                    return None, 0.25
            lane_handle = None
            for slot in range(effective):
                handle = self._try_lock(self.root / f"slot-{slot:02d}.lock")
                if handle is not None:
                    state["next_start_at"] = now + self.start_interval_s
                    return (handle, probe, state.get("generation", 0), lane_handle), 0
            if probe is not None:
                probe.close()
            if lane_handle is not None:
                lane_handle.close()
            return None, 0.25

    def _release(self, lease, succeeded):
        handle, probe, generation, lane_handle = lease
        try:
            if probe is not None:
                with self._state() as state:
                    if state.get("generation", 0) == generation:
                        if succeeded:
                            state.update(
                                probe_required=False,
                                strikes=0,
                                transport_strikes=0,
                                cooldown_until=0,
                            )
        finally:
            handle.close()
            if lane_handle is not None:
                lane_handle.close()
            if probe is not None:
                probe.close()

    @asynccontextmanager
    async def _remote_lease(self, audit_path):
        started_at = _wall_time()
        if audit_path:
            atomic_json(
                audit_path,
                {
                    "status": "waiting",
                    "started_at": started_at,
                    "capacity": 32,
                    "shared_hosts": True,
                },
            )
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(*self._remote_address()), timeout=10
            )
        except (OSError, TimeoutError) as error:
            from reweave.planner.providers.base import NativeTransportError

            raise NativeTransportError(
                "Shared admission endpoint unavailable before model submission",
                request_sent=False,
            ) from error
        record, succeeded = None, False
        try:
            writer.write(self._remote_message("lease"))
            await writer.drain()
            reply = await reader.readline()
            if not reply:
                from reweave.planner.providers.base import NativeTransportError

                raise NativeTransportError(
                    "Shared admission connection closed before model submission",
                    request_sent=False,
                )
            value = json.loads(reply)
            if value.get("status") != "admitted":
                raise RuntimeError("Shared admission refused a native request")
            record = {
                **value,
                "started_at": started_at,
                "admitted_at": _wall_time(),
                "wait_s": _wall_time() - started_at,
                "shared_hosts": True,
            }
            self.effective_capacity = value["capacity"]
            if audit_path:
                atomic_json(audit_path, record)
            yield self
            succeeded = True
        finally:
            try:
                if record is not None:
                    writer.write(json.dumps({"succeeded": succeeded}).encode() + b"\n")
                    await writer.drain()
                    await asyncio.wait_for(reader.readline(), timeout=10)
            finally:
                writer.close()
                await writer.wait_closed()
                if audit_path and record is not None:
                    atomic_json(
                        audit_path,
                        {
                            **record,
                            "status": "completed" if succeeded else "failed",
                            "closed_at": _wall_time(),
                        },
                    )

    @asynccontextmanager
    async def lease(self, audit_path=None):
        if self.remote_endpoint:
            async with self._remote_lease(audit_path):
                yield self
            return
        start, started_at = time.monotonic(), _wall_time()
        lease = None
        if audit_path:
            atomic_json(
                audit_path,
                {
                    "status": "waiting",
                    "started_at": started_at,
                    "capacity": self.capacity,
                },
            )
        while lease is None:
            lease, wait = self._acquire()
            if lease is None:
                await asyncio.sleep(wait)
        _, probe, _, _ = lease
        record = {
            "status": "admitted",
            "started_at": started_at,
            "admitted_at": _wall_time(),
            "wait_s": time.monotonic() - start,
            "capacity": self.effective_capacity,
            "recovery_probe": probe is not None,
        }
        succeeded = False
        try:
            if audit_path:
                atomic_json(audit_path, record)
            yield self
            succeeded = True
        finally:
            self._release(lease, succeeded)
            if audit_path:
                atomic_json(
                    audit_path,
                    {
                        **record,
                        "status": "completed" if succeeded else "failed",
                        "closed_at": _wall_time(),
                    },
                )
