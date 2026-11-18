"""Run owned LIBERO environments over SSH while native model sessions stay local.

Only observation/RPC bytes cross the forwarder. Physical RPCs are never retried.
The remote supervisor owns a process group and a device lease, and closes both
on stdin EOF, including loss of the SSH connection. Closure is independent of
episode outcome and recording finalization.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import fcntl
import json
import os
import select
import shlex
import signal
import socket
import socketserver
import sys
import threading
import time
import uuid
from pathlib import Path

from reweave.utils.persistence import atomic_json

_CONNECTIONS = {}
_CONNECTIONS_LOCK = threading.Lock()


def shared_connections(configuration):
    """Multiplex new channels over one authenticated connection per process."""
    from reweave.utils.ssh import ReconnectingSSH

    key = configuration["connection_file"]
    with _CONNECTIONS_LOCK:
        if key not in _CONNECTIONS:
            _CONNECTIONS[key] = ReconnectingSSH(lambda: connect(configuration))
        return _CONNECTIONS[key]


def connect(configuration):
    """Use an operator's existing authentication files without copying them."""
    import paramiko

    source = json.loads(Path(configuration["connection_file"]).read_text())
    client = paramiko.SSHClient()
    client.load_host_keys(source["known_hosts"])
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    # Environment bridges are separate processes. Serialize authentication, not
    # RPCs, so a burst of new lanes does not trip sshd's pre-auth connection cap.
    path = Path(configuration["connection_file"])
    with path.with_name(path.name + ".connect.lock").open("a") as handshake:
        fcntl.flock(handshake, fcntl.LOCK_EX)
        try:
            client.connect(
                source["hostname"],
                port=source["port"],
                username=source["username"],
                key_filename=source["identity_file"],
                look_for_keys=False,
                allow_agent=False,
                timeout=15,
                auth_timeout=20,
                banner_timeout=20,
            )
        except BaseException:
            client.close()
            raise
    client.get_transport().set_keepalive(10)
    return client


def relay(left, right):
    """Forward each byte once; a broken physical connection is not replayed."""
    readers = [left, right]
    while readers:
        ready, _, _ = select.select(readers, [], [], 1)
        for source in ready:
            target = right if source is left else left
            chunk = source.recv(262144)
            if chunk:
                target.sendall(chunk)
            else:
                readers.remove(source)
                try:
                    if hasattr(target, "shutdown_write"):
                        target.shutdown_write()
                    else:
                        target.shutdown(socket.SHUT_WR)
                except OSError as error:
                    if error.errno not in (errno.ENOTCONN, errno.EPIPE, errno.EBADF):
                        raise


def stop_event(*, heartbeat_s=None):
    """Observe both an orderly stop and the owning parent's death."""
    stopping = threading.Event()
    last_heartbeat = [time.monotonic()]
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda *_: stopping.set())

    def watch():
        try:
            while os.read(sys.stdin.fileno(), 8192):
                last_heartbeat[0] = time.monotonic()
        finally:
            stopping.set()

    threading.Thread(target=watch, daemon=True).start()
    if heartbeat_s is not None:

        def expire():
            while not stopping.wait(1):
                if time.monotonic() - last_heartbeat[0] > heartbeat_s:
                    stopping.set()

        threading.Thread(target=expire, daemon=True).start()
    return stopping


def validate_request(configuration, identity, arguments, slot):
    """Restrict the supervisor to an allocated, explicit LIBERO condition."""
    from reweave.utils.gpu_slots import decode_slot

    if uuid.UUID(identity).hex != identity:
        raise ValueError("Invalid environment owner identity")
    allowed = {
        "--suite",
        "--task",
        "--seed",
        "--reset-state-id",
        "--max-episode-steps",
        "--cuda-device",
    }
    if len(arguments) != len(allowed) * 2 or set(arguments[::2]) != allowed:
        raise ValueError("Expected one explicit value for each environment argument")
    values = dict(zip(arguments[::2], arguments[1::2], strict=True))
    device = int(values["--cuda-device"])
    lanes = configuration.get("slots_per_gpu", 4)
    if type(lanes) is not int or not 1 <= lanes <= 16:
        raise ValueError("Invalid remote environment lane allocation")
    if type(slot) is not int or slot < 0:
        raise ValueError("Invalid environment slot identity")
    # Non-pooled legacy deployments can use GPU indices above three. The
    # extended layout is opted into by a larger explicit lane allocation.
    allocated_device, lane = decode_slot(slot) if lanes > 4 else divmod(slot, 4)
    if (
        device not in configuration["devices"]
        or allocated_device != device
        or lane >= lanes
    ):
        raise ValueError("Environment slot is outside the operator allocation")
    if values["--suite"] not in configuration["suites"]:
        raise ValueError("Suite is outside the experiment scope")
    if not 0 <= int(values["--task"]) < 10:
        raise ValueError("Invalid task index")
    if any(int(values[key]) < 0 for key in ("--seed", "--reset-state-id")):
        raise ValueError("Seed and reset identity must be nonnegative")
    if int(values["--max-episode-steps"]) <= 0:
        raise ValueError("A finite physical budget is required")


async def supervise(configuration, identity, arguments, slot):
    """Close the owned process group before publishing resource closure."""
    from reweave.evolution.infrastructure.processes import drain_process_group
    from reweave.utils.daemon import reserve_free_port

    validate_request(configuration, identity, arguments, slot)
    root = Path(configuration["remote_root"])
    directory = root / "owners" / identity
    directory.mkdir(parents=True, exist_ok=False)
    stopping = stop_event(heartbeat_s=45)
    lease_path = root / "slots" / f"slot-{slot:02d}.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state = {
                "service_instance_id": identity,
                "status": "exited",
                "slot": slot,
                "exit": {
                    "returncode": None,
                    "parent_closed": True,
                    "never_spawned": True,
                },
                "error": "remote device slot already owned",
            }
            atomic_json(directory / "owner.json", state)
            print(json.dumps(state), flush=True)
            raise
        reservation = reserve_free_port()
        command = [
            configuration["python"],
            "-u",
            "-m",
            "reweave.benchmarks.libero.env_server",
            *arguments,
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(reservation.port),
            "--parent-watch",
        ]
        state = {
            "service_instance_id": identity,
            "status": "starting",
            "slot": slot,
            "command": command,
            "started_at": time.time(),
        }
        atomic_json(directory / "owner.json", state)
        process = None
        try:
            if stopping.is_set():
                raise RuntimeError("Parent stopped before environment launch")
            with (directory / "environment.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=configuration["repository"],
                    env={
                        **os.environ,
                        "LIBERO_TYPE": "pro",
                        "MUJOCO_GL": "egl",
                        "PYOPENGL_PLATFORM": "egl",
                        "ROBOT_PLATFORM": "LIBERO",
                        "LIBERO_CONFIG_PATH": configuration["asset_config"],
                        "PYTHONPATH": configuration["repository"],
                        "REWEAVE_SERVICE_INSTANCE_ID": identity,
                        "REWEAVE_RPC_LISTEN_FD": str(reservation.fileno),
                    },
                    stdin=asyncio.subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    pass_fds=(reservation.fileno, lease.fileno()),
                )
                state.update(status="running", pid=process.pid, port=reservation.port)
                atomic_json(directory / "owner.json", state)
                reservation.close()
                print(json.dumps(state), flush=True)
                while process.returncode is None and not stopping.is_set():
                    await asyncio.sleep(0.2)
                process.stdin.close()
                await drain_process_group(process, terminate=stopping.is_set())
        finally:
            reservation.close()
            if process is not None:
                await drain_process_group(process, terminate=True)
                state.update(
                    status="exited",
                    exit={"returncode": process.returncode, "parent_closed": True},
                    closed_at=time.time(),
                )
                atomic_json(directory / "owner.json", state)
                print(json.dumps(state), flush=True)
            else:
                state.update(
                    status="exited",
                    exit={
                        "returncode": None,
                        "parent_closed": True,
                        "never_spawned": True,
                    },
                    closed_at=time.time(),
                )
                atomic_json(directory / "owner.json", state)
                print(json.dumps(state), flush=True)


class ForwardServer(socketserver.ThreadingTCPServer):
    """An episode or shared-service listener; failed channels stay failed."""

    daemon_threads = True
    # Rebind a closed owner's port while old TCP connections are in TIME_WAIT.
    # SO_REUSEPORT stays disabled: another live listener still owns its port.
    allow_reuse_address = True


def listener(
    configuration, remote_port, *, port=0, inherited_fd=None, connections=None
):
    """Create fresh SSH channels for new connections, never for a failed call."""
    import paramiko

    connections = connections or shared_connections(configuration)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            client = None
            channel = None
            try:
                client = connections.borrow()
                channel = client.get_transport().open_channel(
                    "direct-tcpip",
                    ("127.0.0.1", remote_port),
                    self.client_address,
                    timeout=15,
                )
                relay(self.request, channel)
            except (OSError, EOFError, paramiko.SSHException) as error:
                if client is not None:
                    transport = client.get_transport()
                    transport_failed = transport is None or not transport.is_active()
                    opening_failed = channel is None and not isinstance(
                        error, paramiko.ChannelException
                    )
                    if transport_failed or opening_failed:
                        connections.retire(client)
                print(f"RPC channel ended: {type(error).__name__}: {error}", flush=True)
            finally:
                if channel is not None:
                    channel.close()
                if client is not None:
                    connections.release(client)

    server = ForwardServer(("127.0.0.1", port), Handler, bind_and_activate=False)
    if inherited_fd is None:
        server.server_bind()
    else:
        server.socket.close()
        server.socket = socket.socket(fileno=inherited_fd)
        server.server_address = server.socket.getsockname()
    server.server_activate()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def remote_state(configuration, identity):
    """Read the original supervisor's receipt; never re-launch it."""
    path = str(Path(configuration["remote_root"]) / "owners" / identity / "owner.json")
    connections = shared_connections(configuration)
    client = connections.borrow()
    try:
        with client.open_sftp() as sftp, sftp.open(path, "rb") as stream:
            return json.load(stream)
    finally:
        if client.get_transport() is None or not client.get_transport().is_active():
            connections.retire(client)
        connections.release(client)


def reconcile_slots(configuration, root):
    """Reclaim dead local owners only after their original remote group closes."""
    import paramiko

    from reweave.evolution.infrastructure.remote_slots import settle_remote_slot

    for path in sorted((root / "scheduler/online").glob("slot-*.remote.json")):
        original = json.loads(path.read_text())
        if original["status"] == "exited":
            continue
        slot = original["slot"]
        with (path.parent / f"slot-{slot:02d}.lock").open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if json.loads(path.read_text()) != original:
                continue
            try:
                state = remote_state(configuration, original["service_instance_id"])
            except (OSError, EOFError, paramiko.SSHException):
                continue
            if state.get("status") == "exited":
                settle_remote_slot(path, state)
                atomic_json(
                    root
                    / "reports/resource_reconciliation"
                    / f"{original['service_instance_id']}.json",
                    {"original": original, "closed": state, "physical_replay": False},
                )


def bridge(configuration, arguments):
    """Bind a local ProcessDaemon to exactly one remote environment owner."""
    import paramiko

    from reweave.evolution.infrastructure.campaign_slots import campaign_root
    from reweave.evolution.infrastructure.remote_slots import (
        retain_remote_slot,
        settle_remote_slot,
    )
    from reweave.utils.ssh import ReconnectingSSH

    identity = os.environ["REWEAVE_SERVICE_INSTANCE_ID"]
    device = int(arguments[arguments.index("--cuda-device") + 1])
    slot = int(os.environ.get("REWEAVE_CAMPAIGN_SLOT", str(device * 4)))
    validate_request(configuration, identity, arguments, slot)
    output = Path.cwd()
    root = campaign_root(output)
    remote_output = str(Path(configuration["remote_root"]) / "owners" / identity)
    claim = retain_remote_slot(output, slot, identity, remote_output) if root else None
    stopping = stop_event()
    client = None
    channel = None
    server = None
    connections = None
    supervisor_requested = False
    try:
        client = connect(configuration)
        connections = ReconnectingSSH(lambda: connect(configuration))
        connections.client = client
        connections.borrow()  # Pin the supervisor while a forwarding route retires.
        command = [
            configuration["python"],
            "-u",
            "-m",
            "reweave.robots.remote_environment",
            "serve",
            "--config",
            configuration["remote_configuration"],
            "--identity",
            identity,
            "--slot",
            str(slot),
            "--",
            *arguments,
        ]
        command = (
            "cd "
            + shlex.quote(configuration["repository"])
            + " && exec "
            + shlex.join(command)
        )
        # Once exec_command begins, submission can be uncertain even if it raises.
        supervisor_requested = True
        stdin, stdout, stderr = client.exec_command(command, timeout=60)
        channel = stdout.channel
        line = stdout.readline()
        if not line:
            raise RuntimeError(
                "Remote environment failed before handshake: "
                + stderr.read(4096).decode()
            )
        state = json.loads(line)
        if (
            state.get("service_instance_id") != identity
            or state.get("status") != "running"
        ):
            raise ValueError("Remote environment handshake identity differs")
        atomic_json(output / "remote_environment.json", state)
        server = listener(
            configuration,
            state["port"],
            inherited_fd=int(os.environ["REWEAVE_RPC_LISTEN_FD"]),
            connections=connections,
        )
        channel.settimeout(5)
        next_heartbeat = 0
        while not stopping.wait(0.2):
            if channel.exit_status_ready() or not client.get_transport().is_active():
                break
            if time.monotonic() >= next_heartbeat:
                channel.sendall(b"heartbeat\n")
                next_heartbeat = time.monotonic() + 2
        stdin.close()
        channel.shutdown_write()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if client is not None:
            client.close()
        if connections is not None:
            connections.release(client)
            connections.close()
        # Authentication failed before any remote command could be submitted.
        # This is a known absence of physical work, not an unknown remote owner.
        if not supervisor_requested:
            state = {
                "service_instance_id": identity,
                "status": "exited",
                "slot": slot,
                "exit": {"parent_closed": True, "never_spawned": True},
                "submission": "not_submitted",
            }
            settle_remote_slot(claim, state)
            atomic_json(output / "remote_environment_closed.json", state)
        # A dropped connection initiates remote cleanup. Only its closed receipt
        # releases the durable claim; unavailable SSH leaves it quarantined.
        deadline = time.monotonic() + 90
        while supervisor_requested:
            try:
                state = remote_state(configuration, identity)
                if state.get("status") == "exited":
                    settle_remote_slot(claim, state)
                    atomic_json(output / "remote_environment_closed.json", state)
                    break
            except (OSError, EOFError, paramiko.SSHException):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Remote resource closure unverified; original owner remains quarantined"
                )
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "bridge", "forward"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--identity")
    parser.add_argument("--slot", type=int)
    args, remainder = parser.parse_known_args()
    configuration = json.loads(args.config.read_text())
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    if args.mode == "serve":
        asyncio.run(supervise(configuration, args.identity, remainder, args.slot))
    elif args.mode == "bridge":
        bridge(configuration, remainder)
    else:
        stopping = stop_event()
        servers = []
        try:
            for row in configuration["forwards"]:
                servers.append(
                    listener(configuration, row["remote"], port=row["local"])
                )
            stopping.wait()
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for connections in _CONNECTIONS.values():
                connections.close()


if __name__ == "__main__":
    main()
