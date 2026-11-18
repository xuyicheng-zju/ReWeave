"""Host-only, request-bound download of one immutable server sensor snapshot."""

from __future__ import annotations

import fcntl
import io
import json
import os
import re
import select
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_UNPACKED_BYTES = 256 * 1024 * 1024
FETCH_TIMEOUT_S = 120
_FETCH_SESSION = ContextVar("sensor_fetch_session", default=None)
_MANIFEST_FIELDS = {
    "schema_version",
    "transport_config_path",
    "transport_python",
    "remote_observation_root",
    "snapshots",
}
_SNAPSHOT_FIELDS = {"archive_path", "metadata_path", "archive_bytes", "metadata_bytes"}


def _regular_local(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError(
            "sensor evidence requires an existing canonical regular local file"
        )
    return path


def _reference_parts(reference):
    if not isinstance(reference, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}:(?:0|[1-9][0-9]*):[0-9a-f]{32}", reference
    ):
        raise ValueError("invalid server observation reference")
    return reference.split(":")


def _descriptor(manifest, reference):
    _, _, identity = _reference_parts(reference)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_FIELDS
        or manifest["schema_version"] != 1
    ):
        raise ValueError("invalid Host server evidence manifest")
    if (
        not isinstance(manifest["snapshots"], dict)
        or reference not in manifest["snapshots"]
    ):
        raise ValueError("observation reference is not admitted by this episode")
    root = PurePosixPath(manifest["remote_observation_root"])
    if (
        not root.is_absolute()
        or ".." in root.parts
        or str(root) != manifest["remote_observation_root"]
        or root.name != "server_observations"
    ):
        raise ValueError("invalid remote observation root")
    item = manifest["snapshots"][reference]
    if not isinstance(item, dict) or set(item) != _SNAPSHOT_FIELDS:
        raise ValueError("invalid admitted sensor descriptor")
    for name, filename, maximum in [
        ("archive", "rgbd.npz", MAX_ARCHIVE_BYTES),
        ("metadata", "metadata.json", MAX_METADATA_BYTES),
    ]:
        path = item[name + "_path"]
        if not isinstance(path, str) or path != str(root / identity / filename):
            raise ValueError("sensor path escaped its admitted observation")
        size = item[name + "_bytes"]
        if type(size) is not int or not 0 < size <= maximum:
            raise ValueError("sensor evidence size exceeds its bound")
    return item


@contextmanager
def _ssh_handshake(timeout_s):
    """Serialize Host sensor handshakes without serializing SFTP transfers.

    Concurrent episode closeout must not exhaust sshd's unauthenticated
    connection allowance. This never retries sensor reads or environment RPCs.
    """
    path = (
        Path(tempfile.gettempdir()).resolve() / f"reweave-sensor-ssh-{os.getuid()}.lock"
    )
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Host sensor handshake lock is not owned by this user")
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Host sensor SSH handshake queue timed out"
                    ) from None
                time.sleep(0.05)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Host sensor SSH handshake queue timed out")
        yield remaining
    finally:
        os.close(fd)


def _connect(config, timeout_s):
    import paramiko

    with _ssh_handshake(timeout_s) as remaining:
        deadline = time.monotonic() + remaining
        for attempt in range(4):
            client = paramiko.SSHClient()
            try:
                client.load_host_keys(config["known_hosts"])
                client.set_missing_host_key_policy(paramiko.RejectPolicy())
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Host sensor SSH handshake timed out")
                client.connect(
                    config["hostname"],
                    port=config["port"],
                    username=config["username"],
                    key_filename=config["identity_file"],
                    look_for_keys=False,
                    allow_agent=False,
                    timeout=remaining,
                    auth_timeout=remaining,
                    banner_timeout=remaining,
                )
                return client
            except BaseException as error:
                client.close()
                delay = 2**attempt
                # The server has not sent its SSH banner: no authentication,
                # SFTP read or environment operation has been submitted.
                if (
                    type(error) is not paramiko.SSHException
                    or not str(error).startswith("Error reading SSH protocol banner")
                    or attempt == 3
                    or deadline - time.monotonic() <= delay
                ):
                    raise
                time.sleep(delay)


def fetch_files(request, *, connector=None, client=None):
    """Worker implementation; SFTP reads only the two Host-admitted originals."""
    manifest, reference = request["manifest"], request["observation_ref"]
    item = _descriptor(manifest, reference)
    config = json.loads(_regular_local(manifest["transport_config_path"]).read_text())
    output = Path(request["staging_directory"])
    if not output.is_absolute() or output.resolve() != output or not output.is_dir():
        raise ValueError("invalid Host download staging directory")
    started = time.monotonic()
    owned = client is None
    if owned:
        client = (connector or _connect)(config, min(20, FETCH_TIMEOUT_S))
    try:
        sftp = client.open_sftp()
        try:
            sftp.get_channel().settimeout(20)
            for name, filename in [
                ("archive", "rgbd.npz"),
                ("metadata", "metadata.json"),
            ]:
                path = PurePosixPath(item[name + "_path"])
                parent = PurePosixPath("/")
                for component in path.parts[1:-1]:
                    parent /= component
                    mode = sftp.lstat(str(parent)).st_mode
                    if not stat.S_ISDIR(mode):
                        raise ValueError(
                            "remote sensor parent is not a regular directory"
                        )
                info = sftp.lstat(str(path))
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size != item[name + "_bytes"]
                ):
                    raise ValueError(
                        "remote sensor original type or byte count changed"
                    )
                remaining = info.st_size
                with (
                    sftp.open(str(path), "rb") as source,
                    (output / filename).open("xb") as target,
                ):
                    if time.monotonic() - started >= FETCH_TIMEOUT_S:
                        raise TimeoutError("sensor download exceeded its time limit")
                    # Paramiko 5 supports bounded outstanding 32 KiB requests.
                    # Avoid one WAN round trip per sequential read without retries.
                    source.prefetch(file_size=info.st_size, max_concurrent_requests=16)
                    while remaining:
                        if time.monotonic() - started >= FETCH_TIMEOUT_S:
                            raise TimeoutError(
                                "sensor download exceeded its time limit"
                            )
                        chunk = source.read(min(262144, remaining))
                        if not chunk:
                            raise ValueError(
                                "remote sensor original ended before its declared size"
                            )
                        if len(chunk) > remaining:
                            raise ValueError(
                                "remote sensor original exceeds its declared size"
                            )
                        target.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise ValueError(
                            "remote sensor original exceeds its declared size"
                        )
        finally:
            sftp.close()
    finally:
        if owned:
            client.close()


def _serve_fetch_session(source, target, *, connector=None):
    """One authenticated connection for sequential immutable snapshot reads."""
    client = None
    config = None
    try:
        for line in source:
            try:
                request = json.loads(line)
                _descriptor(request["manifest"], request["observation_ref"])
                current = json.loads(
                    _regular_local(
                        request["manifest"]["transport_config_path"]
                    ).read_text()
                )
                if client is None or current != config:
                    if client is not None:
                        client.close()
                    client = None
                    config = current
                    client = (connector or _connect)(config, min(20, FETCH_TIMEOUT_S))
                fetch_files(request, client=client)
                result = {"complete": True}
            except Exception as error:
                if client is not None:
                    client.close()
                    client = None
                result = {"error_type": type(error).__name__}
            target.write(json.dumps(result) + "\n")
            target.flush()
    finally:
        if client is not None:
            client.close()


class _FetchSession:
    def __init__(self, manifest):
        self.manifest = manifest
        self.process = None

    def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
            finally:
                process.stdout.close()

    def fetch(self, request):
        python = Path(self.manifest["transport_python"])
        if not python.is_absolute() or not python.is_file():
            raise ValueError("Host sensor transport interpreter is unavailable")
        try:
            if self.process is None:
                self.process = subprocess.Popen(
                    [
                        str(python),
                        str(Path(__file__).resolve()),
                        "--fetch-session-worker",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], FETCH_TIMEOUT_S)[0]:
                raise TimeoutError("server sensor fetch exceeded its time limit")
            result = json.loads(self.process.stdout.readline())
        except (OSError, ValueError, TimeoutError) as error:
            self.close()
            if isinstance(error, TimeoutError):
                raise
            raise RuntimeError("server sensor fetch failed (EOFError)") from error
        if not isinstance(result, dict):
            self.close()
            raise RuntimeError("server sensor fetch failed (TransportFailure)")
        if result.get("complete") is not True:
            kind = result.get("error_type", "TransportFailure")
            if not isinstance(kind, str) or not re.fullmatch(r"[A-Za-z]+", kind):
                kind = "TransportFailure"
            raise RuntimeError(f"server sensor fetch failed ({kind})")


@contextmanager
def sensor_fetch_session(manifest):
    """Lazily reuse SSH for one storage closeout; never share action ownership."""
    session = _FetchSession(manifest)
    token = _FETCH_SESSION.set(session)
    try:
        yield
    finally:
        _FETCH_SESSION.reset(token)
        session.close()


def _run_fetch_worker(request):
    session = _FETCH_SESSION.get()
    if session is not None and session.manifest == request["manifest"]:
        return session.fetch(request)
    python = Path(request["manifest"]["transport_python"])
    if not python.is_absolute() or not python.is_file():
        raise ValueError("Host sensor transport interpreter is unavailable")
    try:
        result = subprocess.run(
            [str(python), str(Path(__file__).resolve()), "--fetch-worker"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=FETCH_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("server sensor fetch exceeded its time limit") from exc
    if result.returncode:
        # Never forward Paramiko errors, hostnames, authentication filenames or
        # worker stderr into the model-visible development response.
        try:
            error_type = json.loads(result.stdout)["error_type"]
        except (ValueError, KeyError):
            error_type = "TransportFailure"
        if not isinstance(error_type, str) or not re.fullmatch(
            r"[A-Za-z]+", error_type
        ):
            error_type = "TransportFailure"
        raise RuntimeError(f"server sensor fetch failed ({error_type})")


def _validate_snapshot(directory, reference, descriptor):
    from reweave.benchmarks.robodojo.access import CAMERAS
    from reweave.benchmarks.robodojo.env_client import decode_sensor_snapshot

    archive = _regular_local(directory / "rgbd.npz")
    metadata_path = _regular_local(directory / "metadata.json")
    if (
        archive.stat().st_size != descriptor["archive_bytes"]
        or metadata_path.stat().st_size != descriptor["metadata_bytes"]
    ):
        raise ValueError("downloaded sensor size differs from the admitted original")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or set(metadata) != {
        "observation_ref",
        "service_instance_id",
        "physical_step",
        "instruction",
        "state",
        "vision",
    }:
        raise ValueError("sensor metadata contains unexpected fields")
    sid, step, _ = _reference_parts(reference)
    if (
        metadata.get("observation_ref") != reference
        or metadata.get("service_instance_id") != sid
        or metadata.get("physical_step") != int(step)
    ):
        raise ValueError("sensor metadata refers to another episode or snapshot")
    if set(metadata.get("vision", {})) != set(CAMERAS):
        raise ValueError("sensor metadata must describe all three original cameras")
    expected = set()
    for camera, value in metadata["vision"].items():
        if not isinstance(value, dict) or set(value) != {
            "shape",
            "color_dtype",
            "has_depth",
            "intrinsic_matrix",
            "extrinsic_matrix",
            "arrays",
        }:
            raise ValueError("sensor camera metadata contains unexpected fields")
        arrays = value.get("arrays", {})
        if not {"color", "intrinsic_matrix", "extrinsic_matrix"} <= set(arrays) or set(
            arrays
        ) - {
            "color",
            "depth",
            "distance_to_image_plane",
            "intrinsic_matrix",
            "extrinsic_matrix",
        }:
            raise ValueError("sensor metadata has unknown or missing image arrays")
        for field, description in arrays.items():
            if (
                set(description) != {"key", "shape", "dtype"}
                or description["key"] != camera + "__" + field
            ):
                raise ValueError(
                    "sensor metadata array identity differs from its camera"
                )
            expected.add(description["key"] + ".npy")
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as container:
        entries = container.infolist()
        if (
            len(entries) != len(expected)
            or {entry.filename for entry in entries} != expected
            or sum(entry.file_size for entry in entries) > MAX_UNPACKED_BYTES
        ):
            raise ValueError(
                "sensor archive contains extra arrays or exceeds unpacked limit"
            )
    decode_sensor_snapshot(
        {
            "observation_ref": reference,
            "metadata": metadata,
            "archive": raw,
        }
    )
    return metadata


def read_sensor_snapshot(episode_root, evidence_id, observation_ref, cache_root):
    """Materialize one admitted historical original without exposing connection data."""
    started = time.monotonic()
    if not isinstance(evidence_id, str) or not re.fullmatch(
        r"[A-Za-z0-9:_-]+", evidence_id
    ):
        raise ValueError("invalid admitted evidence identity")
    episode_root = Path(episode_root)
    _reference_parts(observation_ref)
    from reweave.evolution.evidence.sensor_retention import read_retention

    retention = read_retention(episode_root)
    if retention and observation_ref not in retention["raw_rgbd_observations"]:
        raise ValueError(
            "Exact RGB-D was not retained for this observation. Read the original model RGB or the explicitly indexed state/video evidence; video is not raw RGB-D."
        )
    manifest_path = episode_root / "server_evidence.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        raise ValueError(
            "This episode has no admitted sensor snapshot archive. Use read_trace "
            "and read_original for its indexed observation files. A file handle or "
            "trajectory JSON pointer is not a sensor snapshot ID."
        )
    manifest_path = _regular_local(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    descriptor = _descriptor(manifest, observation_ref)
    _, _, identity = _reference_parts(observation_ref)
    if retention and retention.get("local_sensor_originals"):
        destination = episode_root / "sensor_originals" / identity
        metadata = _validate_snapshot(destination, observation_ref, descriptor)
        return {
            "observation_ref": observation_ref,
            "cache_hit": True,
            "archive_path": str(destination / "rgbd.npz"),
            "metadata_path": str(destination / "metadata.json"),
            "metadata": metadata,
            "archive_bytes": descriptor["archive_bytes"],
            "metadata_bytes": descriptor["metadata_bytes"],
            "downloaded_bytes": 0,
            "download_elapsed_s": 0.0,
            "elapsed_s": time.monotonic() - started,
            "meaning": "Exact original RGB-D stored with this local episode. No server access or duplicate cache. Load archive_path with numpy.load(..., allow_pickle=False); metadata contains calibration.",
        }
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("sensor cache cannot follow links")
    parent = root / evidence_id
    parent.mkdir(exist_ok=True)
    if parent.resolve() != parent:
        raise ValueError("sensor cache episode cannot follow links")
    destination = parent / identity
    hit = destination.exists()
    download_elapsed_s = 0.0
    if hit:
        if destination.is_symlink() or destination.resolve() != destination:
            raise ValueError("sensor cache cannot follow links")
        binding = json.loads(
            _regular_local(destination / "cache_binding.json").read_text()
        )
        if binding != {"observation_ref": observation_ref, "descriptor": descriptor}:
            raise ValueError(
                "cached sensor original differs from its admitted reference"
            )
        metadata = _validate_snapshot(destination, observation_ref, descriptor)
    else:
        staging = Path(tempfile.mkdtemp(prefix=".sensor-download-", dir=parent))
        try:
            download_started = time.monotonic()
            _run_fetch_worker(
                {
                    "manifest": manifest,
                    "observation_ref": observation_ref,
                    "staging_directory": str(staging),
                }
            )
            download_elapsed_s = time.monotonic() - download_started
            metadata = _validate_snapshot(staging, observation_ref, descriptor)
            if {path.name for path in staging.iterdir()} != {
                "rgbd.npz",
                "metadata.json",
            }:
                raise ValueError("sensor download contains unexpected files")
            (staging / "cache_binding.json").write_text(
                json.dumps(
                    {
                        "observation_ref": observation_ref,
                        "descriptor": descriptor,
                    }
                )
                + "\n"
            )
            for path in staging.iterdir():
                if not path.is_file() or path.is_symlink():
                    raise ValueError("sensor staging contains unexpected entries")
                path.chmod(0o444)
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {
        "observation_ref": observation_ref,
        "cache_hit": hit,
        "archive_path": str(destination / "rgbd.npz"),
        "metadata_path": str(destination / "metadata.json"),
        "metadata": metadata,
        "archive_bytes": descriptor["archive_bytes"],
        "metadata_bytes": descriptor["metadata_bytes"],
        "downloaded_bytes": 0
        if hit
        else descriptor["archive_bytes"] + descriptor["metadata_bytes"],
        "download_elapsed_s": download_elapsed_s,
        "elapsed_s": time.monotonic() - started,
        "meaning": "Exact retained historical RGB-D, not live sensing or a task-success result. Load archive_path with numpy.load(..., allow_pickle=False); metadata describes camera array keys and calibration.",
    }


def _retire_worker(request):
    """Execute only the fixed post-episode file-retirement program over SSH."""
    import shlex

    manifest = request["manifest"]
    plan = request["plan"]
    for reference in plan["raw_rgbd_observations"]:
        _descriptor(manifest, reference)
    config = json.loads(_regular_local(manifest["transport_config_path"]).read_text())
    source = Path(__file__).with_name("sensor_retention.py").read_text()
    program = (
        source
        + "\nimport sys\nr=json.load(sys.stdin)\nprint(json.dumps(retire_unneeded_sensors(r['root'], r['plan'])))\n"
    )
    client = _connect(config, 20)
    try:
        stdin, stdout, stderr = client.exec_command(
            "/usr/bin/python3 -c " + shlex.quote(program), timeout=FETCH_TIMEOUT_S
        )
        stdin.write(
            json.dumps({"root": manifest["remote_observation_root"], "plan": plan})
        )
        stdin.channel.shutdown_write()
        result = stdout.read()
        stderr.read()
        if stdout.channel.recv_exit_status():
            raise RuntimeError(
                "post-episode server storage validation or retirement failed"
            )
        return json.loads(result)
    finally:
        client.close()


def retire_server_sensors(manifest, plan):
    """Host-only storage worker; no live environment or model RPC is available."""
    result = subprocess.run(
        [
            manifest["transport_python"],
            str(Path(__file__).resolve()),
            "--retire-worker",
        ],
        input=json.dumps({"manifest": manifest, "plan": plan}),
        text=True,
        capture_output=True,
        timeout=FETCH_TIMEOUT_S + 10,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "server sensor retirement did not complete; resume storage only"
        )
    return json.loads(result.stdout)


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--fetch-worker"]:
            fetch_files(json.load(sys.stdin))
        elif sys.argv[1:] == ["--fetch-session-worker"]:
            _serve_fetch_session(sys.stdin, sys.stdout)
        elif sys.argv[1:] == ["--retire-worker"]:
            print(json.dumps(_retire_worker(json.load(sys.stdin))))
        else:
            raise SystemExit("Host sensor worker requires its fixed invocation")
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}))
        raise SystemExit(1) from None
