"""Real historical files, fake SFTP; never contact SSH, models or environments."""

import asyncio
import io
import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.robodojo_demo import simulated_observation
from reweave.benchmarks.robodojo.observation_service import ObservationService
from reweave.evolution.evidence import server as sensors


class FakeSFTP:
    def __init__(self):
        self.opened = []
        self.closed = False
        self.timeout = None
        self.truncate = False
        self.prefetched = []
        self.prefetch_error = None

    def get_channel(self):
        return self

    def settimeout(self, timeout):
        self.timeout = timeout

    def lstat(self, path):
        return Path(path).lstat()

    def open(self, path, mode):
        self.opened.append(path)
        data = Path(path).read_bytes()
        owner = self

        class RemoteFile(io.BytesIO):
            def prefetch(self, *, file_size, max_concurrent_requests):
                owner.prefetched.append((path, file_size, max_concurrent_requests))
                if owner.prefetch_error is not None:
                    raise owner.prefetch_error

        return RemoteFile(data[:-1] if self.truncate else data)

    def close(self):
        self.closed = True


@pytest.fixture
def historical(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    store = ObservationService(remote, service_instance_id="retained-test-episode")
    compact = store.publish(simulated_observation(), 0)
    ref = compact["observation_ref"]
    retained = store.retain_observation(ref)
    episode = tmp_path / "episode"
    episode.mkdir()
    config = tmp_path / "connection.json"
    config.write_text(
        json.dumps(
            {
                "identity_file": "HOST-ONLY-private-key",
                "hostname": "HOST-ONLY-host",
            }
        )
    )
    descriptor = {
        "archive_path": retained["archive_path"],
        "metadata_path": retained["metadata_path"],
        "archive_bytes": Path(retained["archive_path"]).stat().st_size,
        "metadata_bytes": Path(retained["metadata_path"]).stat().st_size,
    }
    manifest = {
        "schema_version": 1,
        "transport_config_path": str(config),
        "transport_python": sys.executable,
        "remote_observation_root": str(store.directory),
        "snapshots": {ref: descriptor},
    }
    manifest_path = episode / "server_evidence.json"
    manifest_path.write_text(json.dumps(manifest))
    sftp = FakeSFTP()
    connections = []

    def connector(config, timeout):
        connections.append((config, timeout))
        return SimpleNamespace(open_sftp=lambda: sftp, close=lambda: None)

    def fetch(request):
        return sensors.fetch_files(request, connector=connector)

    monkeypatch.setattr(sensors, "_run_fetch_worker", fetch)
    return SimpleNamespace(
        episode=episode,
        ref=ref,
        store=store,
        descriptor=descriptor,
        manifest=manifest,
        manifest_path=manifest_path,
        cache=tmp_path / "cache",
        sftp=sftp,
        connections=connections,
    )


def update_manifest(case):
    case.manifest_path.write_text(json.dumps(case.manifest))


def assert_no_completed_cache(case):
    assert not list(case.cache.rglob("rgbd.npz"))
    assert not list(case.cache.rglob(".sensor-download-*"))


def test_selected_historical_snapshot_is_exact_readonly_cached_and_private(historical):
    case = historical
    result = sensors.read_sensor_snapshot(
        case.episode, "episode-1", case.ref, case.cache
    )
    assert result["cache_hit"] is False and result["downloaded_bytes"] > 0
    assert case.sftp.opened == [
        case.descriptor["archive_path"],
        case.descriptor["metadata_path"],
    ]
    assert case.sftp.prefetched == [
        (case.descriptor["archive_path"], case.descriptor["archive_bytes"], 16),
        (case.descriptor["metadata_path"], case.descriptor["metadata_bytes"], 16),
    ]
    for key, original in [
        ("archive_path", case.descriptor["archive_path"]),
        ("metadata_path", case.descriptor["metadata_path"]),
    ]:
        assert Path(result[key]).read_bytes() == Path(original).read_bytes()
        assert stat.S_IMODE(Path(result[key]).stat().st_mode) == 0o444
    with np.load(result["archive_path"], allow_pickle=False) as archive:
        assert np.array_equal(
            archive["cam_head__depth"],
            simulated_observation()["vision"]["cam_head"]["depth"],
        )
    encoded = json.dumps(result)
    assert "HOST-ONLY" not in encoded and "transport_config_path" not in encoded
    again = sensors.read_sensor_snapshot(
        case.episode, "episode-1", case.ref, case.cache
    )
    assert again["cache_hit"] is True and again["downloaded_bytes"] == 0
    assert again["download_elapsed_s"] == 0
    assert result["download_elapsed_s"] > 0
    assert again["archive_path"] == result["archive_path"]
    assert len(case.connections) == 1 and len(case.sftp.opened) == 2
    assert len(case.sftp.prefetched) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_ref",
        "traversal",
        "cross_ref_path",
        "extra_manifest",
        "extra_descriptor",
        "too_large_archive",
        "too_large_metadata",
    ],
)
def test_host_manifest_rejects_unadmitted_paths_or_fields_before_network(
    historical, mutation
):
    case = historical
    ref = case.ref
    if mutation == "unknown_ref":
        ref = ref.replace(":0:", ":1:")
    elif mutation == "traversal":
        case.descriptor["archive_path"] = (
            case.manifest["remote_observation_root"] + "/../rgbd.npz"
        )
    elif mutation == "cross_ref_path":
        case.descriptor["archive_path"] = (
            case.manifest["remote_observation_root"] + "/" + "a" * 32 + "/rgbd.npz"
        )
    elif mutation == "extra_manifest":
        case.manifest["command"] = "forbidden"
    elif mutation == "extra_descriptor":
        case.descriptor["command"] = "forbidden"
    elif mutation == "too_large_archive":
        case.descriptor["archive_bytes"] = sensors.MAX_ARCHIVE_BYTES + 1
    else:
        case.descriptor["metadata_bytes"] = sensors.MAX_METADATA_BYTES + 1
    update_manifest(case)
    with pytest.raises(ValueError):
        sensors.read_sensor_snapshot(case.episode, "episode-1", ref, case.cache)
    assert not case.connections
    assert_no_completed_cache(case)


@pytest.mark.parametrize(
    "kind",
    [
        "remote_file_symlink",
        "remote_parent_symlink",
        "local_manifest_symlink",
        "cache_symlink",
    ],
)
def test_sensor_fetch_never_follows_links(historical, tmp_path, kind):
    case = historical
    if kind == "remote_file_symlink":
        original = Path(case.descriptor["archive_path"])
        moved = original.with_suffix(".saved")
        original.rename(moved)
        original.symlink_to(moved)
    elif kind == "remote_parent_symlink":
        original = case.store.directory
        moved = original.with_name("saved")
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
    elif kind == "local_manifest_symlink":
        moved = case.manifest_path.with_suffix(".saved")
        case.manifest_path.rename(moved)
        case.manifest_path.symlink_to(moved)
    else:
        target = tmp_path / "other-cache"
        target.mkdir()
        case.cache.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        sensors.read_sensor_snapshot(case.episode, "episode-1", case.ref, case.cache)
    assert_no_completed_cache(case)


@pytest.mark.parametrize(
    "corruption",
    [
        "cross_ref",
        "extra_metadata",
        "dtype",
        "shape",
        "extra_npz_array",
        "missing_npz_array",
        "unpacked_limit",
    ],
)
def test_original_metadata_and_npz_must_agree_before_publication(
    historical, corruption, monkeypatch
):
    case = historical
    metadata_path = Path(case.descriptor["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    if corruption == "cross_ref":
        metadata["observation_ref"] = case.ref.replace(":0:", ":1:")
    elif corruption == "extra_metadata":
        metadata["connection"] = "not-original"
    elif corruption == "dtype":
        metadata["vision"]["cam_head"]["arrays"]["depth"]["dtype"] = "float32"
    elif corruption == "shape":
        metadata["vision"]["cam_head"]["arrays"]["depth"]["shape"] = [99, 99]
    elif corruption == "unpacked_limit":
        monkeypatch.setattr(sensors, "MAX_UNPACKED_BYTES", 1)
    else:
        archive_path = Path(case.descriptor["archive_path"])
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = dict(archive)
        if corruption == "extra_npz_array":
            arrays["private_extra"] = np.zeros(1)
        else:
            arrays.pop("cam_head__depth")
        np.savez_compressed(archive_path, **arrays)
        case.descriptor["archive_bytes"] = archive_path.stat().st_size
    metadata_path.write_text(json.dumps(metadata))
    case.descriptor["metadata_bytes"] = metadata_path.stat().st_size
    update_manifest(case)
    with pytest.raises(ValueError):
        sensors.read_sensor_snapshot(case.episode, "episode-1", case.ref, case.cache)
    assert_no_completed_cache(case)


def test_interrupted_download_does_not_leave_a_completed_original(historical):
    case = historical
    case.sftp.truncate = True
    with pytest.raises(ValueError, match="ended before"):
        sensors.read_sensor_snapshot(case.episode, "episode-1", case.ref, case.cache)
    assert case.sftp.closed
    assert_no_completed_cache(case)


def test_worker_timeout_cleans_partial_cache_without_network(historical, monkeypatch):
    case = historical

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 120)

    monkeypatch.setattr(subprocess, "run", timeout)
    # Restore the real worker wrapper; the subprocess itself remains synthetic.
    monkeypatch.undo()
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(TimeoutError, match="time limit"):
        sensors.read_sensor_snapshot(case.episode, "episode-1", case.ref, case.cache)
    assert_no_completed_cache(case)


def test_worker_error_cannot_disclose_authentication_details(tmp_path, monkeypatch):
    request = {"manifest": {"transport_python": sys.executable}}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout=json.dumps({"error_type": "AuthenticationException"}),
            stderr="PRIVATE KEY /secret/key on secret-host",
        ),
    )
    with pytest.raises(RuntimeError, match="AuthenticationException") as error:
        sensors._run_fetch_worker(request)
    assert "PRIVATE" not in str(error.value) and "secret" not in str(error.value)


def test_author_tool_resolves_only_admitted_handle_and_registers_originals(
    historical, tmp_path
):
    from reweave.evolution.authoring.tools import AuthorDevelopmentTools
    from tests.unit_tests.harness.evolution.test_bundle_author import workspace

    case = historical
    author_root = tmp_path / "author"
    author_root.mkdir()
    root = workspace(author_root)
    service = AuthorDevelopmentTools(
        root,
        author_root / "audit/development.jsonl",
        evidence_roots={"episode-1": case.episode},
    )
    handle = service.references.identity("trace", "episode-1")
    result = asyncio.run(
        service.model_call(
            "read_sensor_snapshot",
            {"evidence_id": handle, "observation_ref": case.ref},
            case_results=[],
        )
    )
    assert result["originals"]["archive"].startswith("file:")
    metadata = asyncio.run(
        service.model_call(
            "read_original",
            {"reference": result["originals"]["metadata"]},
            case_results=[],
        )
    )
    assert metadata["value"]["observation_ref"] == case.ref
    assert "HOST-ONLY" not in json.dumps(result)
    with pytest.raises(ValueError, match="only evidence_id"):
        service._model_arguments(
            "read_sensor_snapshot",
            {
                "evidence_id": handle,
                "observation_ref": case.ref,
                "host": "not-admitted",
            },
        )
    unadmitted = service.references.identity("trace", "foreign-episode")
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(
            service.model_call(
                "read_sensor_snapshot",
                {"evidence_id": unadmitted, "observation_ref": case.ref},
                case_results=[],
            )
        )


def test_trace_navigation_marks_remote_originals_without_false_missing_images(
    historical,
):
    from reweave.evolution.evidence.trace import _Episode, _observations

    case = historical
    state = {
        "steps": [
            {
                "step_idx": 0,
                "state": {},
                "artifacts": ["cam_head.png"],
                "extras": {"observation_ref": case.ref},
            }
        ]
    }
    (case.episode / "states.json").write_text(json.dumps(state))
    # Minimal published inventory for this navigation-only test.
    from reweave.evolution.evidence.records import EpisodeEvidence, inventory_episode

    evidence = EpisodeEvidence(
        evidence_id="episode-1", files=inventory_episode(case.episode)
    )
    rows, gaps = _observations(
        _Episode(case.episode.parent, evidence, case.episode), [], []
    )
    assert rows[0]["sensor_snapshot"]["read_with"] == "read_sensor_snapshot"
    assert rows[0]["sensor_snapshot"]["observation_ref"] == case.ref
    assert rows[0]["images"][0]["source"] is None
    assert rows[0]["images"][0]["storage"] == "server_original_not_materialized"
    assert not any(gap["kind"] == "declared_image_missing" for gap in gaps)
    assert "transport" not in json.dumps(rows)


def test_prefetch_failure_never_falls_back_retries_or_publishes_cache(historical):
    case = historical
    case.sftp.prefetch_error = TypeError("prefetch API unavailable")
    with pytest.raises(TypeError, match="prefetch API unavailable"):
        sensors.read_sensor_snapshot(case.episode, "episode-1", case.ref, case.cache)
    assert len(case.connections) == 1
    assert case.sftp.opened == [case.descriptor["archive_path"]]
    assert len(case.sftp.prefetched) == 1 and case.sftp.closed
    assert_no_completed_cache(case)


def test_storage_session_reuses_one_authenticated_client_for_many_snapshots(
    historical, tmp_path
):
    requests = []
    for i in range(3):
        staging = tmp_path / f"batch-{i}"
        staging.mkdir()
        requests.append(
            {
                "manifest": historical.manifest,
                "observation_ref": historical.ref,
                "staging_directory": str(staging),
            }
        )
    closed = []
    clients = []

    def connect(config, timeout):
        client = SimpleNamespace(
            open_sftp=lambda: FakeSFTP(), close=lambda: closed.append(1)
        )
        clients.append(client)
        return client

    output = io.StringIO()
    sensors._serve_fetch_session(
        io.StringIO("".join(json.dumps(r) + "\n" for r in requests)),
        output,
        connector=connect,
    )
    assert len(clients) == len(closed) == 1
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {"complete": True}
    ] * 3
    for request in requests:
        assert (Path(request["staging_directory"]) / "rgbd.npz").read_bytes() == Path(
            historical.descriptor["archive_path"]
        ).read_bytes()


def test_storage_session_disconnect_replaces_connection_only_for_next_read(
    historical, tmp_path
):
    clients, closed = [], []
    reads = 0

    class InterruptedSFTP(FakeSFTP):
        def open(self, path, mode):
            nonlocal reads
            reads += 1
            if reads == 3:
                raise EOFError("connection dropped on next snapshot")
            return super().open(path, mode)

    def connect(config, timeout):
        number = len(clients)
        client = SimpleNamespace(
            open_sftp=lambda: InterruptedSFTP(), close=lambda: closed.append(number)
        )
        clients.append(client)
        return client

    requests = []
    for i in range(3):
        staging = tmp_path / f"resume-{i}"
        staging.mkdir()
        requests.append(
            {
                "manifest": historical.manifest,
                "observation_ref": historical.ref,
                "staging_directory": str(staging),
            }
        )
    output = io.StringIO()
    sensors._serve_fetch_session(
        io.StringIO("".join(json.dumps(r) + "\n" for r in requests)),
        output,
        connector=connect,
    )
    assert len(clients) == 2 and closed == [0, 1]
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {"complete": True},
        {"error_type": "EOFError"},
        {"complete": True},
    ]
    assert (tmp_path / "resume-0/rgbd.npz").is_file()
    assert (tmp_path / "resume-2/rgbd.npz").is_file()


def test_host_fetch_session_is_lazy_and_owns_one_worker(historical, monkeypatch):
    processes = []
    original = subprocess.Popen

    def start(*args, **kwargs):
        # A real line-oriented worker process, without SSH or remote writes.
        process = original(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys,json; [print(json.dumps({'complete':True}),flush=True) for line in sys.stdin]",
            ],
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start)
    request = {"manifest": historical.manifest}
    with sensors.sensor_fetch_session(historical.manifest) as _:
        assert not processes
        # The fixture replaces _run_fetch_worker for snapshot tests; use the
        # context's actual session to test pipe lifecycle independently.
        session = sensors._FETCH_SESSION.get()
        session.fetch(request)
        session.fetch(request)
        assert len(processes) == 1 and processes[0].poll() is None
    assert processes[0].poll() == 0
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_sensor_pointer_errors_identify_the_correct_evidence_route(tmp_path):
    with pytest.raises(ValueError, match="invalid server observation reference"):
        sensors.read_sensor_snapshot(
            tmp_path, "trace:001", "file:001#/steps/11", tmp_path / "cache"
        )
    with pytest.raises(ValueError, match="no admitted sensor snapshot archive"):
        sensors.read_sensor_snapshot(
            tmp_path, "trace:001", "episode:11:" + "a" * 32, tmp_path / "cache"
        )
    assert not (tmp_path / "cache").exists()
