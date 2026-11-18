"""Read-only proof construction for RoboDojo post-episode sensor retention."""

import json
from pathlib import Path

PHYSICAL = {"move_pose", "move_joints", "move_to", "set_gripper", "pi0_pick"}
PERCEPTION = {"back_project", "segment", "generate_grasps"}
OTHER = {"model:plan", "view_env_state", "write_text_file", "read_text_file", "finish"}


def plan_retention(episode):
    episode = Path(episode)
    exit_record = json.loads((episode.parent / "exit.json").read_text())
    close = json.loads((episode / "environment_owner_exit.json").read_text())
    transcript = json.loads(next(episode.glob("transcript_*.json")).read_text())
    if (
        exit_record.get("returncode") != 0
        or close.get("returncode") != 0
        or close.get("parent_closed") is not True
        or close.get("recording_close_receipt_verified") is not True
        or transcript.get("cleanup_errors") != []
    ):
        raise ValueError("sensor retention requires closed episode and video writers")
    states = json.loads((episode / "states.json").read_text())["steps"]
    manifest = json.loads((episode / "server_evidence.json").read_text())
    admitted = set(manifest["snapshots"])
    events = [
        json.loads(line)
        for line in (episode / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    current = states[0]["extras"]["observation_ref"]
    model = {}
    perception = {}
    calls = {}
    completed = set()
    contexts = 0
    for i, event in enumerate(events):
        kind = event["kind"]
        if kind == "action_submitted":
            operation = event["operation"]
            if operation not in PHYSICAL | PERCEPTION | OTHER:
                raise ValueError("cannot prove retention for operation " + operation)
            if event["action_id"] in calls:
                raise ValueError("repeated submitted action")
            calls[event["action_id"]] = operation
            if operation in PERCEPTION:
                perception.setdefault(current, []).append(
                    {
                        "event_index": i,
                        "operation": operation,
                        "arguments": event["arguments"],
                        "action_id": event["action_id"],
                    }
                )
        elif kind == "action_completed":
            if calls.get(event["action_id"]) != event["operation"]:
                raise ValueError("unbound completed action")
            completed.add(event["action_id"])
            if event["operation"] in PHYSICAL:
                current = event["result"]["observation_ref"]
        elif kind == "context_built":
            contexts += 1
            context = event["input"]
            reference = context["observation"]["observation_ref"]
            if reference != current:
                raise ValueError(
                    "decision observation differs from settled physical state"
                )
            cameras = [
                item["name"]
                for item in context["current_camera_observation"]["cameras"]
            ]
            images = list(
                {
                    item["path"]: item
                    for pointer, item in event.get("image_payloads", {}).items()
                    if pointer.startswith("/input/messages/")
                }.values()
            )
            output_paths = {
                item["path"]
                for pointer, item in event.get("image_payloads", {}).items()
                if pointer.startswith("/output/")
            }
            if not output_paths <= {item["path"] for item in images}:
                raise ValueError(
                    "output contains independently sourced images requiring a retention binding"
                )
            # Restrict proof to the current three original views. Other context
            # layouts require an explicit binding rather than guessing order.
            if (
                cameras != ["cam_head", "cam_left_wrist", "cam_right_wrist"]
                or len(images) != 3
            ):
                raise ValueError(
                    "model image layout needs explicit original camera binding"
                )
            mapping = {
                camera: {
                    "path": "harness_context/" + image["path"],
                    "bytes": image["bytes"],
                }
                for camera, image in zip(cameras, images)
            }
            if reference in model and model[reference]["images"] != mapping:
                raise ValueError("model original paths changed for one observation")
            model.setdefault(reference, {"images": mapping, "context_events": []})[
                "context_events"
            ].append(i)
    if set(calls) != completed:
        raise ValueError("unsettled operations prevent sensor pruning")
    forwards = []
    if (episode / "vla_inputs.jsonl").exists():
        forwards = [
            json.loads(line)
            for line in (episode / "vla_inputs.jsonl").read_text().splitlines()
        ]
    trace = [
        json.loads(line)
        for line in (episode / "execution_trace.jsonl").read_text().splitlines()
    ]
    if {x["forward_id"] for x in trace if x["event_type"] == "vla_forward"} != {
        x["forward_id"] for x in forwards
    }:
        raise ValueError("unbound VLA inputs prevent sensor pruning")
    for row in forwards:
        perception.setdefault(row["observation_ref"], []).append(
            {"operation": "vla_forward", "forward_id": row["forward_id"]}
        )
    if not set(model) | set(perception) <= admitted:
        raise ValueError("required observation missing from sensor manifest")
    if not {s["extras"]["observation_ref"] for s in states} <= admitted:
        raise ValueError("state lacks admitted sensor evidence")
    return {
        "schema_version": 1,
        "complete": False,
        "source_snapshot_count": len(admitted),
        "writer_exit": close,
        "context_count": contexts,
        "model_observations": model,
        "perception_observations": perception,
        "state_video_observations": sorted(admitted - set(model) - set(perception)),
        "raw_rgbd_observations": sorted(perception),
        "contract": "Keep every original model image byte and actual perception RGB-D/calibration. Other observations retain state and continuous video, not exact RGB-D.",
    }


def verify_model_pixels(episode, sensor_root, plan):
    """Prove every retained camera is the original sensor grid, without inference."""
    import hashlib
    import zipfile

    import numpy as np
    from PIL import Image

    from reweave.benchmarks.robodojo.env_client import decode_sensor_snapshot

    episode, sensor_root = Path(episode), Path(sensor_root)
    proof = {}
    for reference in sorted(
        set(plan["model_observations"]) | set(plan["perception_observations"])
    ):
        directory = sensor_root / reference.split(":")[-1]
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["observation_ref"] != reference:
            raise ValueError(
                "metadata observation identity differs from retained model input"
            )
        raw = (directory / "rgbd.npz").read_bytes()
        decoded = decode_sensor_snapshot(
            {"observation_ref": reference, "metadata": metadata, "archive": raw}
        )
        camera_proof = {}
        with zipfile.ZipFile(directory / "rgbd.npz") as archive:
            for camera, item in (
                plan["model_observations"].get(reference, {}).get("images", {}).items()
            ):
                path = episode / item["path"]
                if (
                    path.resolve() != episode.resolve() / item["path"]
                    or path.is_symlink()
                ):
                    raise ValueError("model original path escapes the episode")
                if path.stat().st_size != item["bytes"]:
                    raise ValueError(
                        "model original encoded bytes differ from recorded length"
                    )
                with Image.open(path) as image:
                    pixels = np.asarray(image)
                    if image.mode != "RGB" or pixels.dtype != np.uint8:
                        raise ValueError(
                            "model original is not an untransformed RGB grid"
                        )
                    original = decoded["vision"][camera]["color"]
                    if (
                        pixels.shape != original.shape
                        or pixels.tobytes() != original.tobytes()
                    ):
                        raise ValueError(
                            "model pixels differ from retained camera original"
                        )
                    item["original_size_wh"] = list(image.size)
                    item["archive_size_wh"] = list(image.size)
                    item["pixel_mapping"] = (
                        "identity: original pixels, no resizing or re-encoding"
                    )
                camera_proof[camera] = hashlib.sha256(
                    archive.read(camera + "__color.npy")
                ).hexdigest()
        proof[reference] = {
            "model_color_npy_sha256": camera_proof,
            "archive_bytes": len(raw),
            "verified_archive_sha256": hashlib.sha256(raw).hexdigest(),
            "verified_metadata_sha256": hashlib.sha256(
                (directory / "metadata.json").read_bytes()
            ).hexdigest(),
            "metadata_bytes": (directory / "metadata.json").stat().st_size,
        }
    # Audit the actual native delivery stream as well as the Host context. This
    # includes recovery/cancelled turns, which need not have a model_response.
    delivered = 0
    visible_turns = 0
    bound_images = {
        (episode / item["path"]).read_bytes()
        for row in plan["model_observations"].values()
        for item in row["images"].values()
    }
    native_events = episode / "native/provider/events.jsonl"
    if not native_events.is_file():
        raise ValueError(
            "native image delivery audit is unavailable; retain all sensors"
        )
    for line in native_events.read_text().splitlines():
        event = json.loads(line)
        message = event.get("message", {})
        if (
            event.get("direction") != "outgoing"
            or message.get("method") != "turn/start"
        ):
            continue
        visible_turns += 1
        for item in message["params"].get("input", []):
            if item["type"] == "text":
                continue
            if item["type"] != "localImage" or "/episode/" not in item["path"]:
                raise ValueError(
                    "native image delivery lacks an episode-local original binding"
                )
            relative = item["path"].rsplit("/episode/", 1)[1]
            original = episode / relative
            if (
                original.resolve() != episode.resolve() / relative
                or original.is_symlink()
            ):
                raise ValueError("native delivered image escapes its episode")
            if original.read_bytes() not in bound_images:
                raise ValueError(
                    "native delivered image is absent from verified model originals"
                )
            delivered += 1
    transcript = json.loads(next(episode.glob("transcript_*.json")).read_text())
    if visible_turns != transcript["stats"]["visible_turns"] or not visible_turns:
        raise ValueError(
            "native delivery audit does not cover the recorded visible turns"
        )
    plan["native_delivery_verification"] = {
        "visible_turns": visible_turns,
        "delivered_images": delivered,
    }
    plan["pixel_verification"] = proof
    return plan


def publish_retention_index(episode, plan):
    """Publish the explicit offline retention contract before unused files retire.

    Existing local image paths, journals, native history and metadata stay intact.
    The server index lists only retained exact RGB-D. State/video-only observations
    remain navigable through this independent retention record.
    """
    from reweave.utils.persistence import atomic_json

    episode = Path(episode)
    if (episode / "episode_evidence.json").exists():
        raise ValueError("cannot change a published historical episode")
    if set(plan.get("pixel_verification", {})) != set(plan["model_observations"]) | set(
        plan["perception_observations"]
    ):
        raise ValueError(
            "all required sensor and model originals must be verified first"
        )
    manifest_path = episode / "server_evidence.json"
    pending = episode / ".sensor_retention_pending.json"
    receipt = episode / "sensor_retention.json"
    if (
        receipt.exists()
        and not pending.exists()
        and json.loads(receipt.read_text()).get("complete")
    ):
        return json.loads(receipt.read_text())
    if pending.exists():
        original = json.loads(pending.read_text())
    else:
        original = json.loads(manifest_path.read_text())
        atomic_json(pending, original)
    keep = set(plan["raw_rgbd_observations"])
    if not keep <= set(original["snapshots"]):
        raise ValueError("retained exact sensors are absent from original index")
    # Existing EnvState and dashboard history readers keep their ordinary paths.
    # These are additional names for the same original inode, not image copies.
    import os

    for state in json.loads((episode / "states.json").read_text())["steps"]:
        reference = state["extras"]["observation_ref"]
        for camera, image in (
            plan["model_observations"].get(reference, {}).get("images", {}).items()
        ):
            source_image = episode / image["path"]
            destination = episode / (camera + ".png") / f"{state['step_idx']:02d}.png"
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                if (
                    destination.is_symlink()
                    or destination.read_bytes() != source_image.read_bytes()
                ):
                    raise ValueError(
                        "historical state image differs from verified model original"
                    )
            else:
                os.link(source_image, destination)
    plan = {**plan, "complete": True}
    # Readers refuse pending publication; interruption resumes from the saved
    # original manifest, never from a shortened set of snapshots.
    atomic_json(
        manifest_path,
        {
            **original,
            "snapshots": {k: v for k, v in original["snapshots"].items() if k in keep},
        },
    )
    atomic_json(receipt, plan)
    records_path = episode / ".artifact_records"
    if records_path.exists():
        records = json.loads(records_path.read_text())
        records.pop("server_evidence.json", None)
        atomic_json(records_path, records)
    pending.rename(episode / "server_evidence.before_retention.json")
    read_retention(episode)
    return plan


def _retire_unneeded_sensors(sensor_root, plan):
    """Retire unneeded NPZ/PNG copies only after Host evidence validation.

    This stdlib-only part can run on the storage host after all writers exit.
    A durable receipt is written before deletion. Retrying repeats only file
    retirement; the actual perception originals and metadata are never touched.
    """
    import hashlib
    import os
    import zipfile

    root = Path(sensor_root).resolve()
    if (
        root.name != "server_observations"
        or not root.is_dir()
        or not plan.get("complete")
    ):
        raise ValueError(
            "a completed retention index and sensor directory are required"
        )
    if (root.parent / "episode_evidence.json").exists():
        raise ValueError("cannot retire published historical sensor evidence")
    closed = plan.get("writer_exit", {})
    if (
        closed.get("returncode") != 0
        or closed.get("parent_closed") is not True
        or closed.get("recording_close_receipt_verified") is not True
    ):
        raise ValueError("sensor retirement requires verified closed writers")
    keep = set(plan["raw_rgbd_observations"])
    all_refs = (
        set(plan["model_observations"])
        | set(plan["perception_observations"])
        | set(plan["state_video_observations"])
    )
    metadata = {}
    for path in root.glob("*/metadata.json"):
        value = json.loads(path.read_text())
        ref = value["observation_ref"]
        if (
            path.parent.name != ref.split(":")[-1]
            or ref in metadata
            or value["service_instance_id"] != closed["service_instance_id"]
        ):
            raise ValueError(
                "sensor metadata identity differs from directory or repeats"
            )
        metadata[ref] = path
    if set(metadata) != all_refs:
        raise ValueError("retirement proof does not cover every server observation")
    receipt = root / "sensor_retention.json"
    if not receipt.exists():
        for ref, expected in plan["pixel_verification"].items():
            archive_path = metadata[ref].with_name("rgbd.npz")
            if archive_path.stat().st_size != expected["archive_bytes"]:
                raise ValueError("verified sensor archive changed before retirement")
            if (
                hashlib.sha256(archive_path.read_bytes()).hexdigest()
                != expected["verified_archive_sha256"]
                or hashlib.sha256(metadata[ref].read_bytes()).hexdigest()
                != expected["verified_metadata_sha256"]
            ):
                raise ValueError(
                    "verified RGB-D or calibration changed before retirement"
                )
            with zipfile.ZipFile(archive_path) as archive:
                for camera, digest in expected["model_color_npy_sha256"].items():
                    if (
                        hashlib.sha256(archive.read(camera + "__color.npy")).hexdigest()
                        != digest
                    ):
                        raise ValueError(
                            "server pixels differ from verified model originals"
                        )
        # Keep the pre-retirement index as audit text, not redundant sensor bytes.
        source_manifest = root / "manifest.jsonl"
        original_index = root / "manifest.before_retention.jsonl"
        index_source = original_index if original_index.exists() else source_manifest
        lines = [json.loads(line) for line in index_source.read_text().splitlines()]
        if {row["observation_ref"] for row in lines} != all_refs:
            raise ValueError("server index differs from retention proof")
        original_index = root / "manifest.before_retention.jsonl"
        if not original_index.exists():
            original_index.write_bytes(source_manifest.read_bytes())
        temporary = root / ".manifest.retention.tmp"
        with temporary.open("w") as stream:
            for row in lines:
                if row["observation_ref"] in keep:
                    stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, source_manifest)
        temporary = root / ".retention.receipt.tmp"
        with temporary.open("w") as stream:
            json.dump({"plan": plan, "storage_complete": False}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, receipt)
    else:
        recorded = json.loads(receipt.read_text())
        if recorded["plan"] != plan:
            raise ValueError("retention plan changed during storage recovery")
    # Required originals are checked on every recovery before any retirement.
    for ref in keep:
        if (
            metadata[ref].with_name("rgbd.npz").stat().st_size
            != plan["pixel_verification"][ref]["archive_bytes"]
        ):
            raise ValueError("required exact perception archive is missing or changed")
        if (
            hashlib.sha256(metadata[ref].with_name("rgbd.npz").read_bytes()).hexdigest()
            != plan["pixel_verification"][ref]["verified_archive_sha256"]
        ):
            raise ValueError("required exact perception archive bytes changed")
    removed = 0
    for ref, path in metadata.items():
        candidates = [] if ref in keep else [path.with_name("rgbd.npz")]
        # Server PNGs are byte copies of model media, or never-used intermediate
        # views. Actual perception RGB also remains inside its retained NPZ.
        camera_names = json.loads(path.read_text())["vision"]
        candidates.extend(path.with_name(camera + ".png") for camera in camera_names)
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError("sensor retirement cannot follow links")
            if candidate.exists():
                removed += candidate.stat().st_size
                candidate.unlink()
    temporary = root / ".retention.receipt.tmp"
    temporary.write_text(json.dumps({"plan": plan, "storage_complete": True}))
    os.replace(temporary, receipt)
    return {
        "removed_file_bytes_this_pass": removed,
        "raw_snapshots_kept": len(keep),
        "model_observations_kept": len(plan["model_observations"]),
    }


def retire_unneeded_sensors(sensor_root, plan):
    """Serialize storage retirement against another post-episode finalizer."""
    import fcntl

    root = Path(sensor_root)
    with (root / ".retention.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _retire_unneeded_sensors(root, plan)


def retain_local_sensors(episode, sensor_root, plan):
    """Publish verified original RGB-D locally before any server retirement.

    local_sensor_originals means offline readers must use these local files;
    missing local evidence is an error, never an implicit network fallback.
    """
    import hashlib
    import os
    import shutil
    import tempfile

    from reweave.evolution.evidence.server import _validate_snapshot

    episode = Path(episode).resolve()
    manifest = json.loads((episode / "server_evidence.json").read_text())
    root = episode / "sensor_originals"
    root.mkdir(exist_ok=True)
    for reference in plan["raw_rgbd_observations"]:
        identity = reference.split(":")[-1]
        target = root / identity
        expected = plan["pixel_verification"][reference]

        def validate(directory):
            _validate_snapshot(directory, reference, manifest["snapshots"][reference])
            for filename, key in (
                ("rgbd.npz", "verified_archive_sha256"),
                ("metadata.json", "verified_metadata_sha256"),
            ):
                if (
                    hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                    != expected[key]
                ):
                    raise ValueError(
                        "local sensor original differs from verified evidence"
                    )

        if target.exists():
            validate(target)
            continue
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
        try:
            for filename in ("rgbd.npz", "metadata.json"):
                shutil.copyfile(
                    Path(sensor_root) / identity / filename, staging / filename
                )
                (staging / filename).chmod(0o444)
            validate(staging)
            os.rename(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    plan["local_sensor_originals"] = True
    return plan


def read_retention(episode):
    """Read a completed retention contract; never treat pending evidence as ready."""
    episode = Path(episode)
    if (episode / ".sensor_retention_pending.json").exists():
        raise ValueError(
            "sensor retention publication is incomplete; resume storage only"
        )
    path = episode / "sensor_retention.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("complete") is not True:
        raise ValueError("sensor retention index is incomplete or unsupported")
    model = set(value["model_observations"])
    raw = set(value["raw_rgbd_observations"])
    video = set(value["state_video_observations"])
    if raw != set(value["perception_observations"]) or video & (raw | model):
        raise ValueError("sensor retention roles contradict each other")
    manifest = json.loads((episode / "server_evidence.json").read_text())
    if set(manifest["snapshots"]) != raw:
        raise ValueError("sensor retention and exact RGB-D index disagree")
    if len(model | raw | video) != value["source_snapshot_count"]:
        raise ValueError("sensor retention omits original observations")
    if value.get("local_sensor_originals"):
        from reweave.evolution.evidence.server import _regular_local

        for reference in raw:
            directory = episode / "sensor_originals" / reference.split(":")[-1]
            for filename, key in (
                ("rgbd.npz", "archive_bytes"),
                ("metadata.json", "metadata_bytes"),
            ):
                path = _regular_local(directory / filename)
                if path.stat().st_size != manifest["snapshots"][reference][key]:
                    raise ValueError("local sensor original changed or is missing")
    for row in value["model_observations"].values():
        for image in row["images"].values():
            original = episode / image["path"]
            if (
                original.resolve() != episode.resolve() / image["path"]
                or original.is_symlink()
            ):
                raise ValueError("retained original image escapes episode")
            if original.stat().st_size != image["bytes"]:
                raise ValueError("retained original model image changed or is missing")
    return value


def _cached_retention_snapshot(manifest, reference, sensor_root):
    """Checkpoint fully downloaded originals; transport failure loses only one copy."""
    import tempfile
    import time

    from reweave.evolution.evidence.server import (
        _descriptor,
        _run_fetch_worker,
        _validate_snapshot,
    )

    descriptor = _descriptor(manifest, reference)
    destination = sensor_root / reference.split(":")[-1]
    if destination.exists():
        _validate_snapshot(destination, reference, descriptor)
        return
    for attempt in range(3):
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=sensor_root) as pending:
            pending = Path(pending).resolve()
            try:
                _run_fetch_worker(
                    {
                        "manifest": manifest,
                        "observation_ref": reference,
                        "staging_directory": str(pending),
                    }
                )
            except (RuntimeError, TimeoutError) as error:
                # These operations only read already-closed sensor originals.
                # Retry in fresh staging; never retry integrity/authentication
                # failures or any environment/model operation.
                transient = str(error) in {
                    "server sensor fetch failed (SSHException)",
                    "server sensor fetch failed (EOFError)",
                    "server sensor fetch failed (TimeoutError)",
                    "server sensor fetch exceeded its time limit",
                }
                if not transient or attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
                continue
            _validate_snapshot(pending, reference, descriptor)
            pending.rename(destination)
            return


def finalize_episode_sensors(attempt):
    """Independent storage closeout after child exit, before offline publication.

    Fully validated snapshot copies survive a transport interruption in Host scratch.
    Resume reuses them and never reruns episode work; complete closeout removes scratch.
    """
    import shutil
    import tempfile
    import time

    from reweave.evolution.evidence.media_storage import finalize_media
    from reweave.evolution.evidence.records import EpisodeEvidence, inventory_episode
    from reweave.evolution.evidence.server import (
        retire_server_sensors,
        sensor_fetch_session,
    )
    from reweave.evolution.evidence.trace import build_episode_context
    from reweave.utils.persistence import atomic_json

    attempt = Path(attempt)
    episode = attempt / "episode"
    if (
        not (episode / "server_evidence.json").exists()
        or (episode / "episode_evidence.json").exists()
    ):
        return
    started = time.monotonic()
    storage_receipt = episode / "sensor_storage.json"
    if storage_receipt.exists() and json.loads(storage_receipt.read_text()).get(
        "complete"
    ):
        return
    # A failed/partial scene keeps all of its original evidence. Storage is not
    # allowed to change an existing episode failure into an infrastructure retry.
    close_path = episode / "environment_owner_exit.json"
    transcripts = list(episode.glob("transcript_*.json"))
    if (
        not close_path.is_file()
        or len(transcripts) != 1
        or json.loads(transcripts[0].read_text()).get("cleanup_errors") != []
        or json.loads((attempt / "exit.json").read_text()).get("returncode") != 0
    ):
        return
    close = json.loads(close_path.read_text())
    if (
        close.get("returncode") != 0
        or close.get("recording_close_receipt_verified") is not True
        or close.get("parent_closed") is not True
    ):
        return
    finalize_media(attempt)
    # Hold the same publication lock as inspect_episode; offline readers cannot
    # publish an inventory halfway through this operation.
    import fcntl

    with (attempt / ".episode.evidence.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (episode / "episode_evidence.json").exists():
            return
        retention_path = episode / "sensor_retention.json"
        pending = episode / ".sensor_retention_pending.json"
        if pending.exists() and not retention_path.exists():
            # Interrupted before the verified receipt: restore only the local
            # index and rebuild the proof. No server retirement has started.
            atomic_json(
                episode / "server_evidence.json", json.loads(pending.read_text())
            )
            pending.unlink()
        manifest = json.loads((episode / "server_evidence.json").read_text())
        with tempfile.TemporaryDirectory(
            prefix="reweave-sensor-retention-"
        ) as temporary:
            # macOS TMPDIR commonly traverses /var -> /private/var. The Host
            # chooses this directory; resolve it before the fetch worker's
            # canonical-path validation instead of weakening that validation.
            temporary = Path(temporary).resolve()
            if retention_path.exists():
                plan = json.loads(retention_path.read_text())
                if (episode / ".sensor_retention_pending.json").exists():
                    plan = publish_retention_index(episode, plan)
                read_retention(episode)
            else:
                try:
                    plan = plan_retention(episode)
                except (ValueError, KeyError) as error:
                    # An unproven binding is explicitly retained, not discarded.
                    atomic_json(
                        storage_receipt,
                        {
                            "complete": True,
                            "sensor_retirement": "not_proven_all_originals_retained",
                            "reason": str(error),
                            "storage_elapsed_s": time.monotonic() - started,
                        },
                    )
                    return
                sensor_root = (attempt / ".scratch/sensor-retention").resolve()
                sensor_root.mkdir(parents=True, exist_ok=True)
                with sensor_fetch_session(manifest):
                    for reference in sorted(
                        set(plan["model_observations"])
                        | set(plan["perception_observations"])
                    ):
                        _cached_retention_snapshot(manifest, reference, sensor_root)
                try:
                    plan = verify_model_pixels(episode, sensor_root, plan)
                except (ValueError, KeyError) as error:
                    atomic_json(
                        storage_receipt,
                        {
                            "complete": True,
                            "sensor_retirement": "not_proven_all_originals_retained",
                            "reason": str(error),
                            "storage_elapsed_s": time.monotonic() - started,
                        },
                    )
                    return
                plan = retain_local_sensors(episode, sensor_root, plan)
                plan = publish_retention_index(episode, plan)
            evidence = EpisodeEvidence(
                evidence_id="storage-validation", files=inventory_episode(episode)
            )
            build_episode_context(
                temporary / "offline", evidence, episode, include_visuals=False
            )
            result = retire_server_sensors(manifest, plan)
        atomic_json(
            storage_receipt,
            {
                "complete": True,
                "storage_elapsed_s": time.monotonic() - started,
                "model_images_reencoded": False,
                "server": result,
            },
        )
        shutil.rmtree(attempt / ".scratch/sensor-retention", ignore_errors=True)
