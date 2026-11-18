# Modified for the ReWeave namespace.
"""Full original archives remain authoritative across native author workspaces."""

import base64
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from reweave.evolution.evidence.records import (
    archive_episode,
    archive_record,
    artifact_record,
    deduplicate_episode_media,
    inspect_episode,
    inventory_episode,
    restore_record,
    store_bytes,
)


def episode(tmp_path):
    source = tmp_path / "episode"
    context = source / "harness_context"
    context.mkdir(parents=True)
    (context / "planner_context.json").write_text('{"system_prompt": "complete"}')
    (context / "capability_catalog.json").write_text("{}")
    (source / "execution_trace.jsonl").write_text(
        "".join(
            json.dumps({"index": i, "message": "action" * 2000}) + "\n"
            for i in range(120)
        )
    )
    (source / "transcript_run.json").write_text(
        json.dumps(
            {"messages": [{"content": "完整证据🙂\r\n" * 3000} for _ in range(120)]},
            ensure_ascii=False,
        )
    )
    (source / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4)
    return source


def test_inventory_batches_durable_id_publications_without_changing_ids(
    tmp_path, monkeypatch
):
    from reweave.evolution.evidence import records as evidence

    for index in range(8):
        folder = tmp_path / f"source-{index}"
        folder.mkdir()
        for item in range(25):
            (folder / f"file-{item}.py").write_text("retained original\n")
    previous = artifact_record(tmp_path / "source-0/file-0.py")
    original = evidence.atomic_json
    publications = []

    def publish(path, value):
        publications.append(path)
        return original(path, value)

    monkeypatch.setattr(evidence, "atomic_json", publish)
    first = inventory_episode(tmp_path)
    assert len(first) == 200 and len(publications) == 8
    assert first["source-0/file-0.py"]["artifact_id"] == previous["artifact_id"]
    assert all(path.name == ".artifact_records" for path in publications)
    publications.clear()
    assert inventory_episode(tmp_path) == first
    assert not publications


def test_journal_stores_repeated_image_strings_once_and_restores_exact_requests(
    tmp_path,
):
    # Include line breaks and invalid encoding: archival must also preserve an
    # illegal candidate request, without normalizing the evidence of its error.
    encoded = "abc\r\n原样🙂+==\r"
    images = [
        {"data_base64": encoded, "media_type": "image/png"},
        {"kind": "binary", "data": encoded, "media_type": "image/png"},
        {"type": "base64", "data": encoded, "media_type": "image/png"},
    ]
    request = {
        "task": "preserve complete history",
        "history/~": [{"images": images, "tool_call_id": str(i)} for i in range(56)],
        "ordinary_text": encoded,
    }
    before = json.dumps(request, ensure_ascii=False)
    archived = archive_record(request, tmp_path)
    assert json.dumps(request, ensure_ascii=False) == before
    assert archived["history/~"][0]["images"][0]["data_base64"] is None
    assert len(archived["image_payloads"]) == 168
    assert len(list((tmp_path / "image_payloads").glob("image-*.txt"))) == 1
    assert restore_record(json.loads(json.dumps(archived)), tmp_path) == request
    assert archive_record(request, tmp_path) == archived
    assert archived["ordinary_text"] == encoded
    assert restore_record({"text": "no images"}, tmp_path) == {"text": "no images"}
    with pytest.raises(ValueError, match="reserved"):
        archive_record(archived, tmp_path)


def test_journal_and_transcript_share_observation_bytes_and_restore_both_encodings(
    tmp_path,
):
    from reweave.evolution.evidence.trace import resolve_trace_reference
    from reweave.evolution.evidence.trajectory import OriginalReference, read_original

    episode = tmp_path / ".archive/episodes/current"
    context = episode / "harness_context"
    media = context / "observations"
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (12, 89, 231)).save(buffer, format="PNG")
    data = buffer.getvalue() + b"\xfb\xff\xff\xff\xff\xff"
    standard = base64.b64encode(data).decode("ascii")
    urlsafe = base64.urlsafe_b64encode(data).decode("ascii")
    assert standard != urlsafe
    observed = store_bytes(data, media, suffix=".png")
    request = {
        "messages": [
            {"source": {"type": "base64", "media_type": "image/png", "data": text}}
            for text in (standard, urlsafe)
        ]
    }
    journal = archive_record(request, context)
    transcript = archive_record(request, episode, media_directory=media)
    assert {ref["encoding"] for ref in journal["image_payloads"].values()} == {
        "base64",
        "base64url",
    }
    assert {ref["path"] for ref in journal["image_payloads"].values()} == {
        "observations/" + observed["path"]
    }
    assert len(list(episode.rglob("*.png"))) == 1
    assert not list(episode.rglob("*.txt"))
    assert restore_record(journal, context) == request
    assert restore_record(transcript, episode) == request
    path = episode / "transcript_test.json"
    path.write_text(json.dumps(transcript))
    publication = artifact_record(path)
    pointer = "/messages/1/source/data"
    assert (
        resolve_trace_reference(
            tmp_path,
            {
                **publication,
                "path": ".archive/episodes/current/transcript_test.json",
                "pointer": pointer,
            },
        )
        == urlsafe
    )
    assert read_original(OriginalReference(**publication, pointer=pointer)) == urlsafe


@pytest.mark.parametrize("corruption", ["codec", "path", "extra", "truncated", "size"])
def test_binary_journal_rejects_invalid_codec_and_escaped_media_paths(
    tmp_path, corruption
):
    archived = archive_record(
        {"image": {"media_type": "image/png", "data_base64": "aGVsbG8="}},
        tmp_path,
    )
    reference = next(iter(archived["image_payloads"].values()))
    if corruption == "codec":
        reference["encoding"] = "normalize-anything"
    elif corruption == "path":
        reference["path"] = "observations/../outside.png"
    elif corruption == "extra":
        reference["unknown"] = True
    elif corruption == "truncated":
        (tmp_path / reference["path"]).write_bytes(b"h")
    else:
        reference["bytes"] += 1
    with pytest.raises(ValueError, match="(?:invalid journal|payload length)"):
        restore_record(archived, tmp_path)


@pytest.mark.parametrize(
    "corruption", ["path", "missing", "symlink", "leaf", "pointer"]
)
def test_journal_rejects_invalid_image_references(tmp_path, corruption):
    archived = archive_record(
        {"images": [{"media_type": "image/png", "data_base64": "exact"}]}, tmp_path
    )
    references = archived["image_payloads"]
    pointer, reference = next(iter(references.items()))
    payload = tmp_path / reference["path"]
    if corruption == "path":
        reference["path"] = "../outside.txt"
    elif corruption == "missing":
        payload.unlink()
    elif corruption == "symlink":
        external = tmp_path / "outside.txt"
        payload.rename(external)
        payload.symlink_to(external)
    elif corruption == "leaf":
        archived["images"][0]["data_base64"] = "overwritten"
    else:
        references[pointer.replace("/0/", "/00/")] = references.pop(pointer)
    with pytest.raises(ValueError):
        restore_record(archived, tmp_path)


def test_settled_media_deduplication_preserves_paths_pixels_modes_and_atomic_replacement(
    tmp_path,
):
    images = [
        tmp_path / name
        for name in ("original.png", "copy.png", "different.png", "readonly.png")
    ]
    for path, data in zip(
        images,
        (b"same-pixels", b"same-pixels", b"next-pixels", b"same-pixels"),
        strict=True,
    ):
        path.write_bytes(data)
    images[-1].chmod(0o444)
    before = {path.name: (path.read_bytes(), path.stat().st_mode) for path in images}
    result = deduplicate_episode_media(tmp_path)
    assert result == {"files_linked": 1, "duplicate_bytes_removed": len(b"same-pixels")}
    assert {
        path.name: (path.read_bytes(), path.stat().st_mode) for path in images
    } == before
    assert images[0].stat().st_ino == images[1].stat().st_ino
    assert images[0].stat().st_ino != images[-1].stat().st_ino
    assert deduplicate_episode_media(tmp_path)["duplicate_bytes_removed"] == 0
    replacement = tmp_path / "replacement.tmp"
    replacement.write_bytes(b"new-pixels")
    replacement.replace(images[0])
    assert images[1].read_bytes() == b"same-pixels"


def test_full_messages_middle_events_unicode_and_images_survive_cross_branch_archive(
    tmp_path,
):
    source = episode(tmp_path)
    original = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    first = tmp_path / "first/.archive/episodes"
    evidence = archive_episode(source, first)
    assert archive_episode(source, first) == evidence
    second = tmp_path / "second/.archive/episodes"
    assert archive_episode(first / evidence.evidence_id, second) == evidence
    for root in (source, first / evidence.evidence_id, second / evidence.evidence_id):
        assert {name: (root / name).read_bytes() for name in original} == original
        assert set(evidence.files) == {name.as_posix() for name in original}
        assert json.loads(
            (root / "episode_evidence.json").read_text()
        ) == evidence.model_dump(mode="json")
        evidence.verify(root)


@pytest.mark.parametrize(
    "missing",
    [
        "execution_trace.jsonl",
        "harness_context/planner_context.json",
        "harness_context/capability_catalog.json",
        "transcript_run.json",
    ],
)
def test_missing_current_decision_evidence_cannot_be_archived(tmp_path, missing):
    source = episode(tmp_path)
    (source / missing).unlink()
    with pytest.raises(ValueError):
        inspect_episode(source)


def test_archive_requires_files_without_rechecking_content_hashes(tmp_path):
    source = episode(tmp_path)
    link = source / "linked.json"
    link.symlink_to(source / "transcript_run.json")
    with pytest.raises(ValueError, match="symlinks"):
        inspect_episode(source)
    link.unlink()
    duplicate = source / "transcript_another.json"
    duplicate.write_text("{}")
    with pytest.raises(ValueError, match="exactly one"):
        inspect_episode(source)
    duplicate.unlink()
    evidence = inspect_episode(source)
    (source / "image.png").write_bytes(b"changed")
    evidence.verify(source)
    (source / "image.png").unlink()
    with pytest.raises(ValueError, match="unavailable"):
        evidence.verify(source)


def test_archive_keeps_copied_files_when_source_is_updated_later(tmp_path, monkeypatch):
    source = episode(tmp_path)
    copytree = shutil.copytree

    def changed_copy(*args, **kwargs):
        result = copytree(*args, **kwargs)
        if Path(args[0]) == source:
            (source / "image.png").write_bytes(b"changed during archival")
        return result

    monkeypatch.setattr(shutil, "copytree", changed_copy)
    captured = archive_episode(source, tmp_path / "archive")
    archived = tmp_path / "archive" / captured.evidence_id / "image.png"
    assert archived.read_bytes() != (source / "image.png").read_bytes()


def test_byte_records_compare_actual_equal_length_bytes_and_keep_published_ids(
    tmp_path,
):
    directory = tmp_path / "images"
    first = store_bytes(b"same-length-a", directory, suffix=".txt")
    second = store_bytes(b"same-length-b", directory, suffix=".txt")
    assert first["bytes"] == second["bytes"]
    assert first["artifact_id"] != second["artifact_id"]
    assert (directory / first["path"]).read_bytes() == b"same-length-a"
    assert (directory / second["path"]).read_bytes() == b"same-length-b"
    with ThreadPoolExecutor(max_workers=4) as workers:
        repeated = list(
            workers.map(
                lambda _: store_bytes(b"same-length-a", directory, suffix=".txt"),
                range(8),
            )
        )
    assert repeated == [first] * 8
    assert len(json.loads((directory / "records.json").read_text())) == 2


def test_artifact_and_episode_publication_reuse_one_id_without_inventorying_metadata(
    tmp_path,
):
    source = episode(tmp_path)
    original = source / "image.png"
    with ThreadPoolExecutor(max_workers=4) as workers:
        records = list(workers.map(lambda _: artifact_record(original), range(8)))
    assert records == [records[0]] * 8
    assert set(records[0]) == {"artifact_id", "path", "bytes"}
    assert not list(source.glob(".*.json"))
    with ThreadPoolExecutor(max_workers=4) as workers:
        inventories = list(workers.map(lambda _: inspect_episode(source), range(8)))
    assert inventories == [inventories[0]] * 8
    assert inventory_episode(source) == inventories[0].files
    assert all(
        not Path(name).name.startswith(".artifact_records")
        for name in inventories[0].files
    )
