"""Finished-file storage preserves original bytes and never replays execution."""

import io
import json
import os

import pytest
from PIL import Image

from reweave.evolution.evidence import media_storage as archive_storage


@pytest.fixture
def attempt(tmp_path):
    root = tmp_path / "attempt"
    episode = root / "episode"
    episode.mkdir(parents=True)
    (root / "exit.json").write_text(json.dumps({"returncode": 0}))
    (episode / "transcript_test.json").write_text(json.dumps({"cleanup_errors": []}))
    for name in ("native", "harness_context"):
        (episode / name).mkdir()
    image = io.BytesIO()
    Image.new("RGB", (640, 480), (17, 91, 203)).save(image, format="PNG")
    a = episode / "native/original.png"
    b = episode / "harness_context/original.png"
    a.write_bytes(image.getvalue())
    b.write_bytes(image.getvalue())
    a.chmod(0o444)
    b.chmod(0o644)
    os.link(b, episode / "harness_context/alias.png")
    return root


def test_original_pixels_encoded_bytes_and_aliases_survive(attempt):
    episode = attempt / "episode"
    paths = sorted(episode.rglob("*.png"))
    originals = {p: p.read_bytes() for p in paths}
    before = archive_storage.storage_usage(episode)
    receipt = archive_storage.finalize_media(attempt)
    assert receipt["complete"] and receipt["storage_elapsed_s"] > 0
    assert len({p.stat().st_ino for p in paths}) == 1
    assert all(p.read_bytes() == originals[p] for p in paths)
    assert all(Image.open(p).size == (640, 480) for p in paths)
    assert receipt["after"]["unique_bytes"] < before["unique_bytes"]
    assert archive_storage.finalize_media(attempt) == receipt


def test_interruption_resumes_storage_only(attempt, monkeypatch):
    original = archive_storage.deduplicate_episode_media
    paths = list((attempt / "episode").rglob("*.png"))
    data = [p.read_bytes() for p in paths]

    def interrupted(_):
        raise InterruptedError("simulated storage interruption")

    monkeypatch.setattr(archive_storage, "deduplicate_episode_media", interrupted)
    with pytest.raises(InterruptedError):
        archive_storage.finalize_media(attempt)
    assert [p.read_bytes() for p in paths] == data
    marker = attempt / "episode/archive_storage.json"
    assert not json.loads(marker.read_text())["complete"]
    monkeypatch.setattr(archive_storage, "deduplicate_episode_media", original)
    assert archive_storage.finalize_media(attempt)["complete"]
    assert [p.read_bytes() for p in paths] == data


def test_published_originals_are_rejected(attempt):
    episode = attempt / "episode"
    (episode / "episode_evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="published"):
        archive_storage.finalize_media(attempt)
    assert (episode / "harness_context/original.png").stat().st_mode & 0o200


def test_unclosed_writers_are_rejected(attempt):
    (attempt / "exit.json").unlink()
    with pytest.raises(FileNotFoundError):
        archive_storage.finalize_media(attempt)
    assert not (attempt / "episode/archive_storage.json").exists()


def test_external_hardlink_is_not_changed(attempt):
    path = attempt / "episode/harness_context/original.png"
    external = attempt / "other-run.png"
    os.link(path, external)
    with pytest.raises(ValueError, match="outside"):
        archive_storage.finalize_media(attempt)
    assert external.stat().st_mode & 0o200
