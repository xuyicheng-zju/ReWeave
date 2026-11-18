# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""The native author receives actual selected video pixels and source references."""

import asyncio
import base64
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from reweave.evolution.authoring.candidate import host_input
from reweave.evolution.authoring.client import _request_development_tool
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.authoring.remote import RemoteWorkspaceTools
from reweave.evolution.authoring.tools import AuthorDevelopmentTools
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.evidence.trace import read_video
from tests.unit_tests.harness.evolution.test_bundle_author import workspace


def video_workspace(tmp_path):
    root = workspace(tmp_path)
    episode = tmp_path / "episode"
    episode.mkdir()
    video = episode / "episode.mp4"
    with imageio.get_writer(video, fps=5) as writer:
        for value in (20, 70, 130, 230):
            writer.append_data(np.full((32, 32, 3), value, dtype=np.uint8))
    identity = "a" * 64
    source = {
        "path": f".archive/episodes/{identity}/episode.mp4",
        **{
            key: value for key, value in artifact_record(video).items() if key != "path"
        },
        "pointer": "",
    }
    index = root / "context/trace" / identity / "index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"media": [{"kind": "video", "source": source}]}))
    private_index = (
        host_input(root, "evidence") / "context/trace" / identity / "index.json"
    )
    private_index.parent.mkdir(parents=True, exist_ok=True)
    private_index.write_bytes(index.read_bytes())
    references = AuthorReferences.load(root)
    references.evidence_roots[identity] = episode
    references.identity("trace", identity)
    return root, {identity: episode}, identity, source


def test_native_mcp_video_response_contains_requested_pixels_and_audited_provenance(
    tmp_path,
):
    root, roots, identity, source = video_workspace(tmp_path)
    original_video = (roots[identity] / "episode.mp4").read_bytes()
    references = AuthorReferences.load(root)
    trace = references.identity("trace", identity)
    service = AuthorDevelopmentTools(root, tmp_path / "audit/video.jsonl", roots)
    service.server.start()
    try:
        metadata = _request_development_tool(
            service.server.url,
            "read_video",
            {"evidence_id": trace, "path": "episode.mp4", "frames": []},
        )
        assert metadata["structuredContent"]["frame_count"] == 4
        remote = RemoteWorkspaceTools(root, root.parent, (), (), service.server.url)
        assert "read_video" in {item["name"] for item in asyncio.run(remote.catalog())}
        result = asyncio.run(
            remote.call(
                "read_video",
                {"evidence_id": trace, "path": "episode.mp4", "frames": [1, 3]},
            )
        )
    finally:
        service.server.stop(timeout_s=None)
    assert not result["isError"]
    manifest = result["structuredContent"]
    assert service.references.trace_reference(manifest["source"]) == source
    assert manifest["physical_step_alignment"] is None
    assert [item["frame_index"] for item in manifest["frames"]] == [1, 3]
    assert [item["nominal_video_time_s"] for item in manifest["frames"]] == [0.2, 0.6]
    images = [item for item in result["content"] if item["type"] == "image"]
    assert len(images) == 2
    audit = [json.loads(line) for line in service.audit.read_text().splitlines()]
    for image, frame in zip(images, audit[-1]["result"]["frames"]):
        assert base64.b64decode(image["data"]) == Path(frame["path"]).read_bytes()
        assert frame["bytes"] == len(base64.b64decode(image["data"]))
    assert audit[-1]["result"]["source"] == source
    assert all(
        set(frame) == {"frame_index", "nominal_video_time_s"}
        for frame in manifest["frames"]
    )
    assert "base64" not in service.audit.read_text()
    assert (roots[identity] / "episode.mp4").read_bytes() == original_video


@pytest.mark.parametrize(
    "path,frames",
    [
        ("../episode.mp4", [0]),
        ("episode.mp4", [4]),
        ("episode.mp4", [2, 1]),
        ("episode.mp4", [True]),
        ("episode.mp4", [1, 1]),
    ],
)
def test_video_does_not_guess_paths_frames_or_alignment(tmp_path, path, frames):
    root, roots, identity, _ = video_workspace(tmp_path)
    with pytest.raises(ValueError):
        read_video(root, roots, evidence_id=identity, path=path, frames=frames)


def test_video_scratch_cannot_redirect_host_write_to_originals(tmp_path):
    root, roots, identity, source = video_workspace(tmp_path)
    original_video = (roots[identity] / "episode.mp4").read_bytes()
    scratch = root / ".scratch"
    scratch.mkdir(exist_ok=True)
    (scratch / "media").symlink_to(roots[identity], target_is_directory=True)
    with pytest.raises(ValueError, match="must not follow links"):
        read_video(root, roots, evidence_id=identity, path="episode.mp4", frames=[0])
    assert (roots[identity] / "episode.mp4").read_bytes() == original_video
    assert len(list(roots[identity].glob("*.mp4"))) == 1
