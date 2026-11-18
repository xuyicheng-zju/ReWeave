"""Native media stays lossless and durable without copying immutable inputs."""

import base64
import errno
import io
import json

import pytest
from PIL import Image

from reweave.evolution.bundle import RequestLimits
from reweave.planner.native_context import prepare_native
from reweave.planner.providers.codex import CodexDecisionSession


def camera(color):
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(stream, format="PNG")
    return stream.getvalue()


def context(raw, *, urlsafe=False):
    encode = base64.urlsafe_b64encode if urlsafe else base64.b64encode
    return {
        "instructions": "current pixels",
        "tools": [],
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": [
                            {
                                "kind": "binary",
                                "media_type": "image/png",
                                "data": encode(raw).decode(),
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_reemphasized_pixels_reuse_one_readonly_file_across_turns(tmp_path):
    raw = camera("red")
    text, first = prepare_native(context(raw), tmp_path, RequestLimits())
    _, second = prepare_native(context(raw, urlsafe=True), tmp_path, RequestLimits())
    assert first == second
    assert first[0].read_bytes() == raw
    assert not first[0].stat().st_mode & 0o222
    assert json.loads(text)["messages"][0]["parts"][0]["content"] == [
        {"image_file": str(first[0])}
    ]
    _, changed = prepare_native(context(camera("blue")), tmp_path, RequestLimits())
    assert changed != first
    assert first[0].read_bytes() == raw
    assert len(list(tmp_path.glob("*.png"))) == 2


def test_prepared_media_and_request_and_launcher_share_bytes_without_audit_access(
    tmp_path,
):
    _, images = prepare_native(
        context(camera("red")), tmp_path / "records/images", RequestLimits()
    )
    source = images[0]
    session = CodexDecisionSession(tmp_path / "provider")
    request = tmp_path / "provider/requests/0123456789abcdef0123456789abcdef"
    archive = request / "images/0.png"
    archive.parent.mkdir(parents=True)
    session._archive_image(source, archive)
    submitted = session._submission_image_path(request, {"path": "head.png"}, 0)
    assert source.stat().st_ino == archive.stat().st_ino == submitted.stat().st_ino
    expected = source.read_bytes()
    source.unlink()
    assert archive.read_bytes() == submitted.read_bytes() == expected
    assert submitted.is_relative_to(session.directory / "launcher/images")
    assert not submitted.is_relative_to(session.directory / "requests")


def test_writable_external_images_keep_an_independent_immutable_snapshot(tmp_path):
    source, target = tmp_path / "caller.png", tmp_path / "archive.png"
    source.write_bytes(camera("red"))
    expected = source.read_bytes()
    CodexDecisionSession(tmp_path / "provider")._archive_image(source, target)
    assert source.stat().st_ino != target.stat().st_ino
    source.write_bytes(camera("blue"))
    assert target.read_bytes() == expected


def test_readonly_media_on_another_filesystem_is_copied_losslessly(
    tmp_path, monkeypatch
):
    _, images = prepare_native(
        context(camera("red")), tmp_path / "records/images", RequestLimits()
    )
    source, target = images[0], tmp_path / "archive.png"

    def cross_device(*args):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr("reweave.planner.providers.codex.os.link", cross_device)
    CodexDecisionSession(tmp_path / "provider")._archive_image(source, target)
    assert source.stat().st_ino != target.stat().st_ino
    assert target.read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob("tmp*"))


def test_other_link_errors_do_not_hide_a_broken_archive(tmp_path, monkeypatch):
    _, images = prepare_native(
        context(camera("red")), tmp_path / "records/images", RequestLimits()
    )
    source, target = images[0], tmp_path / "archive.png"

    def denied(*args):
        raise OSError(errno.EACCES, "archive denied")

    monkeypatch.setattr("reweave.planner.providers.codex.os.link", denied)
    with pytest.raises(OSError, match="archive denied"):
        CodexDecisionSession(tmp_path / "provider")._archive_image(source, target)
    assert not target.exists()


def test_readonly_external_input_is_snapshotted_before_its_owner_changes_it(tmp_path):
    source, target = tmp_path / "external.png", tmp_path / "archive.png"
    source.write_bytes(camera("red"))
    expected = source.read_bytes()
    source.chmod(0o444)
    session = CodexDecisionSession(tmp_path / "provider")
    session._archive_image(source, target)
    assert source.stat().st_ino != target.stat().st_ino
    source.chmod(0o644)
    source.write_bytes(camera("blue"))
    assert target.read_bytes() == expected
