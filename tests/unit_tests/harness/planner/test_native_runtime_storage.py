import json
from pathlib import Path

import pytest

from reweave.planner.providers.base import NativeProtocolError
from reweave.planner.providers.runtime_storage import online_runtime_home


def test_default_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    evidence = tmp_path / "episode"
    evidence.mkdir()
    home, resume = online_runtime_home(evidence, source_root=tmp_path)
    assert not resume
    assert home.parent.parent == tmp_path / ".codex/harness-online"
    (tmp_path / ".reweave-online-runtime.json").write_text("invalid")
    assert online_runtime_home(evidence, source_root=tmp_path) == (home, True)


def test_configured_root(tmp_path):
    evidence = tmp_path / "episode"
    evidence.mkdir()
    root = tmp_path / "private"
    (tmp_path / ".reweave-online-runtime.json").write_text(
        json.dumps({"runtime_root": str(root)})
    )
    home, resume = online_runtime_home(evidence, source_root=tmp_path)
    assert not resume
    assert home.parent.parent == root
    for path in (root, home.parent, home):
        assert path.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("value", ["relative", "", None, 42, "/tmp/../tmp"])
def test_invalid_config(tmp_path, value):
    (tmp_path / ".reweave-online-runtime.json").write_text(
        json.dumps({"runtime_root": value})
    )
    with pytest.raises(NativeProtocolError):
        online_runtime_home(tmp_path, source_root=tmp_path)


def test_nonprivate_root(tmp_path):
    root = tmp_path / "public"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    evidence = tmp_path / "episode"
    evidence.mkdir()
    (tmp_path / ".reweave-online-runtime.json").write_text(
        json.dumps({"runtime_root": str(root)})
    )
    with pytest.raises(NativeProtocolError):
        online_runtime_home(evidence, source_root=tmp_path)
    assert not (evidence / "runtime_home.json").exists()
