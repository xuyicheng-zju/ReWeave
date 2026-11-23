# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reweave.evaluation.result import write_json_atomic


def test_write_json_atomic_replaces_complete_record(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"status": "old"}\n', encoding="utf-8")

    result = write_json_atomic(destination, {"status": "complete", "success": True})

    assert result == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "complete",
        "success": True,
    }
    assert not (tmp_path / ".result.json.tmp").exists()


def test_write_json_atomic_cleans_temporary_file_on_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"status": "old"}\n', encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_json_atomic(destination, {"status": "new"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "old"}
    assert not (tmp_path / ".result.json.tmp").exists()
