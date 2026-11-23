# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reweave.utils.persistence import atomic_text


def test_atomic_text_replaces_complete_record(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"status": "old"}\n', encoding="utf-8")

    atomic_text(destination, '{"status": "complete", "success": true}\n')

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "complete",
        "success": True,
    }
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_text_cleans_temporary_file_on_replace_error(
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
        atomic_text(destination, '{"status": "new"}\n')

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "old"}
    assert list(tmp_path.iterdir()) == [destination]
