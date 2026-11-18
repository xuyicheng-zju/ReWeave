# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Current corpus scopes and immutable local access boundaries."""

import pytest

from reweave.memory import ReadOnlyCorpus, tools


@pytest.mark.parametrize("scope", ["global", "task-family", "task-specific", "results"])
def test_current_scopes_are_read_only(tmp_path, scope):
    root = tmp_path / "memory"
    leaf = root / scope / "reference.md"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("reference text")
    bindings = ReadOnlyCorpus(root).get_common_tool_bindings()
    assert bindings["read_text_file"][1](str(leaf))["content"] == "reference text"
    with pytest.raises(PermissionError, match="writing to memory"):
        bindings["write_text_file"][1](str(leaf), "new text")
    assert leaf.read_text() == "reference text"


@pytest.mark.parametrize("scope", ["suite", "task_only"])
def test_removed_layout_is_rejected_even_beside_current_scopes(tmp_path, scope):
    root = tmp_path / "memory"
    (root / scope).mkdir(parents=True)
    (root / "task-family").mkdir()
    (root / "global").mkdir()
    with pytest.raises(ValueError, match="removed corpus directory"):
        ReadOnlyCorpus(root)
    with pytest.raises(PermissionError, match="reading this memory path"):
        tools.list_dir(str(root / scope), memory_root=root)


@pytest.mark.parametrize("operation", ["read", "write", "list"])
def test_symlink_cannot_escape_corpus_boundary(tmp_path, operation):
    root = tmp_path / "memory"
    (root / "global").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_text("outside bytes")
    link = root / "global" / "escape"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="escapes the corpus"):
        if operation == "read":
            tools.read_text_file(str(link / "note.md"), memory_root=root)
        elif operation == "write":
            tools.write_text_file(str(link / "note.md"), "changed", memory_root=root)
        else:
            tools.list_dir(str(link), memory_root=root)
    assert (outside / "note.md").read_text() == "outside bytes"
