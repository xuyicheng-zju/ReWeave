# Modified for the ReWeave namespace.
"""Repeated reference encoding keeps issued output without republishing catalogs."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import reweave.evolution.authoring.references as reference_module
from reweave.evolution.authoring.references import AuthorReferences


def catalog(tmp_path, name, originals=None):
    return AuthorReferences.create(
        tmp_path / name,
        registry=tmp_path / "run/author_references.json",
        evidence_roots={},
        originals=originals or {},
    )


def test_repeated_encoding_preserves_output_and_never_republishes_catalog(
    tmp_path, monkeypatch
):
    original = "/immutable/episode/results.json"
    metadata = {"artifact_id": "published-file", "bytes": 20}
    references = catalog(tmp_path, "author", {original: metadata})
    raw = {
        "episode_id": str("episode"),
        "reference": {"path": original, **metadata, "pointer": "/result"},
        "value": {"episode_id": "literal evidence"},
    }
    issued = references.encode(raw)
    assert issued == {
        "episode_id": "episode:001",
        "reference": "file:001#/result",
        "value": {"episode_id": "literal evidence"},
    }
    path = references.workspace / ".host_inputs/author_references.json"
    before = path.read_bytes(), references.registry.read_bytes()

    def forbidden_write(*args, **kwargs):
        raise AssertionError("already-issued references require no publication")

    monkeypatch.setattr(reference_module, "atomic_json", forbidden_write)
    for _ in range(3):
        assert references.encode(raw) == issued
        assert references.decode(issued) == raw
    assert references.identity("episode", str("episode")) == "episode:001"
    assert (path.read_bytes(), references.registry.read_bytes()) == before


def test_existing_global_reference_is_persisted_on_first_use_by_a_sibling(tmp_path):
    original = "/immutable/shared.json"
    metadata = {"artifact_id": "published-file", "bytes": 20}
    first = catalog(tmp_path, "first", {original: metadata})
    second = catalog(tmp_path, "second", {original: metadata})
    handle = first.reference(original, **metadata)
    before = second.registry.read_bytes()
    assert second.reference(original, **metadata) == handle
    assert second.registry.read_bytes() == before
    restored = AuthorReferences.load(second.workspace)
    assert restored.original(handle).path == original
    assert restored.original(handle).artifact_id == metadata["artifact_id"]


def test_new_reference_after_noop_persists_without_granting_sibling_access(tmp_path):
    first_path, second_path = "/immutable/first.json", "/immutable/second.json"
    first = catalog(
        tmp_path, "first", {first_path: {"artifact_id": "first-file", "bytes": 10}}
    )
    second = catalog(
        tmp_path, "second", {second_path: {"artifact_id": "second-file", "bytes": 10}}
    )
    issued = first.reference(first_path, "first-file", 10)
    assert first.reference(first_path, "first-file", 10) == issued
    new = second.reference(second_path, "second-file", 10)
    assert issued == "file:001#" and new == "file:002#"
    restored = AuthorReferences.load(second.workspace)
    assert restored.original(new).path == second_path
    with pytest.raises(ValueError, match="unknown author reference"):
        restored.original(issued)
    with pytest.raises(ValueError, match="outside this request"):
        restored.reference(first_path, "first-file", 10)
    assert set(AuthorReferences.load(second.workspace).entries) == {"file:002"}


def test_noop_stale_instances_do_not_overwrite_concurrent_new_references(tmp_path):
    initial = catalog(tmp_path, "author")
    bundle_id = "common-bundle"
    common = initial.identity("bundle", bundle_id)
    stale = [AuthorReferences.load(initial.workspace) for _ in range(12)]

    def allocate(index):
        references = stale[index]
        assert references.identity("bundle", bundle_id) == common
        return references.identity("episode", str(index))

    with ThreadPoolExecutor(max_workers=4) as workers:
        names = list(workers.map(allocate, range(len(stale))))
    restored = AuthorReferences.load(initial.workspace)
    assert len(set(names)) == len(stale)
    assert restored.resolve(common) == bundle_id
    assert [restored.resolve(name) for name in names] == [
        str(index) for index in range(len(stale))
    ]
    registry = json.loads(initial.registry.read_text())
    assert set(registry) == set(restored.entries)
