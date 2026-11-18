# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Readable author references preserve evidence, scope and graph identities."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.contracts import GraphAnalysisResult, OriginalReference
from reweave.evolution.evidence.trajectory import read_original
from tests.unit_tests.harness.evolution.fixtures import file_metadata


def catalog(tmp_path, name="workspace", originals=None):
    return AuthorReferences.create(
        tmp_path / name,
        registry=tmp_path / "run/author_references.json",
        evidence_roots={},
        originals=originals or {},
    )


def test_originals_share_one_file_and_resolve_exact_pointers(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"action":{"x/y":[1,2]}}\n{"action":null}\n')
    metadata = file_metadata(path)
    references = catalog(tmp_path, originals={str(path): metadata})
    raw = [
        {"path": str(path), **metadata, "pointer": pointer}
        for pointer in ("/0/action/x~1y/1", "/1/action", "")
    ]
    compact = references.encode({"references": raw})
    assert compact == {
        "references": ["file:001#/0/action/x~1y/1", "file:001#/1/action", "file:001#"]
    }
    assert references.decode(compact) == {"references": raw}
    assert read_original(references.original(compact["references"][0])) == 2
    assert read_original(references.original(compact["references"][1])) is None
    assert len(references.entries) == 1
    with pytest.raises(ValueError, match="raw artifact"):
        references.decode({"references": raw})
    with pytest.raises(ValueError, match="JSON pointer"):
        references.original("file:001#/0/~invalid")


def test_aliasing_keeps_host_paths_exact_without_touching_originals(
    tmp_path, monkeypatch
):
    path = "/remote-host/archive/trace.jsonl"
    references = catalog(
        tmp_path, originals={path: {"artifact_id": "original-file", "bytes": 10}}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "creating a reference must not resolve or open its original"
        )

    monkeypatch.setattr(Path, "resolve", forbidden)
    reference = {
        "path": path,
        "artifact_id": "original-file",
        "bytes": 10,
        "pointer": "/0/action",
    }
    assert references.decode(references.encode({"reference": reference})) == {
        "reference": reference
    }


def test_scope_is_local_and_names_are_stable_across_rounds(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('{"round":1}')
    b.write_text('{"round":2}')
    first = catalog(tmp_path, "first", {str(a): file_metadata(a)})
    second = catalog(tmp_path, "second", {str(b): file_metadata(b)})
    bundle = str("same implementation")
    assert first.identity("bundle", bundle) == second.identity("bundle", bundle)
    ref = first.reference(str(a), **file_metadata(a))
    with pytest.raises(ValueError, match="unknown author reference"):
        second.original(ref)
    with pytest.raises(ValueError, match="outside this request"):
        second.reference(str(a), **file_metadata(a))
    resumed = AuthorReferences.load(first.workspace)
    assert resumed.original(ref) == OriginalReference(
        path=str(a), **file_metadata(a), pointer=""
    )
    with pytest.raises(ValueError, match="expected episode"):
        resumed.resolve("bundle:001", kind="episode")


def test_literal_program_and_evidence_values_are_not_reference_registrations(tmp_path):
    references = catalog(tmp_path)
    references.identity("bundle", str("implementation"))
    literal = {
        "output": "bundle:001",
        "source": "def run(): return 'file:001#'",
        "instructions": {"value": "bundle:001"},
        "parameters": {"target": "bundle:001", "bundle_id": str("literal")},
        "roles": {"source": "file:001#"},
        "bindings": {"target": "episode:001"},
        "value": {
            "path": str(tmp_path / "foreign.json"),
            "artifact_id": "original-file",
            "bytes": 10,
            "pointer": "",
        },
    }
    assert references.encode(literal) == literal
    assert references.decode(literal) == literal
    assert list(references.entries) == ["bundle:001"]


def test_concurrent_and_stale_catalog_instances_do_not_lose_issued_names(tmp_path):
    initial = catalog(tmp_path)
    stale = AuthorReferences.load(initial.workspace)
    issued = initial.identity("bundle", str("initial"))
    stale.identity("bundle", str("later"))

    def allocate(index):
        references = AuthorReferences.load(initial.workspace)
        return references.identity("episode", str(index))

    with ThreadPoolExecutor(max_workers=4) as workers:
        names = list(workers.map(allocate, range(8)))
    restored = AuthorReferences.load(initial.workspace)
    assert restored.resolve(issued) == str("initial")
    assert len(set(names)) == 8
    assert [restored.resolve(name) for name in names] == [str(i) for i in range(8)]
    assert initial.resolve(names[-1]) == str(7)


def test_normalized_map_and_scaffold_roundtrip_with_domain_names(tmp_path):
    references = catalog(tmp_path)
    version, bundle, episode, condition, partition = [str(n) for n in range(5)]
    value = {
        "versions": {version: bundle},
        "cells": {version: {condition: episode}},
        "validation_cells": {version: {str("missing"): None}},
        "episodes": {
            episode: {
                "receipt_id": episode,
                "condition_id": condition,
                "bundle_id": bundle,
            }
        },
        "node_evidence": {
            "grasped": [
                {"reached_episode_ids": [episode], "completed_episode_ids": [episode]}
            ]
        },
        "edge_interfaces": {
            "pick": str("interface"),
            "retry": str("interface"),
        },
    }
    encoded = references.encode(value)
    assert encoded["cells"] == {"version:001": {"condition:001": "episode:001"}}
    assert encoded["versions"] == {"version:001": "bundle:001"}
    assert encoded["node_evidence"]["grasped"] == [
        {
            "reached_episode_ids": ["episode:001"],
            "completed_episode_ids": ["episode:001"],
        }
    ]
    assert encoded["edge_interfaces"] == {
        "pick": "interface:001",
        "retry": "interface:001",
    }
    assert references.decode(encoded) == value
    scaffold = {
        "policy": {},
        "episodes": [],
        "nodes": {partition: {"value": "bundle:001"}},
        "paths": {episode: [partition, None]},
        "transitions": [
            {
                "source": partition,
                "target": partition,
                "episode_id": episode,
                "frame_index": 1,
            }
        ],
    }
    assert references.decode(references.encode(scaffold)) == scaffold
    semantic = {
        "nodes": {"partition:001": {"meaning": "bundle:001", "predecessors": []}},
        "edges": {"edge": {"source": "partition:001", "target": "partition:001"}},
        "programs": {},
        "episodes": {episode: [{"status": "node", "node_id": "partition:001"}]},
        "unclassified": [],
    }
    assert references.decode(references.encode(semantic)) == semantic


def test_author_schema_rejects_raw_and_wrong_domain_episode_ids(tmp_path):
    references = catalog(tmp_path)
    schema = references.schema(GraphAnalysisResult.model_json_schema())
    witness = schema["$defs"]["TransitionWitness"]["properties"]["episode_id"]
    validate("episode:001", witness)
    for wrong in ("a" * 64, "bundle:001"):
        with pytest.raises(ValidationError):
            validate(wrong, witness)
    original = schema["$defs"]["OriginalReference"]
    validate("file:001#", original)
    validate("file:001#/0/action", original)
    with pytest.raises(ValidationError):
        validate(
            {
                "path": "/tmp/x",
                "artifact_id": "original-file",
                "bytes": 10,
                "pointer": "",
            },
            original,
        )


def test_version_and_validation_records_differ_from_matrix_tables(tmp_path):
    references = catalog(tmp_path)
    version, bundle, condition = (str(value) for value in range(3))
    value = {
        "evaluation": {
            "validation": {"condition_ids": [condition], "target_success_rate": 0.8}
        },
        "versions": [{"version_id": version, "bundle_id": bundle}],
        "navigation": {"versions": {version: {"label": "V000", "bundle_id": bundle}}},
    }
    compact = references.encode(value)
    assert compact["evaluation"]["validation"]["condition_ids"] == ["condition:001"]
    assert compact["versions"] == [
        {"version_id": "version:001", "bundle_id": "bundle:001"}
    ]
    assert compact["navigation"]["versions"] == {
        "version:001": {"label": "V000", "bundle_id": "bundle:001"}
    }
    assert references.decode(compact) == value


def test_version_and_comparison_views_share_the_same_condition_handles(tmp_path):
    references = catalog(tmp_path)
    version, condition, receipt = (str(value) for value in range(3))
    cell = {"receipt_id": receipt, "condition_id": condition}
    selected = {"version_id": version, **cell}
    views = [
        {
            "version_id": version,
            "cells": {condition: cell},
            "validation_cells": {condition: cell},
        },
        {
            "version_id": version,
            "dev": {condition: {"selected": selected, "contrasts": []}},
            "validation": {condition: {"selected": selected, "contrasts": []}},
        },
    ]
    for view in views:
        compact = references.encode(view)
        assert compact["version_id"] == "version:001"
        for pool in set(view) - {"version_id"}:
            assert set(compact[pool]) == {"condition:001"}
        assert references.decode(compact) == view
    assert set(references.entries) == {"version:001", "condition:001", "episode:001"}


@pytest.mark.parametrize(
    "name",
    [
        "source",
        "value",
        "instructions",
        "parameters",
        "signature",
        "bundle_id",
        "nodes",
    ],
)
def test_semantic_identifiers_are_not_field_names(tmp_path, name):
    path = tmp_path / "evidence.json"
    path.write_text("{}")
    metadata = file_metadata(path)
    references = catalog(tmp_path, originals={str(path): metadata})
    reference = {"path": str(path), **metadata, "pointer": ""}
    graph = {
        "nodes": {name: {"references": [reference], "meaning": "evidence"}},
        "edges": {name: {"source": name, "target": name, "references": [reference]}},
        "programs": {name: {"bundle": reference, "mechanism": "evidence"}},
    }
    encoded = references.encode(graph)
    assert encoded["nodes"][name]["references"] == ["file:001#"]
    assert encoded["programs"][name]["bundle"] == "file:001#"
    assert references.decode(encoded) == graph


def test_failed_registration_does_not_publish_partial_aliases(tmp_path):
    references = catalog(tmp_path)
    with pytest.raises(ValueError, match="outside this request"):
        references.encode(
            {
                "bundle_id": "a" * 64,
                "reference": {
                    "path": str(tmp_path / "foreign"),
                    "artifact_id": "foreign-file",
                    "bytes": 10,
                },
            }
        )
    assert references.entries == {}
    assert AuthorReferences.load(references.workspace).entries == {}
    assert references.identity("bundle", "c" * 64) == "bundle:001"


def test_batched_preparation_preserves_names_with_constant_catalog_writes(
    tmp_path, monkeypatch
):
    from reweave.evolution.authoring import references as module

    ordinary = catalog(tmp_path / "ordinary")
    expected = [ordinary.identity("episode", str(i)) for i in range(40)]
    batched = catalog(tmp_path / "batched")
    writes = []
    original = module.atomic_json

    def record_write(path, value):
        writes.append(path)
        original(path, value)

    monkeypatch.setattr(module, "atomic_json", record_write)
    with batched.batch():
        actual = [batched.identity("episode", str(i)) for i in range(40)]
    assert actual == expected
    assert batched.entries == ordinary.entries
    assert AuthorReferences.load(batched.workspace).entries == ordinary.entries
    assert len(writes) == 2


def test_failed_nested_registration_can_be_recovered_inside_batch(tmp_path):
    references = catalog(tmp_path)
    with references.batch():
        assert references.identity("episode", "settled") == "episode:001"
        with pytest.raises(ValueError, match="outside this request"):
            references.encode(
                {
                    "bundle_id": "unpublished",
                    "reference": {
                        "path": str(tmp_path / "foreign"),
                        "artifact_id": "x",
                        "bytes": 1,
                    },
                }
            )
        assert references.identity("bundle", "kept") == "bundle:001"
    restored = AuthorReferences.load(references.workspace)
    assert restored.entries == {"episode:001": "settled", "bundle:001": "kept"}


def test_aborted_batch_does_not_publish_its_partial_names(tmp_path):
    references = catalog(tmp_path)
    with pytest.raises(RuntimeError, match="abort"):
        with references.batch():
            references.identity("bundle", "not-issued")
            raise RuntimeError("abort")
    assert AuthorReferences.load(references.workspace).entries == {}
    assert references.identity("bundle", "actual") == "bundle:001"


def test_original_file_grants_preserve_exact_directory_boundaries():
    from reweave.evolution.authoring.author import _outside_evidence_roots

    roots = [Path("/archive/episode"), Path("/archive/second")]
    paths = [
        Path("/archive/episode"),
        Path("/archive/episode/image.png"),
        Path("/archive/second/nested/trace.jsonl"),
        Path("/archive/episode-extra/private.json"),
        Path("/archive"),
        Path("/other/original.json"),
    ]
    assert _outside_evidence_roots(paths, roots) == tuple(paths[3:])
    assert _outside_evidence_roots(paths, roots) == tuple(
        path for path in paths if not any(path.is_relative_to(root) for root in roots)
    )


def test_episode_virtual_reference_keeps_scope_and_pointer(tmp_path):
    identity = str("episode archive")
    root = tmp_path / "episode"
    root.mkdir()
    path = root / "trace.jsonl"
    path.write_text('{"value":7}\n')
    references = AuthorReferences.create(
        tmp_path / "workspace",
        registry=tmp_path / "registry.json",
        evidence_roots={identity: root},
    )
    raw = {
        "path": f".archive/episodes/{identity}/trace.jsonl",
        **file_metadata(path),
        "pointer": "/0/value",
    }
    handle = references.encode(raw)
    assert references.trace_reference(handle) == raw
    assert read_original(references.original(handle)) == 7
    with pytest.raises(ValueError, match="outside this request"):
        references.encode({**raw, "path": f".archive/episodes/{'f' * 64}/trace.jsonl"})
    with pytest.raises(ValueError, match="JSON pointer"):
        references.reference(str(path), **file_metadata(path), pointer="bad")
    assert json.loads(
        (references.workspace / ".host_inputs/author_references.json").read_text()
    )["entries"]
