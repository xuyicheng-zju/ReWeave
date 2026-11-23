# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Bound archive growth and publication work without discarding evidence."""

import base64
import json
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reweave.evolution.authoring.author import _prepared_evidence_access
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.evidence.preparation import (
    _evidence_roots,
    _history_paths,
    _native_evidence_paths,
)
from reweave.evolution.evidence.records import (
    archive_record,
    artifact_record,
    artifact_records,
    restore_record,
    store_bytes,
)
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.evolution.test_ablation_prompt_contract import (
    prepared_author,
)
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request


def test_nested_reference_allocations_do_not_copy_the_growing_catalog(tmp_path):
    refs = AuthorReferences.create(
        tmp_path / "author", registry=tmp_path / "registry.json", evidence_roots={}
    )

    class AppendOnlyEntries(UserDict):
        def __iter__(self):
            raise AssertionError("nested allocation must not scan the complete catalog")

    with refs.batch():
        refs.entries = AppendOnlyEntries()
        for index in range(1000):
            refs.identity("episode", str(index))
        refs.entries = refs.entries.data
    assert len(AuthorReferences.load(refs.workspace).entries) == 1000


@pytest.mark.parametrize("role", ["graph", "author"])
def test_preparation_bounds_catalog_publications_and_reuses_checked_inputs(
    tmp_path, monkeypatch, role
):
    from reweave.evolution.authoring import references

    writes = []
    original = references.atomic_json

    def publish(path, value):
        writes.append(path)
        original(path, value)

    monkeypatch.setattr(references, "atomic_json", publish)
    if role == "graph":
        request, _ = analysis_request(tmp_path)
        author = NativeGraphAnalysis(
            bundle_loader=lambda _: baseline(),
            policy=AuthorPolicy.model_validate(request.policy),
            evidence_workspace=tmp_path / "archive",
        )
        directory = tmp_path / "analysis"
        author._prepare_inputs(request, directory, resume=False)
        expected_roots = _evidence_roots(request)
        expected = (
            *_native_evidence_paths(request, expected_roots),
            *(
                p
                for p in _history_paths(request)
                if not any(p.is_relative_to(root) for root in expected_roots.values())
            ),
        )
        monkeypatch.setattr(
            "reweave.evolution.authoring.author._evidence_roots",
            lambda _: pytest.fail(
                "prepared access must not rescan historical evidence"
            ),
        )
        roots, files = _prepared_evidence_access(request, directory / "workspace")
        assert roots == expected_roots and files == expected
    else:
        request, author, directory = prepared_author(tmp_path, ("W", "C", "L"))
    assert len(writes) <= 5
    before = list(writes)
    monkeypatch.setattr(author, "_prepare", lambda *_: pytest.fail("already prepared"))
    author._prepare_inputs(request, directory, resume=True)
    assert writes == before
    changed = request.model_copy(update={"record_id": "changed-input"})
    with pytest.raises(ValueError, match="differs from its saved request"):
        author._prepare_inputs(changed, directory, resume=True)


def test_mixed_image_journal_batches_indexes_and_restores_exact_bytes(
    tmp_path, monkeypatch
):
    from reweave.evolution.evidence import records

    writes = []
    original = records.atomic_json

    def publish(path, value):
        writes.append(path)
        original(path, value)

    monkeypatch.setattr(records, "atomic_json", publish)
    value = {
        "images": [
            {
                "media_type": "image/png",
                "data_base64": base64.b64encode(bytes([i])).decode(),
            }
            for i in range(100)
        ]
        + [{"media_type": "image/png", "data_base64": "invalid\r\n🙂"}] * 100
    }
    archived = archive_record(value, tmp_path)
    assert len(writes) == 2 and restore_record(archived, tmp_path) == value
    writes.clear()
    assert archive_record(value, tmp_path) == archived and not writes
    # Reverse store order across concurrent journals: neither holds two locks.
    with ThreadPoolExecutor(max_workers=4) as workers:
        outputs = list(
            workers.map(
                lambda _: archive_record({"images": value["images"][::-1]}, tmp_path),
                range(8),
            )
        )
    assert all(
        restore_record(out, tmp_path)["images"] == value["images"][::-1]
        for out in outputs
    )
    assert not writes


def test_history_metadata_publishes_once_per_directory(tmp_path, monkeypatch):
    from reweave.evolution.evidence import records

    paths = []
    for index in range(4):
        directory = tmp_path / str(index)
        directory.mkdir()
        for number in range(80):
            path = directory / f"{number}.json"
            path.write_text("unchanged original")
            paths.append(path)
    first = artifact_record(paths[0])
    writes = []
    original = records.atomic_json

    def publish(path, value):
        writes.append(path)
        original(path, value)

    monkeypatch.setattr(records, "atomic_json", publish)
    published = artifact_records(paths)
    assert len(writes) == 4 and len(published) == 320
    assert published[0] == first
    writes.clear()
    assert artifact_records(paths) == published and not writes


def test_partial_image_publication_retains_already_written_evidence(
    tmp_path, monkeypatch
):
    original = Path.write_bytes
    calls = []

    def write(path, data):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("synthetic second-file publication failure")
        return original(path, data)

    value = {
        "images": [
            {"media_type": "image/png", "data_base64": base64.b64encode(data).decode()}
            for data in (b"first", b"second")
        ]
    }
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", write)
        with pytest.raises(OSError, match="second-file"):
            archive_record(value, tmp_path)
    index = json.loads((tmp_path / "observations/records.json").read_text())
    assert len(index) == 1
    assert store_bytes(b"first", tmp_path / "observations", suffix=".png") == index[0]
    assert (tmp_path / "observations" / index[0]["path"]).read_bytes() == b"first"
