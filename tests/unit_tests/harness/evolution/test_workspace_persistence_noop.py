# Modified for the ReWeave namespace.
"""Reusing unchanged metadata does not rewrite the complete evolution archive."""

import copy
import json
import stat

import pytest

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    BundleVersion,
    EvaluationRecord,
    EvaluationRequest,
    EvolutionState,
)
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from tests.unit_tests.harness.evolution.fixtures import baseline, protocol
from tests.unit_tests.harness.evolution.test_graph_evolution import record


@pytest.fixture
def anyio_backend():
    return "asyncio"


def writes_to_state(monkeypatch, workspace):
    from reweave.evolution.search import engine as evolution

    original = evolution.atomic_text
    writes = []

    def publish(path, value):
        if path == workspace.state_path:
            writes.append(json.loads(value))
        original(path, value)

    monkeypatch.setattr(evolution, "atomic_text", publish)
    return writes


def test_repeated_existing_version_does_not_republish_unchanged_state(
    tmp_path, monkeypatch
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    workspace.put_bundle(baseline())
    version = BundleVersion(
        record_id=new_record_id(),
        bundle_id=baseline().record_id,
        proposal_id=None,
        branch_id=None,
    )
    with workspace.transaction() as state:
        identity = workspace.add_version(state, version)
        original = copy.deepcopy(state)
        before = workspace.state_path.read_bytes()
        writes = writes_to_state(monkeypatch, workspace)
        for _ in range(5):
            assert workspace.add_version(state, version) == identity
        assert state == original and workspace.state_path.read_bytes() == before
        assert writes == []
        # The no-op still consumes the required Bundle archive.
        (workspace.root / "bundles" / f"{version.bundle_id}.json").unlink()
        with pytest.raises(FileNotFoundError):
            workspace.add_version(state, version)
        assert writes == []


def test_new_version_and_missing_order_are_validated_and_persisted(
    tmp_path, monkeypatch
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    workspace.put_bundle(baseline())
    version = BundleVersion(
        record_id=new_record_id(),
        bundle_id=baseline().record_id,
        proposal_id=None,
        branch_id=None,
    )
    with workspace.transaction() as state:
        writes = writes_to_state(monkeypatch, workspace)
        identity = workspace.add_version(state, version)
        assert len(writes) == 1
        assert json.loads(workspace.state_path.read_text()) == state
        state["order"].clear()
        assert workspace.add_version(state, version) == identity
        assert len(writes) == 2
        persisted = EvolutionState.model_validate_json(workspace.state_path.read_text())
        assert persisted.order == (identity,)
        assert persisted.versions[identity] == version
        original = workspace.state_path.read_bytes()
        state["versions"][identity]["branch_id"] = "modified"
        with pytest.raises(ValueError, match="immutable bundle version changed"):
            workspace.add_version(state, version)
        assert len(writes) == 2 and workspace.state_path.read_bytes() == original


async def complete_evaluation(workspace, state):
    request = EvaluationRequest(
        record_id=new_record_id(),
        bundle_id=baseline().record_id,
        instance=workspace.protocol.dev[0],
        split="dev",
        manifest_id=workspace.protocol.manifest_id,
        execution=workspace.protocol.execution_id,
        conditions=workspace.protocol.execution,
    )
    calls = []

    async def evaluate(directory, before_launch):
        before_launch()
        calls.append(directory)
        return record(request, directory, False)

    evolution = GraphEvolution(workspace, None, None)
    result = await evolution._attempt(
        state,
        "evaluation",
        "evaluation/noop/condition-001/attempt-01",
        request,
        evaluate,
        EvaluationRecord,
    )
    return evolution, request, result, calls


@pytest.mark.anyio
async def test_compact_state_preserves_values_receipts_and_file_directory_durability(
    tmp_path, monkeypatch
):
    from reweave.utils import persistence

    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    with workspace.transaction() as state:
        _, request, _, calls = await complete_evaluation(workspace, state)
        configuration = workspace.root / "configuration.json"
        archived_request = calls[0] / "request.json"
        persistence.atomic_json(configuration, {"original": {"format": "unchanged"}})
        persistence.atomic_json(archived_request, request.model_dump(mode="json"))
        originals = {
            path: path.read_bytes()
            for path in (configuration, archived_request, calls[0] / "result.json")
        }
        assert b'\n  "original"' in originals[configuration]
        expected = EvolutionState.model_validate(state).model_dump(mode="json")
        synced = []
        original_sync = persistence.os.fsync

        def sync(descriptor):
            synced.append(
                "directory"
                if stat.S_ISDIR(persistence.os.fstat(descriptor).st_mode)
                else "file"
            )
            original_sync(descriptor)

        monkeypatch.setattr(persistence.os, "fsync", sync)
        workspace.save(state)
        raw = workspace.state_path.read_text()
        assert raw.count("\n") == 1
        assert len(raw) < len(json.dumps(expected, sort_keys=True, indent=2) + "\n")
        assert json.loads(raw) == expected
        assert (
            EvolutionState.model_validate_json(raw).model_dump(mode="json") == expected
        )
        assert synced == ["file", "directory"]
        assert {path: path.read_bytes() for path in originals} == originals


@pytest.mark.anyio
async def test_cached_receipt_is_still_read_and_verified_without_state_write(
    tmp_path, monkeypatch
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    with workspace.transaction() as state:
        evolution, request, result, calls = await complete_evaluation(workspace, state)
        before = workspace.state_path.read_bytes()
        original = copy.deepcopy(state)
        writes = writes_to_state(monkeypatch, workspace)
        verified = []

        def verify(request, directory):
            verified.append(request.record_id)
            return EvaluationRecord.model_validate_json(
                (directory / "result.json").read_text()
            )

        async def forbidden(*args, **kwargs):
            raise AssertionError("completed receipt must not launch another evaluation")

        for _ in range(3):
            reused = await evolution._attempt(
                state,
                "evaluation",
                "evaluation/noop/condition-001/attempt-01",
                request,
                forbidden,
                EvaluationRecord,
                verify_result=verify,
            )
            assert reused == result
        assert verified == [request.record_id] * 3 and len(calls) == 1
        assert state == original and workspace.state_path.read_bytes() == before
        assert writes == []
        (calls[0] / "result.json").write_text("{}")
        with pytest.raises(ValueError):
            await evolution._attempt(
                state,
                "evaluation",
                "evaluation/noop/condition-001/attempt-01",
                request,
                forbidden,
                EvaluationRecord,
                verify_result=verify,
            )
        assert writes == [] and workspace.state_path.read_bytes() == before


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["prepared", "running", "failed", "completed"])
async def test_receipt_completion_changes_still_publish_state(
    tmp_path, monkeypatch, status
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    with workspace.transaction() as state:
        evolution, request, result, calls = await complete_evaluation(workspace, state)
        attempt = state["attempts"]["evaluation/noop/condition-001/attempt-01"]
        attempt.update(
            status=status, result_id="e" * 64 if status == "completed" else None
        )
        workspace.save(state)
        writes = writes_to_state(monkeypatch, workspace)

        async def forbidden(*args, **kwargs):
            raise AssertionError("published receipt must settle without another launch")

        reused = await evolution._attempt(
            state,
            "evaluation",
            "evaluation/noop/condition-001/attempt-01",
            request,
            forbidden,
            EvaluationRecord,
        )
        assert reused == result and len(calls) == 1
        assert len(writes) == 1
        persisted = EvolutionState.model_validate_json(workspace.state_path.read_text())
        assert (
            persisted.attempts["evaluation/noop/condition-001/attempt-01"].status
            == "completed"
        )
        assert (
            persisted.attempts["evaluation/noop/condition-001/attempt-01"].result_id
            == result.record_id
        )
