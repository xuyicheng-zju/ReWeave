"""Production orchestration must carry validated topology through feedback/recovery."""

import asyncio
import json

import pytest

from reweave.evolution.distillation.engine import SharedEvolution, SharedPlan
from reweave.evolution.distillation.topology import SharedTopology, reconstruct
from tests.unit_tests.harness.evolution.test_shared import (
    Author,
    Evaluator,
    plan,
    task,
)
from tests.unit_tests.harness.evolution.test_shared_topology import (
    graphs_fixture,
    teachers,
)


def shared_plan(graphs):
    value = plan().model_dump(mode="json")
    value.update(
        method="aligned",
        unseen=[],
        sources=[task(n).model_dump(mode="json") for n in graphs],
        teachers=[t.model_dump(mode="json") for t in teachers(graphs)],
        max_rounds=2,
    )
    return SharedPlan.model_validate(value)


def test_joint_topology_reaches_author_and_failed_sharing_splits_before_next_round(
    tmp_path,
):
    graphs = graphs_fixture(tmp_path)
    p = shared_plan(graphs)
    author = Author()
    evaluator = Evaluator(
        lambda r, b: (
            b.record_id not in {"initial", "compact"} or r.instance.task == "beta"
        )
    )
    engine = SharedEvolution(tmp_path / "run", author, evaluator)
    result = asyncio.run(engine.run(p))
    assert len(author.requests) == 2
    first = SharedTopology.model_validate(author.requests[0]["shared_topology"])
    second = SharedTopology.model_validate(author.requests[1]["shared_topology"])
    assert any(len(g.members) > 1 for g in first.mechanisms.values())
    assert second.revision > first.revision
    assert all(len(g.members) == 1 for g in second.mechanisms.values())
    assert result["bundle"]["record_id"] == "bad"
    assert result["source_graphs"] == {
        n: g.model_dump(mode="json") for n, g in graphs.items()
    }
    frozen = SharedTopology.model_validate(result["shared_topology"])
    for name, graph in graphs.items():
        assert reconstruct(frozen, name) == graph
    before = len(evaluator.requests)
    assert asyncio.run(engine.run(p)) == result
    assert len(evaluator.requests) == before


def test_interrupted_author_replays_cached_receipts_and_same_topology_request(tmp_path):
    graphs = graphs_fixture(tmp_path)
    p = shared_plan(graphs)
    evaluator = Evaluator(lambda r, b: b.record_id != "initial")

    class Interrupting(Author):
        async def __call__(self, request, directory):
            if request["round"] == 2:
                raise RuntimeError("synthetic author interruption")
            return await super().__call__(request, directory)

    directory = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interruption"):
        asyncio.run(SharedEvolution(directory, Interrupting(), evaluator).run(p))
    before = len(evaluator.requests)
    original = json.loads((directory / "rounds/002/request.json").read_text())
    author = Author()
    asyncio.run(SharedEvolution(directory, author, evaluator).run(p))
    assert author.requests[1] == original
    # Only the previously unexecuted round-two source cells are new work.
    assert len(evaluator.requests) == before + len(graphs)


def test_native_author_query_service_and_topology_manifest_are_on_actual_call_path(
    tmp_path, monkeypatch
):
    from reweave.evolution.authoring.client import call_development_tool
    from reweave.evolution.distillation.author import NativeSharedAuthor
    from tests.unit_tests.harness.evolution.test_shared_tools import prepared_workspace

    request, workspace, _, topology, prepared = prepared_workspace(tmp_path)
    called = []

    async def check(*args, **kwargs):
        called.append("contract")

    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_runtime_contracts", check
    )
    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_interaction_suite", check
    )

    class SyntheticCodex(NativeSharedAuthor):
        async def _codex(
            self,
            workspace,
            directory,
            readable,
            writable,
            record,
            started,
            query_path,
            begin,
            instructions,
            prompt,
            url,
        ):
            begin()
            assert url
            result = await asyncio.to_thread(
                call_development_tool,
                url,
                "retrieve_graphs",
                {"query_task": "alpha", "top_k": 1},
            )
            assert result["matches"][0]["task"] == "alpha"
            site = {
                "pointers": ["/context/instructions"],
                "explanation": "Retain source-specific guards and phase dependencies in the planner.",
            }
            manifest = {
                "source_constraints": dict.fromkeys(topology.sources, site),
                "interfaces": {
                    m.key: site for g in topology.mechanisms.values() for m in g.members
                },
            }
            scratch = workspace / ".scratch"
            scratch.mkdir(exist_ok=True)
            (scratch / "shared_implementation.json").write_text(json.dumps(manifest))

    author = SyntheticCodex(
        bundle_loader=None,
        policy=prepared.policy,
        evidence_workspace=tmp_path,
        runtime_manifest=prepared.runtime_manifest,
    )
    result = asyncio.run(author(request, workspace.parent))
    assert result["topology"] == topology.model_dump(mode="json")
    assert set(result["implementation"]["source_constraints"]) == set(topology.sources)
    assert len(called) == 2
    assert asyncio.run(author(request, workspace.parent)) == result
    assert len(called) == 2


def test_rejected_constraint_manifest_is_corrected_in_original_session(
    tmp_path, monkeypatch
):
    from reweave.evolution.distillation.author import NativeSharedAuthor
    from tests.unit_tests.harness.evolution.test_shared_tools import prepared_workspace

    request, workspace, _, topology, prepared = prepared_workspace(tmp_path)

    async def check(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_runtime_contracts", check
    )
    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_interaction_suite", check
    )
    turns = []

    class RepairingCodex(NativeSharedAuthor):
        async def _codex(
            self,
            workspace,
            directory,
            readable,
            writable,
            record,
            started,
            query_path,
            begin,
            instructions,
            prompt,
            url,
        ):
            begin()
            turns.append(prompt)
            (directory / "session").mkdir(exist_ok=True)
            (directory / "session/identity.json").write_text(
                json.dumps({"session_id": "same-original-session"})
            )
            (directory / "native_result.json").write_text(
                json.dumps({"session_id": "same-original-session", "turn": len(turns)})
            )
            site = {
                "pointers": ["/context/instructions"],
                "explanation": "All private guards are represented by the instruction mechanism.",
            }
            raw = {
                "source_constraints": dict.fromkeys(topology.sources, site)
                if len(turns) > 1
                else {},
                "interfaces": {
                    m.key: site for g in topology.mechanisms.values() for m in g.members
                },
            }
            (workspace / ".scratch").mkdir(exist_ok=True)
            (workspace / ".scratch/shared_implementation.json").write_text(
                json.dumps(raw)
            )

    author = RepairingCodex(
        bundle_loader=None,
        policy=prepared.policy,
        evidence_workspace=tmp_path,
        runtime_manifest=prepared.runtime_manifest,
    )
    result = asyncio.run(author(request, workspace.parent))
    assert len(turns) == 2 and "admission_feedback" in turns[1]
    assert result["implementation"]["source_constraints"]
    feedback = list((workspace.parent / "audit/corrections").glob("*/feedback.json"))
    assert len(feedback) == 1
    assert "omits source-specific" in json.loads(feedback[0].read_text())["error"]
    assert (feedback[0].parent / "shared_implementation.json").exists()
    assert (
        json.loads((workspace.parent / "session/identity.json").read_text())[
            "session_id"
        ]
        == "same-original-session"
    )


def test_initial_shared_harness_below_loss_threshold_needs_no_author(tmp_path):
    graphs = graphs_fixture(tmp_path)
    p = shared_plan(graphs).model_copy(update={"loss_threshold": 0.05})
    author = Author()
    result = asyncio.run(
        SharedEvolution(tmp_path / "run", author, Evaluator(lambda r, b: True)).run(p)
    )
    assert not author.requests
    assert result["round"] == 0
    assert result["bundle"] == p.initial.model_dump(mode="json")
    assert result["termination"] == "loss_threshold_reached"
    assert result["score"]["delta_size"]["total_bytes"] == 0


@pytest.mark.parametrize("failures", [1, 4])
def test_direct_image_contract_failure_repairs_in_original_native_session(
    tmp_path, monkeypatch, failures
):
    from reweave.evolution.distillation.author import NativeSharedAuthor
    from reweave.runtime.capability import ModelInputRejected
    from tests.unit_tests.harness.evolution.test_shared_tools import prepared_workspace

    request, workspace, _, _, prepared = prepared_workspace(tmp_path)
    request = {
        **request,
        "method": "direct",
        "shared_topology": None,
        "source_graphs": {},
        "teachers": [{**teacher, "graph": None} for teacher in request["teachers"]],
    }
    workspace, _ = prepared.prepare(request, tmp_path / "direct")
    (workspace / "context/source_graphs.json").write_text("{}")
    seen = []
    validations = []

    async def runtime_contracts(*args, **kwargs):
        validations.append(True)
        if len(validations) <= failures:
            raise ModelInputRejected(
                "Native image contains invalid embedded base64 bytes"
            )

    async def interaction(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_runtime_contracts",
        runtime_contracts,
    )
    monkeypatch.setattr(
        "reweave.evolution.distillation.author.check_interaction_suite", interaction
    )

    class Repairing(NativeSharedAuthor):
        async def _codex(
            self,
            workspace,
            directory,
            readable,
            writable,
            record,
            started,
            query_path,
            begin,
            instructions,
            prompt,
            url,
        ):
            begin()
            seen.append(prompt)
            (directory / "session").mkdir(exist_ok=True)
            (directory / "session/identity.json").write_text(
                json.dumps({"session_id": "original-direct"})
            )
            (directory / "native_result.json").write_text(
                json.dumps({"session_id": "original-direct", "turn": len(seen)})
            )

    author = Repairing(
        bundle_loader=None,
        policy=prepared.policy,
        evidence_workspace=tmp_path,
        runtime_manifest=prepared.runtime_manifest,
    )
    result = asyncio.run(author(request, workspace.parent))
    assert result["bundle"] and len(seen) == len(validations) == failures + 1
    assert "valid encoded image bytes" in seen[1]
    assert "do not repeat the initial consolidation" in seen[1]
    assert "Read context/input.json and distill" not in seen[1]
    assert "phase=interaction" in seen[1]
    assert "source constraint manifest" not in seen[1]
    feedback = list((workspace.parent / "audit/corrections").glob("*/feedback.json"))
    assert len(feedback) == failures
    assert json.loads(feedback[0].read_text())["error_type"] == "ModelInputRejected"
    assert (
        json.loads((workspace.parent / "session/identity.json").read_text())[
            "session_id"
        ]
        == "original-direct"
    )
