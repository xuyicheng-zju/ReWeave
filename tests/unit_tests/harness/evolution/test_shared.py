"""Shared construction boundaries, selection, retention and crash recovery."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.evolution.bundle import AgentBundle
from reweave.evolution.contracts import EvaluationRecord, EvolutionIncomplete, Instance
from reweave.evolution.distillation.engine import (
    SharedEvolution,
    SharedPlan,
    SharedTask,
    SharedTeacher,
    candidate_score,
)
from reweave.evolution.distillation.metrics import harness_delta_size, harness_size
from tests.unit_tests.harness.evolution.fixtures import baseline, protocol, usage


def task(name):
    return SharedTask(
        name=name,
        family=name,
        dev=(Instance(record_id=name + "-dev", task=name, initialization={"seed": 0}),),
        test=(
            Instance(record_id=name + "-test", task=name, initialization={"seed": 1}),
        ),
    )


def bundle(name):
    value = baseline().model_dump(mode="json")
    value["record_id"] = name
    value["context"]["instructions"] = name
    return AgentBundle.model_validate(value)


def plan():
    source = task("source")
    teacher = bundle("teacher")
    return SharedPlan(
        record_id="shared-plan",
        conditional_retention=False,
        initialize_new_task_graphs=False,
        mode="consolidation",
        method="direct",
        sources=(source,),
        unseen=(task("unseen"),),
        teachers=(
            SharedTeacher(task="source", bundle=teacher, learned_tasks=("source",)),
        ),
        initial=bundle("initial"),
        execution_id="execution",
        manifest_id="manifest",
        conditions=protocol().execution,
        test_conditions=protocol().execution,
        max_rounds=2,
        loss_threshold=-1.0,
    )


class Author:
    def __init__(self):
        self.requests = []

    async def __call__(self, request, directory):
        self.requests.append(request)
        return {
            "bundle": bundle("compact" if request["round"] == 1 else "bad").model_dump(
                mode="json"
            )
        }


class Evaluator:
    def __init__(self, success=None):
        self.requests = []
        self.success = success or (
            lambda request, b: b.record_id not in {"bad", "initial"}
        )

    async def __call__(self, request, b, directory, *, before_launch):
        before_launch()
        self.requests.append(request)
        success = self.success(request, b)
        return EvaluationRecord(
            record_id=request.record_id + "-result",
            request=request,
            official_success=success,
            failure_origin="none" if success else "agent",
            infrastructure_retryable=None,
            usage=usage(),
            evidence=(
                {"path": "/fixture/evidence", "artifact_id": "fixture", "bytes": 0},
            ),
            episode=None,
            initialization=request.instance.initialization,
        )


def test_unseen_provenance_rejected():
    value = plan().model_dump(mode="json")
    value["initial_learned_tasks"] = ["unseen"]
    with pytest.raises(ValidationError, match="learned unseen"):
        SharedPlan.model_validate(value)
    value["initial_learned_tasks"] = []
    value["teachers"][0]["learned_tasks"] = ["unseen"]
    with pytest.raises(ValidationError, match="unseen-task provenance"):
        SharedPlan.model_validate(value)


def test_different_ids_cannot_hide_test_overlap():
    value = plan().model_dump(mode="json")
    value["sources"][0]["test"][0]["initialization"] = {"seed": 0}
    with pytest.raises(ValidationError, match="distinct"):
        SharedPlan.model_validate(value)


def test_heldout_teacher_evidence_rejected():
    p = plan()
    evaluator = Evaluator()
    request = __import__(
        "reweave.evolution.contracts", fromlist=["EvaluationRequest"]
    ).EvaluationRequest(
        record_id="leaked",
        bundle_id=p.teachers[0].bundle.record_id,
        instance=p.sources[0].test[0],
        split="test",
        manifest_id=p.manifest_id,
        execution=p.execution_id,
        conditions=p.conditions,
    )
    record = asyncio.run(
        evaluator(request, p.teachers[0].bundle, Path("."), before_launch=lambda: None)
    )
    value = p.model_dump(mode="json")
    value["teachers"][0]["evidence"] = [record.model_dump(mode="json")]
    with pytest.raises(ValidationError, match="source development only"):
        SharedPlan.model_validate(value)


def test_full_source_selection_then_frozen_unseen_test(tmp_path):
    p = plan()
    author, evaluator = Author(), Evaluator()
    engine = SharedEvolution(tmp_path, author, evaluator)
    with pytest.raises(ValueError, match="freeze"):
        asyncio.run(engine.test(p))
    result = asyncio.run(engine.run(p))
    assert result["bundle"]["record_id"] == "compact"
    assert all("unseen" not in json.dumps(r) for r in author.requests)
    assert all(r.split == "dev" for r in evaluator.requests)
    # A failing later candidate does not replace the feasible shared student.
    assert (
        json.loads((tmp_path / "rounds/002/score.json").read_text())["eligible"]
        is False
    )
    report = asyncio.run(engine.test(p))
    assert report["unseen"]["unseen"]["successes"] == 1
    assert len(author.requests) == 2
    count = len(evaluator.requests)
    assert asyncio.run(engine.run(p)) == result
    asyncio.run(engine.test(p))
    assert len(evaluator.requests) == count
    assert len(author.requests) == 2


def test_phase_anchor_never_rolls_forward():
    p = plan().model_copy(update={"retention_tolerance": 0.1})
    assert candidate_score(p, bundle("x"), {"source": 0.9}, {"source": 1})["eligible"]
    assert not candidate_score(p, bundle("x"), {"source": 0.8}, {"source": 1})[
        "eligible"
    ]


def test_size_changes_selection_only_after_retention():
    p = plan()
    short, long = bundle("x"), bundle("x" * 1000)
    a = candidate_score(p, short, {"source": 1}, {"source": 1})
    b = candidate_score(p, long, {"source": 1}, {"source": 1})
    assert a["objective"] > b["objective"]
    assert not candidate_score(p, short, {"source": 0}, {"source": 1})["eligible"]
    assert harness_size(long)["total_bytes"] - harness_size(short)["total_bytes"] == 999


def test_no_feasible_candidate_is_not_a_published_success(tmp_path):
    author, evaluator = Author(), Evaluator(lambda request, b: b.record_id == "teacher")
    engine = SharedEvolution(tmp_path, author, evaluator)
    with pytest.raises(EvolutionIncomplete, match="no shared candidate"):
        asyncio.run(engine.run(plan()))
    assert not (tmp_path / "frozen.json").exists()


def test_unknown_cell_stops_selection_and_is_not_replayed(tmp_path):
    class Unknown(Evaluator):
        async def __call__(self, *args, **kwargs):
            result = await super().__call__(*args, **kwargs)
            return result.model_copy(
                update={"official_success": False, "failure_origin": "unknown"}
            )

    evaluator = Unknown()
    engine = SharedEvolution(tmp_path, Author(), evaluator)
    for _ in range(2):
        with pytest.raises(EvolutionIncomplete, match="unsettled"):
            asyncio.run(engine.run(plan()))
    assert len(evaluator.requests) == 1


def test_started_missing_result_needs_recovery_not_repeat(tmp_path):
    class Crash(Evaluator):
        async def __call__(self, *args, **kwargs):
            await super().__call__(*args, **kwargs)
            raise RuntimeError("interrupted")

    evaluator = Crash()
    engine = SharedEvolution(tmp_path, Author(), evaluator)
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(engine.run(plan()))
    with pytest.raises(EvolutionIncomplete, match="owner recovery"):
        asyncio.run(engine.run(plan()))
    assert len(evaluator.requests) == 1


def test_changed_plan_cannot_reuse_run(tmp_path):
    engine = SharedEvolution(tmp_path, Author(), Evaluator())
    asyncio.run(engine.run(plan()))
    with pytest.raises(ValueError, match="published identity"):
        asyncio.run(engine.run(plan().model_copy(update={"size_weight_per_kb": 0.1})))


def test_task_macro_is_not_episode_weighted():
    from reweave.evolution.contracts import EvaluationRequest
    from reweave.evolution.distillation.engine import task_scores

    first, second = task("one"), task("two")
    second = second.model_copy(
        update={
            "dev": tuple(
                Instance(record_id=f"two-{i}", task="two", initialization={"seed": i})
                for i in range(5)
            )
        }
    )
    p = plan()
    evaluator = Evaluator(lambda r, b: r.instance.task == "one")
    records = []
    for t in (first, second):
        for i in t.dev:
            r = EvaluationRequest(
                record_id=i.record_id,
                bundle_id="b",
                instance=i,
                split="dev",
                manifest_id="m",
                execution="e",
                conditions=p.conditions,
            )
            records.append(
                asyncio.run(
                    evaluator(r, bundle("b"), Path("."), before_launch=lambda: None)
                )
            )
    scores = task_scores((first, second), records)
    assert candidate_score(p, bundle("b"), scores, {})["macro_success"] == 0.5


def test_source_encoder_preserves_graph_and_rejects_topology_change(tmp_path):
    from reweave.evolution.authoring.config import AuthorPolicy
    from reweave.evolution.contracts import SemanticGraph
    from reweave.evolution.distillation.conditions import (
        NativeSourceConditions,
        encoding_request,
    )
    from tests.unit_tests.harness.evolution.fixtures import graph_proposal_fields
    from tests.unit_tests.harness.evolution.test_graph_retrieval import with_matching

    fields = graph_proposal_fields(tmp_path / "originals")
    graph = SemanticGraph.model_validate_json(Path(fields["graph"].path).read_text())
    p = plan()
    p = p.model_copy(
        update={"teachers": (p.teachers[0].model_copy(update={"graph": graph}),)}
    )
    request = encoding_request(p)
    assert "unseen" not in json.dumps(request)
    author = NativeSourceConditions(
        bundle_loader=None,
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path,
    )
    workspace = tmp_path / "workspace"
    (workspace / "candidate").mkdir(parents=True)
    path = workspace / "candidate/conditions.json"
    semantics = with_matching(graph).matching.model_dump(mode="json")
    path.write_text(json.dumps({"source": semantics}))
    result = asyncio.run(author.admit(request, workspace, tmp_path / "audit"))
    encoded = SemanticGraph.model_validate(result["graphs"]["source"])
    assert encoded.model_dump(exclude={"matching"}) == graph.model_dump(
        exclude={"matching"}
    )
    semantics["nodes"]["invented"] = semantics["nodes"]["goal"]
    path.write_text(json.dumps({"source": semantics}))
    with pytest.raises(ValidationError, match="canonical nodes"):
        asyncio.run(author.admit(request, workspace, tmp_path / "audit"))


def test_native_shared_workspace_contains_only_source_inputs(tmp_path):
    from types import SimpleNamespace

    from reweave.evolution.authoring.config import AuthorPolicy
    from reweave.evolution.authoring.workspace import native_output
    from reweave.evolution.distillation.author import NativeSharedAuthor
    from reweave.evolution.distillation.conditions import encoding_request

    manifest = SimpleNamespace(
        authoring_contract=SimpleNamespace(
            operations={}, tools=[], native_image_contract=None
        ),
        configuration={"limits": {"episode": {}}},
    )
    author = NativeSharedAuthor(
        bundle_loader=None,
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path,
        runtime_manifest=manifest,
    )
    request = encoding_request(plan())
    request["size_reference"] = plan().initial.model_dump(mode="json")
    workspace, readable = author.prepare(request, tmp_path / "job")
    assert readable == ()
    assert native_output(workspace) == workspace / "candidate"
    author_input = json.loads((workspace / "context/input.json").read_text())
    assert author_input["size_reference"] == "context/size_reference"
    assert (workspace / "context/size_reference").is_dir()
    assert not (tmp_path / "job/preparation").exists()
    for path in workspace.rglob("*"):
        if path.is_file():
            assert "unseen" not in path.read_text()
    assert author.prepare(request, tmp_path / "job")[0] == workspace


def test_zero_source_h0_is_measured_on_fixed_panels_not_assumed_to_fail(tmp_path):
    value = plan().model_dump(mode="json")
    value.update(
        mode="baseline",
        sources=[],
        teachers=[],
        unseen=[],
        max_rounds=0,
        conditional_retention=False,
    )
    p = SharedPlan.model_validate(value)
    author = Author()
    evaluator = Evaluator(lambda request, b: request.instance.task == "fixed")
    engine = SharedEvolution(tmp_path, author, evaluator)
    frozen = asyncio.run(engine.run(p))
    assert not author.requests and not evaluator.requests
    assert frozen["score"] is None
    panels = {"fixed_seed": (task("fixed"),), "ood": (task("new"),)}
    report = asyncio.run(engine.test_panels(p, panels))
    assert report["panels"]["fixed_seed"]["fixed"]["successes"] == 1
    assert report["panels"]["ood"]["new"]["successes"] == 0
    assert len(evaluator.requests) == 2
    assert all(
        r.split == "test" and r.bundle_id == p.initial.record_id
        for r in evaluator.requests
    )
    asyncio.run(engine.test_panels(p, panels))
    assert len(evaluator.requests) == 2


def test_successful_large_harness_keeps_evolving_until_loss_threshold(tmp_path):
    class CompressingAuthor(Author):
        async def __call__(self, request, directory):
            self.requests.append(request)
            name = "x" * 100_000 if request["round"] == 1 else "compact"
            return {"bundle": bundle(name).model_dump(mode="json")}

    p = plan().model_copy(
        update={"unseen": (), "loss_threshold": 0.05, "max_rounds": 10}
    )
    author = CompressingAuthor()
    result = asyncio.run(SharedEvolution(tmp_path, author, Evaluator()).run(p))
    assert len(author.requests) == result["rounds_executed"] == 2
    assert result["round"] == 2
    assert result["termination"] == "loss_threshold_reached"
    assert result["loss_threshold_reached"] is True
    assert result["score"]["loss"] < 0.05
    assert author.requests[1]["size_reference"] == p.initial.model_dump(mode="json")
    first = json.loads((tmp_path / "rounds/001/score.json").read_text())
    assert first["macro_success"] == 1 and first["loss"] > 0.05


def test_loss_equal_to_threshold_exhausts_round_budget_and_reports_unmet(tmp_path):
    class SameAuthor(Author):
        async def __call__(self, request, directory):
            self.requests.append(request)
            return {"bundle": bundle("compact").model_dump(mode="json")}

    p = plan()
    threshold = candidate_score(p, bundle("compact"), {"source": 1}, {"source": 1})[
        "loss"
    ]
    p = p.model_copy(update={"loss_threshold": threshold})
    author = SameAuthor()
    result = asyncio.run(SharedEvolution(tmp_path, author, Evaluator()).run(p))
    assert len(author.requests) == result["rounds_executed"] == 2
    assert result["termination"] == "round_limit"
    assert result["loss_threshold_reached"] is False
    assert result["score"]["loss"] == threshold


@pytest.mark.parametrize(
    "updates", [{"loss_threshold": None}, {"stop_on_source_success": True}]
)
def test_legacy_success_only_policy_cannot_start_new_consolidation(tmp_path, updates):
    author, evaluator = Author(), Evaluator()
    with pytest.raises(ValueError, match="explicit loss_threshold"):
        asyncio.run(
            SharedEvolution(tmp_path, author, evaluator).run(
                plan().model_copy(update=updates)
            )
        )
    assert not author.requests and not evaluator.requests


def test_consolidation_cannot_enable_new_task_success_shortcut():
    raw = plan().model_dump(mode="json")
    raw["new_tasks"] = ["source"]
    with pytest.raises(ValidationError, match="continual mode"):
        SharedPlan.model_validate(raw)


def test_low_loss_never_overrides_source_retention(tmp_path):
    p = plan().model_copy(update={"loss_threshold": 10.0})
    author = Author()
    evaluator = Evaluator(lambda r, b: b.record_id == "teacher")
    with pytest.raises(EvolutionIncomplete, match="no shared candidate"):
        asyncio.run(SharedEvolution(tmp_path, author, evaluator).run(p))
    assert len(author.requests) == p.max_rounds
    assert not (tmp_path / "frozen.json").exists()


def test_delta_size_excludes_base_and_counts_equal_length_replacements():
    base = bundle("alpha")
    identical = base.model_copy(update={"record_id": "new-provenance"})
    assert harness_delta_size(identical, base)["total_bytes"] == 0
    changed = bundle("bravo")
    assert harness_size(changed)["total_bytes"] == harness_size(base)["total_bytes"]
    assert harness_delta_size(changed, base)["text_bytes"] == 5
    assert harness_delta_size(bundle("你好"), base)["text_bytes"] == 6


def test_delta_size_does_not_offset_additions_with_deletions():
    base = bundle("old\n" * 100)
    changed = bundle("new\n")
    delta = harness_delta_size(changed, base)
    assert delta["total_bytes"] == 4 and delta["removed_bytes"] == 400
    assert harness_size(changed)["total_bytes"] < harness_size(base)["total_bytes"]


def test_delta_size_counts_new_workflow_source_and_interfaces():
    from tests.unit_tests.harness.evolution.fixtures import workflow

    base = bundle("base")
    value = base.model_dump(mode="json")
    value["workflows"] = [workflow().model_dump(mode="json")]
    changed = AgentBundle.model_validate(value)
    delta = harness_delta_size(changed, base)
    assert delta["source_bytes"] > 0 and delta["interface_bytes"] > 0
    assert (
        delta["total_bytes"]
        == harness_size(changed)["total_bytes"] - harness_size(base)["total_bytes"]
    )


def test_delta_score_uses_fixed_reference_instead_of_parent():
    p = plan().model_copy(
        update={"size_reference": bundle("original"), "initial": bundle("parent")}
    )
    score = candidate_score(p, bundle("original"), {"source": 1}, {"source": 1})
    assert score["delta_size"]["total_bytes"] == 0
    assert score["loss"] == -p.size_weight_per_kb * score["teacher_delta_mean_kb"]
    assert score["objective"] == 1 - score["loss"]


def test_teacher_delta_mean_is_a_fixed_offset_not_a_moving_parent_target():
    p = plan()
    a = candidate_score(p, bundle("first"), {"source": 1}, {"source": 1})
    b = candidate_score(p, bundle("second"), {"source": 1}, {"source": 1})
    expected = harness_delta_size(p.teachers[0].bundle, p.initial)["size_kb"]
    assert a["teacher_delta_mean_kb"] == b["teacher_delta_mean_kb"] == expected
    for score in (a, b):
        assert score["delta_excess_kb"] == score["delta_size"]["size_kb"] - expected
        assert score["loss"] == p.size_weight_per_kb * score["delta_excess_kb"]
