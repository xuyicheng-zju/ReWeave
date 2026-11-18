"""Conditional losses must affect selection without laundering missing evidence."""

import asyncio

import pytest

from reweave.evolution.distillation.engine import SharedEvolution
from reweave.evolution.distillation.transfer import (
    ConditionalObservation,
    MechanismReference,
    conditional_rates,
    transfer_loss,
)
from tests.unit_tests.harness.evolution.test_shared import (
    Author,
    Evaluator,
    bundle,
    plan,
)


def reference():
    return MechanismReference(
        key="source:grasp",
        task="source",
        entry="object reachable",
        exit="object retained",
        source_bundle_id="teacher",
    )


def rate(value):
    return {
        "lower": value,
        "upper": value,
        "observed_entries": 1,
        "known_outcomes": 1,
        "unknown_entries": 0,
        "absent_entries": 0,
    }


class Assessor:
    def __init__(self):
        self.requests = []

    async def __call__(self, request, directory):
        self.requests.append(request)
        candidate = request["parent"]["record_id"]
        # Both candidates complete the task. The smaller compact candidate
        # loses the required local mechanism; bad preserves it.
        return {
            "rates": {
                "source:grasp": rate(0 if candidate in {"initial", "compact"} else 1)
            }
        }


def test_local_loss_changes_winner_and_remains_source_only(tmp_path):
    p = plan().model_copy(
        update={"conditional_retention": True, "mechanism_references": (reference(),)}
    )
    assessor = Assessor()
    evaluator = Evaluator(lambda request, b: b.record_id != "initial")
    engine = SharedEvolution(tmp_path, Author(), evaluator, assessor)
    result = asyncio.run(engine.run(p))
    assert result["bundle"]["record_id"] == "bad"
    assert result["score"]["conditional_loss"] == 0
    assert len(assessor.requests) == 4
    assert all("unseen" not in str(r) for r in assessor.requests)
    assert result["mechanism_references"][0]["source_bundle_id"] == "bad"
    calls = len(assessor.requests)
    assert asyncio.run(engine.run(p)) == result
    asyncio.run(engine.test(p))
    assert len(assessor.requests) == calls


def test_conditional_mode_cannot_silently_fall_back(tmp_path):
    p = plan().model_copy(update={"conditional_retention": True})
    with pytest.raises(ValueError, match="independent assessor"):
        asyncio.run(SharedEvolution(tmp_path, Author(), Evaluator()).run(p))


def test_loss_uses_positive_drop_and_uncertainty_bounds():
    ref = reference()
    assert (
        transfer_loss((ref,), {ref.key: rate(0.5)}, {ref.key: rate(1)})[
            "conditional_loss"
        ]
        == 0
    )
    uncertain = {**rate(0), "upper": 1, "observed_entries": 0}
    result = transfer_loss((ref,), {ref.key: rate(1)}, {ref.key: uncertain})
    assert result["conditional_loss"] == 1
    assert not result["conditional_complete"]
    with pytest.raises(ValueError, match="missing fixed"):
        transfer_loss((ref,), {ref.key: rate(1)}, {})


def test_observations_require_exact_source_cells_and_original_citations(tmp_path):
    p = plan()
    evaluator = Evaluator()
    rows = asyncio.run(
        SharedEvolution(tmp_path, Author(), evaluator).evaluate(
            p, bundle("teacher"), p.sources, tmp_path / "teacher"
        )
    )
    r = rows[0]
    ref = reference()
    obs = ConditionalObservation(
        reference=ref.key,
        record_id=r.record_id,
        entry="observed",
        outcome="achieved",
        citations=(r.evidence[0].path,),
        explanation="frame 0 to 1",
    )
    tasks = {"source": {"source"}}
    assert conditional_rates((ref,), [obs], rows, tasks)[ref.key]["lower"] == 1
    with pytest.raises(ValueError, match="every source opportunity"):
        conditional_rates((ref,), [], rows, tasks)
    with pytest.raises(ValueError, match="every source opportunity"):
        conditional_rates((ref,), [obs, obs], rows, tasks)
    with pytest.raises(ValueError, match="not original"):
        conditional_rates(
            (ref,), [obs.model_copy(update={"citations": ("test.json",)})], rows, tasks
        )
    absent = obs.model_copy(update={"entry": "absent", "outcome": "unknown"})
    result = conditional_rates((ref,), [absent], rows, tasks)[ref.key]
    assert result["lower"] == 0 and result["upper"] == 1
    with pytest.raises(ValueError, match="cannot assert"):
        conditional_rates(
            (ref,), [obs.model_copy(update={"entry": "unknown"})], rows, tasks
        )


def test_alignment_references_keep_real_paths_and_coupled_programs(tmp_path):
    from reweave.evolution.contracts import HarnessProgram, SemanticGraph
    from reweave.evolution.distillation.engine import SharedTeacher
    from reweave.evolution.distillation.transfer import aligned_references
    from tests.unit_tests.harness.evolution.test_graph_retrieval import corpus_graph

    graph = corpus_graph(tmp_path)
    value = graph.model_dump(mode="json")
    value["programs"] = {
        "p": HarnessProgram(
            bundle=graph.edges["edge0"].references[0],
            pointers=("/context",),
            mechanism="fixture coupled decision",
            uncertainty="fixture",
        ).model_dump(mode="json")
    }
    for edge in value["edges"].values():
        for witness in edge["witnesses"]:
            witness["program_id"] = "p"
    graph = SemanticGraph.model_validate(value)
    teachers = tuple(
        SharedTeacher(
            task=name, bundle=bundle(name), learned_tasks=(name,), graph=graph
        )
        for name in ("a", "b")
    )
    refs, matches = aligned_references(teachers, 1.0)
    assert refs and len(matches) == 2
    assert {r.task for r in refs} == {"a", "b"}
    assert sum(r.weight for r in refs) == pytest.approx(1)
    assert all(r.source_edges and r.programs for r in refs)
    assert all(set(r.source_edges) <= graph.edges.keys() for r in refs)


def test_native_measurement_admission_rejects_foreign_evidence(tmp_path):
    import json

    from reweave.evolution.authoring.config import AuthorPolicy
    from reweave.evolution.distillation.assessment import (
        NativeConditionalAssessment,
        measurement_request,
    )

    p = plan()
    rows = asyncio.run(
        SharedEvolution(tmp_path, Author(), Evaluator()).evaluate(
            p, bundle("teacher"), p.sources, tmp_path / "teacher"
        )
    )
    request = measurement_request(p, bundle("teacher"), rows, (reference(),))
    analyst = NativeConditionalAssessment(
        bundle_loader=None,
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path,
    )
    workspace = tmp_path / "workspace"
    (workspace / "candidate").mkdir(parents=True)
    path = workspace / "candidate/measurement.json"
    observation = ConditionalObservation(
        reference=reference().key,
        record_id=rows[0].record_id,
        entry="observed",
        outcome="unknown",
        citations=(rows[0].evidence[0].path,),
        explanation="occluded after grasp",
    )
    path.write_text(json.dumps([observation.model_dump(mode="json")]))
    result = asyncio.run(analyst.admit(request, workspace, tmp_path / "audit"))
    assert result["rates"][reference().key]["lower"] == 0
    assert result["rates"][reference().key]["upper"] == 1
    path.write_text(
        json.dumps(
            [
                {
                    **observation.model_dump(mode="json"),
                    "citations": ["foreign-test.json"],
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="original episode"):
        asyncio.run(analyst.admit(request, workspace, tmp_path / "audit"))


def test_source_success_with_conditional_loss_continues_to_loss_threshold(tmp_path):
    p = plan().model_copy(
        update={
            "unseen": (),
            "conditional_retention": True,
            "mechanism_references": (reference(),),
            "loss_threshold": 0.05,
            "max_rounds": 10,
        }
    )
    author = Author()
    evaluator = Evaluator(lambda request, b: b.record_id != "initial")
    result = asyncio.run(
        SharedEvolution(tmp_path, author, evaluator, Assessor()).run(p)
    )
    assert len(author.requests) == 2 and result["round"] == 2
    assert result["score"]["conditional_loss"] == 0
    assert result["score"]["loss"] < 0.05
    assert result["termination"] == "loss_threshold_reached"
    assert author.requests[1]["history"][-1]["score"]["conditional_loss"] == 1
