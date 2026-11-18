"""Explicit task-model assessments survive independent author/analysis sessions."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from reweave.evolution.authoring.candidate import (
    AuthorPlan,
    TopologyFeedback,
    candidate_summary,
    read_author_plan,
    read_candidate,
    topology_feedback,
)
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.contracts import (
    AuthoredMutation,
    Candidate,
    EvolutionProtocol,
    GraphAnalysisReceipt,
)
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.search.engine import EvolutionWorkspace
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    evolution_history,
    usage,
)
from tests.unit_tests.harness.evolution.graph_fixtures import FixtureGraphAnalysis
from tests.unit_tests.harness.evolution.test_ablation_prompt_contract import (
    prepared_author,
)
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request
from tests.unit_tests.harness.evolution.test_graph_evolution import (
    child,
    configured,
    graph_search,
    record,
    verified_author,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def plan(**extra):
    return {
        "diagnosis": "A physical precondition remains unverified.",
        "summary": "Inspect fresh evidence before acting.",
        "validation": "Synthetic interface checks only.",
        **extra,
    }


@pytest.mark.parametrize("stance", ["agree", "modify", "refuse"])
def test_material_feedback_is_optional_and_requires_a_reason(stance):
    assert AuthorPlan.model_validate(plan()).topology_feedback == ()
    assert "topology_feedback" not in AuthorPlan.model_validate(plan()).model_dump()
    feedback = {
        "target": "contact_is_possession",
        "stance": stance,
        "reason": "Observed contact alone leaves retention uncertain.",
    }
    assert (
        topology_feedback(json.dumps(plan(topology_feedback=[feedback])))[0].stance
        == stance
    )
    with pytest.raises(ValidationError):
        TopologyFeedback.model_validate({**feedback, "reason": "   "})
    with pytest.raises(ValidationError):
        TopologyFeedback.model_validate({**feedback, "stance": "approved"})
    assert topology_feedback("historical free-text diagnosis") == ()
    assert topology_feedback('"historical JSON string"') == ()


def test_author_feedback_binds_evidence_before_cross_session_handoff(tmp_path):
    request, _, directory = prepared_author(tmp_path, ("W", "C", "L"))
    workspace = directory / "workspace"
    refs = AuthorReferences.load(workspace)
    original = request.evidence[0].evidence[0]
    handle = refs.reference(**original.model_dump(mode="json"))
    value = plan(
        topology_feedback=[
            {
                "target": "contact_is_possession",
                "stance": "modify",
                "reason": "Keep contact and verified retention distinct.",
                "evidence": [handle],
            }
        ]
    )
    schema = json.loads((workspace / "context/plan.schema.json").read_text())
    Draft202012Validator(schema).validate(value)
    atomic_json(workspace / "candidate/plan.json", value)
    published = topology_feedback(candidate_summary(workspace))[0]
    assert published.evidence[0].path == original.path
    assert published.evidence[0].artifact_id == original.artifact_id
    # Admission refuses invented handles rather than carrying ambiguous IDs forward.
    value["topology_feedback"][0]["evidence"] = ["file:999999#"]
    with pytest.raises(ValueError):
        read_author_plan(workspace, json.dumps(value))


def test_host_enforces_real_scope_without_authored_scope_metadata(tmp_path):
    _, _, directory = prepared_author(tmp_path, ("C",))
    workspace = directory / "workspace"
    root = workspace / "candidate"
    atomic_json(root / "plan.json", plan())
    (root / "SKILL.md").unlink()
    (root / "task-family/context/instructions.md").write_text(
        "Use the current public observation."
    )
    assert read_candidate(workspace).novel_scopes((baseline(),)) == {"C"}
    source = next((root / "task-family/loop/src").rglob("*.py"))
    source.write_text(source.read_text() + "\n# forbidden change outside C\n")
    with pytest.raises(ValueError):
        read_candidate(workspace)


@pytest.mark.parametrize("purpose", ["search", "final_selection"])
def test_analysis_receives_only_latest_assessments_with_rebound_references(
    tmp_path, purpose
):
    request, result = analysis_request(tmp_path)
    previous = GraphAnalysisReceipt(
        record_id="previous-analysis",
        request_id=request.record_id,
        analysis=result,
        session_id="previous-independent-session",
        usage=usage(),
        artifacts=(request.graph,),
    )
    source = request.evidence[0].evidence[0]
    feedback = {
        "version_id": request.versions[0].version_id,
        "round_index": 0,
        "items": [
            {
                "target": "contact_is_possession",
                "stance": "refuse",
                "reason": "Retention remains unobserved.",
                "evidence": [{**source.model_dump(mode="json"), "pointer": ""}],
            }
        ],
    }
    history = evolution_history(
        tmp_path / "feedback_history",
        originals={
            source.path: {"artifact_id": source.artifact_id, "bytes": source.bytes}
        },
        navigation={
            "latest_topology_feedback": [feedback, {**feedback, "round_index": -1}]
        },
    )
    request = request.model_copy(
        update={
            "round_index": 1,
            "history": history,
            "purpose": purpose,
            "previous_analysis": artifact(
                tmp_path, "previous.json", previous.model_dump(mode="json")
            ),
        }
    )
    analyzer = NativeGraphAnalysis(
        policy=AuthorPolicy(),
        bundle_loader=lambda _: baseline(),
        evidence_workspace=tmp_path / "archive",
    )
    directory = tmp_path / "analysis"
    analyzer._prepare_inputs(request, directory, resume=False)
    workspace = directory / "workspace"
    inputs = json.loads((workspace / "context/input.json").read_text())
    (review,) = inputs["author_topology_feedback"]
    assert review["round_index"] == 0
    (ref,) = review["items"][0]["evidence"]
    assert AuthorReferences.load(workspace).original(ref).path == source.path
    instructions, _ = analyzer._prompts(request, workspace)
    assert "new_episode_ids and author_topology_feedback" in instructions
    assert "complete, coherently reprojected graph" in " ".join(instructions.split())
    reference = (workspace / "context/GRAPH_REFERENCE.md").read_text()
    assert len(instructions) < len(reference)


@pytest.mark.anyio
async def test_round_history_foregrounds_latest_feedback_without_accumulating_prose(
    tmp_path,
):
    settings = EvolutionProtocol.model_validate(
        {
            **configured(3).model_dump(mode="json"),
            "width_mode": "fixed",
            "max_authors_per_round": 1,
        }
    )
    workspace = EvolutionWorkspace(tmp_path, baseline(), settings)
    atomic_json(tmp_path / "configuration.json", {})
    analyses, authors = [], []

    class Analyser(FixtureGraphAnalysis):
        async def __call__(self, request, directory, *, before_launch):
            analyses.append(request)
            return await super().__call__(
                request, directory, before_launch=before_launch
            )

    async def author(request, directory, *, before_launch):
        before_launch()
        authors.append(request)
        index = request.round_index
        return AuthoredMutation(
            record_id=f"feedback-author-{index}",
            proposal_id=request.record_id,
            candidates=(
                Candidate(
                    branch_id="candidate",
                    bundle=child(str(index)),
                    summary=json.dumps(
                        plan(
                            topology_feedback=[
                                {
                                    "target": "contact_is_possession",
                                    "stance": ("agree", "modify", "refuse")[index],
                                    "reason": f"specific-assessment-round-{index}",
                                }
                            ]
                        )
                    ),
                ),
            ),
            summary="Synthetic edit",
            session_id=f"independent-session-{index}",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {"round": index}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, False)

    result = await graph_search(
        workspace, verified_author(author), evaluator, analyzer=Analyser()
    ).run()
    assert result["stop_reason"] == "round_limit"
    assert (
        len(authors) == 3 and len(analyses) == 4
    )  # final graph, no extra review session
    final = analyses[-1]
    summary = Path(final.history.summary.path).read_text()
    assert "**refuse**" in summary and "specific-assessment-round-2" in summary
    assert "specific-assessment-round-0" not in summary
    originals = json.loads(Path(final.history.originals.path).read_text())["navigation"]
    assert len(originals["latest_topology_feedback"]) == 1
    assert [
        entry["topology_feedback"][0]["stance"]
        for entry in originals["versions"].values()
        if entry.get("topology_feedback")
    ] == ["agree", "modify", "refuse"]
