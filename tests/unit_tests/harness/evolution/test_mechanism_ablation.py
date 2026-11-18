"""Behavioral isolation for mechanism ablations, including native read authority."""

import json

import pytest
from jsonschema import ValidationError, validate

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.authoring.tools import AuthorDevelopmentTools
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    AuthoredMutation,
    Candidate,
    EvolutionProtocol,
    GraphAnalysisReceipt,
)
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.graph.refinement import validate_refinement
from reweave.evolution.search.engine import (
    EvolutionWorkspace,
)
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request
from tests.unit_tests.harness.evolution.test_graph_evolution import (
    child,
    record,
    verified_author,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def controls(**changes):
    return EvolutionProtocol.model_validate(
        {
            **protocol().model_dump(mode="json"),
            **changes,
        }
    )


def test_fixed_topology_initial_analysis_has_the_same_prompt_as_full(tmp_path):
    request, _ = analysis_request(tmp_path)
    fixed = request.model_copy(
        update={
            "protocol": request.protocol.model_copy(
                update={"topology_mode": "fixed_initial"}
            )
        }
    )
    assert NativeGraphAnalysis._prompts(
        fixed, tmp_path
    ) == NativeGraphAnalysis._prompts(request, tmp_path)


def test_fixed_topology_rejects_semantic_edits_but_accepts_updated_evidence(tmp_path):
    request, result = analysis_request(tmp_path)
    prior = GraphAnalysisReceipt(
        record_id="prior",
        request_id=request.record_id,
        analysis=result,
        session_id="prior-native",
        usage=usage(),
        artifacts=(request.graph,),
    )
    ref = artifact(tmp_path, "prior.json", prior.model_dump(mode="json"))
    config = controls(topology_mode="fixed_initial")
    request = request.model_copy(
        update={
            "protocol": config,
            "previous_analysis": ref,
            "round_index": 1,
            "policy": AuthorPolicy(author="codex").model_dump(mode="json"),
        }
    )
    validate_refinement(request, result)
    graph = result.graph.model_copy(
        update={
            "nodes": {
                key: node.model_copy(
                    update={"meaning": "Different physical definition"}
                )
                for key, node in result.graph.nodes.items()
            }
        }
    )
    with pytest.raises(ValueError, match="initial topology is frozen"):
        validate_refinement(request, result.model_copy(update={"graph": graph}))
    author = NativeGraphAnalysis(
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path / "archive",
    )
    author._prepare_inputs(request, tmp_path / "analysis", resume=False)
    schema = json.loads(
        (tmp_path / "analysis/workspace/context/analysis.schema.json").read_text()
    )
    refs = AuthorReferences.load(tmp_path / "analysis/workspace")
    validate(refs.encode(result.model_dump(mode="json")), schema)
    with pytest.raises(ValidationError):
        validate(
            refs.encode(
                result.model_copy(update={"graph": graph}).model_dump(mode="json")
            ),
            schema,
        )
    instructions, _ = author._prompts(request, tmp_path / "analysis/workspace")
    assert "initial topology is fixed" in instructions
    normalized = " ".join(instructions.split())
    # Only structural-change permissions are omitted from the fixed topology.
    for structural in (
        "Subsequent analyses may add",
        "split states",
        "refine edge entry/exit",
        "Refine state/interface granularity",
        "Revise earlier interpretations",
    ):
        assert structural not in normalized
    for retained in (
        "Do not prescribe a next repair, select this task's parent Harness",
        "Author agree / modify / refuse records assess specific claims",
        "Do not adopt a diagnosis because an author asserted it",
        "role=milestone for necessary earned achievements",
        "Programs are separate from physical transitions",
        "Update physical evidence assignments under the fixed topology definitions",
    ):
        assert retained in normalized
    _, final_prompt = author._prompts(
        request.model_copy(update={"purpose": "final_selection"}),
        tmp_path / "analysis/workspace",
    )
    assert "the same fixed topology" in final_prompt
    assert "updated topology" not in final_prompt


@pytest.mark.anyio
async def test_analysis_has_no_session_inquiry_capability(tmp_path, monkeypatch):
    request, _ = analysis_request(tmp_path)
    policy = AuthorPolicy(author="codex")
    request = request.model_copy(
        update={
            "policy": policy.model_dump(mode="json"),
        }
    )

    author = NativeGraphAnalysis(
        bundle_loader=lambda _: baseline(),
        policy=policy,
        evidence_workspace=tmp_path / "archive",
    )
    directory = tmp_path / "analysis"
    author._prepare_inputs(request, directory, resume=False)
    targets = directory / "audit/inputs/session_targets.json"
    assert not targets.exists()
    tools = AuthorDevelopmentTools(
        directory / "workspace", directory / "audit/tools.jsonl"
    )
    assert "ask_session" not in {t.name for t in await tools.catalog()}
    with pytest.raises(ValueError, match="unknown development tool"):
        await tools._call("ask_session", {}, case_results=[])
    captured = []

    async def native(
        workspace,
        directory,
        readable,
        writable,
        record,
        started,
        query_path,
        begin_query,
        instructions,
        native_prompt,
        development_url,
    ):
        captured.append(instructions)
        assert "ask_session" not in instructions
        raise RuntimeError("native probe boundary reached")

    monkeypatch.setattr(author, "_codex", native)
    monkeypatch.setattr(
        AuthorDevelopmentTools, "preflight", lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError, match="native probe boundary reached"):
        await author._execute_once(request, directory, resume=True)
    assert captured


@pytest.mark.anyio
async def test_fixed_topology_keeps_normal_rounds_final_analysis_and_resume(tmp_path):
    from tests.unit_tests.harness.evolution.test_graph_evolution import graph_search

    config = controls(
        topology_mode="fixed_initial", max_authors_per_round=1, max_rounds=2
    )
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    authors, evaluations = [], []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        authors.append(proposal)
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="edit",
                    bundle=child(str(proposal.round_index)),
                    summary="Fixture edit",
                    experience="Fixture",
                ),
            ),
            summary="Fixture",
            session_id=f"native-{proposal.round_index}",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        evaluations.append(request)
        return record(request, directory, False)

    engine = graph_search(workspace, verified_author(author), evaluator)
    state = await engine.run()
    assert len(state["rounds"]) == 2 and len(state["final_projections"]) == 1
    assert len(authors) == 2 and len(evaluations) == 3
    await graph_search(workspace, engine.author, evaluator, recover=True).run()
    assert len(authors) == 2 and len(evaluations) == 3
