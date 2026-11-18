"""Topology removal retains Full author engineering and paid-search semantics."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.author import NativeBundleAuthor
from reweave.evolution.authoring.workspace import native_output
from reweave.evolution.contracts import (
    AuthoredMutation,
    Candidate,
    EvolutionProtocol,
    Proposal,
)
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    globalized,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.test_ablation_prompt_contract import (
    prepared_author,
)
from tests.unit_tests.harness.evolution.test_graph_evolution import (
    child,
    record,
    verified_author,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def settings(**changes):
    return EvolutionProtocol.model_validate(
        {
            **protocol().model_dump(mode="json"),
            "evolution_mode": "no_topology",
            "max_authors_per_round": 1,
            "width_mode": "dynamic",
            "max_rounds": 3,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "change",
    [
        {"max_authors_per_round": 2},
        {"mutation_scopes": ("C",)},
    ],
)
def test_no_topology_rejects_unrelated_mechanism_changes(change):
    with pytest.raises(ValidationError):
        settings(**change)


def test_author_retains_full_engineering_and_evidence(tmp_path):
    full, author, directory = prepared_author(tmp_path, ("W", "C", "L"))
    config = EvolutionProtocol.model_validate(
        {
            **full.task_protocol.model_dump(mode="json"),
            "evolution_mode": "no_topology",
            "width_mode": "dynamic",
            "max_authors_per_round": 1,
        }
    )
    values = {
        **full.model_dump(mode="json"),
        "purpose": "no_topology",
        "task_protocol": config.model_dump(mode="json"),
        **dict.fromkeys(("graph", "execution_graph", "progress_map", "decision")),
    }
    proposal = Proposal.model_validate(values)
    new = NativeBundleAuthor(
        bundle_loader=author.bundle_loader,
        policy=author.policy,
        evidence_workspace=tmp_path / "no_graph_archive",
    )
    destination = tmp_path / "no_graph_author"
    new._prepare_inputs(proposal, destination, resume=False)
    assert (
        native_output(destination / "workspace") == destination / "workspace/candidate"
    )
    before, after = directory / "workspace/context", destination / "workspace/context"
    for name in (
        "ENGINEERING.md",
        "VALIDATION.md",
        "PLAN_AND_EXPERIENCE.md",
        "CONTRACT_CASES.md",
        "INTERACTION.md",
        "runtime.schema.json",
    ):
        assert (before / name).read_bytes() == (after / name).read_bytes(), name
    with_graph = json.loads((before / "plan.schema.json").read_text())
    without_graph = json.loads((after / "plan.schema.json").read_text())
    del with_graph["properties"]["topology_feedback"]
    del with_graph["$defs"]["TopologyFeedback"]
    del with_graph["$defs"]["OriginalReference"]
    assert with_graph == without_graph
    inputs = json.loads((after / "input.json").read_text())
    assert inputs["complete_originals"] is True and inputs["history"]
    assert inputs["topology_disabled"]
    assert not (after / "experiment_access.json").exists()
    assert not (destination / "workspace/.host_inputs/access_policy.json").exists()
    for name in (
        "graph.json",
        "execution_graph.json",
        "progress_map.json",
        "search_review.json",
    ):
        assert not (after / name).exists()
    texts = [(after / "DEVELOPMENT.md").read_text()]
    assert not (after / "AUTHOR_WORKFLOW.md").exists()
    texts.extend(new._prompts(proposal, destination / "workspace"))
    for text in texts:
        for forbidden in (
            "read_graph",
            "current graph",
            "graph progress",
            "topology",
            "search_review.json",
            "semantic and map summaries",
        ):
            assert forbidden not in text, forbidden
    assert "do not start additional model sessions" in texts[0]
    assert "selected globally from all historical Harnesses" in texts[0]
    assert "then newest version on equal success rates" in texts[0]
    assert "Final output uses the same rule" in texts[0]
    assert "plan.probes" in texts[0]
    assert len(texts[0]) < 10000
    with pytest.raises(ValidationError, match="graph inputs"):
        Proposal.model_validate({**values, "graph": full.graph.model_dump(mode="json")})


@pytest.mark.anyio
@pytest.mark.parametrize("failure_origin", ["policy", "infrastructure"])
async def test_chain_keeps_full_dedup_history_and_newest_sr_tie(
    tmp_path, monkeypatch, failure_origin
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), settings())
    (tmp_path / "configuration.json").write_text("{}")
    proposals, evaluations = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError("topology machinery must not run")

    for name in ("_graph", "_analyze", "_finalize_topology"):
        monkeypatch.setattr(GraphEvolution, name, forbidden)
    monkeypatch.setattr("reweave.evolution.graph.selection.policy_parent", forbidden)
    monkeypatch.setattr("reweave.evolution.graph.selection.compute_width", forbidden)

    async def author(p, directory, *, before_launch):
        before_launch()
        proposals.append(p)
        candidate = child("same" if p.round_index % 2 == 0 else "other")
        return AuthoredMutation(
            record_id=f"author-{p.round_index}",
            proposal_id=p.record_id,
            candidates=(
                Candidate(
                    branch_id="edit",
                    bundle=candidate,
                    summary="Measured edit",
                    experience="Synthetic test",
                ),
            ),
            summary="Synthetic test",
            session_id=f"session-{p.round_index}",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        evaluations.append(request)
        result = record(request, directory, False)
        if failure_origin == "infrastructure":
            result = result.model_copy(
                update={
                    "failure_origin": "infrastructure",
                    "infrastructure_retryable": False,
                }
            )
        return result

    engine = GraphEvolution(workspace, verified_author(author), evaluator)
    state = await engine.run()
    assert len(proposals) == 3 and len(evaluations) == 3  # Full content reuse retained.
    assert state["rounds"] == {} and not state.get("final_projections")
    assert all(a["kind"] != "graph_analysis" for a in state["attempts"].values())
    assert (
        proposals[1].parent.bundle_id
        == state["versions"][state["order"][1]]["bundle_id"]
    )
    history = json.loads(Path(proposals[2].history.originals.path).read_text())
    assert any(p.endswith("/native.json") for p in history["files"])
    assert (
        "Observed intervention exposure"
        in Path(proposals[2].history.summary.path).read_text()
    )
    assert any(
        v.get("mutation_review") is not None
        for v in history["navigation"]["versions"].values()
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["best_version"] == state["order"][-1]
    assert (
        report["selection"]["criterion"]
        == "global_official_success_then_coverage_then_newest"
    )
    assert report["semantic_projection"] is None
    await GraphEvolution(workspace, engine.author, evaluator, recover=True).run()
    assert len(proposals) == 3 and len(evaluations) == 3


@pytest.mark.anyio
async def test_global_sr_parent_and_final_selection_keep_old_winners_and_newest_ties(
    tmp_path,
):
    config = settings(max_rounds=4, success_consolidation=False)
    config = config.model_copy(
        update={
            "dev": tuple(
                config.dev[0].model_copy(
                    update={
                        "record_id": f"dev-{seed}",
                        "initialization": {"seed": seed},
                    }
                )
                for seed in (11, 12, 13)
            ),
        }
    )
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    proposals, evaluations = [], []
    candidates = [child(f"candidate-{index}") for index in range(4)]
    successes = {
        bundle.record_id: count
        for bundle, count in zip(
            [baseline(), *candidates],
            [1, 2, 1, 2, 1],
            strict=True,
        )
    }

    async def author(proposal, directory, *, before_launch):
        before_launch()
        proposals.append(proposal)
        return AuthoredMutation(
            record_id=proposal.record_id + "-result",
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="edit",
                    bundle=candidates[proposal.round_index],
                    summary="Synthetic",
                    experience="Synthetic",
                ),
            ),
            summary="Synthetic",
            session_id=proposal.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        evaluations.append(request)
        return record(
            request,
            directory,
            request.instance.initialization["seed"] - 11 < successes[bundle.record_id],
        )

    engine = GraphEvolution(workspace, verified_author(author), evaluator)
    state = await engine.run()
    assert [p.parent.bundle_id for p in proposals] == [
        baseline().record_id,
        candidates[0].record_id,
        candidates[0].record_id,
        candidates[2].record_id,
    ]
    report = json.loads((tmp_path / "report.json").read_text())
    selected = report["scores"][report["best_version"]]
    assert selected["bundle_id"] == candidates[2].record_id
    assert selected["success_rate"] == 2 / 3
    assert report["best_version"] != state["order"][-1]
    assert not state["rounds"] and "final_projections" not in state
    await GraphEvolution(workspace, engine.author, evaluator, recover=True).run()
    assert len(proposals) == 4 and len(evaluations) == 15
    assert (
        json.loads((tmp_path / "report.json").read_text())["best_version"]
        == report["best_version"]
    )


@pytest.mark.anyio
async def test_success_keeps_full_consolidation_without_replacing_measured_result(
    tmp_path,
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), settings())
    (tmp_path / "configuration.json").write_text("{}")
    seen = []

    async def author(p, directory, *, before_launch):
        before_launch()
        seen.append(p.purpose)
        candidate = (
            child("successful")
            if p.purpose == "no_topology"
            else globalized(workspace.bundle(p.parent.bundle_id))
        )
        return AuthoredMutation(
            record_id=p.record_id + "-result",
            proposal_id=p.record_id,
            candidates=(
                Candidate(
                    branch_id="edit",
                    bundle=candidate,
                    summary="Synthetic",
                    experience="Synthetic",
                ),
            ),
            summary="Synthetic",
            session_id=p.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, bundle.record_id != baseline().record_id)

    state = await GraphEvolution(workspace, verified_author(author), evaluator).run()
    assert seen == ["no_topology", "success_consolidation"]
    assert state["stop_reason"] == "target_reached" and len(state["versions"]) == 2
    assert len(list((tmp_path / "publications").glob("*.json"))) == 1
