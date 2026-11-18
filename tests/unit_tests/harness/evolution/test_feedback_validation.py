# Modified for the ReWeave namespace.
"""Conditional physical spending, validation feedback and exact recovery."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import AuthoredMutation, Candidate, EvolutionProtocol
from reweave.evolution.graph.projection import map_row
from reweave.evolution.graph.views import query_map
from reweave.evolution.search.engine import EvolutionWorkspace
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.test_graph_evolution import (
    child,
    graph_search,
    record,
    verified_author,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def gated(rounds=2):
    value = protocol().model_dump(mode="json")
    conditions = [
        {"record_id": f"condition-{s}", "task": "drawer", "initialization": {"seed": s}}
        for s in range(1, 11)
    ]
    value.update(
        dev=conditions[:5],
        validation={"conditions": conditions[5:], "target_success_rate": 0.8},
        target_success_rate=1.0,
        max_rounds=rounds,
        max_authors_per_round=1,
    )
    return EvolutionProtocol.model_validate(value)


def test_frozen_condition_inventory_preserves_h0_identity_when_partitioned():
    config = gated()
    assert config.manifest_id == "fixture-manifest"
    old = config.model_dump(mode="json")
    old.update(
        dev=[i.model_dump(mode="json") for i in config.instances], validation=None
    )
    assert EvolutionProtocol.model_validate(old).manifest_id == config.manifest_id
    for invalid in (
        {"conditions": [], "target_success_rate": 0.8},
        {"conditions": [config.dev[0].model_dump()], "target_success_rate": 0.8},
        {
            "conditions": [
                {
                    "record_id": "other-condition",
                    "task": "other",
                    "initialization": {"seed": 6},
                }
            ],
            "target_success_rate": 0.8,
        },
        {
            "conditions": [config.validation.conditions[0].model_dump()],
            "target_success_rate": 0,
        },
    ):
        with pytest.raises(ValidationError):
            EvolutionProtocol.model_validate(
                {**config.model_dump(), "validation": invalid}
            )


@pytest.mark.anyio
async def test_unqualified_content_never_launches_validation(tmp_path):
    workspace = EvolutionWorkspace(tmp_path, baseline(), gated(0))
    (tmp_path / "configuration.json").write_text("{}")
    launches = []

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        launches.append(request)
        assert request.split == "dev"
        return record(request, directory, request.instance.initialization["seed"] != 5)

    state = await graph_search(workspace, None, evaluator).run()
    assert len(launches) == 5
    assert state["stop_reason"] == "round_limit"
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["target_reached"]
    assert report["validation"]["matrix"][baseline().record_id] == [None] * 5


@pytest.mark.anyio
async def test_failed_gate_returns_to_graph_and_author_then_same_bundle_passes_both(
    tmp_path,
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), gated())
    (tmp_path / "configuration.json").write_text("{}")
    jobs, launches = [], []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        jobs.append(proposal)
        if proposal.round_index == 1:
            assert sum(r.request.split == "validation" for r in proposal.evidence) == 5
            history = Path(proposal.history.summary.path).read_text()
            assert "## Official results" in history
            assert "C007" in history
            assert (
                "| success | success | success | success | success | success | success | failure | failure | failure |"
                in history
            )
            value = json.loads(Path(proposal.progress_map.path).read_text())
            feedback = [
                cell
                for version in value["validation_cells"]
                for cell in map_row(value, version, split="validation").values()
            ]
            assert len(feedback) == 5
            assert all(len(row) == 5 for row in value["cells"].values())
            failed = next(c for c in feedback if not c["official_success"])
            assert query_map(value, view="episode", identity=failed["receipt_id"])[
                "transitions"
            ]
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="repair",
                    bundle=child(str(proposal.round_index)),
                    summary="Repair evidence-backed failure",
                    experience="Fixture lesson",
                ),
            ),
            summary="Fixture diagnosis",
            session_id=proposal.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        launches.append(request)
        label = bundle.context.instructions
        success = bool(label) and (
            request.split == "dev"
            or request.instance.initialization["seed"] <= (7 if label == "0" else 9)
        )
        return record(request, directory, success)

    search = graph_search(workspace, verified_author(author), evaluator)
    state = await search.run()
    assert len(jobs) == 2
    assert len(launches) == 25
    assert sum(r.split == "validation" for r in launches) == 10
    assert state["stop_reason"] == "target_reached"
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["target_reached"]
    scores = {s["bundle_id"]: s for s in report["scores"].values()}
    assert scores[child("0").record_id]["success_rate"] == 1
    assert scores[child("0").record_id]["validation_success_rate"] == 0.4
    assert not scores[child("0").record_id]["target_reached"]
    assert scores[child("1").record_id]["validation_success_rate"] == 0.8
    assert report["scores"][report["best_version"]]["bundle_id"] == child("1").record_id
    assert report["total_usage_including_failed_work"]["episode_launches"] == 25
    first_outcome = json.loads(
        Path(state["rounds"]["0"]["outcome"]["path"]).read_text()
    )
    assert first_outcome["observations"][0]["success"] is False
    assert await search.run() == state
    assert len(launches) == 25


@pytest.mark.anyio
async def test_interrupted_validation_resumes_only_missing_condition(tmp_path):
    workspace = EvolutionWorkspace(tmp_path, baseline(), gated(0))
    (tmp_path / "configuration.json").write_text("{}")
    launches = []
    interrupted = False

    async def evaluator(request, bundle, directory, *, before_launch):
        nonlocal interrupted
        before_launch()
        launches.append((request.split, request.instance.initialization["seed"]))
        if (
            request.split == "validation"
            and request.instance.initialization["seed"] == 8
            and not interrupted
        ):
            interrupted = True
            raise RuntimeError("Fixture validation interruption before result")
        return record(request, directory, True)

    with pytest.raises(RuntimeError, match="validation interruption"):
        await graph_search(workspace, None, evaluator).run()
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert not progress["target_reached"]
    assert None in progress["validation"]["matrix"][baseline().record_id]
    evaluator.recover = evaluator
    state = await graph_search(workspace, None, evaluator, recover=True).run()
    assert state["stop_reason"] == "target_reached"
    assert len(launches) == 11
    assert launches.count(("validation", 8)) == 2
    assert all(launches.count(("dev", seed)) == 1 for seed in range(1, 6))
