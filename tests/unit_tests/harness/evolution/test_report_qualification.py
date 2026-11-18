"""Qualified-version reporting ranks official success before the recency tie."""

import json

import pytest

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import AuthoredMutation, Candidate, EvolutionProtocol
from reweave.evolution.distillation.metrics import harness_size
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


@pytest.mark.anyio
@pytest.mark.parametrize(
    "newer_successes,expected",
    [((True, True, False), "first"), ((True, True, True), "second")],
)
async def test_fixed_k2_qualified_report_uses_score_then_latest(
    tmp_path, newer_successes, expected
):
    raw = protocol().model_dump(mode="json")
    raw.update(
        width_mode="fixed",
        max_authors_per_round=2,
        max_rounds=1,
        target_success_rate=2 / 3,
    )
    original = raw["dev"][0]
    raw["dev"] = [
        dict(original, record_id=f"dev-{seed}", initialization={"seed": seed})
        for seed in (11, 12, 13)
    ]
    declared = EvolutionProtocol.model_validate(raw)
    workspace = EvolutionWorkspace(tmp_path, baseline(), declared)
    (tmp_path / "configuration.json").write_text("{}")
    children = {name: child(name) for name in ("first", "second")}

    async def author(proposal, directory, *, before_launch):
        before_launch()
        name = "first" if proposal.author_index == 0 else "second"
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="qualified",
                    bundle=children[name],
                    summary="Synthetic candidate",
                    experience="Synthetic qualification ordering only.",
                ),
            ),
            summary="Synthetic independent author",
            session_id=name,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        seed = request.instance.initialization["seed"]
        success = (
            False
            if bundle.record_id == baseline().record_id
            else True
            if bundle.record_id == children["first"].record_id
            else newer_successes[seed - 11]
        )
        return record(request, directory, success)

    state = await graph_search(workspace, verified_author(author), evaluator).run()
    report = json.loads((tmp_path / "report.json").read_text())
    assert state["stop_reason"] == "target_reached"
    assert len(state["rounds"]) == 1
    assert len(state["versions"]) == 3
    official = report["scores"][report["official_best_version"]]
    assert official["bundle_id"] == children[expected].record_id
    assert official["success_rate"] == 1.0
    # Both authors finish; the smaller successful implementation beats recency.
    assert (
        harness_size(children["first"])["total_bytes"]
        < harness_size(children["second"])["total_bytes"]
    )
    selected = report["scores"][report["best_version"]]
    assert selected["bundle_id"] == children["first"].record_id
    assert selected["success_rate"] == 1.0
    assert report["selection"]["size_metric"] == "deployed_cwl_utf8_bytes"
    selected_size = report["selection"]["harness_size_bytes"][report["best_version"]]
    assert selected_size == harness_size(children["first"])["total_bytes"]
