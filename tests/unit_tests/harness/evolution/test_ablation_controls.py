"""Search ablations enforce component authority and actual fixed-width execution."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.candidate import (
    read_candidate,
    read_proposal,
    write_bundle_source,
)
from reweave.evolution.authoring.prompts import author_instructions
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    AuthoredMutation,
    BudgetRevision,
    Candidate,
    EvolutionProtocol,
    Proposal,
)
from reweave.evolution.search.engine import EvolutionWorkspace
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    mutated,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace
from tests.unit_tests.harness.evolution.test_graph_evolution import (
    graph_search,
    record,
    verified_author,
)


def restricted_protocol(scopes=("C",), **changes):
    return EvolutionProtocol.model_validate(
        {
            **protocol().model_dump(mode="json"),
            "mutation_scopes": scopes,
            **changes,
        }
    )


def test_default_protocol_serialization_keeps_legacy_contracts_exact():
    value = protocol().model_dump(mode="json")
    assert "width_mode" not in value and "mutation_scopes" not in value
    assert EvolutionProtocol.model_validate(value) == protocol()


@pytest.mark.parametrize("scopes", [(), ("C", "C"), ("C", "W"), ("L", "C")])
def test_authority_requires_nonempty_unique_canonical_components(scopes):
    with pytest.raises(ValidationError, match="nonempty unique"):
        restricted_protocol(scopes)


@pytest.mark.parametrize(
    "field,value", [("width_mode", "fixed"), ("mutation_scopes", ("C",))]
)
def test_budget_revision_cannot_change_ablation_arm(field, value):
    prior = protocol()
    revised = EvolutionProtocol.model_validate(
        {
            **prior.model_dump(mode="json"),
            "record_id": new_record_id(),
            "max_rounds": 2,
            field: value,
        }
    )
    with pytest.raises(ValidationError, match="budget revision may only"):
        BudgetRevision(previous=prior, current=revised)


@pytest.mark.parametrize("scopes", [("C",), ("W", "C")])
def test_proposal_and_native_author_admission_enforce_declared_scopes(tmp_path, scopes):
    root = workspace(tmp_path)
    original = read_proposal(root).model_dump(mode="json")
    original["task_protocol"]["mutation_scopes"] = list(scopes)
    with pytest.raises(ValidationError, match="scopes differ"):
        Proposal.model_validate(original)
    original["scopes"] = list(scopes)
    proposal = Proposal.model_validate(original)
    atomic_json(host_input(root, "proposal.json"), proposal.model_dump(mode="json"))
    assert read_candidate(root).loop == baseline().loop
    assert "Editable components:" in author_instructions(proposal)
    forbidden = "W" if scopes == ("C",) else "L"
    changed = mutated(mutated(baseline(), "C"), forbidden)
    write_bundle_source(changed, root / "candidate")
    with pytest.raises(ValueError, match="forbidden scopes"):
        read_candidate(root, require_plan=False)


@pytest.mark.anyio
@pytest.mark.parametrize("scopes", [("C",), ("W", "C")])
async def test_restricted_search_stops_at_success_without_unrestricted_consolidation(
    tmp_path, scopes
):
    config = restricted_protocol(scopes, max_authors_per_round=1)
    store = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    proposals, evaluations = [], []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        proposals.append(proposal)
        assert proposal.purpose == "mutation" and proposal.scopes == scopes
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="repair",
                    bundle=mutated(baseline(), "C"),
                    summary="Synthetic repair",
                    experience="Synthetic only",
                ),
            ),
            summary="Synthetic repair",
            session_id="fixture",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        evaluations.append(bundle)
        return record(request, directory, bundle.record_id != baseline().record_id)

    state = await graph_search(store, verified_author(author), evaluator).run()
    assert state["stop_reason"] == "target_reached"
    assert len(proposals) == 1 and len(evaluations) == 2
    assert not list((tmp_path / "publications").glob("*.json"))
    assert all(bundle.loop == baseline().loop for bundle in evaluations)


@pytest.mark.anyio
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("scopes", [("C",), ("W", "C", "L")])
async def test_fixed_width_runs_every_sibling_and_recovers_without_new_work(
    tmp_path, width, success, scopes, monkeypatch
):
    from reweave.evolution.graph import selection as graph_selection

    def always_one(*args):
        raise AssertionError("fixed width must not compute a dynamic recommendation")

    monkeypatch.setattr(graph_selection, "compute_width", always_one)
    config = restricted_protocol(
        scopes, width_mode="fixed", max_authors_per_round=width, max_rounds=2
    )
    store = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    authors, evaluations = [], []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        authors.append(proposal.record_id)
        parent = store.bundle(proposal.parent.bundle_id)
        value = parent.model_dump(mode="json")
        value["record_id"] = new_record_id()
        value["context"]["instructions"] += "\n" + proposal.record_id
        from reweave.evolution.bundle import AgentBundle

        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="repair",
                    bundle=AgentBundle.model_validate(value),
                    summary="Synthetic unique candidate",
                    experience="Synthetic only",
                ),
            ),
            summary="Synthetic repair",
            session_id=proposal.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        evaluations.append(request.record_id)
        return record(
            request, directory, success and "author_000" in bundle.context.instructions
        )

    adapter = verified_author(author)
    search = graph_search(store, adapter, evaluator)
    state = await search.run()
    rounds = 1 if success else 2
    assert state["stop_reason"] == ("target_reached" if success else "round_limit")
    assert len(authors) == rounds * width
    assert len(evaluations) == 1 + rounds * width
    for value in state["rounds"].values():
        decision = json.loads(Path(value["decision"]["path"]).read_text())
        assert decision["width"] == width
        assert "dynamic_recommended_width" not in decision["statistics"]
        assert decision["alternatives"] == []
    consolidation = int(success and scopes == ("W", "C", "L"))
    assert search.accounting(state)["author_launches"] == rounds * width + consolidation
    await graph_search(store, adapter, evaluator, recover=True).run()
    assert len(authors) == rounds * width and len(evaluations) == 1 + rounds * width


@pytest.mark.anyio
async def test_custom_author_cannot_bypass_scope_gate_before_rollout(tmp_path):
    config = restricted_protocol(max_authors_per_round=1)
    store = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    calls = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="repair",
                    bundle=mutated(baseline(), "L"),
                    summary="Forbidden L change",
                    experience="Synthetic only",
                ),
            ),
            summary="Synthetic",
            session_id="fixture",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        calls.append(bundle)
        return record(request, directory, False)

    with pytest.raises(ValueError, match="forbidden scopes"):
        await graph_search(store, verified_author(author), evaluator).run()
    assert [b.record_id for b in calls] == [baseline().record_id]
