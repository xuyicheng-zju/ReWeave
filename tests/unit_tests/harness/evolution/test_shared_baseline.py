"""Shared physical inputs are reused across arms without replay or relabeling."""

import json

import pytest

from reweave.evolution.contracts import EvaluationRequest, EvolutionProtocol
from reweave.evolution.evidence.records import EpisodeEvidence, inventory_episode
from reweave.evolution.search.engine import EvolutionWorkspace
from reweave.evolution.search.shared_baseline import seed_shared_baseline
from tests.unit_tests.harness.evolution.fixtures import baseline, protocol
from tests.unit_tests.harness.evolution.test_graph_evolution import graph_search, record


@pytest.fixture
def anyio_backend():
    return "asyncio"


def three_conditions():
    raw = protocol().model_dump(mode="json")
    raw["dev"] = [
        dict(raw["dev"][0], record_id=f"dev-{seed}", initialization={"seed": seed})
        for seed in (11, 12, 13)
    ]
    return EvolutionProtocol.model_validate(raw)


def bank(tmp_path, config, successes):
    paths = []
    for index, (condition, success) in enumerate(zip(config.dev, successes)):
        request = EvaluationRequest(
            record_id=f"bank-request-{index}",
            bundle_id=baseline().record_id,
            instance=condition,
            split="dev",
            manifest_id=config.manifest_id,
            execution=config.execution_id,
            conditions=config.execution,
        )
        directory = tmp_path / f"original-{index}"
        value = record(request, directory, success)
        value = value.model_copy(
            update={
                "episode": EpisodeEvidence(
                    evidence_id=f"synthetic-{index}", files=inventory_episode(directory)
                )
            }
        )
        path = directory / "result.json"
        path.write_text(value.model_dump_json())
        paths.append(path)
    return paths


@pytest.mark.anyio
@pytest.mark.parametrize(
    "successes", [(False, False, False), (True, False, True), (True, True, True)]
)
async def test_three_seed_h0_is_reused_without_evaluator_calls(tmp_path, successes):
    config = three_conditions()
    paths = bank(tmp_path, config, successes)
    original_bytes = [p.read_bytes() for p in paths]

    async def forbidden(*args, **kwargs):
        raise AssertionError("shared H0 must never execute again")

    for arm in ("full", "c", "cw"):
        store = EvolutionWorkspace(tmp_path / arm, baseline(), config)
        seed_shared_baseline(store, paths)
        search = graph_search(store, forbidden, forbidden)
        with store.transaction() as state:
            observed = await search._evaluate(state, "baseline")
            assert [r.official_success for r in observed] == list(successes)
            assert list((await search.matrix(state)).values()) == [successes]
            assert len(state["attempts"]) == 3
        assert not (store.root / "attempts").exists()
    assert [p.read_bytes() for p in paths] == original_bytes


@pytest.mark.parametrize("corruption", ["missing", "order", "execution", "outcome"])
def test_shared_h0_rejects_incomplete_or_mismatched_bank_before_import(
    tmp_path, corruption
):
    config = three_conditions()
    paths = bank(tmp_path, config, (False, False, False))
    if corruption == "missing":
        paths = paths[:2]
    elif corruption == "order":
        paths = paths[::-1]
    else:
        value = json.loads(paths[0].read_text())
        if corruption == "execution":
            value["request"]["execution"] = "different-execution"
        else:
            value.update(official_success=None, failure_origin="unknown")
        paths[0].write_text(json.dumps(value))
    store = EvolutionWorkspace(tmp_path / "arm", baseline(), config)
    with pytest.raises(ValueError):
        seed_shared_baseline(store, paths)
    assert not store.state_path.exists()
