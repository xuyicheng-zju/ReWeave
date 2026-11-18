"""Success publication requires the complete declared development/validation target."""

import json

import pytest

from reweave.evolution.contracts import EvolutionProtocol
from reweave.evolution.search.engine import EvolutionWorkspace
from tests.unit_tests.harness.evolution.fixtures import baseline, protocol
from tests.unit_tests.harness.evolution.test_graph_evolution import graph_search, record


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "successes,validation_success,expected_consolidations",
    [
        ((True, False, False), None, 0),
        ((True, True, False), None, 0),
        ((True, True, True), None, 1),
        ((True, True, True), False, 0),
        ((True, True, True), True, 1),
    ],
)
async def test_partial_three_seed_h0_does_not_launch_success_consolidation(
    tmp_path, successes, validation_success, expected_consolidations
):
    raw = protocol().model_dump(mode="json")
    raw.update(max_rounds=0, width_mode="fixed", max_authors_per_round=1)
    initial = raw["dev"][0]
    raw["dev"] = [
        dict(initial, record_id=f"condition-{seed}", initialization={"seed": seed})
        for seed in (11, 12, 13)
    ]
    if validation_success is not None:
        raw["validation"] = {
            "conditions": [
                dict(
                    initial,
                    record_id="validation-condition",
                    initialization={"seed": 21},
                )
            ],
            "target_success_rate": 1.0,
        }
    declared = EvolutionProtocol.model_validate(raw)
    store = EvolutionWorkspace(tmp_path, baseline(), declared)
    (tmp_path / "configuration.json").write_text("{}")
    calls = []

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        calls.append(request)
        seed = request.instance.initialization["seed"]
        outcome = validation_success if seed == 21 else successes[seed - 11]
        return record(request, directory, outcome)

    async def forbidden_mutation(*args, **kwargs):
        raise AssertionError("zero-round H0 cannot launch mutation authors")

    search = graph_search(store, forbidden_mutation, evaluator)
    state = await search.run()
    consolidations = [
        key for key in state["attempts"] if key.startswith("consolidation/")
    ]
    assert len(consolidations) == expected_consolidations
    assert search.accounting(state)["author_launches"] == expected_consolidations
    assert len(calls) == 3 + int(validation_success is not None)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["target_reached"] is bool(expected_consolidations)
