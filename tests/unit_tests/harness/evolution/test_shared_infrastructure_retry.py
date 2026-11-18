"""Finite shared retries preserve old costs and require actual owner closure."""

import asyncio
from types import SimpleNamespace

import pytest

from reweave.evolution.contracts import EvolutionIncomplete
from reweave.evolution.distillation.engine import SharedEvolution
from reweave.evolution.search.rollout import RolloutEvaluator
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.test_shared import Evaluator, plan


class StartupFailure(Evaluator):
    def __init__(self, *, always=False, origin="infrastructure", retryable=True):
        super().__init__(lambda *_: True)
        self.always = always
        self.origin = origin
        self.retryable = retryable
        self.closed = []

    async def __call__(self, *args, **kwargs):
        result = await super().__call__(*args, **kwargs)
        if self.always or len(self.requests) == 1:
            return result.model_copy(
                update={
                    "official_success": False,
                    "failure_origin": self.origin,
                    "infrastructure_retryable": self.retryable
                    if self.origin == "infrastructure"
                    else None,
                }
            )
        return result

    async def verify_retry_closed(self, request, bundle, directory):
        self.closed.append(directory)


def test_retry_retains_original_and_resume_does_not_replay(tmp_path):
    p = plan()
    evaluator = StartupFailure()
    engine = SharedEvolution(tmp_path, None, evaluator, infrastructure_retries=1)
    first = asyncio.run(engine.evaluate(p, p.initial, p.sources, tmp_path / "eval"))
    cell = tmp_path / "eval/source/000"
    original = (cell / "shared_result.json").read_bytes()
    assert b'"infrastructure"' in original
    assert first[0].official_success
    assert (cell / "retries/001/shared_result.json").is_file()
    assert len(evaluator.requests) == 2
    assert evaluator.requests[0] == evaluator.requests[1]
    assert evaluator.closed == [cell]
    second = asyncio.run(engine.evaluate(p, p.initial, p.sources, tmp_path / "eval"))
    assert second == first
    assert (cell / "shared_result.json").read_bytes() == original
    assert len(evaluator.requests) == 2


@pytest.mark.parametrize(
    "origin,retryable", [("agent", None), ("unknown", None), ("infrastructure", False)]
)
def test_policy_unknown_and_nonretryable_outcomes_are_not_repeated(
    tmp_path, origin, retryable
):
    p = plan()
    evaluator = StartupFailure(origin=origin, retryable=retryable)
    engine = SharedEvolution(tmp_path, None, evaluator, infrastructure_retries=1)
    for _ in range(2):
        rows = asyncio.run(engine.evaluate(p, p.initial, p.sources, tmp_path / "eval"))
        assert rows[0].failure_origin == origin
    assert len(evaluator.requests) == 1
    assert not evaluator.closed


def test_retry_allowance_survives_restarts(tmp_path):
    p = plan()
    evaluator = StartupFailure(always=True)
    for _ in range(3):
        engine = SharedEvolution(tmp_path, None, evaluator, infrastructure_retries=1)
        rows = asyncio.run(engine.evaluate(p, p.initial, p.sources, tmp_path / "eval"))
        assert rows[0].failure_origin == "infrastructure"
    assert len(evaluator.requests) == 2


def test_unsettled_original_never_admits_retry(tmp_path):
    p = plan()
    evaluator = StartupFailure()

    async def unclosed(*args):
        raise EvolutionIncomplete("original worker remains alive")

    evaluator.verify_retry_closed = unclosed
    engine = SharedEvolution(tmp_path, None, evaluator, infrastructure_retries=1)
    with pytest.raises(EvolutionIncomplete, match="remains alive"):
        asyncio.run(engine.evaluate(p, p.initial, p.sources, tmp_path / "eval"))
    assert len(evaluator.requests) == 1
    assert not (tmp_path / "eval/source/000/retries").exists()


def test_production_retry_verifier_requires_remote_closure(tmp_path):
    runner = object.__new__(RolloutEvaluator)
    record = SimpleNamespace(
        failure_origin="infrastructure", infrastructure_retryable=True
    )

    async def recovered(request, bundle, directory, *, before_launch):
        return record

    runner.recover = recovered
    atomic_json(tmp_path / "episode/remote_environment.json", {})
    with pytest.raises(EvolutionIncomplete, match="closure is still unknown"):
        asyncio.run(runner.verify_retry_closed(None, None, tmp_path))
    atomic_json(
        tmp_path / "episode/remote_environment_closed.json",
        {"status": "exited", "exit": {"parent_closed": False}},
    )
    with pytest.raises(EvolutionIncomplete, match="has not closed"):
        asyncio.run(runner.verify_retry_closed(None, None, tmp_path))
    atomic_json(
        tmp_path / "episode/remote_environment_closed.json",
        {"status": "exited", "exit": {"parent_closed": True}},
    )
    assert asyncio.run(runner.verify_retry_closed(None, None, tmp_path)) is record
