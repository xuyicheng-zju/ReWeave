# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Real rollout finalization and search recovery from synthetic exited workers."""

import json
from pathlib import Path

import pytest

from reweave.evolution.bundle import AgentBundle, baseline_loop, new_record_id
from reweave.evolution.contracts import (
    EvaluationRecord,
    EvaluationRequest,
    EvolutionIncomplete,
    EvolutionProtocol,
)
from reweave.evolution.evidence.records import inventory_episode
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    API_ERROR,
    PHYSICAL_FAILURE,
    SUCCESS,
)
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import protocol
from tests.unit_tests.harness.evolution.test_infrastructure_retry import (
    evaluate,
    forbidden_adapter,
)
from tests.unit_tests.harness.evolution.test_parallel_rollout import (
    evaluator as evaluator,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def graph(tmp_path, evaluator):
    # Native receipts count visible provider turns, which cannot certify hidden
    # inference attempts. They remain recoverable without rerunning physical work.
    manifest = evaluator.configuration.runtime_manifest
    bundle = AgentBundle(
        record_id=new_record_id(),
        runtime=manifest.runtime_configuration(),
        loop=baseline_loop(),
    )
    values = protocol().model_dump(mode="json")
    values.update(
        dev=[
            {
                "record_id": "partial-condition",
                "task": "libero:fixture:0",
                "initialization": {
                    "suite": "fixture",
                    "task": 0,
                    "seed": 51,
                    "reset_state_id": 1,
                },
            }
        ],
        execution=manifest.conditions.model_dump(mode="json"),
        infrastructure_retries=2,
        max_rounds=0,
    )
    return EvolutionWorkspace(
        tmp_path / "search", bundle, EvolutionProtocol.model_validate(values)
    )


def request(graph):
    return EvaluationRequest(
        record_id=new_record_id(),
        bundle_id=graph.baseline.record_id,
        instance=graph.protocol.dev[0],
        split="dev",
        manifest_id="b" * 64,
        execution=graph.protocol.execution_id,
        conditions=graph.protocol.execution,
    )


@pytest.mark.parametrize("evidence", ["worker_process", "physical_trace"])
def test_spawn_error_cannot_certify_started_work_as_free(
    graph, evaluator, tmp_path, evidence
):
    directory = tmp_path / "contradictory-spawn"
    directory.mkdir()
    atomic_json(
        directory / "exit.json",
        {"spawn_error": "OSError: process creation failed", "elapsed_s": 1.0},
    )
    if evidence == "worker_process":
        atomic_json(directory / "process.json", {"pid": 123})
    else:
        episode = directory / "episode"
        episode.mkdir()
        (episode / "execution_trace.jsonl").write_text(
            '{"event_type":"control_step"}\n'
        )
    with pytest.raises(EvolutionIncomplete, match="spawn failure.*worker evidence"):
        evaluator._close(request(graph), directory)
    assert not (directory / "result.json").exists()


def exited_worker(directory, evaluation, *, complete=False, unknown_action=False):
    output = directory / "episode"
    output.mkdir(parents=True)
    atomic_json(
        directory / "launch.json",
        {"request": evaluation.model_dump(mode="json"), "pid": None},
    )
    atomic_json(directory / "exit.json", {"returncode": 1, "elapsed_s": 3.5})
    transcript = {
        "run_configuration": {
            "robot_name": "libero",
            **evaluation.instance.initialization,
        },
        "bundle_id": evaluation.bundle_id,
        "termination_class": PHYSICAL_FAILURE if complete else API_ERROR,
        "api_error_retryable": None if complete else True,
        "official_success": False,
        "stats": {"requests": 2} if complete else {},
    }
    atomic_json(output / "transcript_test.json", transcript)
    rows = [{"event_type": "control_step"}] * (20 if complete else 10)
    if unknown_action:
        rows.append({"event_type": "action_outcome_unknown"})
    (output / "execution_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (output / "service_error.log").write_bytes(b"initialization diagnostic\x00\xff\n")
    if complete:
        context = output / "harness_context"
        context.mkdir()
        (context / "planner_context.json").write_text("{}")
        (context / "capability_catalog.json").write_text("{}")
    return transcript


def test_unbound_startup_can_close_as_infrastructure(graph, evaluator, tmp_path):
    evaluation = request(graph)
    directory = tmp_path / "unbound-startup"
    transcript = exited_worker(directory, evaluation)
    transcript["bundle_id"] = None
    atomic_json(directory / "episode/transcript_test.json", transcript)
    record = evaluator._close(evaluation, directory)
    assert record.failure_origin == "infrastructure"
    assert record.infrastructure_retryable is True
    assert record.usage.physical_steps == 10


def test_unbound_startup_never_retries_unknown_actions(graph, evaluator, tmp_path):
    evaluation = request(graph)
    directory = tmp_path / "unbound-unknown"
    transcript = exited_worker(directory, evaluation, unknown_action=True)
    transcript["bundle_id"] = None
    atomic_json(directory / "episode/transcript_test.json", transcript)
    with pytest.raises(EvolutionIncomplete, match="unacknowledged action"):
        evaluator._close(evaluation, directory)
    assert not (directory / "result.json").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("complete", [False, True])
async def test_published_partial_and_complete_receipts_are_reused_verbatim(
    graph, evaluator, tmp_path, monkeypatch, complete
):
    evaluation = request(graph)
    directory = tmp_path / "published"
    exited_worker(directory, evaluation, complete=complete)
    original = evaluator._close(evaluation, directory)
    assert (original.episode is not None) == complete
    before = {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "a published receipt must never be finalized or launched again"
        )

    monkeypatch.setattr(evaluator, "_close", forbidden)
    monkeypatch.setattr(evaluator, "_run", forbidden)
    monkeypatch.setattr(evaluator, "_finalize_storage", forbidden)
    for _ in range(2):
        recovered = await evaluator.recover(
            evaluation, graph.baseline, directory, before_launch=forbidden
        )
        settled = await evaluator._settle(evaluation, directory)
        assert recovered == settled == original
    assert {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    } == before


@pytest.mark.anyio
@pytest.mark.parametrize(
    "conflict",
    ["record_request", "bundle", "missing_evidence", "live_process", "invalid_json"],
)
async def test_published_receipt_recovery_checks_identity_evidence_and_original_owner_first(
    graph, evaluator, tmp_path, monkeypatch, conflict
):
    from reweave.evolution.search import rollout

    evaluation = request(graph)
    directory = tmp_path / "conflict"
    exited_worker(directory, evaluation, complete=False)
    published = evaluator._close(evaluation, directory)
    bundle = graph.baseline
    expected, message = ValueError, None
    if conflict == "record_request":
        raw = json.loads((directory / "result.json").read_text())
        # All physical settings still match; even a changed request identity is
        # a conflict and cannot silently replace the saved receipt.
        raw["request"]["record_id"] = new_record_id()
        atomic_json(directory / "result.json", raw)
        message = "result request changed"
    elif conflict == "bundle":
        bundle = bundle.model_copy(update={"record_id": new_record_id()})
        message = "bundle differs"
    elif conflict == "missing_evidence":
        Path(published.evidence[0].path).unlink()
        message = "evidence"
    elif conflict == "live_process":
        atomic_json(directory / "process.json", {"pid": 12345})
        monkeypatch.setattr(rollout.os, "killpg", lambda *args: None)
        expected, message = EvolutionIncomplete, "process group still exists"
    else:
        (directory / "result.json").write_text("{}")
    before = (directory / "result.json").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "invalid saved evidence must not trigger finalization or replay"
        )

    monkeypatch.setattr(evaluator, "_settle", forbidden)
    with pytest.raises(expected, match=message):
        await evaluator.recover(evaluation, bundle, directory, before_launch=forbidden)
    assert (directory / "result.json").read_bytes() == before


@pytest.mark.anyio
async def test_concurrent_settlement_allocates_only_one_receipt(
    graph, evaluator, tmp_path, monkeypatch
):
    import asyncio
    import time

    evaluation = request(graph)
    directory = tmp_path / "concurrent"
    exited_worker(directory, evaluation, complete=False)
    close = evaluator._close
    calls = []

    def slow_close(*args):
        calls.append(None)
        time.sleep(0.03)
        return close(*args)

    monkeypatch.setattr(evaluator, "_close", slow_close)
    first, second = await asyncio.gather(
        evaluator._settle(evaluation, directory),
        evaluator._settle(evaluation, directory),
    )
    assert len(calls) == 1 and first.record_id == second.record_id
    assert (
        EvaluationRecord.model_validate_json((directory / "result.json").read_text())
        == first
        == second
    )


class InterruptedFinalization(RuntimeError):
    """Stop synthetic process preparation around the real finalization boundary."""


@pytest.mark.anyio
async def test_permanent_partial_api_failure_is_not_replayed(
    graph, evaluator, monkeypatch
):
    launches = []

    async def exited_run(evaluation, bundle, directory, *, before_launch, **kwargs):
        before_launch()
        launches.append(directory)
        transcript = exited_worker(directory, evaluation)
        transcript["api_error_retryable"] = False
        atomic_json(directory / "episode/transcript_test.json", transcript)
        return evaluator._close(evaluation, directory)

    monkeypatch.setattr(evaluator, "_run", exited_run)
    for recover in (False, True):
        search = GraphEvolution(graph, forbidden_adapter, evaluator, recover=recover)
        records = await evaluate(search)
        assert records[0].failure_origin == "infrastructure"
        assert (await search.matrix(json.loads(graph.state_path.read_text())))[
            graph.baseline.record_id
        ] == (None,)
    assert len(launches) == 1
    record = EvaluationRecord.model_validate_json(
        (launches[0] / "result.json").read_text()
    )
    assert record.infrastructure_retryable is False
    assert record.episode is None
    assert record.usage.physical_steps == 10
    record.verify()


@pytest.mark.anyio
@pytest.mark.parametrize("interruption", [None, "before_close", "after_close"])
async def test_partial_api_receipt_retries_and_recovers_without_replaying_work(
    graph, evaluator, monkeypatch, interruption
):
    launched = []
    originals = {}

    async def exited_run(evaluation, bundle, directory, *, before_launch, **kwargs):
        before_launch()
        launched.append(directory)
        exited_worker(directory, evaluation, complete=len(launched) > 1)
        originals.update(
            {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        )
        if len(launched) == 1 and interruption == "before_close":
            raise InterruptedFinalization
        result = evaluator._close(evaluation, directory)
        if len(launched) == 1 and interruption == "after_close":
            originals[directory / "result.json"] = (
                directory / "result.json"
            ).read_bytes()
            raise InterruptedFinalization
        return result

    # Only worker execution is synthetic. Device ownership, _close, receipts,
    # production recover, and GraphEvolution retry/accounting remain real.
    monkeypatch.setattr(evaluator, "_run", exited_run)
    search = GraphEvolution(graph, forbidden_adapter, evaluator)
    if interruption:
        with pytest.raises(InterruptedFinalization):
            await evaluate(search)
        assert len(launched) == 1
        assert (launched[0] / "result.json").exists() == (interruption == "after_close")
        search = GraphEvolution(graph, forbidden_adapter, evaluator, recover=True)
    records = await evaluate(search)
    assert len(launched) == 2 and launched[0] != launched[1]
    assert len(records) == 1 and records[0].failure_origin == "agent"
    assert records[0].episode is not None
    partial = EvaluationRecord.model_validate_json(
        (launched[0] / "result.json").read_text()
    )
    assert partial.failure_origin == "infrastructure" and partial.episode is None
    assert partial.usage.physical_steps == 10
    assert partial.usage.physical_steps_complete is False
    assert partial.usage.model_calls is None and partial.usage.cost_usd is None
    partial.verify()
    actual_files = inventory_episode(launched[0])
    actual_files.pop("result.json")
    assert {
        Path(item.path).relative_to(launched[0]).as_posix(): {
            "artifact_id": item.artifact_id,
            "bytes": item.bytes,
        }
        for item in partial.evidence
    } == actual_files
    assert not (launched[0] / "episode/harness_context").exists()
    assert all(p.read_bytes() == value for p, value in originals.items())

    state = json.loads(graph.state_path.read_text())
    assert len(state["attempts"]) == 2
    assert all(
        a["launches"] == 1 and a["status"] == "completed"
        for a in state["attempts"].values()
    )
    usage = search.accounting(state)
    assert usage["episode_launches"] == 2 and usage["physical_steps"] == 30
    assert usage["reserved_physical_steps"] == 90
    assert usage["physical_steps_complete"] is False
    assert usage["model_calls"] is None
    assert usage["known_model_calls"] == 0
    assert usage["cost_usd"] is None and usage["wall_time_s"] == 7
    receipts = {p / "result.json": (p / "result.json").read_bytes() for p in launched}
    resumed = GraphEvolution(graph, forbidden_adapter, forbidden_adapter, recover=True)
    assert await evaluate(resumed) == records
    assert all(p.read_bytes() == value for p, value in receipts.items())
    assert resumed.accounting(json.loads(graph.state_path.read_text())) == usage


@pytest.mark.anyio
async def test_partial_api_unknown_action_never_starts_an_infrastructure_retry(
    graph, evaluator, monkeypatch
):
    launched = []

    async def exited_run(evaluation, bundle, directory, *, before_launch, **kwargs):
        before_launch()
        launched.append(directory)
        exited_worker(directory, evaluation, unknown_action=True)
        return evaluator._close(evaluation, directory)

    monkeypatch.setattr(evaluator, "_run", exited_run)
    for recover in (False, True):
        search = GraphEvolution(graph, forbidden_adapter, evaluator, recover=recover)
        with pytest.raises(EvolutionIncomplete, match="never automatically rerun"):
            await evaluate(search)
        assert len(launched) == 1
        assert not (launched[0] / "result.json").exists()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "second_termination", [API_ERROR, SUCCESS, ACTION_OUTCOME_UNKNOWN]
)
async def test_multiple_transcripts_never_start_an_infrastructure_retry(
    graph, evaluator, monkeypatch, second_termination
):
    launched = []
    originals = {}

    async def exited_run(evaluation, bundle, directory, *, before_launch, **kwargs):
        before_launch()
        launched.append(directory)
        transcript = exited_worker(directory, evaluation)
        transcript.update(
            termination_class=second_termination,
            official_success=second_termination == SUCCESS,
        )
        atomic_json(directory / "episode/transcript_other.json", transcript)
        originals.update(
            {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        )
        return evaluator._close(evaluation, directory)

    monkeypatch.setattr(evaluator, "_run", exited_run)
    for recover in (False, True):
        search = GraphEvolution(graph, forbidden_adapter, evaluator, recover=recover)
        with pytest.raises(EvolutionIncomplete, match="multiple rollout transcripts"):
            await evaluate(search)
        assert len(launched) == 1
        assert not (launched[0] / "result.json").exists()
        assert all(p.read_bytes() == value for p, value in originals.items())
        state = json.loads(graph.state_path.read_text())
        assert len(state["attempts"]) == 1
        assert search.accounting(state)["episode_launches"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "tail",
    [
        '{"event_type":"action_outcome_unknown"}',
        '{"event_type":"action_outcome_unknown"',
    ],
)
async def test_unterminated_unknown_or_torn_trace_never_retries(
    graph, evaluator, monkeypatch, tail
):
    launched = []
    originals = {}

    async def exited_run(evaluation, bundle, directory, *, before_launch, **kwargs):
        before_launch()
        launched.append(directory)
        exited_worker(directory, evaluation)
        trace = directory / "episode/execution_trace.jsonl"
        trace.write_text(trace.read_text() + tail)
        originals.update(
            {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        )
        return evaluator._close(evaluation, directory)

    monkeypatch.setattr(evaluator, "_run", exited_run)
    for recover in (False, True):
        search = GraphEvolution(graph, forbidden_adapter, evaluator, recover=recover)
        with pytest.raises(EvolutionIncomplete, match="never automatically rerun"):
            await evaluate(search)
        assert len(launched) == 1
        assert not (launched[0] / "result.json").exists()
        assert all(p.read_bytes() == value for p, value in originals.items())
        state = json.loads(graph.state_path.read_text())
        assert len(state["attempts"]) == 1
        assert search.accounting(state)["episode_launches"] == 1


def test_complete_control_record_without_newline_counts_in_partial_lower_bound(
    graph, evaluator, tmp_path
):
    directory = tmp_path / "partial"
    evaluation = request(graph)
    exited_worker(directory, evaluation)
    trace = directory / "episode/execution_trace.jsonl"
    original = trace.read_bytes().removesuffix(b"\n")
    trace.write_bytes(original)
    record = evaluator._close(evaluation, directory)
    assert record.usage.physical_steps == 10
    assert record.usage.physical_steps_complete is False
    assert record.failure_origin == "infrastructure" and record.episode is None
    assert trace.read_bytes() == original
    record.verify()


@pytest.mark.parametrize(
    ("termination", "success"),
    [
        (PHYSICAL_FAILURE, False),
        (SUCCESS, True),
        (ACTION_OUTCOME_UNKNOWN, False),
        (API_ERROR, True),
    ],
)
def test_other_results_cannot_downgrade_missing_decision_evidence(
    graph, evaluator, tmp_path, termination, success
):
    directory = tmp_path / "rejected"
    evaluation = request(graph)
    transcript = exited_worker(directory, evaluation)
    transcript.update(
        termination_class=termination,
        official_success=success,
        api_error_retryable=True if termination == API_ERROR else None,
    )
    atomic_json(directory / "episode/transcript_test.json", transcript)
    with pytest.raises(ValueError, match="missing required decision evidence"):
        evaluator._close(evaluation, directory)
    assert not (directory / "result.json").exists()


@pytest.mark.parametrize("changed", ["bundle", "initialization", "missing_exit"])
def test_partial_api_requires_matching_identity_and_confirmed_exit(
    graph, evaluator, tmp_path, changed
):
    directory = tmp_path / "rejected"
    evaluation = request(graph)
    transcript = exited_worker(directory, evaluation)
    if changed == "bundle":
        transcript["bundle_id"] = "c" * 64
    elif changed == "initialization":
        transcript["run_configuration"] = {
            **transcript["run_configuration"],
            "seed": 52,
        }
    else:
        (directory / "exit.json").unlink()
    atomic_json(directory / "episode/transcript_test.json", transcript)
    with pytest.raises((ValueError, EvolutionIncomplete)):
        evaluator._close(evaluation, directory)
    assert not (directory / "result.json").exists()


def test_partial_native_receipt_invents_neither_model_calls_nor_cost(
    graph, evaluator, tmp_path
):
    directory = tmp_path / "partial"
    evaluation = request(graph)
    transcript = exited_worker(directory, evaluation)
    transcript["stats"] = {"requests": 7}
    atomic_json(directory / "episode/transcript_test.json", transcript)
    record = evaluator._close(evaluation, directory)
    assert record.usage.model_calls is None
    assert record.usage.cost_usd is None
    assert record.episode is None
