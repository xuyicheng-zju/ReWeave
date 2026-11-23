# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Durable native-author transport recovery."""

import asyncio
import json

import pytest

from reweave.evolution.authoring.author import NativeBundleAuthor, cumulative_usage
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.infrastructure.native_retry import (
    NATIVE_RETRY_LIMIT,
    NativeTransportFailure,
    RetryableNativeProviderError,
    is_retryable_native_error,
    native_retry_wait_s,
)


class _Task(NativeBundleAuthor):
    def __init__(self, evidence_workspace, errors):
        super().__init__(
            bundle_loader=None,
            policy=AuthorPolicy(),
            evidence_workspace=evidence_workspace,
        )
        self.errors = list(errors)
        self.calls = 0

    async def _execute(self, proposal, directory, **kwargs):
        self.calls += 1
        if self.calls == 1:
            session = directory / "session"
            session.mkdir(parents=True, exist_ok=True)
            (session / "identity.json").write_text(
                json.dumps({"session_id": "same-session"})
            )
        if self.errors:
            error = self.errors.pop(0)
            if is_retryable_native_error(error):
                raise NativeTransportFailure(directory, error) from error
            raise error
        return "completed"


def test_native_transport_failure_resumes_same_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _attempt: 0
    )
    task = _Task(
        tmp_path / "evidence", [RuntimeError("API Error: 503 Service Unavailable")]
    )
    assert asyncio.run(task(None, tmp_path)) == "completed"
    assert task.calls == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "audit/events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["event"] == "native_transport_retry_scheduled"
    assert events[0]["attempt"] == 1
    assert events[0]["maximum"] == 3
    assert events[0]["error_type"] == "RuntimeError"
    assert events[0]["error_bytes"] == len("API Error: 503 Service Unavailable")
    assert events[0]["retry_wait_s"] == 0
    assert events[0]["same_session"] is True
    assert events[0]["session_id"] == "same-session"
    assert cumulative_usage(tmp_path / "audit").wall_time_complete


@pytest.mark.parametrize(
    "reason",
    [
        "stream closed before response.completed",
        "Transport error: network error: error decoding response body",
    ],
)
def test_codex_closed_response_resumes_same_session_with_existing_budget(
    tmp_path, monkeypatch, reason
):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _attempt: 0
    )
    task = _Task(
        tmp_path / "evidence",
        [
            RuntimeError(
                'Codex author turn did not complete: {"message": '
                f'"stream disconnected before completion: {reason}"}}'
            )
        ],
    )
    assert asyncio.run(task(None, tmp_path)) == "completed"
    event = json.loads(
        (tmp_path / "audit/native_transport_retries.jsonl").read_text().splitlines()[0]
    )
    assert task.calls == 2
    assert event["session_id"] == "same-session"
    assert event["maximum"] == NATIVE_RETRY_LIMIT


def test_provider_declared_transport_failure_resumes_same_session(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _attempt: 0
    )
    task = _Task(
        tmp_path / "evidence",
        [
            RetryableNativeProviderError(
                "Claude Code provider transport failed: "
                "RetriableError: [canceled] http/2 stream closed with "
                "error code CANCEL (0x8)"
            )
        ],
    )
    assert asyncio.run(task(None, tmp_path)) == "completed"
    assert task.calls == 2
    event = json.loads(
        (tmp_path / "audit/native_transport_retries.jsonl").read_text().splitlines()[0]
    )
    assert event["same_session"] is True
    assert event["session_id"] == "same-session"
    assert event["maximum"] == NATIVE_RETRY_LIMIT


def test_provider_declared_transport_failure_has_a_finite_retry_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _attempt: 0
    )
    failures = [
        RetryableNativeProviderError("RetriableError: provider unavailable")
        for _ in range(NATIVE_RETRY_LIMIT + 1)
    ]
    task = _Task(tmp_path / "evidence", failures)
    with pytest.raises(RetryableNativeProviderError):
        asyncio.run(task(None, tmp_path))
    assert task.calls == NATIVE_RETRY_LIMIT + 1


def test_parallel_provider_failures_resume_their_own_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _attempt: 0
    )

    class ParallelTasks(NativeBundleAuthor):
        def __init__(self):
            super().__init__(
                bundle_loader=None,
                policy=AuthorPolicy(),
                evidence_workspace=tmp_path / "evidence",
            )
            self.calls = {}

        async def _execute(self, proposal, directory, **kwargs):
            calls = self.calls.get(directory, 0) + 1
            self.calls[directory] = calls
            if calls == 1:
                session = directory / "session"
                session.mkdir(parents=True)
                (session / "identity.json").write_text(
                    json.dumps({"session_id": directory.name})
                )
                error = RetryableNativeProviderError(
                    "RetriableError: [canceled] http/2 stream closed"
                )
                raise NativeTransportFailure(directory, error) from error
            return directory.name

    tasks = ParallelTasks()
    first = tmp_path / "first"
    second = tmp_path / "second"

    async def run():
        return await asyncio.gather(tasks(None, first), tasks(None, second))

    assert asyncio.run(run()) == ["first", "second"]
    assert tasks.calls == {first: 2, second: 2}
    for directory in (first, second):
        event = json.loads(
            (directory / "audit/native_transport_retries.jsonl")
            .read_text()
            .splitlines()[0]
        )
        assert event["session_id"] == directory.name


def test_native_permanent_failure_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _attempt: 0
    )
    task = _Task(
        tmp_path / "evidence", [RuntimeError("authentication failed: invalid API key")]
    )
    with pytest.raises(RuntimeError, match="authentication failed"):
        asyncio.run(task(None, tmp_path))
    assert task.calls == 1
    assert not (tmp_path / "audit/events.jsonl").exists()


@pytest.mark.parametrize(
    ("terminal_reason", "status", "text", "expected"),
    [
        ("max_turns", None, "maximum turns reached", False),
        ("api_error", 503, "upstream failed", True),
        ("api_error", 400, "invalid request", False),
        (None, None, "API Error: 400 Bad Request", False),
        (None, None, "process exited with status 127", False),
    ],
)
def test_native_sdk_result_classification(terminal_reason, status, text, expected):
    class ResultError(RuntimeError):
        pass

    error = ResultError(text)
    error.terminal_reason = terminal_reason
    error.api_error_status = status
    assert is_retryable_native_error(error) is expected


def test_native_retry_limit_survives_new_task_instance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _attempt: 0
    )
    first = _Task(
        tmp_path / "evidence", [RuntimeError("API Error: 503 Service Unavailable")]
    )
    assert asyncio.run(first(None, tmp_path)) == "completed"

    second = _Task(
        tmp_path / "evidence",
        [RuntimeError("API Error: 503 Service Unavailable") for _ in range(4)],
    )
    with pytest.raises(RuntimeError, match="503 Service Unavailable"):
        asyncio.run(second(None, tmp_path, resume=True))
    assert second.calls == 3
    ledger = tmp_path / "audit/native_transport_retries.jsonl"
    assert len(ledger.read_text().splitlines()) == 3


def test_native_retry_backoff_is_bounded():
    assert native_retry_wait_s(1) == 2
    assert native_retry_wait_s(100) == 60


@pytest.mark.parametrize("wrapper", [RuntimeError, RetryableNativeProviderError])
def test_transient_wrapper_cannot_hide_permanent_cause(wrapper):
    outer = wrapper("provider temporarily unavailable")
    outer.api_error_status = 503
    inner = RuntimeError("request rejected")
    inner.api_error_status = 401
    outer.__cause__ = inner
    assert not is_retryable_native_error(outer)


def test_generic_400_does_not_start_a_speculative_resume(tmp_path):
    task = _Task(tmp_path / "evidence", [RuntimeError("API Error: 400 Bad Request")])
    with pytest.raises(RuntimeError, match="400 Bad Request"):
        asyncio.run(task(None, tmp_path))
    assert task.calls == 1
    assert not (tmp_path / "audit/native_transport_retries.jsonl").exists()


def test_parent_does_not_retry_a_child_transport_failure(tmp_path):
    parent = _Task(tmp_path / "evidence", [])
    child = tmp_path / "critic"

    async def invoke_child(_proposal, _directory, **_kwargs):
        raise NativeTransportFailure(child, RuntimeError("API Error: 503"))

    parent._execute = invoke_child
    with pytest.raises(NativeTransportFailure):
        asyncio.run(parent(None, tmp_path))
    assert not (tmp_path / "audit/native_transport_retries.jsonl").exists()


def test_native_generic_wrapper_does_not_hide_permanent_inner_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.native_retry_wait_s", lambda _attempt: 0
    )

    class ResultError(RuntimeError):
        pass

    try:
        raise RuntimeError("unknown model: unavailable-alias")
    except RuntimeError as inner:
        outer = ResultError("native CLI failed")
        outer.__cause__ = inner
    task = _Task(tmp_path / "evidence", [outer])
    with pytest.raises(ResultError, match="native CLI failed"):
        asyncio.run(task(None, tmp_path))
    assert task.calls == 1
    assert not (tmp_path / "audit/events.jsonl").exists()


def test_claude_upstream_eof_is_retryable_but_permanent_causes_win():
    message = "API Error: Claude upstream stream ended before message_delta, reason=eof"
    assert is_retryable_native_error(RuntimeError(message))
    assert not is_retryable_native_error(RuntimeError("unexpected EOF in JSON"))
    error = RuntimeError(message)
    error.__cause__ = RuntimeError("invalid api key")
    assert not is_retryable_native_error(error)
