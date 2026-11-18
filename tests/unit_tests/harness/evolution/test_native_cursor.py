"""Cursor coding queries retain native identities, unknown costs and exit proof."""

import json
import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from reweave.evolution.authoring.config import CURSOR_LONG_CONTEXT_MODEL, AuthorPolicy
from reweave.evolution.infrastructure import native_cursor as cursor
from reweave.utils.persistence import atomic_json


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("stderr_case", ["none", "current_retry", "old_retry"])
@pytest.mark.parametrize("model", ["grok-4.7-xhigh", CURSOR_LONG_CONTEXT_MODEL])
async def test_native_query_binds_init_and_resumes_same_chat(
    tmp_path, monkeypatch, success, model, stderr_case
):
    workspace = tmp_path / "workspace"
    (workspace / ".cursor").mkdir(parents=True)
    (tmp_path / "audit/queries").mkdir(parents=True)
    identity = str(uuid.uuid4())
    calls, events = [], []
    retry_error = "RetriableError: Agent turn stopped after repeated resume attempts made no progress\n"
    if stderr_case == "old_retry":
        (tmp_path / "audit/cursor_stderr.log").write_text(retry_error)
    monkeypatch.setattr(
        cursor,
        "cursor_workspace",
        lambda *args: (
            "native-launcher",
            {"CURSOR_DATA_DIR": str(workspace / ".scratch/runtime/cursor")},
        ),
    )

    class Admission:
        def __init__(self, route, *, lane):
            assert route == "cursor-login" and lane == "offline"

        @asynccontextmanager
        async def lease(self, path):
            yield

    monkeypatch.setattr(cursor, "ProviderAdmission", Admission)

    long_context = "context=500k" in model

    async def select(*args):
        atomic_json(tmp_path / "session/identity.json", {"session_id": identity})
        return identity

    monkeypatch.setattr(cursor, "_select_long_context", select)

    async def run(command, cwd, environment, stderr, consume):
        calls.append(command)
        if stderr_case == "current_retry":
            with stderr.open("a") as stream:
                stream.write(retry_error)
        assert ("--resume" in command) is (long_context or len(calls) > 1)
        if long_context:
            assert "--model" not in command
            assert command[command.index("--resume") + 1] == identity
        else:
            assert command[command.index("--model") + 1] == model
        consume(
            json.dumps(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": identity,
                }
            ).encode()
        )
        assert (
            json.loads((tmp_path / "session/identity.json").read_text())["session_id"]
            == identity
        )
        if success:
            consume(
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "session_id": identity,
                        "is_error": False,
                    }
                ).encode()
            )
        return 0

    monkeypatch.setattr(cursor, "_run_cli", run)
    owner = SimpleNamespace(
        policy=AuthorPolicy(author="cursor", model=model), native_runtime_root=None
    )
    query_path = tmp_path / "audit/queries/000000.json"

    def begin():
        atomic_json(
            query_path,
            {
                "physical_steps": 0,
                "model_calls": None,
                "wall_time_s": 0,
                "wall_time_complete": False,
                "cost_usd": None,
            },
        )

    args = (
        owner,
        workspace,
        tmp_path,
        (),
        (),
        events.append,
        time.monotonic(),
        query_path,
        begin,
        "instructions",
        "prompt",
        "http://127.0.0.1/tools",
    )
    if not success:
        expected = (
            "declared a retriable turn failure"
            if stderr_case == "current_retry"
            else "without a successful result"
        )
        with pytest.raises(RuntimeError, match=expected) as caught:
            await cursor.cursor_query(*args)
        from reweave.evolution.infrastructure.native_retry import (
            is_retryable_native_error,
        )

        assert is_retryable_native_error(caught.value) is (
            stderr_case == "current_retry"
        )
        assert not (tmp_path / "native_result.json").exists()
        assert (tmp_path / "session/identity.json").exists()
        return
    usage, session_id = await cursor.cursor_query(*args)
    assert session_id == identity
    assert usage.model_calls is None and usage.cost_usd is None
    receipt = json.loads((tmp_path / "native_result.json").read_text())
    assert receipt["native_checkpoint"]["backend"] == "cursor"
    await cursor.cursor_query(*args)
    assert len(calls) == 2
    assert len(events) == 4


def test_mixed_session_and_error_result_never_complete():
    with pytest.raises(ValueError, match="another native session"):
        cursor.cursor_event({"type": "result", "session_id": "wrong"}, "expected")
    assert not cursor.cursor_event(
        {"type": "result", "subtype": "success", "is_error": True}, "expected"
    )
    assert not cursor.cursor_event({"type": "assistant"}, "expected")


def test_cursor_login_and_chat_state_stay_separate_with_explicit_proxy(
    tmp_path, monkeypatch
):
    private = tmp_path / "account/.config/reweave/cursor"
    private.mkdir(parents=True)
    (private / "auth.json").touch()
    workspace = tmp_path / "job/workspace"
    workspace.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    seen = {}
    monkeypatch.setenv("HOME", "/experiment-codex-home")
    monkeypatch.setenv("REWEAVE_CURSOR_PROXY", "http://cursor-proxy:8080")
    monkeypatch.delenv("REWEAVE_CURSOR_ACCOUNT_HOME", raising=False)
    monkeypatch.setattr(
        cursor, "getpwuid", lambda _: SimpleNamespace(pw_dir=str(tmp_path / "account"))
    )
    monkeypatch.setattr(cursor, "CURSOR_PERSISTENT_ROOT", tmp_path / "registry")
    monkeypatch.setattr(cursor.tempfile, "mkdtemp", lambda **_: str(runtime))

    def environment(origin):
        assert origin["HOME"] == str(tmp_path / "account")
        return {
            "HOME": origin["HOME"],
            "CURSOR_CONFIG_DIR": str(private),
            "XDG_CONFIG_HOME": str(private.parent),
            "OPENAI_API_KEY": "must-not-be-inherited",
        }

    def launcher(*args, **kwargs):
        seen.update(kwargs)
        return "launcher", {"CURSOR_DATA_DIR": str(runtime / "cursor")}

    monkeypatch.setattr(cursor, "cursor_environment", environment)
    monkeypatch.setattr(
        cursor, "resolve_cursor_executable", lambda *_: "/installed/cursor/agent"
    )
    monkeypatch.setattr(cursor, "native_author_launcher", launcher)
    monkeypatch.setattr(cursor, "native_output", lambda _: workspace / "candidate")
    (workspace.parent / "launcher").mkdir()
    _, env = cursor.cursor_workspace(
        workspace, workspace.parent, (), (workspace / "candidate",)
    )
    assert env["CURSOR_CONFIG_DIR"] == env["CURSOR_DATA_DIR"] == env["HOME"]
    assert env["XDG_CONFIG_HOME"] == str(private.parent)
    assert env["NODE_USE_ENV_PROXY"] == "1"
    assert env["HTTPS_PROXY"] == env["HTTP_PROXY"] == "http://cursor-proxy:8080"
    assert env["https_proxy"] == env["http_proxy"] == "http://cursor-proxy:8080"
    assert env["ALL_PROXY"] == env["all_proxy"] == "http://cursor-proxy:8080"
    assert {"127.0.0.1", "localhost", "::1"} <= set(env["NO_PROXY"].split(","))
    assert "OPENAI_API_KEY" not in env
    assert private / "auth.json" in seen["readable_roots"]
    assert private not in seen["native_state_roots"]


@pytest.mark.anyio
async def test_native_process_uses_owned_group_and_drains_before_return(
    tmp_path, monkeypatch
):
    calls = []

    class Output:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    process = SimpleNamespace(stdout=Output(), returncode=0)

    async def wait():
        return 0

    process.wait = wait

    async def spawn(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return process

    async def drain(owned, *, terminate):
        assert owned is process
        calls.append("drained")

    monkeypatch.setattr(cursor.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(cursor, "drain_process_group", drain)
    assert (
        await cursor._run_cli(
            ["agent"], tmp_path, {}, tmp_path / "stderr", lambda _: None
        )
        == 0
    )
    assert calls == ["drained"]
