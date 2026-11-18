# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Native publication receipts and executable author admission regressions."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reweave.evolution.authoring.author import (
    NativeBundleAuthor,
    _evidence_roots,
    cumulative_usage,
)
from reweave.evolution.authoring.author import (
    native_completion as _native_completion,
)
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    EvaluationRecord,
    EvaluationRequest,
    Proposal,
    Usage,
    VersionReference,
)
from reweave.evolution.evidence.records import inventory_episode
from reweave.evolution.validation.admission import admit_candidate
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    author_candidate,
    baseline,
    evolution_history,
    mutated,
    protocol,
    usage,
)


def native_completion(directory, measured, session_id, checkpoint):
    """Completed-query fixture includes its authoritative usage journal."""
    atomic_json(
        directory / "audit/queries/000000.json", measured.model_dump(mode="json")
    )
    _native_completion(directory, measured, session_id, checkpoint)


def host_input(workspace: Path, name: str) -> Path:
    """Mirror the production Host-private request location in fixtures."""
    return workspace.parent / "audit/inputs" / name


def test_missing_infrastructure_receipt_has_no_native_physical_root(tmp_path):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    missing = proposal.evidence[0].model_copy(
        update={"failure_origin": "infrastructure", "infrastructure_retryable": False}
    )
    assert _evidence_roots(proposal.model_copy(update={"evidence": (missing,)})) == {}
    assert _evidence_roots(proposal) == {}
    from reweave.evolution.authoring.author import _native_evidence_paths

    assert _native_evidence_paths(proposal, {}) == tuple(
        Path(file.path) for file in proposal.evidence[0].evidence
    )


def workspace(tmp_path):
    from reweave.evolution.authoring import client as author_client
    from reweave.evolution.authoring.references import AuthorReferences

    parent = baseline()
    request = EvaluationRequest(
        conditions=protocol().execution,
        record_id=new_record_id(),
        bundle_id=parent.record_id,
        instance=protocol().dev[0],
        split="dev",
        manifest_id=protocol().manifest_id,
        execution=protocol().execution_id,
    )
    record = EvaluationRecord(
        record_id=new_record_id(),
        request=request,
        official_success=False,
        failure_origin="agent",
        infrastructure_retryable=None,
        usage=usage(),
        evidence=(artifact(tmp_path, "fixture.json", {}),),
        episode=None,
        initialization=request.instance.initialization,
    )
    history = evolution_history(tmp_path)
    from tests.unit_tests.harness.evolution.fixtures import graph_proposal_fields

    proposal = Proposal(
        **graph_proposal_fields(tmp_path),
        record_id="batch_000000",
        parent=VersionReference(version_id="b" * 64, bundle_id=parent.record_id),
        references=(VersionReference(version_id="b" * 64, bundle_id=parent.record_id),),
        round_index=0,
        author_index=0,
        scopes=("W", "C", "L"),
        history=history,
        evidence=(record,),
    )
    root = tmp_path / "workspace"
    (root / ".archive").mkdir(parents=True)
    (root / "context").mkdir()
    AuthorReferences.create(
        root, registry=tmp_path / "author_references.json", evidence_roots={}
    )
    staged_catalog = root / ".host_inputs/author_references.json"
    published_catalog = host_input(root, "author_references.json")
    published_catalog.parent.parent.mkdir(parents=True, exist_ok=True)
    staged_catalog.parent.rename(published_catalog.parent)
    atomic_json(host_input(root, "executed_sources.json"), {})
    atomic_json(
        host_input(root, "references.json"),
        {parent.record_id: parent.model_dump(mode="json")},
    )
    atomic_json(host_input(root, "proposal.json"), proposal.model_dump(mode="json"))
    (root / "context/EVOLUTION_HISTORY.md").write_text(
        Path(history.summary.path).read_text()
    )
    atomic_json(
        root / "context/history_originals.json", {"files": {}, "navigation": {}}
    )
    atomic_json(
        root / "context/input.json",
        {
            "kind": "mutation",
            "implementation": {"executed_sources": {}},
            "index": [],
        },
    )
    atomic_json(root / "context/experiment_access.json", {"episodes": []})
    client_source = root / "context/host_client.py"
    client_source.write_bytes(Path(author_client.__file__).read_bytes())
    (root / "context/dev.py").write_text(author_client.development_client())
    atomic_json(
        root / "context/gateway.json",
        {
            "operations": {
                "read_terminal_state": {
                    "effect": "read",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            },
            "tools": [
                {
                    "name": "read_terminal_state",
                    "description": "Read the synthetic environment's terminal state",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ],
            "episode_limits": {
                "program_calls": 6,
                "gateway_calls": 6,
                "wall_time_s": 10,
            },
        },
    )
    author_candidate(parent, root / "candidate")
    (root / "candidate/task-family/context/instructions.md").write_text(
        "verify progress"
    )
    atomic_json(
        root / ".scratch/tests/interaction.json",
        [
            {
                "name": "context-through-default-loop",
                "system_prompt": "Control using public tools.",
                "task_message": "Inspect the scene.",
                "model_responses": [
                    {
                        "tool_calls": [
                            {
                                "name": "read_terminal_state",
                                "arguments": {},
                            }
                        ]
                    }
                ],
                "operations": [
                    {
                        "name": "read_terminal_state",
                        "arguments": {},
                        "result": {"terminated": True},
                        "environment_state": {
                            "official_success": True,
                            "terminated": False,
                            "truncated": False,
                        },
                    }
                ],
                "assertions": [{"path": ["coverage", "C"], "equals": True}],
            }
        ],
    )
    atomic_json(
        root / "candidate/plan.json",
        {
            "diagnosis": "The available trace cannot distinguish stale observation from incorrect control.",
            "summary": "Ask the Planner to verify progress before continuing.",
            "validation": "Context contract and synthetic full-loop interaction passed; physical efficacy remains untested.",
            "scopes": ["C"],
            "cross_scope_reason": None,
        },
    )
    atomic_json(
        tmp_path / "input_inventory.json",
        {"context": inventory_episode(root / "context"), "archive": {}},
    )
    (tmp_path / "audit").mkdir(exist_ok=True)
    return root


def test_candidate_roundtrip_runs_real_loop_worker(tmp_path):
    root = workspace(tmp_path)
    assert asyncio.run(admit_candidate(root)).same_harness(mutated(baseline(), "C"))


def test_workflow_teaching_example_uses_complete_physical_feedback(tmp_path):
    from jsonschema import validate

    from reweave.evolution.authoring.examples import author_examples
    from reweave.evolution.bundle import AgentBundle
    from reweave.evolution.validation.admission import check_runtime_contracts
    from reweave.runtime.capability import PHYSICAL_FEEDBACK_SCHEMA

    example = author_examples()["W"]["workflow"]
    for case in example["contract_cases"]:
        for response in case["responses"]:
            if response["operation"] == "release":
                validate(response["result"], PHYSICAL_FEEDBACK_SCHEMA)
    root = workspace(tmp_path)
    contract = json.loads((root / "context/gateway.json").read_text())
    contract["operations"] = {
        "release": {
            "effect": "physical",
            "input_schema": {
                "type": "object",
                "properties": {"max_steps": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
        "view_env_state": {
            "effect": "read",
            "input_schema": {
                "type": "object",
                "properties": {"step": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
    }
    payload = baseline().model_dump(mode="json")
    payload["workflows"] = [example]
    asyncio.run(check_runtime_contracts(AgentBundle.model_validate(payload), contract))


@pytest.mark.parametrize(
    "filename,contents",
    [
        ("context/context.json", []),
        (
            "context/context.json",
            {"context_fields": [], "instructions": "hidden override"},
        ),
    ],
)
def test_bad_json_is_explicit_rejection(tmp_path, filename, contents):
    root = workspace(tmp_path)
    atomic_json(root / "candidate" / filename, contents)
    with pytest.raises(ValueError):
        asyncio.run(admit_candidate(root))


def test_plan_must_explain_actual_changed_scope(tmp_path):
    root = workspace(tmp_path)
    (root / "candidate/task-family/context/instructions.md").write_text(
        "Check measured displacement."
    )
    plan_path = root / "candidate/plan.json"
    plan = json.loads(plan_path.read_text())
    plan["scopes"] = ["W"]
    atomic_json(plan_path, plan)
    with pytest.raises(ValueError, match="plan scopes"):
        asyncio.run(admit_candidate(root))


def test_final_plan_rejects_no_op_before_runtime_checks(tmp_path, monkeypatch):
    root = workspace(tmp_path)
    author_candidate(baseline(), root / "candidate")
    atomic_json(
        root / "candidate/plan.json",
        {
            "diagnosis": "The H0 trajectory is inconclusive.",
            "summary": "Retain a reference unchanged.",
            "validation": "No changed behavior was tested.",
            "scopes": [],
            "cross_scope_reason": None,
        },
    )
    runtime_checks_called = False

    async def runtime_checks(*args, **kwargs):
        nonlocal runtime_checks_called
        runtime_checks_called = True

    monkeypatch.setattr(
        "reweave.evolution.validation.admission.check_runtime_contracts", runtime_checks
    )
    with pytest.raises(ValueError, match="differ from every reference"):
        asyncio.run(admit_candidate(root))
    assert not runtime_checks_called


def test_cumulative_native_usage_preserves_failed_work_and_unknown(tmp_path):
    atomic_json(
        tmp_path / "queries/000000.json",
        Usage(physical_steps=0, model_calls=4, wall_time_s=15, cost_usd=2).model_dump(
            mode="json"
        ),
    )
    atomic_json(
        tmp_path / "queries/000001.json",
        Usage(physical_steps=0, model_calls=3, wall_time_s=10, cost_usd=1).model_dump(
            mode="json"
        ),
    )
    assert cumulative_usage(tmp_path).cost_usd == 3
    assert cumulative_usage(tmp_path).model_calls == 7
    atomic_json(
        tmp_path / "queries/000002.json",
        Usage(
            physical_steps=0, model_calls=None, wall_time_s=7, cost_usd=None
        ).model_dump(mode="json"),
    )
    measured = cumulative_usage(tmp_path)
    assert (
        measured.wall_time_s == 32
        and measured.cost_usd is None
        and measured.model_calls is None
    )


class Journal:
    def __init__(self, directory):
        directory.mkdir(exist_ok=True)

    def bind(self, identity):
        assert identity == "native-id"

    def checkpoint(self):
        return {"bytes": 1}

    def verify(self, checkpoint, *, exact):
        assert exact and checkpoint == self.checkpoint()


def test_completed_query_loads_without_rechecking_output_snapshot(
    tmp_path, monkeypatch
):
    root = workspace(tmp_path)
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.ClaudeSessionStore", Journal
    )
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    (root / "candidate/task-family/context/instructions.md").write_text("late write")
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    assert author._load_completion(tmp_path)["session_id"] == "native-id"


def test_import_timeout_requires_correction_without_publishing_invalid_code(
    tmp_path, monkeypatch
):
    workspace(tmp_path)
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.ClaudeSessionStore", Journal
    )

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("protected worker", 30)

    monkeypatch.setattr(
        "reweave.runtime.sandbox.ProgramSandbox.validate_program_import", timeout
    )
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    from reweave.evolution.authoring.author import AuthorCorrectionRequired

    with pytest.raises(AuthorCorrectionRequired, match="timed out"):
        asyncio.run(author._publish(tmp_path))
    assert any(
        "admission_rejected" in line
        for line in (tmp_path / "audit/events.jsonl").read_text().splitlines()
    )
    assert not (tmp_path / "result.json").exists()
    feedback = json.loads(
        (tmp_path / "audit/corrections/000000/feedback.json").read_text()
    )
    assert feedback["error_type"] == "TimeoutExpired"
    assert "protected worker" in feedback["traceback"]
    assert (tmp_path / "native_result.json").exists()


def test_intermediate_native_result_is_not_a_publish_receipt(tmp_path, monkeypatch):
    import uuid
    from types import SimpleNamespace

    from claude_agent_sdk import ResultMessage, SystemMessage

    directory = tmp_path.resolve()
    root = workspace(directory)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    identity = str(uuid.uuid4())

    async def query(*, prompt, options):
        await options.session_store.append(
            {"session_id": identity, "project_key": "review"},
            [
                {
                    "uuid": str(uuid.uuid4()),
                    "type": "assistant",
                    "message": {"content": "Intermediate native turn"},
                }
            ],
        )
        yield SystemMessage(subtype="init", data={"session_id": identity})
        yield ResultMessage(
            subtype="success",
            duration_ms=5,
            duration_api_ms=2,
            is_error=False,
            num_turns=1,
            session_id=identity,
            total_cost_usd=1,
        )
        raise RuntimeError("subsequent native turn failed after intermediate result")

    monkeypatch.setattr("claude_agent_sdk.query", query)
    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeAgentOptions", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.claude_workspace_options",
        lambda *a, **kw: {},
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    with pytest.raises(RuntimeError, match="subsequent native turn"):
        asyncio.run(author._execute(proposal, directory, resume=True))
    assert not (directory / "native_result.json").exists()


def test_completed_receipt_reuses_provider_output_without_journal_gate(
    tmp_path, monkeypatch
):
    workspace(tmp_path)

    class ChangedJournal(Journal):
        def verify(self, checkpoint, *, exact):
            raise ValueError("native transcript differs from completion")

    monkeypatch.setattr(
        "reweave.evolution.authoring.author.ClaudeSessionStore", ChangedJournal
    )
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    asyncio.run(author._publish(tmp_path))
    assert (tmp_path / "result.json").exists()


def test_fixture_violation_cannot_be_swallowed_by_generated_workflow(tmp_path):
    from reweave.evolution.bundle import AgentBundle
    from reweave.evolution.validation.admission import check_runtime_contracts
    from tests.unit_tests.harness.evolution.fixtures import workflow

    root = workspace(tmp_path)
    contract = json.loads((root / "context/gateway.json").read_text())
    contract["operations"] = {
        "inspect_scene": {"effect": "read", "input_schema": {"type": "object"}}
    }
    atomic_json(root / "context/gateway.json", contract)
    payload = baseline().model_dump(mode="json")
    program = workflow().model_dump(mode="json")
    program["permissions"] = program["observations"] = ["inspect_scene"]
    program["program"]["files"][0]["source"] = """def run(runtime, context):
    try:
        runtime.observe("inspect_scene", {})
    except Exception:
        pass
    return {"status": "failed", "summary": "caught", "output": {}}
"""
    program["contract_cases"] = [
        {
            "arguments": {},
            "observation": {},
            "responses": [{"operation": "WRONG", "arguments": {}, "result": {}}],
            "expected_status": "failed",
        }
    ]
    payload["workflows"] = [program]
    with pytest.raises(ValueError, match="contract fixture"):
        asyncio.run(
            check_runtime_contracts(AgentBundle.model_validate(payload), contract)
        )


def test_native_journal_without_identity_cannot_start_another_session(
    tmp_path, monkeypatch
):
    root = workspace(tmp_path)
    prior = root / ".scratch/runtime/claude/projects/session.jsonl"
    prior.parent.mkdir(parents=True)
    prior.write_text('{"type":"assistant"}\n')
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    launches = []
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    with pytest.raises(ValueError, match="recoverable identity"):
        asyncio.run(
            author._execute(
                proposal,
                tmp_path,
                resume=True,
                before_launch=lambda: launches.append(1),
            )
        )
    assert launches == []
    assert not list((tmp_path / "audit/queries").glob("*.json"))


@pytest.mark.parametrize("kind", ["main", "metadata", "subagent"])
def test_actual_claude_query_records_provider_completion(tmp_path, monkeypatch, kind):
    import uuid
    from types import SimpleNamespace

    from claude_agent_sdk import ResultMessage, SystemMessage

    directory = tmp_path.resolve()
    root = workspace(directory)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    identity = str(uuid.uuid4())
    writes = []
    import reweave.evolution.authoring.author as author_module

    real_atomic_json = author_module.atomic_json

    def tracked_write(path, value):
        writes.append(path)
        return real_atomic_json(path, value)

    monkeypatch.setattr(author_module, "atomic_json", tracked_write)

    async def query(*, prompt, options):
        key = {"session_id": identity, "project_key": "review"}
        if kind == "subagent":
            key["subpath"] = "subagents/agent-a"
        row = (
            {"type": "tag", "value": "metadata only"}
            if kind == "metadata"
            else {
                "uuid": "main",
                "type": "assistant",
                "message": {"content": "Analyzed this query"},
            }
        )
        await options.session_store.append(key, [row])
        yield SystemMessage(subtype="init", data={"session_id": identity})
        assert (
            options.mcp_servers["author"]["url"]
            == options.env["REWEAVE_AUTHOR_TOOLS_URL"]
        )
        before_deltas = len(writes)
        for index in range(128):
            yield SystemMessage(subtype="thinking_tokens", data={"index": index})
        assert len(writes) == before_deltas
        yield ResultMessage(
            subtype="success",
            duration_ms=5,
            duration_api_ms=2,
            is_error=False,
            num_turns=1,
            session_id=identity,
            total_cost_usd=1,
        )

    monkeypatch.setattr("claude_agent_sdk.query", query)
    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeAgentOptions", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.claude_workspace_options",
        lambda *a, **kw: {},
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    if kind in {"main", "metadata"}:
        assert (
            asyncio.run(author._execute(proposal, directory, resume=True))
            .candidates[0]
            .bundle
            is not None
        )
        assert (directory / "native_result.json").exists()
        raw = [
            json.loads(line)
            for line in (directory / "audit/events.jsonl").read_text().splitlines()
        ]
        assert [
            row["data"]["index"]
            for row in raw
            if row.get("subtype") == "thinking_tokens"
        ] == list(range(128))
    else:
        with pytest.raises(
            ValueError, match="no new main|native conversation is missing"
        ):
            asyncio.run(author._execute(proposal, directory, resume=True))
        assert not (directory / "native_result.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "fresh",
        "new",
        "replay",
        "metadata",
        "budget",
        "interrupt",
        "interrupt_twice",
        "rate_limit",
        "quota",
    ],
)
def test_actual_codex_query_accepts_provider_completion_without_growth_gate(
    tmp_path, monkeypatch, kind
):
    import threading
    import time
    from types import SimpleNamespace

    monkeypatch.setenv("REWEAVE_PROVIDER_ADMISSION_ROOT", str(tmp_path / "admission"))
    directory = tmp_path.resolve()
    root = workspace(directory)
    (directory / "audit/queries").mkdir()
    native = directory / "codex-native.jsonl"

    def message(text):
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        }

    previous = message("prior turn")
    native.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "codex-thread"}})
        + "\n"
        + (json.dumps(previous) + "\n" if kind not in {"fresh", "budget"} else "")
    )

    if kind not in {"fresh", "budget"}:
        atomic_json(directory / "session/identity.json", {"session_id": "codex-thread"})

    interrupted = kind.startswith("interrupt")
    ready = asyncio.Event()
    interrupt_started = asyncio.Event()
    wake = threading.Event()
    interrupts = []
    drained = []

    class Turn:
        async def interrupt(self):
            interrupts.append("native-turn")
            interrupt_started.set()
            await asyncio.sleep(0.01)
            wake.set()

        async def stream(self):
            if kind == "quota":
                yield {
                    "method": "error",
                    "payload": {
                        "error": {
                            "codex_error_info": "other",
                            "message": "The ChatGPT user has reached their Subscription Sharing usage limit.",
                        },
                        "will_retry": False,
                    },
                }
            if kind == "rate_limit":
                yield {
                    "method": "error",
                    "payload": {
                        "error": {
                            "additional_details": "last status: 429 Too Many Requests",
                            "codex_error_info": {
                                "response_stream_disconnected": {
                                    "http_status_code": 429
                                }
                            },
                        },
                        "will_retry": True,
                    },
                }
            if interrupted:
                ready.set()
                await asyncio.to_thread(wake.wait)
            row = (
                message("new query evidence")
                if kind in {"new", "fresh"}
                else previous
                if kind == "replay"
                else {"type": "event_msg", "payload": {"type": "metadata"}}
            )
            with native.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            from openai_codex.generated.v2_all import Turn as NativeTurn
            from openai_codex.generated.v2_all import (
                TurnCompletedNotification,
                TurnStatus,
            )
            from openai_codex.models import Notification

            yield Notification(
                method="turn/completed",
                payload=TurnCompletedNotification(
                    thread_id="codex-thread",
                    turn=NativeTurn(
                        id="native-turn",
                        items=[],
                        status=TurnStatus.interrupted
                        if interrupted
                        else TurnStatus.completed,
                    ),
                ),
            )
            drained.append(True)

    class Thread:
        id = "codex-thread"
        materialized = kind not in {"fresh", "budget"}

        async def read(self, **kwargs):
            if not self.materialized:
                raise ValueError(
                    "includeTurns is unavailable before first user message"
                )
            return SimpleNamespace(
                thread=SimpleNamespace(
                    id=self.id, path=str(native), model_provider="openai"
                )
            )

        async def turn(self, *args, **kwargs):
            self.materialized = True
            return Turn()

    connections = []

    class Client:
        async def thread_start(self, **kwargs):
            connections.append(("start", None))
            return Thread()

        async def thread_resume(self, *args, **kwargs):
            connections.append(("resume", args[0]))
            return Thread()

        async def close(self):
            wake.set()

    def create_client(**kwargs):
        assert (
            kwargs["mcp_servers"]["author"]
            == kwargs["environment"]["REWEAVE_AUTHOR_TOOLS_URL"]
        )
        access = json.loads((root / "context/experiment_access.json").read_text())
        assert "native_cli" not in access
        assert not (root / "context/NATIVE_EXPERIMENTS.md").exists()
        return Client()

    monkeypatch.setattr(
        "reweave.planner.codex_connection.create_codex_client", create_client
    )
    monkeypatch.setattr(
        "reweave.planner.codex_connection.codex_executable", lambda: sys.executable
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(author="codex"),
    )
    path = directory / "audit/queries/000000.json"

    def begin():
        if kind == "budget":
            raise RuntimeError("budget rejected before query")
        atomic_json(path, usage().model_dump(mode="json"))

    def run():
        async def query():
            work = asyncio.create_task(
                author._codex(
                    root,
                    directory,
                    (),
                    (),
                    lambda _: None,
                    time.monotonic(),
                    path,
                    begin,
                    "Test instructions",
                    "Test native prompt",
                    "http://127.0.0.1:12345/test-author/",
                )
            )
            if interrupted:
                await ready.wait()
                work.cancel()
                if kind == "interrupt_twice":
                    await interrupt_started.wait()
                    work.cancel()
            return await work

        return asyncio.run(query())

    if kind == "budget":
        with pytest.raises(RuntimeError, match="budget rejected"):
            run()
        assert not (directory / "session/identity.json").exists()
        assert not path.exists()
    elif kind == "rate_limit":
        with pytest.raises(RuntimeError, match="HTTP 429"):
            run()
        assert interrupts == ["native-turn"]
        assert drained == [True]
        assert not (directory / "native_result.json").exists()
        assert (directory / "session/transcript.jsonl").exists()
        control = tmp_path / "admission/codex-auth/state.json"
        assert json.loads(control.read_text())["probe_required"]
    elif kind == "quota":
        from reweave.utils.provider_admission import NativeQuotaExhausted

        with pytest.raises(NativeQuotaExhausted):
            run()
        assert not interrupts
        assert drained == [True]
        assert not (directory / "native_result.json").exists()
        assert (directory / "session/transcript.jsonl").exists()
        control = tmp_path / "admission/codex-auth/state.json"
        assert json.loads(control.read_text())["quota_blocked"]
        assert (directory / "audit/native_quota_exhausted.json").exists()
    elif interrupted:
        with pytest.raises(asyncio.CancelledError):
            run()
        assert interrupts == ["native-turn"]
        assert drained == [True]
        assert not (directory / "native_result.json").exists()
        assert (
            json.loads((directory / "session/identity.json").read_text())["session_id"]
            == "codex-thread"
        )
        assert (directory / "session/transcript.jsonl").exists()
    else:
        run()
        assert (directory / "native_result.json").exists()
    assert connections == [
        ("start", None) if kind in {"fresh", "budget"} else ("resume", "codex-thread")
    ]


def test_interrupted_query_reconciles_budget_without_fabricating_measurement(
    tmp_path, monkeypatch
):
    from reweave.evolution.authoring.author import reconcile_interrupted_timing
    from reweave.evolution.contracts import EvolutionProtocol
    from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution

    values = protocol().model_dump(mode="json")
    graph = EvolutionWorkspace(
        tmp_path, baseline(), EvolutionProtocol.model_validate(values)
    )
    search = GraphEvolution(graph, None, None)
    directory = tmp_path / "attempts/interrupted"
    audit = directory / "audit"
    partial = Usage(
        physical_steps=0,
        model_calls=None,
        wall_time_s=2,
        wall_time_complete=False,
        cost_usd=None,
    )
    atomic_json(audit / "queries/000000.json", partial.model_dump(mode="json"))
    atomic_json(audit / "usage.json", partial.model_dump(mode="json"))
    (audit / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "native_query_started",
                "query": "000000.json",
                "started_at": 100,
            }
        )
        + "\n"
    )
    state = {
        "rounds": {},
        "attempts": {
            "interrupted": {
                "launches": 1,
                "kind": "author",
                "status": "failed",
                "last_failure": None,
            }
        },
    }
    assert search.accounting(state)["unresolved_wall_time"]
    monkeypatch.setattr("reweave.evolution.authoring.author.time.time", lambda: 130)
    reconcile_interrupted_timing(directory)
    measured = cumulative_usage(audit)
    assert measured.wall_time_s == 2 and not measured.wall_time_complete
    assert measured.wall_time_charge_s == 30
    assert measured.model_calls is None and measured.cost_usd is None
    assert not search.accounting(state)["unresolved_wall_time"]
    monkeypatch.setattr("reweave.evolution.authoring.author.time.time", lambda: 170)
    reconcile_interrupted_timing(directory)
    assert cumulative_usage(audit) == measured
    totals = search.accounting(state)
    assert totals["wall_time_s"] == 2 and totals["charged_wall_time_s"] == 30
    assert not totals["wall_time_complete"] and not totals["unresolved_wall_time"]


@pytest.mark.parametrize("started_at", [None, 200])
def test_interrupted_timing_rejects_missing_start_or_backwards_clock(
    tmp_path, monkeypatch, started_at
):
    from reweave.evolution.authoring.author import reconcile_interrupted_timing

    audit = tmp_path / "audit"
    atomic_json(
        audit / "queries/000000.json",
        Usage(
            physical_steps=0,
            model_calls=None,
            wall_time_s=1,
            wall_time_complete=False,
            cost_usd=None,
        ).model_dump(mode="json"),
    )
    event = (
        []
        if started_at is None
        else [
            {
                "event": "native_query_started",
                "query": "000000.json",
                "started_at": started_at,
            }
        ]
    )
    (audit / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in event)
    )
    monkeypatch.setattr("reweave.evolution.authoring.author.time.time", lambda: 100)
    with pytest.raises(ValueError, match="original start evidence"):
        reconcile_interrupted_timing(tmp_path)


def test_interrupted_host_admission_preserves_unknown_time_but_can_publish(
    tmp_path, monkeypatch
):
    workspace(tmp_path)
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.ClaudeSessionStore", Journal
    )
    native_completion(
        tmp_path, usage(), "native-id", Journal(tmp_path / "session").checkpoint()
    )
    atomic_json(
        tmp_path / "audit/checks/000000.json",
        {
            "wall_time_s": 2,
            "wall_time_complete": False,
            "started_at": 100,
        },
    )
    monkeypatch.setattr("reweave.evolution.authoring.author.time.time", lambda: 150)
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    result = asyncio.run(author._publish(tmp_path))
    note = tmp_path / "workspace/candidate/SKILL.md"
    assert result.candidates[0].experience == note.read_text()
    assert any(a.path == str(note.resolve()) for a in result.artifacts)
    assert result.candidates[0].bundle.same_harness(mutated(baseline(), "C"))
    assert all(
        section in json.loads(result.candidates[0].summary)
        for section in ("diagnosis", "summary", "validation")
    )
    assert not result.usage.wall_time_complete
    assert result.usage.wall_time_charge_s >= 51
    assert result.usage.wall_time_charge_s > result.usage.wall_time_s
    old = json.loads((tmp_path / "audit/checks/000000.json").read_text())
    assert old["wall_time_s"] == 2 and not old["wall_time_complete"]
    assert old["wall_time_charge_s"] == 50


@pytest.mark.parametrize("historical_source", [False, True])
@pytest.mark.parametrize("feedback_validation", [False, True])
@pytest.mark.parametrize("active_execution", [False, True])
def test_batch_preparation_preserves_execution_and_history(
    tmp_path, monkeypatch, historical_source, feedback_validation, active_execution
):
    import shutil

    from reweave.evolution.contracts import (
        EvidenceFile,
        EvolutionProtocol,
        VersionReference,
    )
    from reweave.evolution.evidence.records import inspect_episode
    from tests.unit_tests.harness.evolution.fixtures import file_metadata
    from tests.unit_tests.harness.evolution.test_trace_context import _episode

    official_task_instruction = "Open the drawer"
    task_rule_supplement = (
        "Project task-rule supplement: return both controlled arms near start."
    )
    resolved_task_instruction = (
        official_task_instruction + "\n\n" + task_rule_supplement
    )
    _, original = _episode(
        tmp_path,
        task_languages=[resolved_task_instruction] * 4,
        task_rule_supplements=[task_rule_supplement] * 4,
    )
    from reweave.evolution.bundle import AgentBundle

    value = mutated(baseline(), "W").model_dump(mode="json")
    value["context"]["skills"] = [
        {
            "skill_id": "retained",
            "scope": "global",
            "text": "# Preserve this experience\n",
        }
    ]
    parent = AgentBundle.model_validate(value)
    atomic_json(
        original / "harness_context/planner_context.json",
        {"system_prompt": "complete", "tools": []},
    )
    atomic_json(original / "harness_context/capability_catalog.json", {})
    (original / "harness_context/source/reweave/benchmarks").mkdir(parents=True)
    (original / "harness_context/source/reweave/benchmarks/executed.py").write_text(
        "# executed fixture\n"
    )
    (original / "harness_context/source/reweave/evolution/authoring").mkdir(
        parents=True
    )
    (original / "harness_context/source/reweave/evolution/validation").mkdir(
        parents=True
    )
    (
        original / "harness_context/source/reweave/evolution/authoring/prompts.py"
    ).write_text("# offline author implementation must remain private\n")
    (
        original / "harness_context/source/reweave/evolution/validation/admission.py"
    ).write_text("# offline admission implementation must remain private\n")
    (
        original / "harness_context/source/reweave/evolution/authoring/candidate.py"
    ).write_text("# offline candidate implementation must remain private\n")
    (original / "harness_context/source/reweave/session").mkdir(parents=True)
    (original / "harness_context/source/reweave/session/base.py").write_text(
        "# online state and image artifact implementation\n"
    )
    atomic_json(
        original / "harness_context/bundle_contract.json",
        {
            "bundle_id": parent.record_id,
            "operations": {},
            "episode_limits": {"wall_time_s": 1200.0},
        },
    )
    unused = original / "unread_audit.txt"
    unused.write_text("Unrelated archived attachment")
    evidence = inspect_episode(original)
    request = EvaluationRequest(
        conditions=protocol().execution,
        record_id=new_record_id(),
        bundle_id=parent.record_id,
        instance=protocol().dev[0],
        split="dev",
        manifest_id=protocol().manifest_id,
        execution=protocol().execution_id,
    )
    record = EvaluationRecord(
        record_id=new_record_id(),
        request=request,
        official_success=False,
        failure_origin="agent",
        infrastructure_retryable=None,
        usage=usage(),
        evidence=tuple(
            EvidenceFile(path=str(original / name), **digest)
            for name, digest in evidence.files.items()
        ),
        episode=evidence,
        initialization=request.instance.initialization,
    )
    records = [record]
    bundles = {parent.record_id: parent}
    if historical_source:
        reference_bundle = mutated(parent, "C")
        bundles[reference_bundle.record_id] = reference_bundle
        reference_bundle_root = tmp_path / "reference_bundle_episode"
        shutil.copytree(
            original,
            reference_bundle_root,
            ignore=shutil.ignore_patterns(
                "episode_evidence.json", ".artifact_records", ".artifact_records.lock"
            ),
        )
        (
            reference_bundle_root
            / "harness_context/source/reweave/benchmarks/executed.py"
        ).write_text("# other archived implementation\n")
        gateway_path = reference_bundle_root / "harness_context/bundle_contract.json"
        gateway = json.loads(gateway_path.read_text())
        atomic_json(
            gateway_path,
            {
                **gateway,
                "bundle_id": reference_bundle.record_id,
                "episode_limits": {"wall_time_s": None},
            },
        )
        reference_bundle_episode = inspect_episode(reference_bundle_root)
        records.append(
            record.model_copy(
                update={
                    "record_id": new_record_id(),
                    "request": request.model_copy(
                        update={
                            "record_id": new_record_id(),
                            "bundle_id": reference_bundle.record_id,
                            "execution": "historical-no-wall-time",
                            "conditions": request.conditions.model_copy(
                                update={"wall_time_s": None}
                            ),
                        }
                    ),
                    "episode": reference_bundle_episode,
                    "evidence": tuple(
                        EvidenceFile(path=str(reference_bundle_root / name), **digest)
                        for name, digest in reference_bundle_episode.files.items()
                    ),
                }
            )
        )
    prior_author = artifact(
        tmp_path, "prior_author.json", {"hypothesis": "Unverified old cause"}
    )
    prior_bundle = artifact(
        tmp_path, f"bundles/{parent.record_id}.json", parent.model_dump(mode="json")
    )
    historical_files = {
        item.path: {"artifact_id": item.artifact_id, "bytes": item.bytes}
        for item in (
            artifact(tmp_path, f"history/item-{index:03d}.json", {"index": index})
            for index in range(128)
        )
    }
    history_navigation = {
        "versions": {
            "b" * 64: {
                "label": "V000",
                "bundle_id": parent.record_id,
                "parent_version_id": None,
                "references": [],
            }
        },
        "bundles": [
            {
                "bundle_id": parent.record_id,
                "bundle": {**prior_bundle.model_dump(mode="json"), "pointer": ""},
            }
        ],
        "condition_index": [
            {
                "label": "C000",
                "condition_id": request.instance.record_id,
                "split": "dev",
            }
        ],
        "attempts": {
            "A000": {
                "receipt": {**prior_author.model_dump(mode="json"), "pointer": ""},
                "evidence_id": None,
            }
        },
    }
    history = evolution_history(
        tmp_path,
        originals={
            prior_author.path: {
                "artifact_id": prior_author.artifact_id,
                "bytes": prior_author.bytes,
            },
            prior_bundle.path: {
                "artifact_id": prior_bundle.artifact_id,
                "bytes": prior_bundle.bytes,
            },
            **historical_files,
        },
        navigation=history_navigation,
    )
    Path(history.summary.path).write_bytes(b"# Shared history\r\nVerbatim text  \r\n")
    history = history.model_copy(
        update={
            "summary": EvidenceFile(
                path=history.summary.path,
                **file_metadata(Path(history.summary.path)),
            )
        }
    )
    from tests.unit_tests.harness.evolution.fixtures import graph_proposal_fields

    fields = graph_proposal_fields(tmp_path)
    configured = protocol().model_dump(mode="json")
    configured.update(target_success_rate=0.75, author_wall_time_s=1800.0)
    if historical_source:
        configured["record_id"] = "historical-null-protocol"
        configured["execution_id"] = "historical-no-wall-time"
        configured["execution"]["wall_time_s"] = None
    if feedback_validation:
        configured["validation"] = {
            "conditions": [
                {
                    "record_id": f"validation-condition-{seed}",
                    "task": "drawer",
                    "initialization": {"seed": seed},
                }
                for seed in (2, 3)
            ],
            "target_success_rate": 0.5,
        }
        # Dynamic width is defined only when dev success ends the search.
        configured["width_mode"] = "fixed"
    fields["task_protocol"] = EvolutionProtocol.model_validate(configured)
    active_manifest = None
    if active_execution:
        from reweave.evolution.contracts import ExecutionRevision
        from reweave.evolution.runtime_manifest import (
            RuntimeAuthoringContract,
            RuntimeManifest,
            capture_baseline,
        )
        from reweave.planner.bundle_runtime import BundleRuntime
        from reweave.runtime.kernel import EpisodeLimits
        from tests.unit_tests.harness.evolution.test_runtime_manifest import fixture

        actual = fixture(tmp_path / "active")
        capture_baseline(tmp_path / "active/baseline", **actual)
        active_manifest = RuntimeManifest.model_validate_json(
            (tmp_path / "active/baseline/runtime.json").read_text()
        )
        limits = EpisodeLimits(
            **fields["task_protocol"].execution.model_dump(
                include={"program_calls", "gateway_calls", "wall_time_s"}
            )
        )
        runtime = BundleRuntime(
            parent, actual["toolkit"], "CPU authoring fixture", episode_limits=limits
        )
        active_contract = RuntimeAuthoringContract.from_runtime(runtime)
        configuration = active_manifest.configuration.copy()
        configuration["limits"] = {"episode": limits.model_dump(mode="json")}
        current_task_rule = {
            "task": "drawer",
            "task_rule_supplement": "Current published rule: keep the drawer stable.",
            "source_urls": ["https://example.test/current-drawer-rule"],
            "review_status": "supplemented",
        }
        configuration["task_rule_supplements"] = {"drawer": current_task_rule}
        previous = fields["task_protocol"]
        active_manifest = active_manifest.model_copy(
            update={
                "conditions": previous.execution,
                "configuration": configuration,
                "authoring_contract": active_contract,
            }
        )
        current = previous.model_copy(
            update={
                "record_id": "active-execution-protocol",
                "execution_id": active_manifest.execution_id,
                "execution_runtime": active_manifest.runtime_configuration(),
            }
        )
        active_manifest = active_manifest.model_copy(
            update={
                "execution_revision": ExecutionRevision(
                    previous=previous,
                    current=current,
                    reason="Synthetic CPU environment change, no physical claim.",
                )
            }
        )
        fields["task_protocol"] = current
        assert actual["toolkit"].calls == 0
        assert actual["toolkit"].terminal is False
    proposal = Proposal(
        **fields,
        record_id="batch_000000",
        parent=VersionReference(version_id="b" * 64, bundle_id=parent.record_id),
        references=(VersionReference(version_id="b" * 64, bundle_id=parent.record_id),),
        round_index=0,
        author_index=0,
        history=history,
        scopes=("W", "C", "L"),
        evidence=tuple(records),
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=bundles.__getitem__,
        policy=AuthorPolicy(),
        runtime_manifest=active_manifest,
    )
    root = tmp_path / "batch/workspace"
    from reweave.evolution.authoring import references as author_references

    history_publications = []

    def record_catalog_write(path, value):
        if Path(path).name == "author_references.json" and "entries" in value:
            history_publications.append(
                sum(
                    isinstance(entry, dict) and entry.get("path") in historical_files
                    for entry in value["entries"].values()
                )
            )
        atomic_json(path, value)

    monkeypatch.setattr(author_references, "atomic_json", record_catalog_write)
    gateway_originals = {
        Path(f.path): Path(f.path).read_bytes()
        for r in records
        for f in r.evidence
        if f.path.endswith("harness_context/bundle_contract.json")
    }
    unused.unlink()
    author._prepare_inputs(proposal, root.parent, resume=False)
    assert all(p.read_bytes() == data for p, data in gateway_originals.items())
    assert (
        json.loads((root / "context/gateway.json").read_text())["episode_limits"][
            "wall_time_s"
        ]
        == configured["execution"]["wall_time_s"]
    )
    if historical_source:
        gateway = json.loads(gateway_path.read_text())
        for name, change in (
            ("operations", {"operations": {"undeclared_tool": {}}}),
            (
                "request_limits",
                {
                    "episode_limits": {
                        **gateway["episode_limits"],
                        "request_limits": {"max_images": 1},
                    }
                },
            ),
            (
                "note_closeout_s",
                {"episode_limits": {**gateway["episode_limits"], "note_closeout_s": 1}},
            ),
        ):
            atomic_json(gateway_path, {**gateway, **change})
            if active_execution:
                author._prepare(proposal, tmp_path / f"rejected-{name}/workspace")
            else:
                with pytest.raises(
                    ValueError, match="different frozen gateway conditions"
                ):
                    author._prepare(proposal, tmp_path / f"rejected-{name}/workspace")
        gateway_path.write_bytes(gateway_originals[gateway_path])
        catalog_path = reference_bundle_root / "harness_context/capability_catalog.json"
        catalog_original = catalog_path.read_bytes()
        atomic_json(catalog_path, {"undeclared_capability": {}})
        if active_execution:
            author._prepare(proposal, tmp_path / "rejected-catalog/workspace")
        else:
            with pytest.raises(ValueError, match="different capability catalogs"):
                author._prepare(proposal, tmp_path / "rejected-catalog/workspace")
        catalog_path.write_bytes(catalog_original)
    # A large history must not fsync the complete catalog once per original.
    assert set(history_publications) == {0, len(historical_files)}
    assert (root / "candidate/plan.json").is_file()
    assert (root / "candidate/SKILL.md").read_text() == ""
    assert (root / "candidate/task-family/workflows").is_dir()
    assert not (root / "candidate/branches").exists()
    assert not (root / "candidate/mutation.json").exists()
    from reweave.evolution.authoring.references import AuthorReferences

    catalog = AuthorReferences.load(root)
    entry = json.loads((root / "context/input.json").read_text())["index"][0]
    task_instruction = json.loads(
        (root / "context/task_instructions.json").read_text()
    )[0]
    assert task_instruction["text"] == resolved_task_instruction
    assert task_instruction["official_text"] == official_task_instruction
    assert task_instruction["task_rule_supplement"] == task_rule_supplement
    current_rules = root / "context/current_task_rules.json"
    if active_execution:
        assert json.loads(current_rules.read_text())["values"] == [current_task_rule]
        instructions, request_text = author._prompts(proposal, root)
        assert (
            "ONE complete candidate under context/input.json's objective"
            in instructions
        )
        assert not (root / "context/AUTHOR_WORKFLOW.md").exists()
        guide = " ".join((root / "context/DEVELOPMENT.md").read_text().split())
        assert "remaining task and budgets" in guide
        assert "Use your own investigation" in request_text
        assert current_task_rule["task_rule_supplement"] in instructions
        assert current_task_rule["source_urls"][0] in instructions
        assert current_task_rule["task_rule_supplement"] not in request_text
        assert "Read context/current_task_rules.json" in request_text
        assert (
            task_instruction["task_rule_supplement"]
            != current_task_rule["task_rule_supplement"]
        )
    else:
        assert not current_rules.exists()
    assert (root / entry["trace_context"]).is_file()
    assert Path(entry["original_directory"]).is_dir()
    assert not (root / "context/experiment_access.json").exists()
    assert (root / "context/EVOLUTION_HISTORY.md").is_file()
    assert (root / "context/history_originals.json").is_file()
    assert not (root / "context/parent.json").exists()
    assert not (root / "context/proposal.json").exists()
    assert host_input(root, "references.json").is_file()
    assert not host_input(root, "parent.json").exists()
    assert (root / "candidate/task-family/context/program.json").is_file()
    assert (root / "candidate/task-family/loop/program.json").is_file()
    assert host_input(root, "proposal.json").is_file()
    assert not (root / ".host_inputs").exists()
    assert not (root / "context/diagnosis.md").exists()
    assert not (root / "context/route.json").exists()
    assert not (root / "context/round_decision.json").exists()
    for name, original in (
        ("graph.json", proposal.graph),
        ("execution_graph.json", proposal.execution_graph),
        ("progress_map.json", proposal.progress_map),
    ):
        assert catalog.read_json(root / "context" / name) == json.loads(
            Path(original.path).read_text()
        )
    prepared = json.loads((root / "context/input.json").read_text())
    assert prepared["search_review"] == "context/search_review.json"
    assert (root / prepared["search_review"]).is_file()
    evaluation = prepared["evaluation"]
    assert evaluation["task"] == "drawer"
    assert evaluation["dev"] == {
        "condition_ids": [catalog.identity("condition", protocol().dev[0].record_id)],
        "target_success_rate": 0.75,
    }
    assert evaluation["execution"] == configured["execution"]
    assert evaluation["author_wall_time_s"] == 1800.0
    if feedback_validation:
        assert evaluation["validation"] == {
            "condition_ids": [
                catalog.identity("condition", item.record_id)
                for item in proposal.task_protocol.validation.conditions
            ],
            "target_success_rate": 0.5,
        }
    else:
        assert evaluation["validation"] is None
    assert "initialization" not in json.dumps(evaluation)
    assert "search" not in evaluation
    assert "max_authors_per_round" not in evaluation
    assert prepared["parent"] == catalog.encode(proposal.parent.model_dump(mode="json"))
    assert prepared["comparisons"] == {
        "tool": "read_graph",
        "layer": "map",
        "view": "comparison",
        "identity": catalog.identity("version", proposal.parent.version_id),
    }
    assert not {"parent_bundle_id", "parent_version_id", "donors"} & set(prepared)
    for ref in proposal.references:
        source = (
            root / "context/references" / catalog.identity("version", ref.version_id)
        )
        assert not (source / "bundle.json").exists()
        assert (source / "task-family/workflows").is_dir()
        assert (source / "task-family/context/context.json").is_file()
        assert (source / "task-family/context/program.json").is_file()
        assert (source / "task-family/loop/loop.json").is_file()
        assert (source / "task-family/loop/program.json").is_file()
    source = (
        root
        / "context/references"
        / catalog.identity("version", proposal.parent.version_id)
    )
    expected = {
        path.relative_to(source): path.read_bytes()
        for scope in ("workflows", "context", "loop")
        for path in (source / scope).rglob("*")
        if path.is_file()
    }
    actual = {
        path.relative_to(root / "candidate"): path.read_bytes()
        for scope in ("workflows", "context", "loop")
        for path in (root / "candidate" / scope).rglob("*")
        if path.is_file()
    }
    assert actual == expected
    from reweave.evolution.evidence import trace as trace_context

    monkeypatch.setattr(
        trace_context,
        "_iter_jsonl",
        lambda *args: pytest.fail(
            "settled episode navigation must not reparse originals"
        ),
    )
    second = tmp_path / "second/workspace"
    author._prepare_inputs(
        proposal.model_copy(update={"record_id": "batch_000001", "author_index": 1}),
        second.parent,
        resume=False,
    )
    second_input = json.loads((second / "context/input.json").read_text())
    assert second_input["evaluation"] == evaluation
    assert second_input["references"] == prepared["references"]
    assert (second / "candidate/task-family/context/program.json").is_file()
    assert second_input["parent"] == prepared["parent"]
    navigation_path = entry["trace_context"]
    assert (root / navigation_path).read_bytes() == (
        second / navigation_path
    ).read_bytes()
    assert (root / "context/EVOLUTION_HISTORY.md").read_bytes() == (
        second / "context/EVOLUTION_HISTORY.md"
    ).read_bytes()
    assert (root / "context/EVOLUTION_HISTORY.md").read_bytes() == Path(
        history.summary.path
    ).read_bytes()
    public_history = json.loads((root / "context/history_originals.json").read_text())
    resolved_history = {
        original.path: {"artifact_id": original.artifact_id, "bytes": original.bytes}
        for handle in public_history["files"]
        for original in (catalog.original(handle),)
    }
    assert historical_files.items() <= resolved_history.items()
    public_navigation = public_history["navigation"]
    assert public_navigation == catalog.encode(history_navigation)
    assert (
        "V000"
        == public_navigation["versions"][catalog.identity("version", "b" * 64)]["label"]
    )
    assert public_navigation["condition_index"][0]["condition_id"] == catalog.identity(
        "condition", request.instance.record_id
    )
    assert public_navigation["attempts"]["A000"]["evidence_id"] is None
    assert parent.record_id not in json.dumps(public_history)
    assert str(tmp_path) not in json.dumps(public_history)
    from reweave.evolution.evidence.trajectory import read_original

    bundle_reference = public_navigation["bundles"][0]["bundle"]
    assert read_original(catalog.original(bundle_reference)) == parent.model_dump(
        mode="json"
    )
    assert public_history["files"][bundle_reference]["name"] == "bundle.json"
    raw_navigation = (
        Path("evidence/context/trace") / records[0].episode.evidence_id / "index.json"
    )
    assert host_input(root, str(raw_navigation)).samefile(
        host_input(second, str(raw_navigation))
    )
    assert catalog.read_json(
        root / "context/execution_graph.json"
    ) == AuthorReferences.load(second).read_json(
        second / "context/execution_graph.json"
    )
    assert not (root / "candidate/task-family/context/program.json").samefile(
        second / "candidate/task-family/context/program.json"
    )
    (root / "candidate/task-family/context/instructions.md").write_text(
        "One author's independent implementation"
    )
    assert (
        second / "candidate/task-family/context/instructions.md"
    ).read_text() == parent.context.instructions
    assert (
        root
        / "context/references"
        / catalog.identity("version", proposal.parent.version_id)
        / "task-family/context/instructions.md"
    ).read_text() == parent.context.instructions
    author._prepare_inputs(proposal, root.parent, resume=True)
    assert (
        root / "candidate/task-family/context/instructions.md"
    ).read_text() == "One author's independent implementation"
    assert prepared["capability_catalog"] == "context/capability_catalog.json"
    assert json.loads((root / prepared["capability_catalog"]).read_text()) == (
        active_contract.model_dump(mode="json")["capability_catalog"]
        if active_execution
        else {}
    )
    if active_execution:
        assert (
            json.loads((root / "context/gateway.json").read_text())["operations"]
            == (active_contract.model_dump(mode="json")["operations"])
        )
    from jsonschema import validate

    schema = json.loads((root / "context/runtime.schema.json").read_text())[
        "physical_budget"
    ]
    validate(None, schema)  # Unsupported environments expose unknown, not zero.
    validate(
        {
            "unit": "environment_control_step",
            "limit": 310,
            "used": 47,
            "remaining": 263,
        },
        schema,
    )
    assert not (root / ".archive/episodes").exists()
    assert not (root / ".archive/history").exists()
    assert "path" not in prepared["index"][0]
    sources = json.loads(
        (root / prepared["implementation"]["executed_sources"]).read_text()
    )
    assert len(sources) == len(records)
    private_sources = json.loads(host_input(root, "executed_sources.json").read_text())
    for entry in prepared["index"]:
        paths = sources[entry["executed_source"]]
        source = private_sources[
            catalog.resolve(entry["executed_source"], kind="source")
        ]
        assert (Path(source["path"]) / "reweave/benchmarks/executed.py").is_file()
        assert paths == sorted(source["files"])
        assert all(
            path.startswith(
                ("reweave/benchmarks/", "reweave/planner/", "reweave/runtime/")
            )
            or path
            in {
                "reweave/evolution/bundle.py",
                "reweave/session/__init__.py",
                "reweave/session/base.py",
            }
            for path in source["files"]
        )
        assert "reweave/evolution/authoring/prompts.py" not in source["files"]
        assert "reweave/evolution/validation/admission.py" not in source["files"]
        assert "reweave/evolution/authoring/candidate.py" not in source["files"]
        assert "reweave/session/base.py" in source["files"]
    assert not (root / "context/author_source").exists()
    assert not (root / "context/executed_source").exists()
    assert not (root / "context/mutation.schema.json").exists()
    client = (root / "context/host_client.py").read_text()
    assert "call_development_tool" in client
    assert "AuthorDevelopmentTools" not in client
    assert "simulate_candidate" not in client
    assert not list((root / "context/trace").rglob("overview-*.jpg"))
    mutation_trace = json.loads(
        (root / prepared["index"][0]["trace_context"]).read_text()
    )
    assert mutation_trace["visuals_included"] is False
    local = json.loads((root / "context/history_originals.json").read_text())["files"]
    prior_handle = catalog.reference(
        prior_author.path, prior_author.artifact_id, prior_author.bytes
    )
    assert prior_handle in local
    assert (
        Path(catalog.original(prior_handle).path).read_bytes()
        == Path(prior_author.path).read_bytes()
    )
    assert len(list((root / "context/references").iterdir())) == 1
    if not (active_execution or historical_source or feedback_validation):
        # Simulate the preparation crash boundary: audit inputs were committed,
        # but the prepared tree had not become the visible workspace yet.
        saved_sources = host_input(root, "executed_sources.json").read_bytes()
        saved_inventory = (root / "context/source_inventory.json").read_bytes()
        root.rename(root.parent / "interrupted_preparation")
        author._prepare_inputs(proposal, root.parent, resume=True)
        assert host_input(root, "executed_sources.json").read_bytes() == saved_sources
        assert (root / "context/source_inventory.json").read_bytes() == saved_inventory
        recovered_catalog = AuthorReferences.load(root)
        recovered_input = json.loads((root / "context/input.json").read_text())
        for episode in recovered_input["index"]:
            assert recovered_catalog.resolve(
                episode["executed_source"], kind="source"
            ) in json.loads(saved_sources)


@pytest.mark.parametrize("field", ["diagnosis", "summary", "validation"])
@pytest.mark.parametrize("invalid", [None, "", "   "])
def test_each_mutation_requires_current_diagnosis_change_and_validation(
    tmp_path, field, invalid
):
    from reweave.evolution.authoring.candidate import AuthorPlan

    root = workspace(tmp_path)
    path = root / "candidate/plan.json"
    value = json.loads(path.read_text())
    if invalid is None:
        value.pop(field)
    else:
        value[field] = invalid
    with pytest.raises(ValueError):
        AuthorPlan.model_validate(value)
    atomic_json(path, value)
    with pytest.raises(ValueError):
        asyncio.run(admit_candidate(root))


@pytest.mark.parametrize(
    "phase,complete,allowed",
    [
        ("sdk_initialization", True, True),
        ("native_query", True, False),
        ("sdk_initialization", False, False),
        (None, True, False),
    ],
)
def test_fresh_session_requires_terminal_initialization_failure(
    tmp_path, phase, complete, allowed
):
    from reweave.evolution.authoring.author import failed_initializations

    audit = tmp_path / "audit"
    original = Usage(
        physical_steps=0,
        model_calls=None,
        wall_time_s=3,
        wall_time_complete=complete,
        cost_usd=None,
    )
    atomic_json(audit / "queries/000000.json", original.model_dump(mode="json"))
    (audit / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "native_query_failed",
                "query": "000000.json",
                "phase": phase,
            }
        )
        + "\n"
    )
    if allowed:
        assert failed_initializations(audit) == ["000000.json"]
    else:
        with pytest.raises(ValueError, match="recoverable identity"):
            failed_initializations(audit)
    assert cumulative_usage(audit).model_calls is None
    assert cumulative_usage(audit).cost_usd is None
    assert (audit / "queries/000000.json").read_text() == json.dumps(
        original.model_dump(mode="json"), indent=2, sort_keys=True
    ) + "\n"


def test_initialize_crash_is_recorded_and_explicitly_retried(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    namespace = {}
    exec(
        compile(
            'def initialize():\n    raise RuntimeError("CLI aborted before prompt")\n',
            "/sdk/claude_agent_sdk/_internal/query.py",
            "exec",
        ),
        namespace,
    )

    async def query(**kwargs):
        namespace["initialize"]()
        yield

    monkeypatch.setattr("claude_agent_sdk.query", query)
    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeAgentOptions", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(
        "reweave.evolution.authoring.author.claude_workspace_options",
        lambda *a, **kw: {},
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    launches = []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="CLI aborted"):
            asyncio.run(
                author._execute(
                    proposal,
                    tmp_path,
                    resume=True,
                    before_launch=lambda: launches.append(1),
                )
            )
    assert len(launches) == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "audit/events.jsonl").read_text().splitlines()
    ]
    failures = [
        event for event in events if event.get("event") == "native_query_failed"
    ]
    assert len(failures) == 2
    assert all(event["phase"] == "sdk_initialization" for event in failures)
    assert any(
        event.get("event") == "native_initialization_restarted" for event in events
    )
    assert len(list((tmp_path / "audit/queries").glob("*.json"))) == 2
    assert cumulative_usage(tmp_path / "audit").cost_usd is None


@pytest.mark.parametrize(
    "manifest",
    [
        {"files": {}},
        {"files": {}, "navigation": []},
        {"files": {}, "navigation": {}, "legacy": {}},
    ],
)
def test_history_manifest_rejects_removed_or_malformed_navigation(tmp_path, manifest):
    from reweave.evolution.authoring.references import history_originals

    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    Path(proposal.history.originals.path).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="files and navigation maps"):
        history_originals(proposal)


@pytest.mark.parametrize("boundary", ["published", "resume", "unpublished_workspace"])
@pytest.mark.parametrize("change", ["author_index", "budget", "boolean_setting"])
def test_same_proposal_id_cannot_reuse_different_actual_input(
    tmp_path, monkeypatch, boundary, change
):
    from reweave.evolution.authoring.candidate import read_proposal
    from reweave.evolution.contracts import AuthoredMutation, Candidate

    root = workspace(tmp_path)
    original = read_proposal(root)
    conditions = original.task_protocol.execution.model_copy(
        update={"model_sampling": {"use_cache": True}}
    )
    original = original.model_copy(
        update={
            "task_protocol": original.task_protocol.model_copy(
                update={"execution": conditions}
            )
        }
    )
    atomic_json(host_input(root, "proposal.json"), original.model_dump(mode="json"))
    if change == "author_index":
        changed = original.model_copy(
            update={"author_index": original.author_index + 1}
        )
    else:
        changed_conditions = conditions.model_copy(
            update={"model_requests": conditions.model_requests + 1}
            if change == "budget"
            else {"model_sampling": {"use_cache": 1}}
        )
        changed = original.model_copy(
            update={
                "task_protocol": original.task_protocol.model_copy(
                    update={"execution": changed_conditions}
                )
            }
        )
    assert changed.record_id == original.record_id
    author = NativeBundleAuthor(
        policy=AuthorPolicy(),
        bundle_loader=lambda _: baseline(),
        evidence_workspace=tmp_path / "evidence",
    )
    if boundary == "published":
        receipt = AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=original.record_id,
            candidates=(
                Candidate(
                    branch_id="repair",
                    bundle=mutated(baseline(), "C"),
                    summary="Synthetic candidate",
                    experience="Unverified fixture",
                ),
            ),
            summary="Synthetic completion",
            session_id="native-session",
            usage=usage(),
            artifacts=(original.history.summary,),
        )
        atomic_json(tmp_path / "result.json", receipt.model_dump(mode="json"))
        assert author.verify_result(original, tmp_path) == receipt
    elif boundary == "unpublished_workspace":
        root.rename(tmp_path / "prepared_but_unpublished")
    else:
        author._prepare_inputs(original, tmp_path, resume=True)
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        author,
        "_prepare",
        lambda *args: pytest.fail(
            "different input must not launch preparation or any native query"
        ),
    )
    with pytest.raises(ValueError, match="input differs"):
        if boundary == "published":
            author.verify_result(changed, tmp_path)
        else:
            author._prepare_inputs(changed, tmp_path, resume=True)
    assert {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before
