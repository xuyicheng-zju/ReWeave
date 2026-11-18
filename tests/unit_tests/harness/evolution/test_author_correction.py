# Modified for the ReWeave namespace.
"""Invalid native output must be repaired in the same session, never evaluated."""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from copy import deepcopy
from types import SimpleNamespace

import pytest

from reweave.evolution.authoring.author import NativeBundleAuthor
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.references import AuthorReferences
from reweave.evolution.contracts import Proposal
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import baseline, mutated
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace


@pytest.mark.parametrize("backend", ["claude_code", "codex"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_failed_admission_repairs_same_session_and_keeps_all_attempts(
    tmp_path, monkeypatch, backend, interrupt
):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    source = root / "candidate/task-family/context/src/context.py"
    original = source.read_text()
    identity = str(uuid.uuid4())
    prompts, sessions = [], []
    failures = 4  # No arbitrary three-repair cap.
    recovery_clock = [1000.0]

    def write_output(prompt, resumed):
        sessions.append(resumed)
        prompts.append(prompt)
        audit = tmp_path / "audit"
        query_wall = sum(
            json.loads(path.read_text())["wall_time_s"]
            for path in (audit / "queries").glob("*.json")
        )
        prequery_wall = sum(
            json.loads(path.read_text())["wall_time_s"]
            for path in (audit / "preparation_usage").glob("*.json")
        )
        check_wall = sum(
            json.loads(path.read_text())["wall_time_s"]
            for path in (audit / "checks").glob("*.json")
        )
        live = json.loads((audit / "usage.json").read_text())
        assert live["wall_time_s"] == pytest.approx(
            prequery_wall + query_wall + check_wall
        )
        assert not live["wall_time_complete"]
        assert not (tmp_path / "result.json").exists()
        if len(prompts) > 1:
            assert resumed == identity
            assert "Host final admission rejected" in prompt
            reference = json.loads(prompt.split("\n\n", 1)[1])
            from pathlib import Path

            from tests.unit_tests.harness.evolution.fixtures import file_metadata

            evidence_original = AuthorReferences.load(root).original(
                reference["complete_report"]
            )
            feedback_path = Path(evidence_original.path)
            assert file_metadata(feedback_path) == {
                "artifact_id": evidence_original.artifact_id,
                "bytes": evidence_original.bytes,
            }
            feedback = feedback_path.read_text()
            assert reference["error_type"] == json.loads(feedback)["error_type"]
            assert "SyntaxError" in feedback and "context.py" in feedback
            assert "Traceback" in feedback
            if interrupt:
                previous = json.loads((audit / "checks/000000.json").read_text())
                assert previous["wall_time_charge_s"] == 100.0
                # The old check's interval must already be closed; time spent
                # in this repair cannot be charged a second time to that check.
                recovery_clock[0] = 2000.0
        else:
            assert resumed is None
        source.write_text(
            f"def broken(:  # attempt {len(prompts)}\n"
            if len(prompts) <= failures
            else original
        )

    def assert_correction_access(kwargs):
        assert tmp_path / "audit/corrections" in kwargs["readable_roots"]
        assert tmp_path / "audit/corrections" not in kwargs.get("writable_roots", ())

    if backend == "claude_code":
        from claude_agent_sdk import ResultMessage, SystemMessage

        async def query(*, prompt, options):
            write_output(prompt, options.resume)
            await options.session_store.append(
                {"session_id": identity, "project_key": "correction"},
                [
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "assistant",
                        "message": {"content": f"Attempt {len(prompts)}"},
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

        monkeypatch.setattr("claude_agent_sdk.query", query)
        monkeypatch.setattr(
            "claude_agent_sdk.ClaudeAgentOptions", lambda **kw: SimpleNamespace(**kw)
        )

        def workspace_options(*args, **kwargs):
            assert_correction_access(kwargs)
            return {}

        monkeypatch.setattr(
            "reweave.evolution.authoring.author.claude_workspace_options",
            workspace_options,
        )
    else:
        native = tmp_path / "native-codex.jsonl"
        native.write_text(
            json.dumps({"type": "session_meta", "payload": {"id": identity}}) + "\n"
        )

        class Turn:
            async def stream(self):
                with native.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "type": "response_item",
                                "payload": {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": f"Attempt {len(prompts)}",
                                        }
                                    ],
                                },
                            }
                        )
                        + "\n"
                    )
                yield {
                    "method": "turn/completed",
                    "payload": {"turn": {"status": "completed"}},
                }

        class Thread:
            id = identity

            def __init__(self, resumed):
                self.resumed = resumed

            async def read(self, **kwargs):
                return SimpleNamespace(
                    thread=SimpleNamespace(
                        id=self.id, path=str(native), model_provider="openai"
                    )
                )

            async def turn(self, prompt, **kwargs):
                write_output(prompt, self.resumed)
                return Turn()

        class Client:
            async def thread_start(self, **kwargs):
                return Thread(None)

            async def thread_resume(self, session_id, **kwargs):
                assert session_id == identity
                return Thread(session_id)

            async def close(self):
                pass

        def create_client(**kwargs):
            assert_correction_access(kwargs)
            return Client()

        monkeypatch.setattr(
            "reweave.planner.codex_connection.create_codex_client", create_client
        )

    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(author=backend),
    )
    stopped = False

    def before_launch():
        nonlocal stopped
        if interrupt and len(prompts) == 1 and not stopped:
            stopped = True
            raise asyncio.CancelledError()
        if interrupt and len(prompts) == 1 and stopped:
            current = json.loads((tmp_path / "audit/usage.json").read_text())
            query_wall = json.loads(
                (tmp_path / "audit/queries/000000.json").read_text()
            )["wall_time_s"]
            prequery_wall = sum(
                json.loads(path.read_text())["wall_time_s"]
                for path in (tmp_path / "audit/preparation_usage").glob("*.json")
            )
            # The launch-budget callback must see the closed interruption,
            # including when native_result still existed at recovery entry.
            assert current["wall_time_charge_s"] == pytest.approx(
                prequery_wall + query_wall + 100.0
            )

    if interrupt:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                author(proposal, tmp_path, resume=True, before_launch=before_launch)
            )
        assert not (tmp_path / "result.json").exists()
        assert len(list((tmp_path / "audit/corrections").glob("*/feedback.json"))) == 1
        assert len(list((tmp_path / "audit/queries").glob("*.json"))) == 1
        check_path = tmp_path / "audit/checks/000000.json"
        interrupted_check = json.loads(check_path.read_text())
        interrupted_check.update(wall_time_complete=False, started_at=900.0)
        atomic_json(check_path, interrupted_check)
        monkeypatch.setattr(
            "reweave.evolution.authoring.author.time.time", lambda: recovery_clock[0]
        )
    result = asyncio.run(
        author.recover(proposal, tmp_path, before_launch=before_launch)
    )
    assert (
        result.candidates[0].bundle.same_harness(mutated(baseline(), "C"))
        and result.session_id == identity
    )
    assert sessions == [None] + [identity] * failures
    assert len(list((tmp_path / "audit/queries").glob("*.json"))) == failures + 1
    rejected = sorted(
        (tmp_path / "audit/corrections").glob(
            "*/output/task-family/context/src/context.py"
        )
    )
    assert [p.read_text() for p in rejected] == [
        f"def broken(:  # attempt {i}\n" for i in range(1, failures + 1)
    ]
    assert result.usage.wall_time_complete is (not interrupt)
    audit = tmp_path / "audit"
    query_wall = sum(
        json.loads(path.read_text())["wall_time_s"]
        for path in (audit / "queries").glob("*.json")
    )
    prequery_wall = sum(
        json.loads(path.read_text())["wall_time_s"]
        for path in (audit / "preparation_usage").glob("*.json")
    )
    check_wall = sum(
        json.loads(path.read_text())["wall_time_s"]
        for path in (audit / "checks").glob("*.json")
    )
    assert result.usage.wall_time_s == pytest.approx(
        prequery_wall + query_wall + check_wall
    )
    completion = json.loads((tmp_path / "native_result.json").read_text())
    assert completion["usage"]["wall_time_s"] <= prequery_wall + query_wall
    assert not completion["usage"]["wall_time_complete"]
    if interrupt:
        first_check = json.loads((audit / "checks/000000.json").read_text())
        assert first_check["wall_time_charge_s"] == 100.0
        assert result.usage.wall_time_charge_s == pytest.approx(
            prequery_wall + query_wall + check_wall - first_check["wall_time_s"] + 100.0
        )
    assert result.usage.cost_usd == (failures + 1 if backend == "claude_code" else None)
    if backend == "claude_code":
        assert result.usage.model_calls == failures + 1
    again = asyncio.run(
        author.recover(
            proposal,
            tmp_path,
            before_launch=lambda: pytest.fail("accepted output must not reauthor"),
        )
    )
    assert again == result
    for artifact in again.artifacts:
        artifact.verify()


def test_contract_failure_keeps_complete_case_evidence_in_same_session_feedback(
    tmp_path, monkeypatch
):
    from claude_agent_sdk import ResultMessage, SystemMessage

    from reweave.evolution.evidence.records import inventory_episode

    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    gateway_path = root / "context/gateway.json"
    gateway = json.loads(gateway_path.read_text())
    input_schema = {
        "type": "object",
        "properties": {"case": {"type": "integer"}},
        "required": ["case"],
        "additionalProperties": False,
    }
    gateway["operations"]["inspect_scene"] = {
        "effect": "read",
        "input_schema": input_schema,
    }
    gateway["tools"].append(
        {
            "name": "inspect_scene",
            "description": "Read the selected synthetic scene case.",
            "input_schema": input_schema,
        }
    )
    atomic_json(gateway_path, gateway)
    atomic_json(
        tmp_path / "input_inventory.json",
        {"context": inventory_episode(root / "context"), "archive": {}},
    )
    marker = "complete-case-evidence-" * 15000
    messages = [
        {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": "Inspect the scene."}],
        }
    ]
    valid_cases = [
        {
            "input": {"state": {"case": index}, "messages": messages, "tools": []},
            "responses": [
                {
                    "method": "observe",
                    "name": "inspect_scene",
                    "arguments": {"case": index},
                    "result": {"case": index, "marker": marker},
                }
            ],
            "expected": {
                "instructions": marker,
                "messages": messages,
                "tools": [],
                "state": {"case": index},
            },
        }
        for index in range(3)
    ]
    broken_cases = deepcopy(valid_cases)
    broken_cases[1]["responses"][0]["name"] = "wrong_operation"
    broken_cases[1]["responses"].append(deepcopy(valid_cases[2]["responses"][0]))
    config = root / "candidate/task-family/context/context.json"
    atomic_json(config, {"contract_cases": broken_cases})
    (root / "candidate/task-family/context/src/context.py").write_text(
        "def build(runtime, context):\n"
        "    try:\n"
        '        observed = runtime.observe("inspect_scene", {"case": context["state"].get("case", 0)})\n'
        "    except Exception:\n"
        '        return {"caught": True}\n'
        '    return {"instructions": observed["marker"], "messages": context["messages"],\n'
        '            "tools": context["tools"], "state": {"case": observed["case"]}}\n'
    )
    atomic_json(
        root / ".scratch/tests/interaction.json",
        [
            {
                "name": "context-retrieves-complete-scene-feedback",
                "system_prompt": "Control using public tools.",
                "task_message": "Inspect the scene.",
                "model_responses": [
                    {
                        "instructions_contains": [marker],
                        "tool_calls": [
                            {
                                "name": "read_terminal_state",
                                "arguments": {},
                            }
                        ],
                    }
                ],
                "operations": [
                    {
                        "name": "inspect_scene",
                        "arguments": {"case": 0},
                        "result": {"case": 0, "marker": marker},
                    },
                    {
                        "name": "read_terminal_state",
                        "arguments": {},
                        "result": {"terminated": True},
                        "environment_state": {
                            "official_success": True,
                            "terminated": False,
                            "truncated": False,
                        },
                    },
                ],
                "assertions": [{"path": ["coverage", "C"], "equals": True}],
            }
        ],
    )
    plan_path = root / "candidate/plan.json"
    plan = json.loads(plan_path.read_text())
    plan["scopes"] = ["C"]
    atomic_json(plan_path, plan)
    identity = str(uuid.uuid4())
    sessions, feedbacks = [], []

    async def query(*, prompt, options):
        sessions.append(options.resume)
        if len(sessions) == 2:
            reference = json.loads(prompt.split("\n\n", 1)[1])
            from pathlib import Path

            from tests.unit_tests.harness.evolution.fixtures import file_metadata

            evidence_original = AuthorReferences.load(root).original(
                reference["complete_report"]
            )
            path = Path(evidence_original.path)
            assert file_metadata(path) == {
                "artifact_id": evidence_original.artifact_id,
                "bytes": evidence_original.bytes,
            }
            feedback = json.loads(path.read_text())
            assert marker not in prompt
            assert len(prompt) < 2000
            saved = json.loads(
                (tmp_path / "audit/corrections/000000/feedback.json").read_text()
            )
            assert feedback == saved
            records = feedback["contract_case_results"]
            assert [(r["scope"], r["case_index"]) for r in records] == [
                ("C", 0),
                ("C", 1),
            ]
            assert records[0]["validation_status"] == "passed"
            assert records[0]["result"] == valid_cases[0]["expected"]
            assert records[0]["calls"][0]["result"]["marker"] == marker
            failed = records[1]
            assert failed["validation_status"] == "failed"
            assert failed["result"] == {"caught": True}
            assert failed["calls"][0]["name"] == "inspect_scene"
            assert failed["calls"][0]["error"]["type"] == "ValueError"
            assert "result" not in failed["calls"][0]
            assert failed["unconsumed_responses"] == broken_cases[1]["responses"][1:]
            assert "Traceback" in feedback["traceback"]
            feedbacks.append(feedback)
            atomic_json(config, {"contract_cases": valid_cases})
        else:
            assert len(sessions) == 1
        await options.session_store.append(
            {"session_id": identity, "project_key": "contract-correction"},
            [
                {
                    "uuid": str(uuid.uuid4()),
                    "type": "assistant",
                    "message": {"content": f"Attempt {len(sessions)}"},
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
    result = asyncio.run(author.recover(proposal, tmp_path, before_launch=lambda: None))
    assert sessions == [None, identity]
    assert len(feedbacks) == 1
    assert result.session_id == identity
    assert result.usage.model_calls == 2 and result.usage.cost_usd == 2
    assert len(list((tmp_path / "audit/checks").glob("*.json"))) == 2
    assert len(list((tmp_path / "audit/queries").glob("*.json"))) == 2
    archived = (
        tmp_path / "audit/corrections/000000/output/task-family/context/context.json"
    )
    assert json.loads(archived.read_text())["contract_cases"] == broken_cases
    for artifact in result.artifacts:
        artifact.verify()


@pytest.mark.parametrize("operation", ["publish", "recover"])
def test_unconfirmed_native_descendants_block_publication_and_recovery(
    tmp_path, operation
):
    root = workspace(tmp_path)
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=5)
    atomic_json(
        tmp_path / "launcher/process.json",
        {"pid": process.pid, "descendants_drained": False},
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    with pytest.raises(RuntimeError, match="no confirmed drain"):
        asyncio.run(
            author._publish(tmp_path)
            if operation == "publish"
            else author.recover(
                proposal,
                tmp_path,
                before_launch=lambda: pytest.fail(
                    "unconfirmed writers must not resume"
                ),
            )
        )
    assert not (tmp_path / "result.json").exists()
    assert not list((tmp_path / "audit/queries").glob("*.json"))
    assert not list((tmp_path / "audit/checks").glob("*.json"))


@pytest.mark.parametrize(
    "damage", [None, "extra_file", "missing_empty_directory", "symlink", "hardlink"]
)
def test_correction_preserves_archive_without_using_it_as_resume_gate(tmp_path, damage):
    from reweave.evolution.authoring.author import native_completion
    from reweave.evolution.contracts import Usage
    from reweave.evolution.infrastructure.session import ClaudeSessionStore
    from tests.unit_tests.harness.evolution.fixtures import file_metadata

    root = workspace(tmp_path)
    candidate = root / "candidate"
    source = candidate / "task-family/context/src/context.py"
    (candidate / "task-family/workflows/empty_workflow").mkdir()
    (candidate / "invalid_link").symlink_to("context/src/context.py")
    os.mkfifo(candidate / "invalid_fifo")
    os.link(source, candidate / "invalid_hardlink.py")
    identity = str(uuid.uuid4())
    store = ClaudeSessionStore(tmp_path / "session")
    store.bind(identity)
    asyncio.run(
        store.append(
            {"session_id": identity, "project_key": "archive-integrity"},
            [
                {
                    "uuid": "completed",
                    "type": "assistant",
                    "message": {"content": "Done"},
                }
            ],
        )
    )
    atomic_json(tmp_path / "session/identity.json", {"session_id": identity})
    native_completion(
        tmp_path,
        Usage(physical_steps=0, model_calls=1, wall_time_s=1, cost_usd=1),
        identity,
        store.checkpoint(),
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(),
    )
    author._save_correction(
        tmp_path,
        tmp_path / "audit/checks/000000.json",
        ValueError("candidate contains invalid entries"),
        case_results=[],
    )
    archive = tmp_path / "audit/corrections/000000/output"
    archived_source = archive / "task-family/context/src/context.py"
    assert not (archive / "invalid_link").exists()
    assert not (archive / "invalid_fifo").exists()
    assert (archive / "invalid_hardlink.py").stat().st_nlink == 1
    assert archived_source.stat().st_nlink == 1
    if damage == "extra_file":
        (archive / "not_in_completion.py").write_text("unexpected = True\n")
    elif damage == "missing_empty_directory":
        (archive / "task-family/workflows/empty_workflow").rmdir()
    elif damage == "symlink":
        archived_source.unlink()
        archived_source.symlink_to(source)
        assert archived_source.read_bytes() == source.read_bytes()
    elif damage == "hardlink":
        os.link(archived_source, tmp_path / "same_bytes.py")
        assert archived_source.read_bytes() == source.read_bytes()
    reference = author._resume_correction(tmp_path)
    feedback = tmp_path / "audit/corrections/000000/feedback.json"
    assert reference == {
        "complete_report": {
            "path": str(feedback.resolve()),
            **file_metadata(feedback),
            "pointer": "",
        },
        "error_type": "ValueError",
        "contract_case_results": [],
    }
    assert not (tmp_path / "native_result.json").exists()


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("remote_tool", [False, True])
@pytest.mark.parametrize("drained", [False, True])
def test_process_receipt_distinguishes_completed_drain_from_live_owner(
    tmp_path, monkeypatch, platform, remote_tool, drained
):
    from reweave.evolution.authoring.remote import RemoteCodexSession

    directory = tmp_path / "audit/remote_tools/000000" if remote_tool else tmp_path
    verify = (
        RemoteCodexSession._verify_process_exit
        if remote_tool
        else NativeBundleAuthor._verify_process_exit
    )
    monkeypatch.setattr("reweave.evolution.authoring.author.sys.platform", platform)
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; print('ready', flush=True); sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as unrelated:
        assert unrelated.stdout.readline().strip() == "ready"
        # A completed historical launcher can have the same numeric PID as an
        # unrelated live process. Its terminal drain receipt remains valid.
        atomic_json(
            directory / "launcher/process.json",
            {"pid": unrelated.pid, "descendants_drained": drained},
        )
        if drained:
            verify(tmp_path)
        else:
            with pytest.raises(
                RuntimeError, match="previous native process still exists"
            ):
                verify(tmp_path)
        assert unrelated.poll() is None
        unrelated.communicate(input="", timeout=5)
        assert unrelated.returncode == 0


def test_missing_process_receipt_has_no_historical_pid_to_check(tmp_path, monkeypatch):
    def unexpected_probe(*args):
        pytest.fail("a missing process receipt must not probe any PID")

    monkeypatch.setattr("reweave.evolution.authoring.author.os.kill", unexpected_probe)
    NativeBundleAuthor._verify_process_exit(tmp_path)


@pytest.mark.parametrize(
    ("platform", "fields", "blocked"),
    [
        ("linux", {}, True),
        ("linux", {"descendants_drained": False}, True),
        ("linux", {"descendants_drained": None}, True),
        ("linux", {"descendants_drained": "true"}, True),
        ("linux", {"descendants_drained": 1}, True),
        ("linux", {"descendants_drained": -1}, True),
        ("linux", {"descendants_drained": True}, False),
        ("darwin", {}, False),
        ("darwin", {"descendants_drained": False}, True),
    ],
)
def test_linux_process_receipt_requires_explicit_true_drain(
    tmp_path, monkeypatch, platform, fields, blocked
):
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=5)
    atomic_json(tmp_path / "launcher/process.json", {"pid": process.pid, **fields})
    monkeypatch.setattr("reweave.evolution.authoring.author.sys.platform", platform)
    if blocked:
        with pytest.raises(RuntimeError, match="no confirmed drain"):
            NativeBundleAuthor._verify_process_exit(tmp_path)
    else:
        NativeBundleAuthor._verify_process_exit(tmp_path)
