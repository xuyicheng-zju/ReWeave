"""Completed native decisions retain exact inputs, disposition and ACK links."""

import json

import pytest

from reweave.evolution.contracts import OriginalReference
from reweave.evolution.evidence.records import EpisodeEvidence, inventory_episode
from reweave.evolution.evidence.trace import (
    build_episode_context,
    resolve_trace_reference,
)
from reweave.evolution.evidence.trajectory import read_original
from tests.unit_tests.harness.evolution.test_trace_context import _episode, _view


def _native_episode(tmp_path, *, phase="completed", completion_status="completed"):
    request_id = "native-request"
    session_id = "native-session"
    proposals = [
        {"name": "pull", "arguments_json": '{"target":"drawer"}'},
        {"name": "release", "arguments_json": "{}"},
        {"name": "pull", "arguments_json": "incomplete original JSON"},
    ]
    ids = [f"call_{request_id}_{index}" for index in range(3)]
    messages = [
        {
            "role": "assistant",
            "content": {"text": "Inspect then act.", "tool_calls": proposals},
            "request_id": request_id,
            "program_call_id": "root",
            "native_session_id": session_id,
        },
        *[
            {
                "role": "tool",
                "tool_call_id": ids[index],
                "provider_tool_call_id": f"{request_id}_{index}",
                "program_call_id": "root",
                "name": proposals[index]["name"],
                "content": json.dumps(result),
            }
            for index, result in [
                (1, {"active_execution_ref": "release", "step": 3}),
                (0, {"active_execution_ref": "workflow", "step": 2}),
                (
                    2,
                    {
                        "source": "program_feedback",
                        "execution_performed": False,
                        "feedback": {"error": "Tool arguments are not complete JSON"},
                    },
                ),
            ]
        ],
    ]
    evidence, root = _episode(tmp_path, messages=messages)
    context = root / "harness_context"
    context.mkdir()
    response = {
        "kind": "model_response",
        "controller_request_unit": "native_turn",
        "request_id": request_id,
        "program_call_id": "root",
        "response": {
            "kind": "response",
            "parts": [
                {"part_kind": "text", "content": "Inspect then act."},
                *[
                    {
                        "part_kind": "tool-call",
                        "tool_name": proposals[index]["name"],
                        "tool_call_id": f"{request_id}_{index}",
                        "args": json.loads(proposals[index]["arguments_json"]),
                    }
                    for index in range(2)
                ],
            ],
            "metadata": {
                "request_id": request_id,
                "program_call_id": "root",
                "tool_call_ids": {
                    f"{request_id}_{index}": identity
                    for index, identity in enumerate(ids)
                },
            },
        },
    }
    events = [
        {
            "kind": "model_call_rejected",
            "request_id": request_id,
            "call": {
                "id": ids[2],
                "provider_id": f"{request_id}_2",
                **proposals[2],
                "arguments": proposals[2]["arguments_json"],
                "validation_error": "Tool arguments are not complete JSON",
            },
            "execution_performed": False,
        },
        response,
    ]
    for index, execution_ref in enumerate(["workflow", "release"]):
        identity = {
            "action_id": f"action-{index}",
            "operation": proposals[index]["name"],
            "planner_call_id": ids[index],
            "workflow_invocation_id": None,
        }
        events.extend(
            [
                {**identity, "kind": "action_submitted", "effect": "physical"},
                {
                    **identity,
                    "kind": "operation_settled",
                    "result": {"active_execution_ref": execution_ref},
                },
                {
                    **identity,
                    "kind": "action_completed",
                    "result": {"active_execution_ref": execution_ref},
                },
            ]
        )
    (context / "bundle_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    provider = root / "native/provider"
    request = provider / "requests/saved-request"
    request.mkdir(parents=True)
    (request / "input.json").write_text(
        json.dumps({"text": "Exact incremental observation", "images": []})
    )
    (request / "native_completion.json").write_text(
        json.dumps({"id": "completed-turn", "status": completion_status})
    )
    (request / "result.json").write_text(json.dumps(messages[0]["content"]))
    (provider / "session.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "requests": {
                    request_id: {
                        "directory": "saved-request",
                        "phase": phase,
                        "turn_id": "completed-turn",
                    },
                    "prepared-only": {
                        "directory": "prepared-request",
                        "phase": "prepared",
                        "turn_id": None,
                    },
                },
            }
        )
    )
    (root / "native/records").mkdir()
    (root / "native/records/episode.md").write_text("Complete native episode record")
    (root / "native/adaptation").mkdir()
    (root / "native/adaptation/receipt.json").write_text(
        json.dumps({"mode": "adaptive", "revisions": []})
    )
    evidence = EpisodeEvidence(
        evidence_id=evidence.evidence_id, files=inventory_episode(root)
    )
    return evidence, root, messages, events


def test_native_decisions_link_exact_input_arguments_rejection_and_physical_ack(
    tmp_path,
):
    evidence, root, messages, _ = _native_episode(tmp_path)
    originals = {name: (root / name).read_bytes() for name in evidence.files}
    result = build_episode_context(tmp_path, evidence, include_visuals=False)
    timeline = _view(tmp_path, evidence, "timeline")
    calls = timeline["planner_calls"]
    assert len(calls) == 3
    assert [call["proposal_status"] for call in calls] == [
        "accepted",
        "accepted",
        "rejected",
    ]
    assert [call["execution_ref"] for call in calls] == ["workflow", "release", None]
    assert not {gap["kind"] for gap in timeline["gaps"]} & {
        "feedback_without_request",
        "kernel_planner_call_missing",
        "decision_input_link_not_unique",
        "execution_without_planner_feedback_link",
    }
    for index, call in enumerate(calls):
        assert (
            resolve_trace_reference(tmp_path, call["request"])
            == messages[0]["content"]["tool_calls"][index]
        )
        assert resolve_trace_reference(tmp_path, call["decision_inputs"][0]) == {
            "text": "Exact incremental observation",
            "images": [],
        }
        assert (
            resolve_trace_reference(tmp_path, call["native_completion"])["status"]
            == "completed"
        )
    assert resolve_trace_reference(tmp_path, calls[0]["arguments_reference"]) == {
        "target": "drawer"
    }
    assert (
        resolve_trace_reference(tmp_path, calls[2]["arguments_reference"])
        == "incomplete original JSON"
    )
    assert (
        resolve_trace_reference(tmp_path, calls[2]["disposition"])["validation_error"]
        == "Tool arguments are not complete JSON"
    )
    manifest = json.loads((tmp_path / result["path"]).read_text())
    assert read_original(
        OriginalReference.model_validate(
            {
                **manifest["native"]["episode_record"],
                "path": str(tmp_path / manifest["native"]["episode_record"]["path"]),
            }
        )
    ) == ("Complete native episode record")
    assert manifest["native"]["adaptation_receipt"] is not None
    assert manifest["coverage"]["physical_steps"] == 3
    assert manifest["coverage"]["planner_calls"] == 3
    assert all((root / name).read_bytes() == data for name, data in originals.items())


@pytest.mark.parametrize(
    ("phase", "completion_status"),
    [("prepared", "completed"), ("failed", "failed"), ("completed", "failed")],
)
def test_native_prepared_or_failed_turn_never_becomes_confirmed_decision(
    tmp_path, phase, completion_status
):
    evidence, _, _, _ = _native_episode(
        tmp_path, phase=phase, completion_status=completion_status
    )
    build_episode_context(tmp_path, evidence, include_visuals=False)
    timeline = _view(tmp_path, evidence, "timeline")
    assert timeline["planner_calls"] == []
    assert any(
        gap["kind"] == "native_decision_not_confirmed" for gap in timeline["gaps"]
    )


def test_native_transcript_without_host_response_keeps_explicit_missingness(tmp_path):
    evidence, root, _, events = _native_episode(tmp_path)
    (root / "harness_context/bundle_events.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n" for row in events if row["kind"] != "model_response"
        )
    )
    evidence = EpisodeEvidence(
        evidence_id=evidence.evidence_id, files=inventory_episode(root)
    )
    build_episode_context(tmp_path, evidence, include_visuals=False)
    timeline = _view(tmp_path, evidence, "timeline")
    assert timeline["planner_calls"] == []
    assert any(
        gap["kind"] == "native_decision_not_confirmed" for gap in timeline["gaps"]
    )


def test_native_root_and_local_decisions_join_by_explicit_request_and_program(tmp_path):
    evidence, root, messages, events = _native_episode(tmp_path)
    local_message = json.loads(json.dumps(messages[0]))
    local_message["request_id"] = "local-request"
    local_message["program_call_id"] = "local-program"
    local_message["content"]["tool_calls"] = local_message["content"]["tool_calls"][:1]
    local_response = json.loads(json.dumps(events[1]))
    local_response.update(request_id="local-request", program_call_id="local-program")
    local_response["response"]["parts"] = local_response["response"]["parts"][:2]
    local_response["response"]["parts"][1]["tool_call_id"] = "local-request_0"
    local_response["response"]["metadata"] = {
        "request_id": "local-request",
        "program_call_id": "local-program",
        "tool_call_ids": {"local-request_0": "call_local-request_0"},
    }
    transcript = json.loads((root / "transcript_test.json").read_text())
    transcript["messages"].insert(0, local_message)
    (root / "transcript_test.json").write_text(json.dumps(transcript))
    (root / "harness_context/bundle_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in [*events, local_response])
    )
    provider = root / "native/provider"
    session = json.loads((provider / "session.json").read_text())
    session["requests"]["local-request"] = {
        "directory": "local-request-files",
        "phase": "completed",
        "turn_id": "local-turn",
    }
    (provider / "session.json").write_text(json.dumps(session))
    local = provider / "requests/local-request-files"
    local.mkdir()
    (local / "input.json").write_text(json.dumps({"text": "Exact local input"}))
    (local / "native_completion.json").write_text(
        json.dumps({"id": "local-turn", "status": "completed"})
    )
    evidence = EpisodeEvidence(
        evidence_id=evidence.evidence_id, files=inventory_episode(root)
    )
    build_episode_context(tmp_path, evidence, include_visuals=False)
    calls = _view(tmp_path, evidence, "timeline")["planner_calls"]
    assert len(calls) == 4
    local_call = next(
        call for call in calls if call["program_call_id"] == "local-program"
    )
    assert local_call["tool_call_id"] == "call_local-request_0"
    assert local_call["execution_ref"] is None
    assert resolve_trace_reference(tmp_path, local_call["decision_inputs"][0]) == {
        "text": "Exact local input"
    }


def test_native_proposal_without_host_disposition_is_not_invented(tmp_path):
    evidence, root, _, events = _native_episode(tmp_path)
    events = [row for row in events if row["kind"] != "model_call_rejected"]
    (root / "harness_context/bundle_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    evidence = EpisodeEvidence(
        evidence_id=evidence.evidence_id, files=inventory_episode(root)
    )
    build_episode_context(tmp_path, evidence, include_visuals=False)
    timeline = _view(tmp_path, evidence, "timeline")
    assert len(timeline["planner_calls"]) == 2
    assert all(
        call["proposal_status"] == "accepted" for call in timeline["planner_calls"]
    )
    assert any(
        gap["kind"] == "native_proposal_without_host_disposition"
        for gap in timeline["gaps"]
    )


def test_cached_navigation_without_native_contract_is_rebuilt(tmp_path):
    evidence, _, _, _ = _native_episode(tmp_path)
    result = build_episode_context(tmp_path, evidence, include_visuals=False)
    path = tmp_path / result["path"]
    manifest = json.loads(path.read_text())
    manifest.pop("native")
    path.write_text(json.dumps(manifest))
    build_episode_context(tmp_path, evidence, include_visuals=False)
    assert "native" in json.loads(path.read_text())
