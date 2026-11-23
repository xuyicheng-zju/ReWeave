import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from reweave.planner.providers.base import (
    NativeCompletedInvalidDecision,
    NativeProtocolError,
    NativeSubmissionUnknown,
    NativeTurnFailed,
)
from reweave.planner.providers.codex import CodexDecisionSession
from reweave.planner.providers.cursor import CursorDecisionSession, _parse_decision

SCHEMA = {
    "type": "object",
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}


class FakeTransport:
    def set_observer(self, observer):
        pass


class InvalidDecisionTest(unittest.TestCase):
    def usage_completion(self, fault=None):
        with tempfile.TemporaryDirectory() as td:
            obj = CodexDecisionSession.__new__(CodexDecisionSession)
            obj._request_directory = lambda request_id: Path(td)
            obj._save = lambda: None
            obj._record = lambda *args, **kwargs: None
            obj._state = {
                "event_stream_complete": fault != "event_gap",
                "token_usage": {"stale": 1},
            }
            message = {
                "type": "agentMessage",
                "id": "m",
                "phase": "final_answer",
                "text": '{"text":"unchanged"}',
            }
            request = {
                "request_id": "r",
                "client_id": "c",
                "turn_id": "t",
                "agent_messages": [message],
                "turn_evidence": {
                    "t": {
                        "items": {
                            "m": {
                                "type": "agentMessage",
                                "completed": fault != "unobserved",
                            }
                        }
                    }
                },
            }
            turn = {
                "id": "t",
                "status": "failed",
                "itemsView": "full",
                "items": [],
                "error": {
                    "message": "stream disconnected before completion: failed to parse ResponseCompleted: missing field `reasoning_tokens`"
                },
            }
            if fault == "other_error":
                turn["error"] = {"message": "unrelated failure"}
            if fault == "old_message":
                request["turn_evidence"] = {"old-turn": request["turn_evidence"]["t"]}
            if fault == "native_effect":
                turn["items"] = [
                    {"id": "effect", "type": "commandExecution", "status": "completed"}
                ]
            if fault == "invalid_json":
                message["text"] = "no executable decision"
            (Path(td) / "attempts/c").mkdir(parents=True)
            if fault:
                with self.assertRaises(NativeTurnFailed):
                    obj._finish(request, {"output_schema": SCHEMA}, turn)
                self.assertEqual(obj._state["token_usage"], {"stale": 1})
            else:
                self.assertEqual(
                    obj._finish(request, {"output_schema": SCHEMA}, turn),
                    {"text": "unchanged"},
                )
                self.assertEqual(request["phase"], "completed")
                self.assertEqual(request["status"], "failed")
                self.assertEqual(request["usage_warning"], turn["error"])
                self.assertIsNone(obj._state["token_usage"])
                self.assertTrue(obj._state["token_usage_incomplete"])
            self.assertEqual(
                json.loads((Path(td) / "native_completion.json").read_text()), turn
            )

    def test_completed_content_survives_only_the_specific_usage_metadata_error(self):
        self.usage_completion()
        for fault in [
            "event_gap",
            "unobserved",
            "other_error",
            "old_message",
            "native_effect",
            "invalid_json",
        ]:
            with self.subTest(fault=fault):
                self.usage_completion(fault)

    def codex_valid_decision(self, text):
        with tempfile.TemporaryDirectory() as td:
            obj = CodexDecisionSession.__new__(CodexDecisionSession)
            obj._request_directory = lambda request_id: Path(td)
            obj._save = lambda: None
            obj._record = lambda *args, **kwargs: None
            obj._state = {"event_stream_complete": True}
            request = {
                "request_id": "r",
                "client_id": "c",
                "turn_id": "t",
                "agent_messages": [],
            }
            turn = {
                "id": "t",
                "status": "completed",
                "itemsView": "full",
                "items": [
                    {
                        "type": "agentMessage",
                        "id": "m",
                        "phase": "final_answer",
                        "text": text,
                    }
                ],
            }
            (Path(td) / "attempts/c").mkdir(parents=True)
            self.assertEqual(
                obj._finish(request, {"output_schema": SCHEMA}, turn),
                {"text": "unchanged"},
            )
            self.assertEqual(
                json.loads((Path(td) / "native_completion.json").read_text()), turn
            )

    def test_formatted_json_preserves_payload_and_raw_completion(self):
        for text in (
            '{"text":"unchanged"}',
            '```json\n{"text":"unchanged"}\n```',
            'I will use the structured proposal.\n\n{"text":"unchanged"}',
            'I will use the structured proposal.\n\n```json\n{"text":"unchanged"}\n```',
        ):
            with self.subTest(text=text):
                self.codex_valid_decision(text)
                self.assertEqual(_parse_decision(text, SCHEMA), {"text": "unchanged"})

    def codex(
        self,
        extra,
        *,
        text="not json",
        view="full",
        stream_complete=True,
        observe_final=True,
    ):
        with tempfile.TemporaryDirectory() as td:
            obj = CodexDecisionSession.__new__(CodexDecisionSession)
            obj._request_directory = lambda request_id: Path(td)
            obj._save = lambda: None
            obj._record = lambda *args, **kwargs: None
            obj._state = {"event_stream_complete": stream_complete}
            observed = (
                {"m": {"type": "agentMessage", "completed": True}}
                if observe_final
                else {}
            )
            request = {
                "request_id": "r",
                "client_id": "c",
                "agent_messages": [],
                "turn_evidence": {"t": {"items": observed}},
            }
            turn = {
                "id": "t",
                "status": "completed",
                "itemsView": view,
                "items": [
                    {
                        "type": "agentMessage",
                        "id": "m",
                        "phase": "final_answer",
                        "text": text,
                    },
                    *extra,
                ],
            }
            (Path(td) / "attempts" / "c").mkdir(parents=True)
            with self.assertRaises(NativeProtocolError) as cm:
                obj._finish(request, {"output_schema": SCHEMA}, turn)
            self.assertTrue((Path(td) / "native_completion.json").exists())
            return type(cm.exception), request

    def test_formatting_does_not_accept_ambiguous_or_invalid_decisions(self):
        for text in (
            'First {"text":"first"}\nThen {"text":"second"}',
            'Proposal {"text":"unchanged"}\nMore instructions.',
            'Proposal {"wrong_key":"unchanged"}',
            '[{"text":"unchanged"}]',
            'Proposal {"incomplete": {"text":"unchanged"}',
        ):
            with self.subTest(text=text):
                self.assertIs(
                    self.codex([], text=text)[0], NativeCompletedInvalidDecision
                )
                with self.assertRaises(NativeProtocolError):
                    _parse_decision(text, SCHEMA)

    def test_codex_completed_invalid_and_effects(self):
        kind, request = self.codex([])
        self.assertIs(kind, NativeCompletedInvalidDecision)
        self.assertTrue(request["invalid_effects_settled"])
        kind, request = self.codex(
            [{"id": "tool", "type": "commandExecution", "status": "inProgress"}]
        )
        self.assertIs(kind, NativeProtocolError)
        self.assertFalse(request["invalid_effects_settled"])
        kind, request = self.codex(
            [{"id": "tool", "type": "commandExecution", "status": "completed"}]
        )
        self.assertIs(kind, NativeProtocolError)
        self.assertIs(self.codex([], view="summary")[0], NativeCompletedInvalidDecision)
        self.assertIs(
            self.codex([], view="summary", stream_complete=False)[0],
            NativeProtocolError,
        )
        self.assertIs(
            self.codex([], view="summary", observe_final=False)[0], NativeProtocolError
        )

    def test_cursor_completed_invalid_vs_unknown_and_tool(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                obj = CursorDecisionSession(
                    Path(td),
                    transport=FakeTransport(),
                    model="grok-4.7-high",
                    reasoning_effort="high",
                )
                obj._state["session_id"] = "session"
                obj._state["start"] = {"instructions": "test"}
                obj._connected = True

                async def rpc(method, params):
                    obj._chunks = ["not json"]
                    return {"stopReason": "end_turn"}

                obj._rpc = rpc
                with self.assertRaises(NativeCompletedInvalidDecision):
                    await obj.decide("input", (), SCHEMA, "r")
                with self.assertRaises(NativeCompletedInvalidDecision):
                    await obj.decide("input", (), SCHEMA, "r")
                self.assertEqual(obj._state["requests"]["r"]["phase"], "invalid")

                async def with_tool(method, params):
                    obj._tool_updates = 1
                    return {"stopReason": "end_turn"}

                obj._rpc = with_tool
                with self.assertRaises(NativeProtocolError) as cm:
                    await obj.decide("input", (), SCHEMA, "tool")
                self.assertIs(type(cm.exception), NativeProtocolError)
                obj._state["requests"]["unknown"] = {
                    "request_id": "unknown",
                    "directory": obj._state["requests"]["r"]["directory"],
                    "phase": "submitting",
                }
                with self.assertRaises(NativeSubmissionUnknown):
                    await obj.decide("input", (), SCHEMA, "unknown")
                obj._release()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
