"""Cursor CLI subscription decisions, without a live model call."""

from __future__ import annotations

import asyncio
import copy
import json
import stat

import pytest

from reweave.planner.cursor_connection import cursor_environment, cursor_model_spec
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.planner.providers.base import (
    NativeProtocolError,
    NativeSubmissionUnknown,
)
from reweave.planner.providers.cursor import CursorDecisionSession, _redact_wire
from reweave.planner.providers.cursor_stdio import CursorStdioTransport

SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {"type": "array"},
    },
    "required": ["text", "tool_calls"],
    "additionalProperties": False,
}


def _options():
    return [
        {
            "id": "mode",
            "currentValue": "agent",
            "options": [{"value": "agent"}, {"value": "plan"}, {"value": "ask"}],
        },
        {
            "id": "model",
            "currentValue": "grok-4.7",
            "options": [{"value": "grok-4.7"}],
        },
        {
            "id": "context",
            "currentValue": "256k",
            "options": [{"value": "256k"}, {"value": "500k"}],
        },
        {
            "id": "reasoning_effort",
            "currentValue": "high",
            "options": [
                {"value": "low"},
                {"value": "medium"},
                {"value": "high"},
                {"value": "xhigh"},
            ],
        },
        {
            "id": "fast",
            "currentValue": "true",
            "options": [{"value": "false"}, {"value": "true"}],
        },
    ]


class FakeCursorTransport:
    def __init__(self):
        self.calls = []
        self.options = _options()
        self.answer = '{"text": "ok", "tool_calls": []}'
        self.stop_reason = "end_turn"
        self.tool_call = False
        self.metadata = {
            "protocol": "cursor-acp",
            "executable": "agent",
            "authentication": "cursor_login",
        }
        self._observer = None

    def set_observer(self, observer):
        self._observer = observer

    async def connect(self, cwd):
        self.calls.append(("connect", str(cwd)))

    async def close(self):
        self.calls.append(("close", None))

    async def notify(self, method, params):
        self.calls.append((method, params))

    async def next_event(self):
        await asyncio.Future()

    async def request(self, method, params):
        self.calls.append((method, copy.deepcopy(params)))
        if method == "initialize":
            return {
                "protocolVersion": 1,
                "authMethods": [{"id": "cursor_login", "name": "Cursor Login"}],
            }
        if method == "authenticate":
            return {}
        if method == "session/new":
            return {"sessionId": "sess-1", "configOptions": copy.deepcopy(self.options)}
        if method == "session/load":
            return {"configOptions": copy.deepcopy(self.options)}
        if method == "session/set_config_option":
            for item in self.options:
                if item["id"] == params["configId"]:
                    item["currentValue"] = params["value"]
            return {"configOptions": copy.deepcopy(self.options)}
        if method == "session/prompt":
            if self.tool_call:
                self._observer(
                    {
                        "direction": "incoming",
                        "message": {
                            "method": "session/update",
                            "params": {"update": {"sessionUpdate": "tool_call"}},
                        },
                    }
                )
            self._observer(
                {
                    "direction": "incoming",
                    "message": {
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"text": self.answer},
                            }
                        },
                    },
                }
            )
            return {"stopReason": self.stop_reason}
        raise AssertionError(method)


def _prompts(transport):
    return [params for method, params in transport.calls if method == "session/prompt"]


def _sets(transport):
    return [
        (params["configId"], params["value"])
        for method, params in transport.calls
        if method == "session/set_config_option"
    ]


def test_grok_47_xhigh_is_the_non_fast_subscription_variant():
    spec = cursor_model_spec("grok-4.7-xhigh")
    assert spec == {
        "catalog_id": "grok-4.7-xhigh",
        "model": "grok-4.7",
        "reasoning_effort": "xhigh",
        "fast": False,
        "context": "256k",
    }
    fast = cursor_model_spec("grok-4.7-xhigh-fast")
    assert fast["fast"] is True and fast["reasoning_effort"] == "xhigh"
    with pytest.raises(ValueError):
        cursor_model_spec("grok-4.7-turbo")


def test_cursor_environment_stays_inside_the_private_echo_directory(tmp_path):
    home = tmp_path / "home"
    env = cursor_environment(
        {
            "PATH": "/bin",
            "HOME": str(home),
            "CURSOR_API_KEY": "secret",
            "CURSOR_AUTH_TOKEN": "tok",
        },
        system="Darwin",
    )
    private = home / ".config" / "reweave" / "cursor"
    assert env["PATH"] == "/bin"
    assert env["HOME"] == str(home)
    assert env["CURSOR_CONFIG_DIR"] == str(private)
    assert "CURSOR_API_KEY" not in env
    assert "secret" not in json.dumps(env)
    assert oct(private.stat().st_mode & 0o777) == "0o700"
    assert oct(private.parent.stat().st_mode & 0o777) == "0o700"
    assert "XDG_CONFIG_HOME" not in env


def test_linux_login_must_be_a_private_file_in_the_echo_directory(tmp_path):
    home = tmp_path / "home"
    with pytest.raises(ValueError, match="private reweave directory"):
        cursor_environment({"HOME": str(home), "PATH": "/bin"}, system="Linux")
    private = home / ".config" / "reweave" / "cursor"
    auth = private / "auth.json"
    auth.write_text("{}\n")
    auth.chmod(0o644)
    env = cursor_environment({"HOME": str(home), "PATH": "/bin"}, system="Linux")
    assert env["XDG_CONFIG_HOME"] == str(private.parent)
    assert env["CURSOR_CONFIG_DIR"] == str(private)
    assert oct(auth.stat().st_mode & 0o777) == "0o600"
    auth.unlink()
    auth.symlink_to(private / "missing.json")
    with pytest.raises(ValueError, match="regular file"):
        cursor_environment({"HOME": str(home), "PATH": "/bin"}, system="Linux")


def test_cursor_planner_defaults_to_grok_47_xhigh(tmp_path):
    planner = NativeBundlePlanner(
        output_dir=tmp_path, provider="cursor", reasoning_effort="medium"
    )
    assert planner.model == "grok-4.7-xhigh"
    assert planner.reasoning_effort == "xhigh"
    configuration = planner.runtime_configuration()
    assert configuration["authentication"] == "cursor_login"
    assert configuration["billing"] == "subscription"
    assert configuration["protocol"] == "cursor-acp"
    assert configuration["model_parameters"] == {
        "model": "grok-4.7",
        "reasoning_effort": "xhigh",
        "fast": False,
        "context": "256k",
        "mode": "ask",
    }
    with pytest.raises(ValueError):
        NativeBundlePlanner(
            output_dir=tmp_path,
            provider="cursor",
            model="grok-4.7-xhigh",
            service_tier="fast",
        )


def test_subscription_session_pins_model_and_replays_the_same_decision(tmp_path):
    transport = FakeCursorTransport()
    session = CursorDecisionSession(
        tmp_path / "episode",
        transport=transport,
        model="grok-4.7-xhigh",
        reasoning_effort="xhigh",
    )
    image = tmp_path / "frame.png"
    image.write_bytes(b"png-bytes")

    async def exercise():
        await session.start("Standing instructions.", tmp_path)
        first = await session.decide("look", (image,), SCHEMA, "request-1")
        second = await session.decide("look", (image,), SCHEMA, "request-1")
        follow = await session.decide("again", (), SCHEMA, "request-2")
        await session.close()
        return first, second, follow

    first, second, follow = asyncio.run(exercise())
    assert first == second == {"text": "ok", "tool_calls": []}
    assert follow == first
    initialize = next(
        params for method, params in transport.calls if method == "initialize"
    )
    assert initialize["clientCapabilities"]["_meta"]["parameterizedModelPicker"] is True
    authenticate = next(
        params for method, params in transport.calls if method == "authenticate"
    )
    assert authenticate == {"methodId": "cursor_login"}
    created = next(
        params for method, params in transport.calls if method == "session/new"
    )
    assert created["mcpServers"] == []
    assert _sets(transport) == [
        ("mode", "ask"),
        ("reasoning_effort", "xhigh"),
        ("fast", "false"),
    ]
    prompts = _prompts(transport)
    assert len(prompts) == 2
    assert prompts[0]["prompt"][0]["text"] == "Standing instructions."
    assert "tool_calls" in prompts[0]["prompt"][1]["text"]
    assert prompts[0]["prompt"][2]["mimeType"] == "image/png"
    assert prompts[1]["prompt"][0]["type"] == "text"
    assert prompts[1]["prompt"][0]["text"].startswith("Return one JSON object")
    wire = (tmp_path / "episode" / "events.jsonl").read_text()
    assert prompts[0]["prompt"][2]["data"] not in wire


def test_image_bytes_are_redacted_from_the_wire_log():
    data = "cG5nLWJ5dGVz"
    redacted = _redact_wire(
        {
            "method": "session/prompt",
            "params": {
                "prompt": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "mimeType": "image/png", "data": data},
                ]
            },
        }
    )
    assert data not in json.dumps(redacted)
    assert redacted["params"]["prompt"][1]["data"].startswith("<redacted ")


def test_unconfirmed_prompt_is_not_replayed(tmp_path):
    transport = FakeCursorTransport()
    session = CursorDecisionSession(
        tmp_path / "episode", transport=transport, model="grok-4.7-xhigh"
    )

    async def exercise():
        await session.start("Instructions.", tmp_path)
        await session.decide("look", (), SCHEMA, "request-1")
        session._state["requests"]["request-1"]["phase"] = "submitting"
        with pytest.raises(NativeSubmissionUnknown):
            await session.decide("look", (), SCHEMA, "request-1")

    asyncio.run(exercise())
    assert len(_prompts(transport)) == 1


def test_changed_request_input_is_rejected(tmp_path):
    transport = FakeCursorTransport()
    session = CursorDecisionSession(
        tmp_path / "episode", transport=transport, model="grok-4.7-xhigh"
    )

    async def exercise():
        await session.start("Instructions.", tmp_path)
        await session.decide("look", (), SCHEMA, "request-1")
        with pytest.raises(NativeProtocolError):
            await session.decide("different", (), SCHEMA, "request-1")

    asyncio.run(exercise())


def test_cursor_stdio_uses_acp_without_an_api_key(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    script = tmp_path / "agent"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"open({str(report)!r}, 'w').write(json.dumps({{'argv': sys.argv[1:], 'api_key': 'CURSOR_API_KEY' in os.environ}}))\n"
        "msg = json.loads(sys.stdin.readline())\n"
        "sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'ok': True}})+'\\n')\n"
        "sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    pass\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "home"
    private = home / ".config" / "reweave" / "cursor"
    private.mkdir(parents=True)
    (private / "auth.json").write_text("{}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CURSOR_API_KEY", "secret-key")
    transport = CursorStdioTransport(tmp_path / "runtime", executable=str(script))
    seen = []

    async def exercise():
        transport.set_observer(lambda event: seen.append(event))
        await transport.connect(tmp_path)
        result = await transport.request("initialize", {"protocolVersion": 1})
        await transport.close()
        return result

    assert asyncio.run(exercise()) == {"ok": True}
    assert json.loads(report.read_text()) == {"argv": ["acp"], "api_key": False}
    assert seen[0]["direction"] == "outgoing"
    assert "secret-key" not in json.dumps(seen)
