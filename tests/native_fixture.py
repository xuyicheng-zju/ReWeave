"""Scripted native decisions for integration tests; no provider request is made.

The callback syntax reuses Pydantic message objects as test values only. Every
decision still traverses production NativeBundlePlanner, editable C/L/W and
Kernel. Callbacks receive only the newly delivered native input, never a
reconstructed provider history.
"""

from __future__ import annotations

import base64
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pydantic_ai.messages import ModelMessagesTypeAdapter, TextPart, ToolCallPart

from reweave.planner.native_loop import NativeBundlePlanner
from tests.native_robot import Robot


class ScriptedNativeSession:
    def __init__(self, callback, directory):
        self.callback = callback
        self.directory = directory
        self.session_id = "SIMULATED-scripted-native-session"
        self.context_generation = 0
        self.inputs = []

    async def start(self, instructions, cwd):
        self.cwd = cwd
        self.developer_instructions = instructions

    async def decide(self, text, images, output_schema, request_id):
        self.inputs.append((text, images, request_id))
        instructions = (self.cwd / "AGENTS.md").read_text()
        catalog = json.loads(
            instructions.split("# Tool catalog\n\n```json\n", 1)[1].split("\n```", 1)[0]
        )
        # Host guidance deltas precede the actual JSON observation payload.
        observation_start = text.rfind('{"messages":')
        payload = (
            json.loads(text[observation_start:])
            if observation_start >= 0
            else {
                "messages": [
                    {
                        "kind": "request",
                        "parts": [{"part_kind": "user-prompt", "content": text}],
                    }
                ],
                "available_tools": [],
            }
        )

        def restore(value):
            if isinstance(value, dict):
                if set(value) == {"image_file"}:
                    return {
                        "kind": "binary",
                        "media_type": "image/png",
                        "data": base64.b64encode(
                            Path(value["image_file"]).read_bytes()
                        ).decode(),
                    }
                return {key: restore(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restore(item) for item in value]
            return value

        messages = ModelMessagesTypeAdapter.validate_python(
            restore(payload["messages"])
        )
        allowed = set(payload["available_tools"])
        objective = None
        for message in payload["messages"]:
            for part in message.get("parts", []):
                content = part.get("content")
                if isinstance(content, str) and content.startswith('{"observation":'):
                    objective = json.loads(content)["objective"]
        info = SimpleNamespace(
            instructions=instructions,
            developer_instructions=self.developer_instructions,
            native_images=images,
            native_text=text,
            objective=objective,
            context_guidance=instructions.split(
                "# Context guidance and task experience\n\n", 1
            )[1].split("\n\n# Task\n\n", 1)[0],
            function_tools=[
                SimpleNamespace(
                    name=item["name"],
                    description=item["description"],
                    parameters_json_schema=item["input_schema"],
                )
                for item in catalog
                if item["name"] in allowed
            ],
            model_settings={"reasoning_effort": "medium"},
        )
        result = self.callback(messages, info)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            return result
        return {
            "text": "\n".join(
                part.content for part in result.parts if isinstance(part, TextPart)
            ),
            "tool_calls": [
                {
                    "name": part.tool_name,
                    "arguments_json": part.args
                    if isinstance(part.args, str)
                    else json.dumps(part.args),
                }
                for part in result.parts
                if isinstance(part, ToolCallPart)
            ],
        }

    async def close(self):
        pass


class ScriptedNativePlanner(NativeBundlePlanner):
    def __init__(
        self,
        model,
        *,
        dashboard_events=None,
        max_tokens=None,
        output_dir=Path("."),
        **kwargs,
    ):
        callback = model.function if hasattr(model, "function") else model
        self.scripted_sessions = []

        def factory(directory):
            session = ScriptedNativeSession(callback, directory)
            self.scripted_sessions.append(session)
            return session

        super().__init__(output_dir=output_dir, session_factory=factory, **kwargs)


class RecordingSink:
    def __init__(self):
        self.events = []

    @property
    def enabled(self):
        return True

    def emit(self, event):
        self.events.append(event)


class PhysicalRobot(Robot):
    """Real Toolkit dispatch over an in-memory actuator, with no model API."""

    def __init__(self, path):
        super().__init__(path)
        self.moves = 0
        self.state.latest_record = lambda: None
        self.add_tool(
            "move_once",
            {
                "name": "move_once",
                "description": "Move the synthetic actuator one step.",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.move_once,
        )

    def move_once(self):
        self.moves += 1
        return {"steps_used": 1, "position": self.moves}

    def get_env_state(self, **kwargs):
        image = io.BytesIO()
        Image.new("RGB", (2, 2), "blue").save(image, format="PNG")
        return {"state": {"position": self.moves}, "_image_bytes": image.getvalue()}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
