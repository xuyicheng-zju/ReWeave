"""Persistent Claude Code decision turns; physical proposals are never executed here."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import NoReturn

import jsonschema

from reweave.planner.codex_connection import claude_environment, claude_executable

from .base import (
    DurableDecisionSession,
    NativeAuthenticationError,
    NativeProtocolError,
    NativeSubmissionUnknown,
    NativeTransportError,
    NativeTurnFailed,
    _write_json,
)


class ClaudeDecisionSession(DurableDecisionSession):
    """One durable episode conversation driven through the Claude Code CLI.

    Each decision starts the CLI against the same private config directory and
    resumes the pinned session, so no SDK task or subprocess outlives a call.
    A request whose user message reached the transcript is never resubmitted.
    The model gets only read tools (plus edits to declared source globs); every
    task effect is a proposal in the structured result.
    """

    PROVIDER = "claude"
    DEFAULT_MODEL = "claude-sonnet-4-6"
    BASE_INSTRUCTIONS = (
        "You are the online decision model of a task harness. "
        "Follow the supplied task instructions and use the observations and feedback in this session. "
        "Return the requested structured decision directly. "
        "Do not perform repository onboarding, development preflight, or routine source/log inspection before decisions. "
        "File tools are available for a necessary, targeted harness repair only. "
        "All task effects must be proposed to the Host; you have no shell."
    )
    READ_TOOLS = ("Read", "Glob", "Grep")
    EDIT_TOOLS = ("Edit", "Write")

    def __init__(
        self,
        directory: Path,
        *,
        model: str | None = None,
        editable_paths: tuple[str, ...] = (),
        client_factory=None,
        reasoning_effort: str = "low",
    ):
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported native reasoning effort")
        self.reasoning_effort = reasoning_effort
        self._model = self.resolve_model(model)
        self.editable_paths = tuple(editable_paths)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._state_path = self.directory / "session.json"
        if self._state_path.exists():
            self._state = self._load_state()
        else:
            self._state = {
                **self._identity(),
                "session_id": str(uuid.uuid4()),
                "start": None,
                "requests": {},
            }
        self._client_factory = client_factory
        self._context_generation = 0
        self._lock = asyncio.Lock()
        self._lease = None

    @property
    def config_directory(self) -> Path:
        """Private CLI home: keeps the operator's settings and memory out."""
        return self.directory / "claude_home"

    async def start(self, instructions: str, cwd: Path) -> None:
        async with self._lock:
            self._claim()
            cwd = Path(cwd).resolve()
            if not cwd.is_dir():
                raise NativeProtocolError(
                    "Native episode workspace must be an existing directory"
                )
            start = {"instructions": instructions, "cwd": str(cwd)}
            if self._state["start"] not in (None, start):
                raise NativeProtocolError(
                    "An episode cannot change its initial native session inputs"
                )
            if self._state["start"] is None:
                # Claude does not auto-load AGENTS.md. Pin the contract text once
                # so a resumed session keeps its original system prompt; later
                # changes arrive as deltas in decision messages.
                contract = cwd / "AGENTS.md"
                (self.directory / "system_prompt.md").write_text(
                    self.BASE_INSTRUCTIONS
                    + "\n\n"
                    + instructions
                    + (
                        "\n\n# AGENTS.md at episode start\n\n" + contract.read_text()
                        if contract.is_file()
                        else ""
                    )
                )
            self._state["start"] = start
            self._save()
            self._record("session_started", session_id=self.session_id)

    def _transcript(self) -> Path | None:
        found = list(
            (self.config_directory / "projects").glob(f"*/{self.session_id}.jsonl")
        )
        return found[0] if found else None

    def _user_message_recorded(self, client_id: str) -> bool:
        transcript = self._transcript()
        if transcript is None:
            return False
        for line in transcript.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "user" and entry.get("uuid") == client_id:
                return True
        return False

    def _options(self, output_schema: dict):
        import claude_agent_sdk

        resume = self._transcript() is not None
        editable = [
            f"{tool}({path})"
            for tool in self.EDIT_TOOLS
            for path in self.editable_paths
        ]
        tools = [*self.READ_TOOLS, *(self.EDIT_TOOLS if editable else ())]
        return claude_agent_sdk.ClaudeAgentOptions(
            cli_path=claude_executable(),
            cwd=self._state["start"]["cwd"],
            model=self.model,
            system_prompt=(self.directory / "system_prompt.md").read_text(),
            tools=tools,
            allowed_tools=[*self.READ_TOOLS, *editable],
            permission_mode="dontAsk",
            setting_sources=[],
            strict_mcp_config=True,
            thinking={"type": "disabled"},
            effort=self.reasoning_effort,
            # No slash commands/skills so the prompt carries only the harness tools.
            extra_args={"disable-slash-commands": None},
            output_format={"type": "json_schema", "schema": output_schema},
            resume=self.session_id if resume else None,
            session_id=None if resume else self.session_id,
            env={
                **claude_environment(),
                "CLAUDE_CONFIG_DIR": str(self.config_directory),
            },
            max_buffer_size=1 << 30,
        )

    def _content(self, request: dict, inputs: dict) -> list[dict]:
        directory = self._request_directory(request["request_id"])
        blocks = [{"type": "text", "text": inputs["text"]}]
        for index, image in enumerate(inputs["images"]):
            path = self._image_path(directory, image, index)
            media_type = mimetypes.guess_type(path.name)[0]
            if media_type not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
                raise NativeProtocolError(
                    "Native images must be PNG, JPEG, GIF or WEBP"
                )
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(path.read_bytes()).decode(),
                    },
                }
            )
        return blocks

    async def decide(
        self, text: str, images: tuple[Path, ...], output_schema: dict, request_id: str
    ) -> dict:
        async with self._lock:
            self._claim()
            if not isinstance(request_id, str) or not request_id:
                raise NativeProtocolError(
                    "A native decision requires a nonempty request id"
                )
            if self._state["start"] is None:
                raise NativeProtocolError(
                    "Start the episode native session before deciding"
                )
            inputs = self._input(text, images, output_schema)
            request = self._state["requests"].get(request_id)
            if request is None:
                if any(
                    r["phase"] == "submitting" for r in self._state["requests"].values()
                ):
                    raise NativeSubmissionUnknown(
                        "An earlier native request still requires settlement"
                    )
                request = {
                    "request_id": request_id,
                    "directory": uuid.uuid4().hex,
                    "client_id": str(uuid.uuid4()),
                    "phase": "prepared",
                }
                directory = self.directory / "requests" / request["directory"]
                for index, image in enumerate(inputs["images"]):
                    target = self._image_path(directory, image, index)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(image["path"], target)
                _write_json(directory / "input.json", inputs)
                self._state["requests"][request_id] = request
                self._save()
                self._record("decision_prepared", request_id=request_id)
            else:
                directory = self._request_directory(request_id)
                if not self._same_input(directory, inputs):
                    raise NativeProtocolError(
                        "Native request id was reused with different input"
                    )
            if request["phase"] == "completed":
                return copy.deepcopy(
                    json.loads((directory / "result.json").read_text())
                )
            if request["phase"] == "failed":
                raise NativeTurnFailed(request["status"], request["error"])
            if request["phase"] == "invalid":
                raise NativeProtocolError(
                    "The original completed native decision was invalid"
                )
            if request["phase"] == "submitting":
                # The transcript is the only evidence of submission. Present means
                # the model may have acted; absent means nothing was sent.
                if self._user_message_recorded(request["client_id"]):
                    self._record("submission_unresolved", request_id=request_id)
                    raise NativeSubmissionUnknown(
                        "Original native request may have been processed"
                    )
                request["phase"] = "prepared"
                self._save()
            return await self._submit(request, inputs, directory)

    async def _submit(self, request: dict, inputs: dict, directory: Path) -> dict:
        import claude_agent_sdk

        request["phase"] = "submitting"
        self._save()
        content = self._content(request, inputs)
        message = {
            "type": "user",
            "uuid": request["client_id"],
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
            "session_id": self.session_id,
        }

        async def prompt():
            yield message

        factory = self._client_factory or claude_agent_sdk.ClaudeSDKClient
        options = self._options(inputs["output_schema"])
        sent = False
        result = None
        try:
            async with factory(options) as client:
                await client.query(prompt(), self.session_id)
                sent = True
                self._record("decision_submitted", request_id=request["request_id"])
                async for frame in client.receive_response():
                    kind = type(frame).__name__
                    if kind == "SystemMessage" and frame.subtype == "compact_boundary":
                        self._context_generation += 1
                    elif kind == "ResultMessage":
                        result = frame
        except claude_agent_sdk.ResultError as error:
            self._fail(request, directory, error)
        except claude_agent_sdk.CLINotFoundError as error:
            request["phase"] = "prepared"
            self._save()
            raise NativeProtocolError(str(error)) from error
        except claude_agent_sdk.ClaudeSDKError as error:
            if not sent:
                request["phase"] = "prepared"
                self._save()
            self._record(
                "event_gap", request_id=request["request_id"], error=str(error)
            )
            raise NativeTransportError(str(error), request_sent=sent) from error
        if result is None:
            self._record("event_gap", request_id=request["request_id"])
            raise NativeTransportError(
                "Claude ended the stream without a result", request_sent=True
            )
        return self._finish(request, inputs, directory, result)

    def _fail(self, request: dict, directory: Path, source) -> NoReturn:
        """Archive a failed result; ``source`` is a ResultError or error ResultMessage."""
        error = {
            "errors": list(source.errors or []),
            "api_error_status": source.api_error_status,
            "terminal_reason": source.terminal_reason,
        }
        request.update(phase="failed", status=source.subtype or "error", error=error)
        self._save()
        _write_json(directory / "native_completion.json", request)
        if error["api_error_status"] in {401, 403}:
            raise NativeAuthenticationError(
                f"Claude provider rejected the credential (HTTP {error['api_error_status']})"
            )
        raise NativeTurnFailed(request["status"], error)

    def _finish(self, request: dict, inputs: dict, directory: Path, result) -> dict:
        _write_json(
            directory / "native_completion.json",
            {
                "subtype": result.subtype,
                "is_error": result.is_error,
                "num_turns": result.num_turns,
                "usage": result.usage,
                "total_cost_usd": result.total_cost_usd,
                "terminal_reason": result.terminal_reason,
            },
        )
        self._state["token_usage"] = result.usage
        if result.is_error:
            self._fail(request, directory, result)
        decision = result.structured_output
        try:
            if not isinstance(decision, dict):
                raise ValueError("A native decision must be a JSON object")
            jsonschema.validate(decision, inputs["output_schema"])
        except (ValueError, jsonschema.ValidationError) as error:
            request.update(phase="invalid", status="completed")
            self._save()
            raise NativeProtocolError(
                "Completed native output did not satisfy the decision schema"
            ) from error
        _write_json(directory / "result.json", decision)
        request.update(phase="completed", status="completed")
        self._save()
        self._record("decision_completed", request_id=request["request_id"])
        return copy.deepcopy(decision)

    async def close(self) -> None:
        async with self._lock:
            self._release()
