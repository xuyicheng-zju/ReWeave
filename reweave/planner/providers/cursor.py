"""Persistent Cursor decision turns. The CLI calls the model; the Host executes proposals."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import uuid
from pathlib import Path

import jsonschema

from reweave import __version__
from reweave.planner.cursor_connection import DEFAULT_CURSOR_MODEL, cursor_model_spec

from reweave.utils.persistence import atomic_text

from .base import (
    DurableDecisionSession,
    NativeAuthenticationError,
    NativeCompletedInvalidDecision,
    NativeProtocolError,
    NativeRpcError,
    NativeSubmissionUnknown,
    NativeTransportError,
    NativeTurnFailed,
    _canonical,
    _decision_text,
)
from .cursor_stdio import CursorStdioTransport
from .transport import NativeTransport

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _redact_wire(message: dict) -> dict:
    """Keep image bytes out of the episode wire log."""
    params = message.get("params")
    prompt = params.get("prompt") if isinstance(params, dict) else None
    if not isinstance(prompt, list):
        return message
    redacted = []
    for block in prompt:
        if isinstance(block, dict) and block.get("type") == "image":
            redacted.append(
                {
                    **block,
                    "data": f"<redacted {len(block.get('data', ''))} chars>",
                }
            )
        else:
            redacted.append(block)
    return {**message, "params": {**params, "prompt": redacted}}


class CursorDecisionSession(DurableDecisionSession):
    """One ACP session on the installed Cursor CLI.

    ``agent acp`` is the stdio JSON-RPC process for subscription model calls.
    Authentication is the existing ``cursor_login``. API keys are not forwarded.
    Ask mode returns a structured decision and does not execute task tools.
    """

    PROVIDER = "cursor"
    DEFAULT_MODEL = DEFAULT_CURSOR_MODEL

    def __init__(
        self,
        directory: Path,
        *,
        transport: NativeTransport | None = None,
        model: str | None = None,
        reasoning_effort: str = "xhigh",
        service_tier: str = "default",
    ):
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported native reasoning effort")
        if service_tier not in {"default", "fast"}:
            raise ValueError("Unsupported native service tier")
        self._model = self.resolve_model(model)
        self.spec = cursor_model_spec(self._model)
        if self.spec["reasoning_effort"] not in {None, reasoning_effort}:
            raise ValueError("Cursor reasoning_effort does not match the model id")
        if service_tier == "fast" and not self.spec["fast"]:
            raise ValueError(
                "Cursor fast tier requires a fast model id such as grok-4.7-xhigh-fast"
            )
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._state_path = self.directory / "session.json"
        if self._state_path.exists():
            self._state = self._load_state()
        else:
            self._state = {
                **self._identity(),
                "model_spec": self.spec,
                "session_id": None,
                "start": None,
                "creation_pending": False,
                "instructions_delivered": False,
                "requests": {},
            }
        self._transport = (
            transport if transport is not None else CursorStdioTransport(self.directory)
        )
        self._transport.set_observer(self._observe)
        self._connected = False
        self._context_generation = 0
        self._lock = asyncio.Lock()
        self._lease = None
        self._collecting = False
        self._chunks: list[str] = []
        self._tool_updates = 0

    @classmethod
    def resolve_model(cls, model: str | None) -> str:
        if model is None:
            return cls.DEFAULT_MODEL
        cursor_model_spec(model)
        return model

    def _observe(self, event: dict) -> None:
        message = event["message"]
        self._record(
            "wire", direction=event["direction"], message=_redact_wire(message)
        )
        if not self._collecting or event["direction"] != "incoming":
            return
        if message.get("method") != "session/update":
            return
        update = message.get("params", {}).get("update", {})
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            text = (update.get("content") or {}).get("text")
            if isinstance(text, str):
                self._chunks.append(text)
        elif kind in {"tool_call", "tool_call_update"}:
            self._tool_updates += 1

    async def _rpc(self, method: str, params: dict) -> dict:
        try:
            return await self._transport.request(method, params)
        except NativeTransportError:
            self._connected = False
            self._record("event_gap", operation=method)
            raise

    async def start(self, instructions: str, cwd: Path) -> None:
        async with self._lock:
            self._claim()
            resolved = str(Path(cwd).resolve())
            if self._state["start"] is None:
                self._state["start"] = {"cwd": resolved, "instructions": instructions}
                self._save()
            elif self._state["start"]["cwd"] != resolved:
                raise NativeProtocolError("Cursor changed the episode workspace")
            elif self._state["start"]["instructions"] != instructions:
                self._state["start"]["instructions"] = instructions
                self._state["instructions_delivered"] = False
                self._save()
            if not self._connected:
                await self._connect()

    async def _connect(self) -> None:
        await self._transport.close()
        await self._transport.connect(Path(self._state["start"]["cwd"]))
        initialized = await self._rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientInfo": {"name": "reweave", "version": __version__},
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                    "_meta": {"parameterizedModelPicker": True},
                },
            },
        )
        methods = [
            item.get("id")
            for item in initialized.get("authMethods", [])
            if isinstance(item, dict)
        ]
        if "cursor_login" not in methods:
            raise NativeAuthenticationError(
                "Cursor CLI did not offer the logged-in subscription"
            )
        try:
            await self._rpc("authenticate", {"methodId": "cursor_login"})
        except NativeRpcError as error:
            self._record("authentication_unavailable", code=error.error.get("code"))
            raise NativeAuthenticationError(
                "Cursor subscription login is not available"
            ) from error
        if self.session_id:
            opened = await self._rpc(
                "session/load",
                {
                    "sessionId": self.session_id,
                    "cwd": self._state["start"]["cwd"],
                    "mcpServers": [],
                },
            )
        elif self._state["creation_pending"]:
            raise NativeSubmissionUnknown("Cursor session creation was not confirmed")
        else:
            self._state["creation_pending"] = True
            self._save()
            try:
                opened = await self._rpc(
                    "session/new",
                    {"cwd": self._state["start"]["cwd"], "mcpServers": []},
                )
            except NativeTransportError as error:
                if not error.request_sent:
                    self._state["creation_pending"] = False
                    self._save()
                raise
            identity = opened.get("sessionId")
            if not isinstance(identity, str) or not identity:
                raise NativeProtocolError("Cursor did not return a session identity")
            self._state["session_id"] = identity
            self._state["creation_pending"] = False
            self._save()
        await self._select_model(opened.get("configOptions"))
        self._state["transport_metadata"] = self._transport.metadata
        self._state["authentication"] = "cursor_login"
        self._connected = True
        self._save()
        self._record("session_started", session_id=self.session_id)

    async def _select_model(self, options) -> None:
        if not isinstance(options, list):
            raise NativeProtocolError("Cursor did not return session configuration")
        wanted = {"model": self.spec["model"], "mode": "ask"}
        if self.spec["context"] is not None:
            wanted["context"] = self.spec["context"]
        wanted["reasoning_effort"] = self.reasoning_effort
        wanted["fast"] = "true" if self.spec["fast"] else "false"
        current = {item.get("id"): item for item in options if isinstance(item, dict)}
        for key, value in wanted.items():
            option = current.get(key)
            if option is None:
                raise NativeProtocolError(f"Cursor did not expose {key}")
            if option.get("currentValue") == value:
                continue
            allowed = {
                item.get("value")
                for item in option.get("options") or []
                if isinstance(item, dict)
            }
            if allowed and value not in allowed:
                raise NativeProtocolError(f"Cursor rejected {key}={value}")
            updated = await self._rpc(
                "session/set_config_option",
                {
                    "sessionId": self.session_id,
                    "configId": key,
                    "value": value,
                },
            )
            options = updated.get("configOptions")
            if not isinstance(options, list):
                raise NativeProtocolError(
                    "Cursor did not confirm session configuration"
                )
            current = {
                item.get("id"): item for item in options if isinstance(item, dict)
            }
            if current.get(key, {}).get("currentValue") != value:
                raise NativeProtocolError(f"Cursor did not apply {key}={value}")
        self._state["model_selection"] = {
            key: current[key]["currentValue"] for key in wanted if key in current
        }
        self._save()

    def _describe_inputs(
        self, text: str, images: tuple[Path, ...], output_schema: dict
    ) -> tuple[dict, list[tuple[str, bytes]]]:
        jsonschema.validators.validator_for(output_schema).check_schema(output_schema)
        stored = []
        blobs = []
        for index, image in enumerate(images):
            source = Path(image).resolve()
            mime = _MIME.get(source.suffix.lower())
            if mime is None or not source.is_file():
                raise NativeProtocolError(
                    "Cursor decision image is not a supported file"
                )
            data = source.read_bytes()
            name = f"{index}{source.suffix.lower()}"
            stored.append(
                {
                    "path": str(source),
                    "archive": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "mime": mime,
                }
            )
            blobs.append((name, data))
        return {
            "text": text,
            "images": stored,
            "output_schema": copy.deepcopy(output_schema),
        }, blobs

    def _archive_inputs(self, directory: Path, blobs: list[tuple[str, bytes]]) -> None:
        for name, data in blobs:
            target = directory / "images" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def _prompt(self, inputs: dict, directory: Path) -> list[dict]:
        blocks = []
        if not self._state["instructions_delivered"]:
            blocks.append(
                {
                    "type": "text",
                    "text": self._state["start"]["instructions"],
                }
            )
        blocks.append(
            {
                "type": "text",
                "text": (
                    "Return one JSON object and no other text. "
                    "The object must satisfy this JSON Schema:\n"
                    + json.dumps(inputs["output_schema"], ensure_ascii=False)
                    + "\n\n"
                    + inputs["text"]
                ),
            }
        )
        for image in inputs["images"]:
            archived = directory / "images" / image["archive"]
            blocks.append(
                {
                    "type": "image",
                    "mimeType": image["mime"],
                    "data": base64.b64encode(archived.read_bytes()).decode("ascii"),
                }
            )
        return blocks

    def _same_input(self, directory: Path, inputs: dict) -> bool:
        original = json.loads((directory / "input.json").read_text())
        if _canonical(original) != _canonical(inputs):
            return False
        return all(
            hashlib.sha256(
                (directory / "images" / item["archive"]).read_bytes()
            ).hexdigest()
            == item["sha256"]
            for item in original["images"]
        )

    async def decide(
        self, text: str, images: tuple[Path, ...], output_schema: dict, request_id: str
    ) -> dict:
        async with self._lock:
            self._claim()
            if not isinstance(request_id, str) or not request_id:
                raise NativeProtocolError(
                    "A native decision requires a nonempty request id"
                )
            if self.session_id is None:
                raise NativeProtocolError(
                    "Start the episode native session before deciding"
                )
            inputs, blobs = self._describe_inputs(text, images, output_schema)
            request = self._state["requests"].get(request_id)
            if request is None:
                if any(
                    item["phase"] == "submitting"
                    for item in self._state["requests"].values()
                ):
                    raise NativeSubmissionUnknown(
                        "An earlier native request still requires settlement"
                    )
                request = {
                    "request_id": request_id,
                    "directory": uuid.uuid4().hex,
                    "phase": "prepared",
                }
                directory = self.directory / "requests" / request["directory"]
                self._archive_inputs(directory, blobs)
                atomic_text(
                    directory / "input.json",
                    json.dumps(inputs, ensure_ascii=False, allow_nan=False) + "\n",
                )
                self._state["requests"][request_id] = request
                self._save()
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
            if request["phase"] == "invalid":
                error_class = (
                    NativeCompletedInvalidDecision
                    if request.get("invalid_effects_settled") is True
                    else NativeProtocolError
                )
                raise error_class("The original completed native decision was invalid")
            if request["phase"] == "failed":
                raise NativeTurnFailed(request["status"], request["error"])
            if request["phase"] == "submitting":
                raise NativeSubmissionUnknown(
                    "The original Cursor prompt may already have been sent"
                )
            if not self._connected:
                await self._connect()
            request["phase"] = "submitting"
            self._save()
            self._chunks = []
            self._tool_updates = 0
            self._collecting = True
            try:
                try:
                    response = await self._rpc(
                        "session/prompt",
                        {
                            "sessionId": self.session_id,
                            "prompt": self._prompt(inputs, directory),
                        },
                    )
                except NativeTransportError as error:
                    if not error.request_sent:
                        request["phase"] = "prepared"
                        self._save()
                    raise
                except NativeRpcError as error:
                    request.update(
                        phase="failed", status="rpcRejected", error=error.error
                    )
                    self._save()
                    raise
            finally:
                self._collecting = False
            stop = response.get("stopReason")
            if stop != "end_turn":
                request.update(
                    phase="failed",
                    status=str(stop),
                    error={"stopReason": stop},
                )
                self._save()
                raise NativeTurnFailed(str(stop), {"stopReason": stop})
            if self._tool_updates:
                request.update(
                    phase="invalid",
                    error={"reason": "tool_call"},
                )
                self._save()
                raise NativeProtocolError("Cursor model-only session invoked a tool")
            try:
                decision = _parse_decision(
                    "".join(self._chunks), inputs["output_schema"]
                )
            except NativeProtocolError as error:
                request.update(
                    phase="invalid",
                    status="completed",
                    invalid_effects_settled=True,
                    error={"reason": str(error)},
                )
                self._save()
                raise NativeCompletedInvalidDecision(str(error)) from error
            atomic_text(
                directory / "result.json",
                json.dumps(decision, ensure_ascii=False, allow_nan=False) + "\n",
            )
            request["phase"] = "completed"
            self._state["instructions_delivered"] = True
            self._save()
            self._record("decision_completed", request_id=request_id)
            return decision

    async def close(self) -> None:
        async with self._lock:
            try:
                await self._transport.close()
            finally:
                self._connected = False
                self._release()


def _parse_decision(text: str, schema: dict) -> dict:
    try:
        value = json.loads(_decision_text(text))
    except json.JSONDecodeError:
        raise NativeProtocolError("Cursor decision was not JSON") from None
    if not isinstance(value, dict):
        raise NativeProtocolError("Cursor decision must be a JSON object")
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as error:
        raise NativeProtocolError(
            "Cursor decision did not match the output schema"
        ) from error
    return value
