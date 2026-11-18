"""Persistent Codex decision turns; physical proposals are never executed here."""

from __future__ import annotations

import asyncio
import copy
import errno
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

import jsonschema

from reweave import __version__
from reweave.planner.codex_connection import codex_connection_config, codex_model_config

from .base import (
    DurableDecisionSession,
    NativeAuthenticationError,
    NativeCompletedInvalidDecision,
    NativeProtocolError,
    NativeRecoveryBlocked,
    NativeRpcError,
    NativeSubmissionUnknown,
    NativeTransportError,
    NativeTurnFailed,
    NativeTurnRecoveryRequired,
    _decision_text,
    _same_bytes,
    _write_json,
)
from .stdio import CodexStdioTransport
from .transport import NativeTransport


class CodexDecisionSession(DurableDecisionSession):
    """One durable episode thread using the installed app-server protocol.

    The artifact directory belongs to the Host, outside writable harness assets.
    Only one owner may operate it. Cached decisions remain readable after close.
    Lost submissions are read back, never replayed. A verified failed turn may
    permit a separately budgeted short continuation in this same thread.
    """

    PROVIDER = "codex"
    DEFAULT_MODEL = "gpt-5.4"
    MAX_CONTINUATIONS = 2
    GENERIC_BAD_REQUEST_CONTINUATIONS = 1
    RECOVERY_TEXT = (
        "The previous decision turn failed before returning a complete decision. Continue "
        "the pending decision using this session's existing observation and task "
        "context. Preserve existing workspace edits; do not repeat earlier native "
        "commands or task actions. No proposals from that failed turn were executed "
        "by the Host. Return one complete decision using the required output schema."
    )
    # Online task decisions keep native history and optional source revision,
    # without inheriting the CLI's default repository-development workflow.
    BASE_INSTRUCTIONS = (
        "You are the online decision model of a task harness. "
        "Follow the supplied task instructions and use the observations and feedback in this session. "
        "Return the requested structured decision directly. "
        "Do not perform repository onboarding, development preflight, skill discovery, "
        "or routine source/log inspection before decisions. "
        "Native file tools are available for a necessary, targeted harness repair only. "
        "All task effects must be proposed to the Host; never execute them from the shell."
    )

    def __init__(
        self,
        directory: Path,
        *,
        transport: NativeTransport | None = None,
        model: str | None = None,
        reasoning_effort: str = "medium",
        service_tier: str = "default",
        codex_auth_mode: str = "auth",
    ):
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported native reasoning effort")
        if service_tier not in {"default", "fast"}:
            raise ValueError("Unsupported native service tier")
        self._model = self.resolve_model(model)
        self.codex_auth_mode = codex_auth_mode
        self.model_provider = codex_connection_config(codex_auth_mode)["model_provider"]
        self.reasoning_effort, self.service_tier = reasoning_effort, service_tier
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._state_path = self.directory / "session.json"
        if self._state_path.exists():
            self._state = self._load_state()
        else:
            self._state = {
                **self._identity(),
                "session_id": None,
                "start": None,
                "creation_pending": False,
                "requests": {},
                "event_stream_complete": True,
            }
        self._transport = (
            transport
            if transport is not None
            else CodexStdioTransport(self.directory, codex_auth_mode=codex_auth_mode)
        )
        from reweave.utils.provider_admission import ProviderAdmission

        self._admission = (
            ProviderAdmission("codex-" + codex_auth_mode) if transport is None else None
        )
        self._rate_interrupts = set()
        self._rate_turns = set()
        self._transport.set_observer(self._observe)
        self._connected = False
        self._context_generation = 0
        self._lock = asyncio.Lock()
        self._lease = None

    def _identity(self) -> dict:
        return {**super()._identity(), "model_provider": self.model_provider}

    def _save(self) -> None:
        for request in self._state["requests"].values():
            attempts = request.setdefault("attempts", {})
            attempts[request["client_id"]] = {
                key: copy.deepcopy(request.get(key))
                for key in ("client_id", "turn_id", "phase", "status", "error")
            }
        _write_json(self._state_path, self._state)

    def _observe(self, event: dict) -> None:
        self._record("wire", **event)
        message = event.get("message", {})
        params = message.get("params", {})
        if (
            self._admission is not None
            and event.get("direction") == "incoming"
            and message.get("method") == "error"
        ):
            self._admission.note_quota_exhausted(params.get("error", {}))
            self._admission.note_transport_failure(params.get("error", {}))
        if (
            self._admission is not None
            and event.get("direction") == "incoming"
            and message.get("method") == "error"
            and self._admission.note_rate_limit(params.get("error", {}))
        ):
            turn_id = params.get("turnId")
            if turn_id and turn_id not in self._rate_turns:
                self._rate_turns.add(turn_id)
                task = asyncio.create_task(self._interrupt_rate_limited(turn_id))
                self._rate_interrupts.add(task)
                task.add_done_callback(self._rate_interrupts.discard)
        if (
            event.get("direction") == "incoming"
            and params.get("threadId") == self.session_id
            and params.get("turnId")
            and message.get("method") in {"error", "item/started", "item/completed"}
        ):
            active = [
                request
                for request in self._state["requests"].values()
                if request.get("turn_id") == params["turnId"]
                or (request["phase"] == "submitting" and request["turn_id"] is None)
            ]
            if len(active) == 1:
                evidence = (
                    active[0]
                    .setdefault("turn_evidence", {})
                    .setdefault(params["turnId"], {"errors": [], "items": {}})
                )
                if message["method"] == "error":
                    evidence["errors"].append(copy.deepcopy(params.get("error", {})))
                else:
                    item = params.get("item", {})
                    if isinstance(item.get("id"), str):
                        evidence["items"][item["id"]] = {
                            "type": item.get("type"),
                            "completed": message["method"] == "item/completed",
                        }
                self._save()
        if (
            event.get("direction") == "incoming"
            and message.get("method") == "thread/compacted"
            and self.session_id is not None
            and message.get("params", {}).get("threadId") == self.session_id
        ):
            self._context_generation += 1
        if (
            event.get("direction") == "incoming"
            and message.get("method") == "thread/started"
            and self._state["creation_pending"]
            and self.session_id is None
        ):
            thread = message.get("params", {}).get("thread", {})
            identity = thread.get("id")
            if isinstance(identity, str) and identity:
                self._state["session_id"] = identity
                self._save()

    async def _interrupt_rate_limited(self, turn_id):
        try:
            await asyncio.wait_for(
                self._rpc(
                    "turn/interrupt",
                    {
                        "threadId": self.session_id,
                        "turnId": turn_id,
                    },
                ),
                timeout=5,
            )
            self._record("rate_limit_interrupt_requested", turn_id=turn_id)
        except Exception as error:
            self._record(
                "rate_limit_interrupt_failed",
                turn_id=turn_id,
                error_type=type(error).__name__,
            )

    async def _rpc(self, method: str, params: dict) -> dict:
        try:
            return await self._transport.request(method, params)
        except NativeTransportError:
            self._connected = False
            self._state["event_stream_complete"] = False
            self._save()
            self._record("event_gap", operation=method)
            raise

    async def _connect(self) -> None:
        if self._connected:
            return
        await self._transport.close()
        await self._transport.connect(Path(self._state["start"]["cwd"]))
        initialized = await self._rpc(
            "initialize",
            {
                "clientInfo": {
                    "name": "reweave",
                    "version": __version__,
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await self._transport.notify("initialized", {})
        if self.codex_auth_mode == "auth":
            account = await self._rpc("account/read", {"refreshToken": False})
            if (account.get("account") or {}).get("type") != "chatgpt":
                self._record("authentication_unavailable", required="chatgpt")
                raise NativeAuthenticationError(
                    "Codex requires the existing local ChatGPT account"
                )
        self._state["transport_metadata"] = self._transport.metadata
        self._state["server_metadata"] = initialized
        self._state["authentication"] = (
            "local-chatgpt" if self.codex_auth_mode == "auth" else "api-key"
        )
        await self._disable_native_skills()
        self._connected = True
        self._save()

    async def _disable_native_skills(self) -> None:
        """Disable every discovered native skill in this episode's private home.

        The catalog budget does not disable discovery. Read the installed CLI's
        actual inventory and verify the resulting enablement without a model turn.
        Harness task-family experience travels in C and is unaffected.
        """
        cwd = self._state["start"]["cwd"]

        async def inventory():
            response = await self._rpc(
                "skills/list", {"cwds": [cwd], "forceReload": True}
            )
            rows = response.get("data")
            if not isinstance(rows, list) or len(rows) != 1:
                raise NativeProtocolError(
                    "Codex returned invalid native skill inventory"
                )
            skills = rows[0].get("skills")
            if (
                not isinstance(skills, list)
                or rows[0].get("errors")
                or any(
                    not isinstance(skill, dict)
                    or not isinstance(skill.get("path"), str)
                    or not Path(skill["path"]).is_absolute()
                    or type(skill.get("enabled")) is not bool
                    for skill in skills
                )
            ):
                raise NativeProtocolError(
                    "Codex returned invalid native skill inventory"
                )
            return skills

        skills = await inventory()
        disabled = [
            {"path": path, "enabled": False}
            for path in sorted({skill["path"] for skill in skills})
        ]
        if any(skill["enabled"] for skill in skills):
            native_home = self._transport.metadata.get("native_home")
            if not isinstance(native_home, str) or not Path(native_home).is_absolute():
                raise NativeProtocolError("Codex native runtime locator is unavailable")
            await self._rpc(
                "config/batchWrite",
                {
                    "filePath": str(Path(native_home) / "config.toml"),
                    "reloadUserConfig": True,
                    "edits": [
                        {
                            "keyPath": "skills.config",
                            "value": disabled,
                            "mergeStrategy": "replace",
                        }
                    ],
                },
            )
            if any(skill["enabled"] for skill in await inventory()):
                raise NativeProtocolError("Codex did not disable its native skills")
        self._state["disabled_native_skills"] = disabled

    def _request_service_tier(self) -> dict:
        """Read an operator-owned, experiment-scoped transport routing policy."""
        for parent in (self.directory, *self.directory.parents):
            path = parent / "native_request_policy.json"
            if not path.is_file():
                continue
            raw = path.read_bytes()
            policy = json.loads(raw)
            scope = Path(policy["scope_root"]).resolve()
            if scope != parent or not self.directory.is_relative_to(scope):
                raise NativeProtocolError(
                    "Native tier policy has a different experiment scope"
                )
            if (
                policy["model"] != self.model
                or policy["reasoning_effort"] != self.reasoning_effort
            ):
                raise NativeProtocolError(
                    "Native tier policy changes the frozen model or effort"
                )
            tier = policy["service_tier"]
            if tier not in {"default", "fast"} or not policy.get(
                "operator_authorization"
            ):
                raise NativeProtocolError(
                    "Native tier policy is not an authorized supported tier"
                )
            return {
                "service_tier": tier,
                "configured_service_tier": self.service_tier,
                "policy_path": str(path),
                "policy_sha256": hashlib.sha256(raw).hexdigest(),
            }
        return {
            "service_tier": self.service_tier,
            "configured_service_tier": self.service_tier,
        }

    def _thread_params(self) -> dict:
        tier = self._request_service_tier()["service_tier"]
        return {
            "model": self.model,
            "modelProvider": self.model_provider,
            "cwd": self._state["start"]["cwd"],
            "developerInstructions": self._state["start"]["instructions"],
            "approvalPolicy": "never",
            "sandbox": "workspace-write",
            "serviceTier": tier,
            "baseInstructions": self.BASE_INSTRUCTIONS,
            "config": {
                **codex_model_config(self.model),
                "features.fast_mode": tier == "fast",
                # The complete fixed episode contract is sent explicitly as
                # developerInstructions. Keep AGENTS.md as a readable audit
                # artifact without injecting a duplicate project-doc copy.
                "project_doc_max_bytes": 0,
                "project_root_markers": ["AGENTS.md"],
                # Native skill discovery is separate from the complete task
                # experience supplied by C and from native file-tool access.
                "skills.max_context_tokens": 1,
                "skills.config": self._state.get("disabled_native_skills", []),
            },
        }

    def _verify_thread(self, response: dict) -> None:
        identity = response.get("thread", {}).get("id")
        if not isinstance(identity, str) or not identity:
            raise NativeProtocolError("Codex did not return a thread identity")
        if self.session_id is not None and identity != self.session_id:
            raise NativeProtocolError("Codex resumed a different episode thread")
        # Preserve the returned identity even when an effective-model mismatch
        # makes it unusable. Retrying must never create another thread.
        self._state["session_id"] = identity
        self._save()
        if (
            response.get("model") != self.model
            or response.get("modelProvider") != self.model_provider
        ):
            raise NativeProtocolError("Codex changed the required model or provider")
        if Path(response.get("cwd", "")).resolve() != Path(self._state["start"]["cwd"]):
            raise NativeProtocolError("Codex changed the episode workspace")
        self._state["effective_thread"] = copy.deepcopy(response)
        self._state["creation_pending"] = False
        self._save()

    async def _resume(self) -> None:
        await self._connect()
        self._verify_thread(
            await self._rpc(
                "thread/resume", {"threadId": self.session_id, **self._thread_params()}
            )
        )
        self._record("session_resumed", session_id=self.session_id)

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
            self._state["start"] = start
            self._save()
            if self.session_id is not None:
                await self._resume()
                return
            if self._state["creation_pending"]:
                raise NativeSubmissionUnknown(
                    "Original native thread creation was not confirmed"
                )
            await self._connect()
            self._state["creation_pending"] = True
            self._save()
            try:
                response = await self._rpc(
                    "thread/start",
                    {
                        **self._thread_params(),
                        "developerInstructions": instructions,
                        "ephemeral": False,
                        "dynamicTools": [],
                        "allowProviderModelFallback": False,
                    },
                )
            except NativeTransportError as error:
                if not error.request_sent:
                    self._state["creation_pending"] = False
                    self._save()
                raise
            except NativeRpcError:
                self._state["creation_pending"] = False
                self._save()
                raise
            self._verify_thread(response)
            self._record("session_started", session_id=self.session_id)

    def _managed_image(self, source: Path) -> bool:
        """Only the episode's indexed Host-owned media pool is append-only."""
        pool = self.directory.parent / "records/images"
        if (
            pool.is_symlink()
            or source.is_symlink()
            or source.parent != pool.absolute()
            or source.stat().st_mode & 0o222
        ):
            return False
        index = pool / "records.json"
        if not index.is_file() or index.is_symlink():
            return False
        records = json.loads(index.read_text())
        return any(
            record["path"] == source.name
            and record["bytes"] == source.stat().st_size
            and source.name.startswith(record["artifact_id"] + ".")
            for record in records
        )

    def _archive_image(self, source: Path, target: Path) -> None:
        """Link immutable Host media; snapshot writable external caller inputs.

        Linking any external source, even read-only, would let its owner rewrite
        the original submitted evidence. Only the episode's indexed native media
        pool is Host-managed and append-only. A different filesystem needs a
        separate byte snapshot.
        """
        if self._managed_image(source):
            try:
                os.link(source, target)
                return
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _submission_image_path(self, directory: Path, image: dict, index: int) -> Path:
        # The native process can read its Host-owned launcher directory, but
        # cannot read request audits. Expose only the submitted immutable media
        # through that existing read-only boundary, including on session resume.
        source = self._image_path(directory, image, index)
        target = self.directory / "launcher/images" / directory.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except FileExistsError:
            if not _same_bytes(source, target):
                raise NativeProtocolError(
                    "Native submission image differs from its archived input"
                ) from None
        return target

    async def decide(
        self, text: str, images: tuple[Path, ...], output_schema: dict, request_id: str
    ) -> dict:
        async with self._lock, AsyncExitStack() as admission_stack:
            self._claim()
            if not isinstance(request_id, str) or not request_id:
                raise NativeProtocolError(
                    "A native decision requires a nonempty request id"
                )
            if self.session_id is None:
                raise NativeProtocolError(
                    "Start the episode native session before deciding"
                )
            inputs = self._input(text, images, output_schema)
            request = self._state["requests"].get(request_id)
            if request is None:
                # Never submit a later request while an earlier turn may exist.
                if any(
                    r["phase"]
                    in {
                        "submitting",
                        "submitted",
                        "recovery_prepared",
                        "recovery_blocked",
                    }
                    for r in self._state["requests"].values()
                ):
                    raise NativeSubmissionUnknown(
                        "An earlier native request still requires settlement"
                    )
                request = {
                    "request_id": request_id,
                    "directory": uuid.uuid4().hex,
                    "client_id": str(uuid.uuid4()),
                    "phase": "prepared",
                    "turn_id": None,
                    "agent_messages": [],
                    "service_tier_routing": self._request_service_tier(),
                }
                directory = self.directory / "requests" / request["directory"]
                for index, image in enumerate(inputs["images"]):
                    target = self._image_path(directory, image, index)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self._archive_image(Path(image["path"]), target)
                _write_json(directory / "input.json", inputs)
                self._state["requests"][request_id] = request
                self._save()
                self._record(
                    "decision_prepared",
                    request_id=request_id,
                    request_directory=request["directory"],
                )
            else:
                directory = self._request_directory(request_id)
                if not self._same_input(directory, inputs):
                    raise NativeProtocolError(
                        "Native request id was reused with different input"
                    )
            if request["phase"] == "completed":
                return copy.deepcopy(
                    json.loads(
                        (
                            self._request_directory(request_id) / "result.json"
                        ).read_text()
                    )
                )
            if request["phase"] == "failed":
                if request.get("status") == "rpcRejected":
                    raise NativeRpcError("turn/start", request["error"])
                if not self._connected:
                    await self._resume()
                return await self._settle(
                    request,
                    inputs,
                    json.loads((directory / "native_completion.json").read_text()),
                )
            if request["phase"] == "recovery_blocked":
                raise NativeRecoveryBlocked(
                    request.get("error"),
                    reason=request["recovery_reason"],
                    continuations=request.get("continuations", 0),
                )
            if request["phase"] == "invalid":
                error_class = (
                    NativeCompletedInvalidDecision
                    if request.get("status") == "completed"
                    and request.get("invalid_effects_settled") is True
                    else NativeProtocolError
                )
                raise error_class("The original completed native decision was invalid")
            if request["phase"] == "recovery_prepared" and self._connected:
                # Reset the owned CLI's stream/cache state only after a failed
                # turn was verified and the Host granted a charged continuation.
                # Resuming below keeps the same durable thread and all history.
                try:
                    await self._transport.close()
                finally:
                    self._connected = False
                self._record("recovery_connection_reset", request_id=request_id)
            if not self._connected:
                await self._resume()
            if request["phase"] == "recovery_prepared":
                request.update(
                    client_id=str(uuid.uuid4()),
                    turn_id=None,
                    phase="prepared",
                    agent_messages=[],
                    continuations=request.get("continuations", 0) + 1,
                )
                request.pop("status", None)
                request.pop("error", None)
                self._save()
            if request["phase"] in {"submitting", "submitted"}:
                turn = await self._recover_turn(request)
                if turn["status"] != "inProgress":
                    return await self._settle(request, inputs, turn)
            else:
                if self._admission is not None:
                    await admission_stack.enter_async_context(
                        self._admission.lease(
                            self.directory / "provider_admission.json"
                        )
                    )
                request["phase"] = "submitting"
                self._save()
                try:
                    response = await self._rpc(
                        "turn/start",
                        {
                            "threadId": self.session_id,
                            "model": self.model,
                            "effort": self.reasoning_effort,
                            "serviceTier": request.get("service_tier_routing", {}).get(
                                "service_tier", self.service_tier
                            ),
                            "clientUserMessageId": request["client_id"],
                            "input": [{"type": "text", "text": self.RECOVERY_TEXT}]
                            if request.get("continuations", 0)
                            else [{"type": "text", "text": text}]
                            + [
                                {
                                    "type": "localImage",
                                    "path": str(
                                        self._submission_image_path(
                                            self._request_directory(request_id),
                                            image,
                                            index,
                                        )
                                    ),
                                }
                                for index, image in enumerate(inputs["images"])
                            ],
                            "outputSchema": inputs["output_schema"],
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
                turn = response.get("turn", {})
                if not isinstance(turn.get("id"), str):
                    raise NativeProtocolError(
                        "Codex turn submission did not return an identity"
                    )
                request.update(turn_id=turn["id"], phase="submitted")
                self._save()
                self._record(
                    "decision_submitted", request_id=request_id, turn_id=turn["id"]
                )
                if turn.get("status") != "inProgress":
                    return await self._settle(request, inputs, turn)
            return await self._wait(request, inputs)

    async def _recover_turn(self, request: dict) -> dict:
        summary = await self._rpc("thread/read", {"threadId": self.session_id})
        if summary.get("thread", {}).get("id") != self.session_id:
            raise NativeProtocolError("Codex read returned a different episode thread")
        turns = []
        cursor = None
        seen = set()
        while True:
            params = {"threadId": self.session_id, "itemsView": "full", "limit": 100}
            if cursor is not None:
                params["cursor"] = cursor
            page = await self._rpc("thread/turns/list", params)
            turns.extend(page.get("data", []))
            cursor = page.get("nextCursor")
            if cursor is None:
                break
            if cursor in seen:
                raise NativeProtocolError(
                    "Codex returned a repeated turn-history cursor"
                )
            seen.add(cursor)
        matches = [
            turn
            for turn in turns
            if (request["turn_id"] is not None and turn.get("id") == request["turn_id"])
            or (
                request["turn_id"] is None
                and any(
                    item.get("type") == "userMessage"
                    and item.get("clientId") == request["client_id"]
                    for item in turn.get("items", [])
                )
            )
        ]
        if len(matches) != 1:
            self._record(
                "submission_unresolved",
                request_id=request["request_id"],
                matches=len(matches),
            )
            raise NativeSubmissionUnknown(
                "Original native request could not be uniquely identified"
            )
        turn = matches[0]
        request["turn_id"] = turn["id"]
        if (
            request["phase"] in {"submitting", "submitted"}
            or turn.get("status") == "inProgress"
        ):
            request["phase"] = "submitted"
        self._save()
        self._record(
            "decision_recovered", request_id=request["request_id"], turn_id=turn["id"]
        )
        return turn

    async def _wait(self, request: dict, inputs: dict) -> dict:
        while True:
            try:
                event = await self._transport.next_event()
            except NativeTransportError:
                self._connected = False
                self._state["event_stream_complete"] = False
                self._save()
                self._record(
                    "event_gap",
                    operation="turn/events",
                    request_id=request["request_id"],
                )
                raise
            params = event.get("params", {})
            if params.get("threadId") != self.session_id:
                continue
            method = event.get("method")
            if method == "thread/tokenUsage/updated":
                self._state["token_usage"] = params.get("tokenUsage")
                self._save()
            if (
                params.get("turnId") == request["turn_id"]
                and method == "item/completed"
            ):
                item = params.get("item", {})
                if item.get("type") == "agentMessage":
                    request["agent_messages"].append(item)
                    self._save()
            if (
                method == "turn/completed"
                and params.get("turn", {}).get("id") == request["turn_id"]
            ):
                return await self._settle(request, inputs, params["turn"])

    @staticmethod
    def _error_variant(error: dict) -> tuple[str, int | None]:
        info = error.get("codexErrorInfo")
        if isinstance(info, str):
            return info.lower(), None
        if isinstance(info, dict) and len(info) == 1:
            name, detail = next(iter(info.items()))
            return name.lower(), detail.get("httpStatusCode") if isinstance(
                detail, dict
            ) else None
        return "", None

    def _recovery_reason(self, request: dict, turn: dict) -> str | None:
        if turn.get("status") != "failed":
            return None
        error = turn.get("error") or {}
        kind, status = self._error_variant(error)
        stream_kinds = {
            "responsestreamdisconnected",
            "responsestreamconnectionfailed",
            "responsetoomanyfailedattempts",
        }
        if kind in stream_kinds and (
            status is None or status in {408, 429} or status >= 500
        ):
            return "provider_stream_failure"
        if kind == "httpconnectionfailed" and (
            status is None or status in {408, 429} or status >= 500
        ):
            return "provider_http_connection_failure"
        if kind in {"serveroverloaded", "internalservererror"}:
            return "provider_service_failure"
        if kind == "other":
            message = error.get("message", "")
            from reweave.utils.provider_admission import is_native_transport_failure

            if message == "request timed out":
                return "provider_request_timeout"
            if is_native_transport_failure(error):
                return "provider_untyped_stream_transport_failure"
            compact_prefix = "Error running remote compact task: "
            compact = isinstance(message, str) and message.startswith(compact_prefix)
            evidence = request.get("turn_evidence", {}).get(turn["id"], {})
            stream_failure = any(
                self._error_variant(previous)[0] in stream_kinds
                and (
                    self._error_variant(previous)[1] is None
                    or self._error_variant(previous)[1] in {408, 429}
                    or self._error_variant(previous)[1] >= 500
                )
                for previous in evidence.get("errors", [])
            )
            if compact:
                message = message.removeprefix(compact_prefix)
                # Native compaction wraps a typed stream error as `other`.
                # Require matching turn evidence, rather than guessing from an
                # arbitrary error string or an earlier turn's disconnection.
                if stream_failure and message.startswith(
                    "stream disconnected before completion: "
                ):
                    return "provider_compaction_stream_failure"
            try:
                generic_bad_request = json.loads(message) == {"detail": "Bad Request"}
            except (ValueError, TypeError):
                generic_bad_request = False
            if generic_bad_request:
                return (
                    "stream_failure_then_generic_bad_request_root_cause_unknown"
                    if stream_failure
                    else "generic_bad_request_root_cause_unknown"
                )
        return None

    def _native_effects(self, request: dict, turn: dict) -> dict:
        # Compaction changes only provider-managed conversation state. A failed
        # compaction is not an unsettled file/tool/physical effect; every
        # recovery still verifies the failed turn and original current input.
        passive = {
            "userMessage",
            "agentMessage",
            "reasoning",
            "plan",
            "contextCompaction",
        }
        observed = request.get("turn_evidence", {}).get(turn["id"], {}).get("items", {})
        effects = {
            identity: {
                "type": item.get("type"),
                "settled": item.get("completed", False),
            }
            for identity, item in observed.items()
            if item.get("type") not in passive
        }
        for item in turn.get("items", []):
            if item.get("type") not in passive:
                effects[item.get("id", "unknown")] = {
                    "type": item.get("type"),
                    "settled": item.get("status")
                    in {"completed", "failed", "declined"},
                }
        return effects

    @staticmethod
    def _has_current_input(request: dict, turn: dict) -> bool:
        return any(
            item.get("type") == "userMessage"
            and item.get("clientId") == request["client_id"]
            for item in turn.get("items", [])
        )

    async def _settle(self, request: dict, inputs: dict, turn: dict) -> dict:
        try:
            return self._finish(request, inputs, turn)
        except NativeTurnFailed as failure:
            if self._recovery_reason(request, turn) is None:
                raise
            # A failed notification can omit items. Verify the durable turn,
            # including native command settlement, before authorizing sampling.
            verified = await self._recover_turn(request)
            if verified.get("status") == "inProgress":
                return await self._wait(request, inputs)
            if verified.get("status") == "completed":
                return self._finish(request, inputs, verified)
            if verified.get("itemsView") != "full" or not isinstance(
                verified.get("items"), list
            ):
                raise NativeProtocolError(
                    "Recovery requires the failed turn's complete native item history"
                )
            attempt_dir = (
                self._request_directory(request["request_id"])
                / "attempts"
                / request["client_id"]
            )
            _write_json(attempt_dir / "verified_completion.json", verified)
            reason = self._recovery_reason(request, verified)
            if reason is None:
                return self._finish(request, inputs, verified)
            effects = self._native_effects(request, verified)
            continuations = request.get("continuations", 0)
            # A bare 400 has no established transport cause. Permit one audited
            # same-session continuation after verifying retained input and all
            # effects, but do not treat a persistent invalid request as transient.
            default_limit = (
                self.GENERIC_BAD_REQUEST_CONTINUATIONS
                if reason == "generic_bad_request_root_cause_unknown"
                else self.MAX_CONTINUATIONS
            )
            blocked = (
                "unsettled_native_effects"
                if any(not item["settled"] for item in effects.values())
                else "missing_current_input"
                if not self._has_current_input(request, verified)
                else "continuations_exhausted"
                if continuations >= request.get("continuation_limit", default_limit)
                else None
            )
            request.update(
                phase="recovery_blocked" if blocked else "recovery_prepared",
                status="failed",
                error=verified.get("error"),
                recovery_reason=blocked or reason,
                continuations=continuations,
            )
            self._save()
            self._record(
                "decision_recovery_blocked"
                if blocked
                else "decision_recovery_required",
                request_id=request["request_id"],
                turn_id=verified["id"],
                continuation=continuations + 1,
                reason=blocked or reason,
                native_effects=effects,
            )
            if blocked:
                raise NativeRecoveryBlocked(
                    verified.get("error"), reason=blocked, continuations=continuations
                ) from failure
            raise NativeTurnRecoveryRequired(
                verified.get("error"), reason=reason, continuation=continuations + 1
            ) from failure

    def recovery_status(self, request_id: str) -> dict:
        """Return durable recovery identity without sampling or provider RPC."""
        request = self._state["requests"].get(request_id)
        if request is None:
            raise NativeProtocolError("Recovery request is not part of this session")
        return {
            "session_id": self.session_id,
            "request_id": request_id,
            "turn_id": request["turn_id"],
            "phase": request["phase"],
            "continuations": request.get("continuations", 0),
            "reason": request.get("recovery_reason"),
        }

    async def authorize_recovery(
        self, request_id: str, turn_id: str, nonce: str, *, session_id: str
    ) -> dict:
        """Admit one explicit, budgeted continuation; never replay an unknown turn.

        An operator's exact identity/nonce token may follow a verified interrupted
        turn or an external repair, including an authentication repair. It does
        not bypass native effects settlement, manufacture a missing submission or
        reset cumulative counts.
        """
        async with self._lock:
            self._claim()
            status = self.recovery_status(request_id)
            request = self._state["requests"][request_id]
            if session_id != self.session_id or turn_id != request["turn_id"]:
                raise NativeProtocolError(
                    "Recovery authorization identity does not match"
                )
            if not isinstance(nonce, str) or not nonce or len(nonce) > 256:
                raise NativeProtocolError(
                    "Recovery authorization requires a bounded nonce"
                )
            if nonce in request.get("recovery_nonces", []):
                return {
                    **status,
                    "authorized": False,
                    "will_submit": request["phase"] == "recovery_prepared"
                    and request.get("authorization_nonce") == nonce,
                }
            if request["phase"] == "completed":
                result = json.loads(
                    (self._request_directory(request_id) / "result.json").read_text()
                )
                return {
                    **status,
                    "authorized": False,
                    "will_submit": False,
                    "result": result,
                }
            if request["phase"] not in {"failed", "recovery_blocked"} or not turn_id:
                raise NativeSubmissionUnknown(
                    "Only a known failed turn permits manual recovery"
                )
            if not self._connected:
                await self._resume()
            verified = await self._recover_turn(request)
            inputs = json.loads(
                (self._request_directory(request_id) / "input.json").read_text()
            )
            if verified.get("status") == "completed":
                result = self._finish(request, inputs, verified)
                return {
                    **self.recovery_status(request_id),
                    "authorized": False,
                    "will_submit": False,
                    "result": result,
                }
            if verified.get("status") == "inProgress":
                result = await self._wait(request, inputs)
                return {
                    **self.recovery_status(request_id),
                    "authorized": False,
                    "will_submit": False,
                    "result": result,
                }
            if (
                verified.get("status") not in {"failed", "interrupted"}
                or verified.get("itemsView") != "full"
                or not isinstance(verified.get("items"), list)
            ):
                raise NativeProtocolError(
                    "Manual recovery requires a verified terminal turn with full history"
                )
            effects = self._native_effects(request, verified)
            blocked = (
                "unsettled_native_effects"
                if any(not item["settled"] for item in effects.values())
                else "missing_current_input"
                if not self._has_current_input(request, verified)
                else None
            )
            if blocked:
                request.update(phase="recovery_blocked", recovery_reason=blocked)
                self._save()
                raise NativeRecoveryBlocked(
                    verified.get("error"),
                    reason=blocked,
                    continuations=request.get("continuations", 0),
                )
            request.setdefault("recovery_nonces", []).append(nonce)
            request.update(
                phase="recovery_prepared",
                recovery_reason="explicit_operator_authorization",
                continuation_limit=request.get("continuations", 0) + 1,
                authorization_nonce=nonce,
            )
            self._save()
            self._record(
                "decision_recovery_authorized",
                request_id=request_id,
                turn_id=turn_id,
                nonce=nonce,
                native_effects=effects,
            )
            return {
                **self.recovery_status(request_id),
                "authorized": True,
                "will_submit": True,
            }

    def _finish(self, request: dict, inputs: dict, turn: dict) -> dict:
        directory = self._request_directory(request["request_id"])
        attempt_completion = (
            directory / "attempts" / request["client_id"] / "native_completion.json"
        )
        if not attempt_completion.exists():
            _write_json(attempt_completion, turn)
        _write_json(directory / "native_completion.json", turn)
        status = turn.get("status")
        messages = {
            item["id"]: item
            for item in [*request["agent_messages"], *turn.get("items", [])]
            if item.get("type") == "agentMessage" and isinstance(item.get("id"), str)
        }
        finals = [
            item for item in messages.values() if item.get("phase") == "final_answer"
        ]
        if not finals:
            finals = [item for item in messages.values() if item.get("phase") is None]
        observed = request.get("turn_evidence", {}).get(turn["id"], {}).get("items", {})
        usage_error = (
            status == "failed"
            and (turn.get("error") or {}).get("message")
            == "stream disconnected before completion: failed to parse ResponseCompleted: missing field `reasoning_tokens`"
            and self._state.get("event_stream_complete") is True
            and bool(finals)
            and observed.get(finals[-1].get("id"), {}).get("completed") is True
            and not self._native_effects(request, turn)
        )
        if status in {"failed", "interrupted"}:
            request.update(phase="failed", status=status, error=turn.get("error"))
            self._save()
            if not usage_error:
                raise NativeTurnFailed(status, turn.get("error"))
        elif status != "completed":
            raise NativeProtocolError("Codex returned an unknown terminal turn status")
        try:
            if not finals:
                raise ValueError("No final native assistant message was provided")
            result = json.loads(_decision_text(finals[-1]["text"]))
            if not isinstance(result, dict):
                raise ValueError("A native decision must be a JSON object")
            jsonschema.validate(result, inputs["output_schema"])
        except (ValueError, KeyError, jsonschema.ValidationError) as error:
            if usage_error:
                raise NativeTurnFailed(status, turn.get("error")) from error
            effects = self._native_effects(request, turn)
            # A completion notification may contain only a summary.  In that
            # case an empty item list is evidence of no effects only if the
            # event stream was continuous and observed this turn's final item.
            observed = (
                request.get("turn_evidence", {}).get(turn["id"], {}).get("items", {})
            )
            full_history = turn.get("itemsView") == "full" and isinstance(
                turn.get("items"), list
            )
            final_observed = (
                bool(finals)
                and observed.get(finals[-1].get("id"), {}).get("completed") is True
            )
            complete_evidence = full_history or (
                self._state.get("event_stream_complete") is True and final_observed
            )
            settled = complete_evidence and not effects
            request.update(
                phase="invalid", status="completed", invalid_effects_settled=settled
            )
            self._save()
            error_class = (
                NativeCompletedInvalidDecision if settled else NativeProtocolError
            )
            raise error_class(
                "Completed native output did not satisfy the decision schema"
            ) from error
        _write_json(directory / "result.json", result)
        request.update(phase="completed", status=status, final_item_id=finals[-1]["id"])
        if usage_error:
            request["usage_warning"] = turn["error"]
            self._state["token_usage"] = None
            self._state["token_usage_incomplete"] = True
            self._record(
                "usage_metadata_unavailable",
                request_id=request["request_id"],
                turn_id=turn["id"],
            )
        self._save()
        self._record(
            "decision_completed",
            request_id=request["request_id"],
            turn_id=request["turn_id"],
        )
        return copy.deepcopy(result)

    async def cancel(self) -> None:
        """Interrupt only the known active turn; never turn cancellation into replay."""
        active = [
            request
            for request in self._state["requests"].values()
            if request["phase"] in {"submitting", "submitted"}
        ]
        if not active:
            return
        request = active[0]
        if request["turn_id"] is None:
            raise NativeSubmissionUnknown(
                "Cannot interrupt an unidentified native turn"
            )
        if not self._connected:
            await self._resume()
        await self._rpc(
            "turn/interrupt",
            {"threadId": self.session_id, "turnId": request["turn_id"]},
        )
        self._record(
            "interrupt_requested",
            request_id=request["request_id"],
            turn_id=request["turn_id"],
        )

    async def close(self) -> None:
        for task in self._rate_interrupts:
            task.cancel()
        await asyncio.gather(*self._rate_interrupts, return_exceptions=True)
        try:
            await self._transport.close()
        finally:
            self._connected = False
            # Closing the transport wakes an outstanding decide(). Let it record
            # its submission boundary before another process can claim the state.
            async with self._lock:
                self._release()
