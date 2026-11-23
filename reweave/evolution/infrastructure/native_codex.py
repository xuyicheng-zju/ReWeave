# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Codex app-server backend for durable native author and analysis sessions."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reweave.evolution.authoring.workspace import native_output
from reweave.evolution.contracts import Usage
from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.evolution.infrastructure.native_accounting import (
    complete_query,
    cumulative_usage,
)
from reweave.evolution.infrastructure.native_retry import native_retry_wait_s
from reweave.evolution.infrastructure.native_tool_failure import (
    NativeToolStartupError,
    tool_startup_failure,
)
from reweave.evolution.infrastructure.session import (
    CodexSessionStore,
    native_event_field,
    serialize_native_event,
)
from reweave.planner.providers.base import NativeTransportError
from reweave.utils.persistence import atomic_json
from reweave.utils.provider_admission import NativeQuotaExhausted


async def _interrupt_codex_turn(turn):
    """A completed-turn race must not hide the original 429 or lose its journal."""
    from openai_codex.errors import InvalidRequestError

    try:
        await asyncio.wait_for(turn.interrupt(), timeout=5)
        return True
    except InvalidRequestError as error:
        if error.code == -32600 and error.message == "no active turn to interrupt":
            return False
        raise


@dataclass
class _TurnOutcome:
    status: Any = None
    error: Any = None
    rate_limited: bool = False
    tool_startup_failed: bool = False
    quota_exhausted: bool = False
    transport_reported: bool = False


async def _consume_turn(turn, record, admission, outcome: _TurnOutcome) -> None:
    async for event in turn.stream():
        record(event)
        outcome.tool_startup_failed |= tool_startup_failure(event)
        if native_event_field(event, "method") == "error" and not outcome.rate_limited:
            error = serialize_native_event(
                native_event_field(native_event_field(event, "payload"), "error")
            )
            if admission.note_quota_exhausted(error):
                outcome.quota_exhausted = True
            elif admission.note_rate_limit(error):
                outcome.rate_limited = True
                await _interrupt_codex_turn(turn)
            elif not outcome.transport_reported:
                outcome.transport_reported = admission.note_transport_failure(error)
        if native_event_field(event, "method") == "turn/completed":
            completed = native_event_field(native_event_field(event, "payload"), "turn")
            outcome.status = native_event_field(completed, "status")
            outcome.error = native_event_field(completed, "error")


async def _read_transcript(thread) -> Path:
    response = await thread.read(include_turns=True)
    if response.thread.id != thread.id or not response.thread.path:
        raise ValueError("native Codex transcript identity is missing")
    return Path(response.thread.path)


async def _capture(store: CodexSessionStore, thread_id: str, transcript: Path) -> None:
    await run_blocking(store.bind, thread_id)
    await run_blocking(store.capture, transcript)


def _drain_codex_supervisor(sync_client, receipt: Path) -> None:
    """Wait on the SDK-owned process, never a later process with the same PID."""
    process = sync_client._proc
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    # The SDK itself kills this trusted supervisor after only two seconds.
    # Its child drain and final receipt fsync must finish before SDK.close().
    process.wait()
    value = json.loads(receipt.read_text())
    if value.get("pid") != process.pid or value.get("descendants_drained") is not True:
        raise RuntimeError("native Codex supervisor exited without its confirmed drain")


async def _close_codex_client(client, directory: Path) -> None:
    """Let the trusted Linux owner settle before the SDK's short close timeout."""

    async def close():
        sync_client = getattr(getattr(client, "_client", None), "_sync", None)
        if sys.platform == "linux" and sync_client is not None:
            await run_blocking(
                _drain_codex_supervisor,
                sync_client,
                directory / "launcher/process.json",
            )
        await client.close()

    cleanup = asyncio.create_task(close())
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()
        raise


async def _settle_cancelled_turn(
    turn, stream, thread, store, client, directory
) -> None:
    """Keep the SDK reader alive until an interrupted turn actually settles."""

    async def settle():
        try:
            await _interrupt_codex_turn(turn)
            await stream
            await _capture(store, thread.id, await _read_transcript(thread))
        finally:
            await _close_codex_client(client, directory)
            await asyncio.gather(stream, return_exceptions=True)

    cleanup = asyncio.create_task(settle())
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    cleanup.result()


class CodexSession:
    """Codex query backend of NativeSession: one thread per durable job."""

    async def _codex(self, workspace, directory, *args, **kwargs):
        from reweave.utils.provider_admission import ProviderAdmission

        admission = ProviderAdmission(
            "codex-" + self.policy.codex_auth_mode, lane="offline"
        )
        reconnects = 0
        while True:
            admitted = False
            try:
                async with admission.lease(directory / "audit/provider_admission.json"):
                    admitted = True
                    return await self._codex_query(
                        workspace, directory, *args, admission=admission, **kwargs
                    )
            except NativeTransportError as error:
                if admitted or error.request_sent is not False:
                    raise
                # A gate replacement can close queued sockets before any native
                # query exists. Keep waiting in this job; submitted turns retain
                # their separate, finite same-session recovery budget.
                reconnects += 1
                wait_s = native_retry_wait_s(min(reconnects, 6))
                atomic_json(
                    directory
                    / "audit/provider_admission_reconnects"
                    / f"{uuid.uuid4().hex}.json",
                    {
                        "timestamp": time.time(),
                        "error": str(error),
                        "request_sent": False,
                        "retry_wait_s": wait_s,
                    },
                )
                await asyncio.sleep(wait_s)

    async def _open_codex_thread(self, client, store, settings, workspace, directory):
        """Resume the bound thread or start one new author/analyst thread."""
        if (directory / "audit/inputs/inquiry_target.json").exists():
            raise ValueError(
                "session inquiry is retired; historical results remain readable"
            )
        identity_path = directory / "session/identity.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())["session_id"]
            await run_blocking(store.bind, identity)
            thread = await client.thread_resume(identity, **settings)
            before = await thread.read(include_turns=True)
            if before.thread.model_provider != settings["model_provider"]:
                raise ValueError(
                    "native author provider differs from its existing session"
                )
            if before.thread.id != thread.id or not before.thread.path:
                raise ValueError(
                    "native Codex transcript identity is missing before query"
                )
            await run_blocking(store.capture, Path(before.thread.path))
            return thread
        if (
            any((workspace / ".scratch/runtime/codex").rglob("*.jsonl"))
            or store.path.exists()
        ):
            raise ValueError(
                "native Codex evidence exists without a recoverable identity; refusing a new thread"
            )
        # A fresh thread is materialized by its first user message.
        return await client.thread_start(ephemeral=False, **settings)

    async def _codex_query(
        self,
        workspace,
        directory,
        readable,
        writable,
        record,
        started,
        query_path,
        begin_query,
        instructions,
        native_prompt,
        development_url,
        admission,
    ):
        import openai_codex
        from openai_codex.generated.v2_all import ReasoningEffort

        from reweave.planner.codex_connection import (
            codex_connection_config,
            create_codex_client,
        )

        client = create_codex_client(
            workspace=workspace,
            output=native_output(workspace),
            control=directory / "launcher",
            mcp_servers={"author": development_url} if development_url else {},
            codex_auth_mode=self.policy.codex_auth_mode,
            model=self.policy.model,
            environment={"REWEAVE_AUTHOR_TOOLS_URL": development_url}
            if development_url
            else {},
            readable_roots=readable,
            writable_roots=writable,
            native_runtime_root=self.native_runtime_root,
            resume=(directory / "session/identity.json").exists(),
        )
        store = CodexSessionStore(directory / "session")
        try:
            settings = {
                "approval_mode": openai_codex.ApprovalMode.deny_all,
                "cwd": str(workspace),
                "developer_instructions": instructions,
                "model": self.policy.model,
                "model_provider": codex_connection_config(self.policy.codex_auth_mode)[
                    "model_provider"
                ],
                "sandbox": openai_codex.Sandbox.full_access,
            }
            thread = await self._open_codex_thread(
                client, store, settings, workspace, directory
            )
            begin_query()
            atomic_json(directory / "session/identity.json", {"session_id": thread.id})
            turn = await thread.turn(
                native_prompt,
                model=self.policy.model,
                effort=ReasoningEffort(self.policy.reasoning_effort),
            )
            outcome = _TurnOutcome()
            # SDK streaming waits on a blocking notification queue. Cancelling
            # its coroutine unregisters that queue before the worker wakes up.
            stream = asyncio.create_task(
                _consume_turn(turn, record, admission, outcome)
            )
            try:
                await asyncio.shield(stream)
            except asyncio.CancelledError:
                await _settle_cancelled_turn(
                    turn, stream, thread, store, client, directory
                )
                raise
            transcript = await _read_transcript(thread)
            usage = Usage(
                physical_steps=0,
                model_calls=None,
                wall_time_complete=False,
                wall_time_s=time.monotonic() - started,
                cost_usd=None,
            )
            atomic_json(query_path, usage.model_dump(mode="json"))
            usage = cumulative_usage(directory / "audit")
            self._update_usage(directory)
            await self._settle_codex_turn(
                outcome, admission, store, thread.id, transcript, directory
            )
            await run_blocking(
                complete_query, directory, usage, thread.id, store, transcript
            )
            return usage, thread.id
        finally:
            await _close_codex_client(client, directory)

    @staticmethod
    async def _settle_codex_turn(
        outcome: _TurnOutcome, admission, store, thread_id, transcript, directory
    ) -> None:
        """Retain the transcript of every unfinished turn, then raise its cause."""
        if outcome.quota_exhausted:
            await _capture(store, thread_id, transcript)
            atomic_json(
                directory / "audit/native_quota_exhausted.json",
                {
                    "failure_origin": "infrastructure",
                    "session_id": thread_id,
                    "shared_admission_blocked": True,
                    "automatic_retry_allowed": False,
                },
            )
            raise NativeQuotaExhausted(
                "Subscription Sharing usage limit reached; shared admission "
                "is blocked until the account budget is restored"
            )
        if outcome.tool_startup_failed:
            await _capture(store, thread_id, transcript)
            atomic_json(
                directory / "audit/native_tool_failure.json",
                {
                    "failure_origin": "infrastructure",
                    "session_id": thread_id,
                    "helper": "codex-code-mode-host",
                    "format_correction_allowed": False,
                },
            )
            raise NativeToolStartupError(
                "codex-code-mode-host could not start; repair native runtime "
                "before resuming this session; no output-format correction"
            )
        if outcome.rate_limited:
            await _capture(store, thread_id, transcript)
            raise RuntimeError(
                "Codex provider rate limit: HTTP 429; native turn interrupted, shared cooldown required"
            )
        if native_event_field(outcome.status, "value", outcome.status) != "completed":
            if not outcome.transport_reported:
                admission.note_transport_failure(serialize_native_event(outcome.error))
            await _capture(store, thread_id, transcript)
            raise RuntimeError(
                "Codex author turn did not complete: "
                + json.dumps(serialize_native_event(outcome.error), ensure_ascii=False)
            )
