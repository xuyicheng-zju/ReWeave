"""Owned Cursor CLI process using its ACP stdio JSON-RPC server."""

from __future__ import annotations

from pathlib import Path

from reweave.planner.cursor_connection import (
    cursor_environment,
    resolve_cursor_executable,
)

from .base import NativeAuthenticationError, NativeProtocolError, NativeTransportError
from .stdio import StdioJsonRpcTransport


class CursorStdioTransport(StdioJsonRpcTransport):
    """One ``agent acp`` process. Model calls use the logged-in subscription."""

    PROTOCOL = "cursor-acp"
    # Ask-mode calls still need a reply when the CLI asks the client something.
    # These answers refuse tool execution and questions; they never run robot I/O.
    CLIENT_RESULTS = {
        "session/request_permission": {
            "outcome": {"outcome": "selected", "optionId": "reject-once"}
        },
        "cursor/ask_question": {"outcome": {"outcome": "skipped"}},
        "cursor/create_plan": {
            "outcome": {"outcome": "rejected", "reason": "model-only session"}
        },
    }

    def __init__(self, directory: Path, *, executable: str = "agent"):
        super().__init__(directory, executable=executable)

    async def connect(self, cwd: Path) -> None:
        if self._process is not None:
            raise NativeProtocolError("Native transport is already connected")
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            environment = cursor_environment()
        except ValueError as error:
            raise NativeAuthenticationError(str(error)) from error
        try:
            executable = resolve_cursor_executable(self.executable, environment)
        except ValueError as error:
            raise NativeTransportError(str(error), request_sent=False) from error
        self._metadata["executable"] = executable
        self._metadata["authentication"] = "cursor_login"
        self._metadata["config_dir"] = environment["CURSOR_CONFIG_DIR"]
        await self._spawn(executable, "acp", cwd=cwd, environment=environment)
