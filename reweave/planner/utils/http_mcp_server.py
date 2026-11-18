# Attribution retained in NOTICE. Modified for the ReWeave namespace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process streamable-HTTP transport for a Toolkit or native MCP Server.

The Codex CLI accepts streamable-HTTP MCP servers via
``mcp_servers.<name>.url``. This module builds an MCP
:class:`~mcp.server.lowlevel.Server` and serves it through a
:class:`~mcp.server.streamable_http_manager.StreamableHTTPSessionManager`
on a background uvicorn thread — no separate subprocess, no second
``Toolkit`` instance.

Usage::

    server = HttpMcpServer(toolkit)
    server.start()  # binds 127.0.0.1 on a free port
    codex_url = server.url  # e.g. "http://127.0.0.1:54321/mcp/"
    ...
    server.stop()
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any

import httpx
import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from reweave.planner.base import strip_mcp_prefix
from reweave.tools.toolkit import Toolkit
from reweave.utils.logging import get_logger

logger = get_logger("mcp_http")

SERVER_NAME = "harness"


def _toolkit_to_mcp_content(
    tr: Any,
) -> tuple[list[types.TextContent | types.ImageContent], bool]:
    """Translate a :class:`ToolResult` into MCP content blocks + isError."""
    blocks = getattr(tr, "content_blocks", None)
    if blocks is None:
        return [types.TextContent(type="text", text=str(tr))], False

    out: list[types.TextContent | types.ImageContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            out.append(types.TextContent(type="text", text=block.get("text", "")))
        elif block_type == "image":
            src = block.get("source", {})
            out.append(
                types.ImageContent(
                    type="image",
                    data=src.get("data", ""),
                    mimeType=src.get("media_type", "image/png"),
                )
            )
    result_dict = getattr(tr, "result", None)
    is_error = isinstance(result_dict, dict) and bool(result_dict.get("error"))
    return out, is_error


def _toolkit_server(toolkit: Toolkit) -> Server:
    """Expose the live Planner toolkit as a native MCP server."""
    mcp_app: Server = Server(SERVER_NAME, version="0.1.0")
    tool_execution_lock = asyncio.Lock()

    @mcp_app.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools: list[types.Tool] = []
        for spec in toolkit.get_tools_spec():
            tools.append(
                types.Tool(
                    name=str(spec["name"]),
                    description=str(spec.get("description", "")),
                    inputSchema=spec.get("input_schema", {"type": "object"}),
                )
            )
        return tools

    @mcp_app.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        lookup = strip_mcp_prefix(name)
        async with tool_execution_lock:
            tr = await asyncio.get_running_loop().run_in_executor(
                None, toolkit.execute_tool, lookup, arguments or {}
            )
        content, is_error = _toolkit_to_mcp_content(tr)
        return types.CallToolResult(content=content, isError=is_error)

    return mcp_app


def _build_asgi_app(source: Toolkit | Server) -> Any:
    """Serve an existing native Server or adapt the live Planner toolkit."""
    mcp_app = source if isinstance(source, Server) else _toolkit_server(source)
    session_manager = StreamableHTTPSessionManager(
        app=mcp_app,
        stateless=True,
        json_response=True,
    )

    # minimal ASGI wrapper
    async def asgi_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    async with session_manager.run():
                        await send({"type": "lifespan.startup.complete"})
                        shutdown_event = await receive()
                        # Use an explicit check rather than ``assert`` —
                        # ``python -O`` strips asserts and would turn a
                        # protocol violation into a silent mis-handle.
                        if shutdown_event["type"] != "lifespan.shutdown":
                            break
                        await send({"type": "lifespan.shutdown.complete"})
                    break
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    break
            return
        if scope["type"] == "http":
            await session_manager.handle_request(scope, receive, send)

    return asgi_app


def _wait_for_ready(url: str, *, timeout_s: float) -> None:
    """POST an MCP ``initialize`` request, retrying connection failures."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hc", "version": "0"},
        },
    }
    transport = httpx.HTTPTransport(retries=10)
    with httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(timeout_s, connect=2),
        trust_env=False,
    ) as c:
        resp = c.post(url, json=body, headers={"Accept": "application/json"})
        if not (resp.is_success and "result" in resp.json()):
            raise RuntimeError(
                f"HttpMcpServer not ready: code: {resp.status_code}; content: {resp.text}"
            )


class HttpMcpServer:
    """Run streamable-HTTP over a Toolkit or an existing native MCP Server.

    The server runs on a background daemon thread with its own asyncio loop;
    callers must invoke :meth:`start` before reading :attr:`url` and
    :meth:`stop` to release the port before the process exits.
    """

    def __init__(
        self,
        source: Toolkit | Server,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/mcp",
    ) -> None:
        self._source = source
        self._host = host
        self._port = port
        self._path = path if path.startswith("/") else f"/{path}"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}{self._path}/"

    def start(self, *, ready_timeout_s: float = 30.0) -> str:
        """Launch uvicorn in a background thread and block until it's serving."""
        if self._thread is not None:
            return self.url

        # Keep ownership from bind through serving. Selecting and closing a
        # free port first races when several independent authors start together.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind((self._host, self._port))
            listener.listen()
        except BaseException:
            listener.close()
            raise
        self._socket = listener
        self._port = listener.getsockname()[1]

        app = _build_asgi_app(self._source)

        async def routed_app(scope, receive, send):
            if scope["type"] == "http" and scope["path"].rstrip(
                "/"
            ) != self._path.rstrip("/"):
                await send(
                    {"type": "http.response.start", "status": 404, "headers": []}
                )
                await send({"type": "http.response.body", "body": b"Not found"})
                return
            await app(scope, receive, send)

        config = uvicorn.Config(
            routed_app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [listener]},
            name="mcp-http-server",
            daemon=True,
        )
        self._thread.start()

        try:
            _wait_for_ready(self.url, timeout_s=ready_timeout_s)
        except BaseException:
            self.stop()
            raise
        logger.info("HttpMcpServer ready at %s", self.url)
        return self.url

    def stop(self, *, timeout_s: float | None = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise RuntimeError(
                    f"HttpMcpServer thread did not exit within {timeout_s:g}s; "
                    "server references remain available for shutdown"
                )
        self._server = None
        self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
