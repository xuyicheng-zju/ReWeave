# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Backend-independent MCP tools for trusted authoring operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mcp import types
from mcp.server.lowlevel import Server


@dataclass(frozen=True)
class AuthorTool:
    """A trusted author operation with its native MCP input and result contract."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[types.CallToolResult]]


def create_author_mcp_server(name: str, tools: list[AuthorTool]) -> Server:
    """Expose typed author operations through the native MCP protocol.

    The MCP server validates each input schema and converts handler exceptions
    to tool errors. Results must use the native CallToolResult so error state
    cannot be silently lost in a backend-specific dictionary translation.
    """
    by_name = {tool.name: tool for tool in tools}
    if len(by_name) != len(tools):
        raise ValueError("author MCP tool names must be unique")
    server = Server(name, version="1.0.0")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in tools
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if name not in by_name:
            raise ValueError(f"unknown author tool: {name}")
        result = await by_name[name].handler(arguments)
        if not isinstance(result, types.CallToolResult):
            raise TypeError("author tool handler must return native CallToolResult")
        return result

    return server
