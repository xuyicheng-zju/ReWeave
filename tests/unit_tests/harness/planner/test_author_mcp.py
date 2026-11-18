# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Exercise author errors through native MCP dispatch, never bare handlers."""

import asyncio

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from reweave.planner.utils.author_mcp import AuthorTool, create_author_mcp_server


@pytest.mark.parametrize("outcome", ["success", "error", "exception", "old_dictionary"])
def test_author_mcp_preserves_native_error_semantics(outcome):
    calls = []

    async def handler(arguments):
        calls.append(arguments)
        if outcome == "exception":
            raise ValueError("source failed validation")
        if outcome == "old_dictionary":
            return {"isError": True, "content": []}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="validation result")],
            isError=outcome == "error",
        )

    tool = AuthorTool(
        name="validate",
        description="Validate source",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=handler,
    )
    server = create_author_mcp_server("author", [tool])

    async def exercise():
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            assert [item.name for item in listed.tools] == ["validate"]
            result = await client.call_tool("validate", {})
            assert result.isError is (outcome != "success")
            assert (await client.call_tool("validate", {"extra": 1})).isError is True
            assert (await client.call_tool("unknown", {})).isError is True
            if outcome == "old_dictionary":
                assert "native CallToolResult" in result.content[0].text
            elif outcome == "exception":
                assert "source failed validation" in result.content[0].text

    asyncio.run(exercise())
    assert calls == [{}]
    with pytest.raises(ValueError, match="unique"):
        create_author_mcp_server("author", [tool, tool])
