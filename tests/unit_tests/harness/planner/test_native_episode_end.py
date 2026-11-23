# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Native W/C/L stops after terminal ACK and never replays unknown actions."""

import asyncio
import json

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from reweave.dashboard.events import NullDashboardEventSink
from reweave.planner.base import (
    ACTION_OUTCOME_UNKNOWN,
    PHYSICAL_FAILURE,
    classify_episode_termination,
)
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.planner.utils.http_mcp_server import HttpMcpServer
from reweave.runtime.capability import PhysicalOutcomeUnknown
from reweave.session.base import EnvState
from reweave.tools.toolkit import Toolkit
from reweave.utils.logging import init_output_dir


class TerminalRobot(Toolkit):
    """A real Toolkit over an in-memory actuator and persisted StepRecords."""

    def __init__(self, directory, flag, *, unknown=False):
        init_output_dir(directory)
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=EnvState(directory),
            filesystem_tools_enabled=False,
        )
        self.flag, self.unknown = flag, unknown
        self.actions = 0
        self.trace = []
        self.error = PhysicalOutcomeUnknown("synthetic submitted action lost ACK")
        self.add_tool(
            "end_episode",
            {
                "name": "end_episode",
                "description": "End an unsuccessful synthetic episode in one action.",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.end_episode,
        )

    def end_episode(self):
        self.actions += 1
        if self.unknown:
            raise self.error
        self.trace.append("physical_ack")
        return {"steps_used": 1, "official_success": False}

    def get_env_state(self, *, command, result, elapsed_s):
        flags = {
            "terminated": self.flag == "terminated",
            "truncated": self.flag == "truncated",
        }
        with self.state.record_step(
            state={"actions": self.actions},
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            **flags,
        ):
            pass
        return {**flags, "official_success": False}

    def solved(self):
        return False


class TerminalSession:
    def __init__(self, directory):
        self.session_id = "synthetic-terminal-session"
        self.context_generation = 0
        self.requests = []
        self.closed = False

    async def start(self, instructions, cwd):
        self.cwd = cwd

    async def decide(self, text, images, schema, request_id):
        self.requests.append(request_id)
        return {
            "text": "Execute once, then stop at the terminal acknowledgement.",
            "tool_calls": [
                {"name": "end_episode", "arguments_json": "{}"},
                {"name": "end_episode", "arguments_json": "{}"},
            ],
        }

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("flag", ["terminated", "truncated"])
def test_acknowledged_unsuccessful_end_stops_without_another_decision(tmp_path, flag):
    robot = TerminalRobot(tmp_path, flag)
    sessions = []

    def factory(directory):
        session = TerminalSession(directory)
        sessions.append(session)
        return session

    result = NativeBundlePlanner(output_dir=tmp_path, session_factory=factory).solve(
        system_prompt="", user_message="initial", toolkit=robot, max_turns=3
    )
    assert (
        result.error is result.termination_class is result.api_error_retryable is None
    )
    assert result.finish_result is None and robot.solved() is False
    assert (
        classify_episode_termination(
            planner_termination_class=None, official_success=False
        )
        == PHYSICAL_FAILURE
    )
    assert robot.actions == len(sessions[0].requests) == 1
    assert sessions[0].closed and result.stats["visible_turns"] == 1
    assert robot.trace == ["physical_ack"]


def test_initial_terminal_record_submits_no_native_query(tmp_path):
    robot = TerminalRobot(tmp_path, "terminated")
    robot.get_env_state(command={}, result={}, elapsed_s=0)
    session = TerminalSession(tmp_path)
    result = NativeBundlePlanner(
        output_dir=tmp_path, session_factory=lambda _: session
    ).solve(system_prompt="", user_message="initial", toolkit=robot, max_turns=3)
    assert result.error is result.termination_class is None
    assert robot.actions == 0 and session.requests == []
    assert session.closed and result.stats["visible_turns"] == 0


def test_unknown_action_stops_native_and_blocks_later_proposals(tmp_path):
    robot = TerminalRobot(tmp_path, "terminated", unknown=True)
    session = TerminalSession(tmp_path)
    result = NativeBundlePlanner(
        output_dir=tmp_path, session_factory=lambda _: session
    ).solve(system_prompt="", user_message="initial", toolkit=robot, max_turns=3)
    assert result.termination_class == ACTION_OUTCOME_UNKNOWN
    assert result.error == str(robot.error) and result.api_error_retryable is None
    assert robot.physical_outcome_error is robot.error and robot.actions == 1
    assert len(session.requests) == 1 and session.closed
    with pytest.raises(PhysicalOutcomeUnknown) as blocked:
        robot.execute_tool("end_episode", {})
    assert blocked.value is robot.error and robot.actions == 1


def test_http_mcp_preserves_terminal_ack_and_blocks_later_physical_tool(tmp_path):
    robot = TerminalRobot(tmp_path, "terminated")

    async def call(url):
        async with streamable_http_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                first = await session.call_tool("end_episode", {})
                second = await session.call_tool("end_episode", {})
                assert not first.isError and not second.isError
                return json.loads(first.content[0].text), json.loads(
                    second.content[0].text
                )

    server = HttpMcpServer(robot)
    try:
        first, second = asyncio.run(call(server.start()))
    finally:
        server.stop()
    assert first["terminated"] and first["capability_result"]["steps_used"] == 1
    assert first["official_success"] is False
    assert second["status"] == second["execution"] == "not_executed"
    assert not second.get("error") and robot.actions == 1
