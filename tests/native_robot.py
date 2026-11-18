"""Toolkit-only robot fixture shared by native W/C/L runtime tests."""

from types import SimpleNamespace

from reweave.dashboard.events import NullDashboardEventSink
from reweave.tools.toolkit import Toolkit, readonly


class Robot(Toolkit):
    def __init__(self, path):
        from reweave.utils.logging import init_output_dir

        init_output_dir(path)
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=SimpleNamespace(output_dir=path),
            filesystem_tools_enabled=False,
        )
        self.calls = 0
        self.terminal = False
        self.add_tool(
            "read_terminal_state",
            {
                "name": "read_terminal_state",
                "description": "Read the synthetic environment's final state.",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.read_terminal_state,
        )
        self.add_tool(
            "inspect_scene",
            {
                "name": "inspect_scene",
                "description": "Inspect scene",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.inspect_scene,
        )

    @readonly
    def inspect_scene(self):
        self.calls += 1
        return {"position": self.calls}

    def solved(self):
        return False

    def episode_ended(self):
        return self.terminal

    @readonly
    def read_terminal_state(self):
        self.terminal = True
        return {"truncated": True}
