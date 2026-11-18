"""Session tool definitions change only with authority or native compaction."""

import json

import pytest

from reweave.evolution.bundle import AgentBundle
from reweave.planner.native_context import baseline_context
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.planner.providers import NativeTransportError
from reweave.runtime import execution
from reweave.runtime.kernel import EpisodeLimits
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.planner.test_native_loop import FakeSession


@pytest.mark.parametrize(
    "change",
    ["unchanged", "definition", "schema", "availability", "compaction", "retry"],
)
def test_catalog_delivery_tracks_current_tools_and_native_context(
    tmp_path, monkeypatch, change
):
    instances, contexts = [], []
    original_build = execution.build_context

    async def build(*args, **kwargs):
        prepared = await original_build(*args, **kwargs)
        if len(contexts) == 2:
            if change == "definition":
                # Model a current tool definition at a settled revision boundary.
                prepared["tools"][0]["description"] += " Updated current guidance."
            elif change == "schema":
                prepared["tools"][0]["input_schema"]["title"] = "Current tool schema"
            elif change == "availability":
                prepared["tools"] = [
                    tool
                    for tool in prepared["tools"]
                    if tool["name"] == "read_terminal_state"
                ]
        contexts.append(prepared)
        return prepared

    async def sleep(seconds):
        pass

    monkeypatch.setattr(execution, "build_context", build)
    monkeypatch.setattr("reweave.planner.native_loop.asyncio.sleep", sleep)

    class Session(FakeSession):
        def __init__(self, directory):
            super().__init__(directory)
            self.identities = []
            self.retried = False
            self.prompts = {}
            instances.append(self)

        async def decide(self, text, images, output_schema, request_id):
            self.inputs.append((text, images, request_id))
            self.prompts[request_id] = (self.cwd / "AGENTS.md").read_text()
            if request_id not in self.identities:
                self.identities.append(request_id)
            step = len(self.identities)
            if step == 2:
                if change == "compaction":
                    self.context_generation += 1
                elif change == "retry" and not self.retried:
                    self.context_generation += 1
                    self.retried = True
                    raise NativeTransportError("resume the same submitted request")
            return {
                "text": "Use the current definitions.",
                "tool_calls": [
                    {
                        "name": "inspect_scene" if step < 3 else "read_terminal_state",
                        "arguments_json": "{}",
                    }
                ],
            }

    bundle = baseline().model_dump(mode="json")
    bundle["context"] = baseline_context().model_dump(mode="json")
    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=Session)
    planner.bind_bundle(
        AgentBundle.model_validate(bundle),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Task",
        user_message="Inspect",
        toolkit=Robot(tmp_path),
        max_turns=3,
    )
    assert result.error is None, result.error
    (session,) = instances
    assert result.stats["visible_turns"] == 3
    assert session.starts == 1
    if change == "retry":
        assert len(session.inputs) == 4
        assert session.inputs[1] == session.inputs[2]
    unique_inputs = {identity: text for text, _, identity in session.inputs}
    delivered = [
        json.loads(text.rsplit("\n\n", 1)[-1]) for text in unique_inputs.values()
    ]
    assert not any("tools" in message for message in delivered)
    for (identity, text), message, context in zip(
        unique_inputs.items(), delivered, contexts, strict=True
    ):
        assert message["available_tools"] == [tool["name"] for tool in context["tools"]]
        catalog_text = (
            session.prompts[identity]
            .split("# Tool catalog\n\n```json\n", 1)[1]
            .split("\n```", 1)[0]
        )
        assert json.loads(catalog_text) == context["tools"]
    changed = change in {"definition", "schema", "availability"}
    assert ("Host updated AGENTS.md" in list(unique_inputs.values())[-1]) == changed
    events = [
        json.loads(line)
        for line in (tmp_path / "harness_context/bundle_events.jsonl")
        .read_text()
        .splitlines()
    ]
    archived = [event["output"] for event in events if event["kind"] == "context_built"]
    assert len(archived) == 3
    assert [entry["tools"] for entry in archived] == [
        context["tools"] for context in contexts
    ]
