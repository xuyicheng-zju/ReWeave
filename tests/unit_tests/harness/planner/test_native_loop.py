"""CPU native decisions execute through actual L/C/Kernel, without a provider."""

import base64
import io
import json

import pytest
from PIL import Image

from reweave.evolution.bundle import AgentBundle, RequestLimits
from reweave.planner.native_context import baseline_context, prepare_native
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.planner.providers.base import NativeTurnFailed
from reweave.runtime.capability import ModelInputRejected
from reweave.runtime.kernel import EpisodeLimits
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline


class FakeSession:
    def __init__(self, directory):
        self.directory = directory
        self.session_id = "same-native-episode"
        self.context_generation = 0
        self.inputs = []
        self.starts = 0
        self.closed = False

    async def start(self, instructions, cwd):
        self.starts += 1
        self.instructions = instructions
        self.cwd = cwd

    async def decide(self, text, images, output_schema, request_id):
        self.inputs.append((text, images, request_id))
        return {
            "text": "Inspect then finish.",
            "tool_calls": [
                {
                    "name": "inspect_scene"
                    if len(self.inputs) == 1
                    else "read_terminal_state",
                    "arguments_json": "{}",
                }
            ],
        }

    async def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "model,effort,tier,selected",
    [
        (None, "medium", "fast", "gpt-5.4"),
        ("gpt-5.4", "low", "default", "gpt-5.4"),
    ],
)
def test_selected_settings_reach_default_session_and_runtime_evidence(
    tmp_path, model, effort, tier, selected
):
    from reweave.planner.base import build_planner

    planner = build_planner(
        "native",
        output_dir=tmp_path,
        model=model,
        reasoning_effort=effort,
        native_service_tier=tier,
    )
    session = planner.session_factory(tmp_path / "provider")
    assert (session.model, session.reasoning_effort, session.service_tier) == (
        selected,
        effort,
        tier,
    )
    assert planner.runtime_configuration()["model"] == selected
    assert planner.runtime_configuration()["service_tier"] == tier
    assert planner.sampling_settings() == {"reasoning_effort": effort}


@pytest.mark.parametrize("model", [None, "gpt-5.4"])
def test_native_real_loop_uses_one_session_and_incremental_feedback(tmp_path, model):
    instances = []

    def factory(directory):
        session = FakeSession(directory)
        instances.append(session)
        return session

    robot = Robot(tmp_path)
    bundle = baseline().model_dump(mode="json")
    bundle["context"] = baseline_context().model_dump(mode="json")
    planner = NativeBundlePlanner(
        output_dir=tmp_path, session_factory=factory, model=model
    )
    planner.bind_bundle(
        AgentBundle.model_validate(bundle),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Task instructions",
        user_message="Inspect",
        toolkit=robot,
        max_turns=3,
    )
    assert result.error is None, result.error
    assert robot.calls == 1 and robot.terminal
    (session,) = instances
    assert session.starts == 1 and session.closed
    assert session.instructions == (session.cwd / "AGENTS.md").read_text()
    assert "Task instructions" in session.instructions
    agents = (session.cwd / "AGENTS.md").read_text()
    assert agents.count("Task instructions") == 1
    assert agents.count("Decide directly from the supplied observation messages") == 1
    assert "Do not browse the repository" in agents
    assert "necessary targeted harness revision" in agents
    assert "Use native tools to read the episode record" not in agents
    assert not list(session.cwd.rglob("execution_trace.jsonl"))
    assert not list(session.cwd.rglob("episode.md"))
    assert len(session.inputs) == 2
    assert (
        '"position": 1' in session.inputs[1][0]
        or '\\"position\\": 1' in session.inputs[1][0]
    )
    assert "Current task guidance:" not in session.inputs[1][0]
    assert "Task instructions" not in session.inputs[0][0]
    assert '"tools":' not in session.inputs[0][0]
    assert result.stats["model_calls"] is None
    assert result.stats["visible_turns"] == 2
    record = (tmp_path / "native/records/episode.md").read_text()
    assert record.count("## Completed native turn") == 2
    assert record.count("## Prepared observation") == 2
    context = json.loads(
        (tmp_path / "harness_context/planner_context.json").read_text()
    )
    assert context["model"] == ("gpt-5.4" if model is None else model)


def test_native_refusal_keeps_evidence_and_executes_no_proposals(tmp_path):
    class Refused(FakeSession):
        async def decide(self, *args):
            raise NativeTurnFailed("failed", {"message": "provider refusal"})

    robot = Robot(tmp_path)
    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=Refused)
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            wall_time_s=2, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Task", user_message="Inspect", toolkit=robot, max_turns=3
    )
    assert result.termination_class == "api_error"
    assert result.api_error_retryable is True
    assert robot.calls == 0
    assert result.stats["visible_turns"] == 1
    assert "provider refusal" in result.error
    assert not (tmp_path / "native/recovery/blocked.json").exists()


def test_native_images_are_actual_pixels_and_recorded_as_files(tmp_path):
    output = io.BytesIO()
    Image.new("RGB", (3, 3), (10, 30, 50)).save(output, "PNG")
    encoded = base64.b64encode(output.getvalue()).decode()
    context = {
        "instructions": "Read pixels",
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": [
                            "current observation",
                            {
                                "kind": "binary",
                                "media_type": "image/png",
                                "data": encoded,
                            },
                        ],
                    }
                ],
            }
        ],
        "tools": [],
    }
    text, images = prepare_native(context, tmp_path, RequestLimits())
    assert len(images) == 1 and images[0].read_bytes() == output.getvalue()
    assert encoded not in text and str(images[0]) in text
    assert "current observation" in text
    context["messages"][0]["parts"][0]["content"][1] = {
        "kind": "image-url",
        "url": "https://example.com/unknown.png",
    }
    with pytest.raises(ModelInputRejected, match="embedded bytes"):
        prepare_native(context, tmp_path, RequestLimits())


def test_native_visible_budget_shared_and_not_internal_model_calls(tmp_path):
    class Reads(FakeSession):
        async def decide(self, text, images, schema, request_id):
            return {
                "text": "Read",
                "tool_calls": [{"name": "inspect_scene", "arguments_json": "{}"}],
            }

    robot = Robot(tmp_path)
    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=Reads)
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Task", user_message="Inspect", toolkit=robot, max_turns=2
    )
    assert result.termination_class == "agent_execution_failure"
    assert robot.calls == 2
    ledger = [
        json.loads(line)
        for line in (tmp_path / "native/native_turns.jsonl").read_text().splitlines()
    ]
    assert len(ledger) == 2 and result.stats["visible_turns"] == 2
    assert result.stats["requests"] is None


def test_connection_recovery_keeps_live_loop_and_request_identity(
    tmp_path, monkeypatch
):
    from reweave.planner import native_loop
    from reweave.planner.providers.base import NativeTransportError

    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(native_loop.asyncio, "sleep", sleep)
    instances = []

    class Reconnected(FakeSession):
        async def decide(self, text, images, schema, request_id):
            self.inputs.append((text, images, request_id))
            if len(self.inputs) == 2:
                raise NativeTransportError("temporary disconnect")
            return {
                "text": "Continue",
                "tool_calls": [
                    {
                        "name": "inspect_scene"
                        if len(self.inputs) == 1
                        else "read_terminal_state",
                        "arguments_json": "{}",
                    }
                ],
            }

    def factory(directory):
        session = Reconnected(directory)
        instances.append(session)
        return session

    robot = Robot(tmp_path)
    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=factory)
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Task", user_message="Inspect", toolkit=robot, max_turns=2
    )
    assert result.error is None, result.error
    (session,) = instances
    assert robot.calls == 1 and robot.terminal
    assert len(session.inputs) == 3
    assert session.inputs[1] == session.inputs[2]
    assert result.stats["visible_turns"] == 2
    assert waits == [2]


def test_native_cli_bootstraps_standard_harness_without_capture(tmp_path):
    robot = Robot(tmp_path)
    planner = NativeBundlePlanner(
        output_dir=tmp_path, session_factory=FakeSession, timeout_s=30
    )
    result = planner.solve(
        system_prompt="Task instructions",
        user_message="Inspect",
        toolkit=robot,
        max_turns=3,
    )
    assert result.error is None, result.error
    assert robot.calls == 1 and robot.terminal
    assert result.stats["visible_turns"] == 2


def test_transient_connection_before_session_start_keeps_episode(tmp_path, monkeypatch):
    import reweave.planner.native_loop as module
    from reweave.planner.providers.base import NativeTransportError

    instances = []

    class Reconnecting(FakeSession):
        async def start(self, instructions, cwd):
            self.starts += 1
            if self.starts == 1:
                raise NativeTransportError("not connected", request_sent=False)
            self.instructions, self.cwd = instructions, cwd

    def factory(directory):
        session = Reconnecting(directory)
        instances.append(session)
        return session

    sleeps = []

    async def pause(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(module.asyncio, "sleep", pause)
    robot = Robot(tmp_path)
    planner = NativeBundlePlanner(
        output_dir=tmp_path, session_factory=factory, timeout_s=30
    )
    result = planner.solve(
        system_prompt="Task", user_message="Inspect", toolkit=robot, max_turns=3
    )
    assert result.error is None, result.error
    assert instances[0].starts == 2 and len(instances[0].inputs) == 2
    assert robot.calls == 1 and result.stats["visible_turns"] == 2
    assert sleeps == [2]
