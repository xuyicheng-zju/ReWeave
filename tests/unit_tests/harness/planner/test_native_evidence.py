"""Native controller context belongs to the complete offline decision archive."""

import json

import pytest

from reweave.evolution.bundle import AgentBundle
from reweave.evolution.evidence.records import inspect_episode
from reweave.memory.notes import planner_memory_guidance
from reweave.planner.native_context import baseline_context
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.planner.providers import NativeSessionError
from reweave.robots.execution_trace import ExecutionTrace
from reweave.runtime.kernel import EpisodeLimits
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline, workflow
from tests.unit_tests.harness.memory.test_episode_notes import NoteRobot
from tests.unit_tests.harness.planner.test_native_loop import FakeSession


@pytest.mark.parametrize("export_context", [False, True])
def test_native_context_and_toolkit_export_complete_the_decision_archive(
    tmp_path, export_context
):
    episode = tmp_path / "episode"
    robot = Robot(episode)
    # A lifecycle tool is registered on the Toolkit but is not callable through
    # the executable bundle. The archive must record the actual controller API.
    robot.add_tool(
        "reset",
        {
            "name": "reset",
            "description": "Reset the synthetic environment.",
            "input_schema": {"type": "object", "properties": {}},
        },
        lambda: {},
    )
    ExecutionTrace(episode / "execution_trace.jsonl")
    context_path = episode / "harness_context/planner_context.json"
    captured = []

    class ReadsInitialContext(FakeSession):
        async def start(self, instructions, cwd):
            captured.append(json.loads(context_path.read_text()))
            assert (
                captured[0]["effective_instructions"] == (cwd / "AGENTS.md").read_text()
            )
            assert instructions == captured[0]["effective_instructions"]
            assert captured[0]["tools"]
            await super().start(instructions, cwd)

        async def decide(self, text, images, output_schema, request_id):
            delivered = json.loads(text)
            if "tools" in delivered:
                assert delivered["tools"] == captured[0]["tools"]
            else:
                assert delivered["available_tools"] == [
                    tool["name"] for tool in captured[0]["tools"]
                ]
            return await super().decide(text, images, output_schema, request_id)

    bundle = baseline().model_dump(mode="json")
    bundle["context"] = baseline_context().model_dump(mode="json")
    bundle["workflows"] = [workflow().model_dump(mode="json")]
    planner = NativeBundlePlanner(
        output_dir=episode,
        session_factory=ReadsInitialContext,
        adaptation_mode="adaptive",
    )
    planner.bind_bundle(
        AgentBundle.model_validate(bundle),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    exported_catalog = None
    if export_context:
        robot.export_harness_context(str(episode))
        exported_catalog = (
            episode / "harness_context/capability_catalog.json"
        ).read_bytes()

    result = planner.solve(
        system_prompt="Original task instructions.",
        user_message="Inspect then stop.",
        toolkit=robot,
        max_turns=3,
    )
    assert result.error is None, result.error
    (context,) = captured
    assert json.loads(context_path.read_text()) == context
    assert context["backend"] == "native"
    assert context["system_prompt"] == "Original task instructions."
    assert context["task_message"] == "Inspect then stop."
    assert context["model"] == "gpt-5.4"
    assert context["reasoning_effort"] == "medium"
    assert context["service_tier"] == "default"
    assert context["model_settings"] == planner.sampling_settings()
    assert context["controller_request_unit"] == "native_turn"
    assert context["adaptation_mode"] == "adaptive"
    for key, value in planner.runtime_configuration().items():
        assert context[key] == value
    assert "The Host alone executes them" in context["effective_instructions"]
    assert (
        context["effective_instructions"]
        == (episode / "native/workspace/AGENTS.md").read_text()
    )
    names = {tool["name"] for tool in context["tools"]}
    assert {"read_image", "workflow_probe", "inspect_scene"} <= names
    assert "reset" not in names
    assert "full_history" not in context
    assert "provider_history" not in context
    assert not (episode / "harness_context/model_attempts.jsonl").exists()
    assert robot.calls == 1 and robot.terminal
    # The CLI owns the final transcript; retain the actual returned evidence.
    (episode / "transcript_native.json").write_text(
        json.dumps(
            {
                "messages": result.messages,
                "stats": result.stats,
                "error": result.error,
                "planner_termination_class": result.termination_class,
                "system_prompt": context["system_prompt"],
                "task_message": context["task_message"],
            }
        )
    )
    if export_context:
        assert (
            episode / "harness_context/capability_catalog.json"
        ).read_bytes() == exported_catalog
        evidence = inspect_episode(episode)
        assert "harness_context/planner_context.json" in evidence.files
        assert "harness_context/capability_catalog.json" in evidence.files
        assert "native/records/episode.md" in evidence.files
        assert "native/adaptation/receipt.json" in evidence.files
        evidence.verify(episode)
    else:
        with pytest.raises(ValueError, match="capability_catalog.json"):
            inspect_episode(episode)


def test_initial_context_survives_native_start_failure_without_a_model_turn(tmp_path):
    context_path = tmp_path / "harness_context/planner_context.json"
    started = []

    class FailsAtStart(FakeSession):
        async def start(self, instructions, cwd):
            context = json.loads(context_path.read_text())
            assert context["effective_instructions"] == (cwd / "AGENTS.md").read_text()
            started.append(context)
            raise NativeSessionError("synthetic session startup failure")

    robot = Robot(tmp_path)
    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=FailsAtStart)
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            wall_time_s=2, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Initial instruction",
        user_message="Task request",
        toolkit=robot,
        max_turns=3,
    )
    assert result.termination_class == "planner_timeout"
    assert "deadline" in result.error
    blocked = json.loads((tmp_path / "native/recovery/blocked.json").read_text())
    assert blocked["status"] == "terminated"
    assert blocked["termination_class"] == "planner_timeout"
    assert blocked["reason"] == "synthetic session startup failure"
    assert blocked["original_deadline"] == blocked["active_deadline"]
    assert blocked["error_type"] == "NativeSessionError"
    assert blocked["controller_attempts"] == 0
    assert blocked["automatic_sampling"] is False
    assert blocked["automatic_replay"] is False
    assert result.stats["visible_turns"] == 0
    assert robot.calls == 0
    assert len(started) == 1
    assert started[0]["adaptation_mode"] == "static"
    assert json.loads(context_path.read_text()) == started[0]


@pytest.mark.parametrize("adaptation_mode", ["static", "adaptive"])
def test_native_uses_existing_notes_guidance_and_keeps_it_in_initial_evidence(
    tmp_path, adaptation_mode
):
    episode = tmp_path / "episode"
    corpus = tmp_path / "published"
    published = corpus / "global" / "fixed.md"
    published.parent.mkdir(parents=True)
    published.write_text("Published knowledge remains read-only.")
    robot = NoteRobot(episode, corpus, filesystem_tools_enabled=False)
    note = robot.episode_notes.drafts / "task-family" / "working-notes.md"
    guidance = planner_memory_guidance(robot, adaptation_mode=adaptation_mode)
    assert "CLOSE OUT FAILED APPROACHES" in guidance
    assert "BEFORE FINISH" in guidance
    assert str(robot.episode_notes.drafts) in guidance
    if adaptation_mode == "static":
        assert guidance == planner_memory_guidance(robot)
        assert "C skills are fixed" in guidance
        assert "Only your draft text is writable" in guidance
    else:
        assert "private C skills" in guidance
        assert "settled revision boundaries" in guidance
        assert "C skills are fixed" not in guidance
        assert "Only your draft text is writable" not in guidance
    seen = []

    class WritesNote(FakeSession):
        async def start(self, instructions, cwd):
            assert guidance in (cwd / "AGENTS.md").read_text()
            context = json.loads(
                (episode / "harness_context/planner_context.json").read_text()
            )
            assert context["effective_instructions"] == (cwd / "AGENTS.md").read_text()
            assert context["adaptation_mode"] == adaptation_mode
            await super().start(instructions, cwd)

        async def decide(self, text, images, output_schema, request_id):
            seen.append(request_id)
            delivered = json.loads(text)
            names = (
                {tool["name"] for tool in delivered["tools"]}
                if "tools" in delivered
                else set(delivered["available_tools"])
            )
            assert {"write_text_file", "read_text_file", "move"} <= names
            return {
                "text": "Retain evidence before ending.",
                "tool_calls": [
                    {
                        "name": "write_text_file" if len(seen) == 1 else "move",
                        "arguments_json": json.dumps(
                            {
                                "path": str(note),
                                "content": "No winning recipe has been verified.",
                            }
                            if len(seen) == 1
                            else {}
                        ),
                    }
                ],
            }

    robot.export_harness_context(str(episode))
    catalog_path = episode / "harness_context/capability_catalog.json"
    exported_catalog = catalog_path.read_bytes()
    planner = NativeBundlePlanner(
        output_dir=episode,
        session_factory=WritesNote,
        adaptation_mode=adaptation_mode,
    )
    planner.bind_bundle(
        baseline(),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100, note_closeout_s=5
        ),
    )
    result = planner.solve(
        system_prompt="Use public evidence.",
        user_message="Synthetic episode.",
        toolkit=robot,
        max_turns=3,
    )
    assert result.error is None, result.error
    assert len(seen) == result.stats["visible_turns"] == 2
    assert note.read_text() == "No winning recipe has been verified."
    assert published.read_text() == "Published knowledge remains read-only."
    assert len(robot.capture_calls) == 1
    assert catalog_path.read_bytes() == exported_catalog
    assert json.loads((robot.episode_notes.root / "closeout.json").read_text()) == {
        "status": "already_written"
    }
    initial = json.loads((episode / "harness_context/planner_context.json").read_text())
    assert (
        initial["effective_instructions"]
        == (episode / "native/workspace/AGENTS.md").read_text()
    )


@pytest.mark.parametrize("adaptation_mode", ["static", "adaptive"])
def test_notes_guidance_does_not_advertise_unavailable_note_tools(
    tmp_path, adaptation_mode
):
    robot = Robot(tmp_path)
    assert planner_memory_guidance(robot, adaptation_mode=adaptation_mode) == ""
    with pytest.raises(ValueError, match="adaptation_mode"):
        planner_memory_guidance(robot, adaptation_mode="invalid")
