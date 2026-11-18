"""Current C guidance remains readable after native context compaction."""

from reweave.evolution.bundle import AgentBundle
from reweave.planner.native_loop import NativeBundlePlanner
from reweave.runtime.kernel import EpisodeLimits
from tests.native_robot import Robot
from tests.unit_tests.harness.evolution.fixtures import baseline
from tests.unit_tests.harness.planner.test_native_loop import FakeSession


def test_context_updates_replace_durable_prompt_without_losing_native_rules(tmp_path):
    seen = []

    class ReadsPrompt(FakeSession):
        async def decide(self, text, images, output_schema, request_id):
            prompt = (tmp_path / "native/workspace/AGENTS.md").read_text()
            step = len(seen) + 1
            assert f"Current guidance {step}" in prompt
            assert "The Host alone executes them" in prompt
            assert "Initial task guidance" in self.instructions
            assert "Current guidance 1" in self.instructions
            if step > 1:
                assert f"Current guidance {step}" not in self.instructions
            assert prompt.count("Initial task guidance") == 1
            if step > 1:
                assert f"Current guidance {step - 1}" not in prompt
                assert f"+Current guidance {step}" in text
                assert "Host updated AGENTS.md" in text
            else:
                assert "Current guidance" not in text
            seen.append(prompt)
            return await super().decide(text, images, output_schema, request_id)

    bundle = baseline().model_dump(mode="json")
    bundle["context"]["program"] = {
        "entrypoint": "context:build",
        "files": [
            {
                "path": "context.py",
                "source": """def build(runtime, context):
    step = context["state"].get("step", 0) + 1
    return {"instructions": "Current guidance " + str(step),
            "messages": context["messages"], "state": {"step": step},
            "tools": context["tools"]}
""",
            }
        ],
    }
    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=ReadsPrompt)
    planner.bind_bundle(
        AgentBundle.model_validate(bundle),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Initial task guidance",
        user_message="Inspect",
        toolkit=Robot(tmp_path),
        max_turns=3,
    )
    assert result.error is None, result.error
    assert len(seen) == 2
    assert (tmp_path / "native/workspace/AGENTS.md").read_text() == seen[-1]
