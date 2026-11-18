# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Shipped examples execute through production admission, not a second runtime."""

import asyncio
import base64

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter

from reweave.evolution.authoring.examples import author_examples
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.validation.admission import check_runtime_contracts
from tests.unit_tests.harness.evolution.fixtures import baseline


def _example_fixture():
    examples = author_examples()
    payload = baseline().model_dump(mode="json")
    payload.update(
        context=examples["C"]["context"],
        loop=examples["L"]["loop"],
        workflows=[examples["W"]["workflow"]],
    )
    contract = {
        "episode_limits": {
            "program_calls": 100,
            "gateway_calls": 100,
            "wall_time_s": 30,
        },
        "operations": {
            "release": {
                "effect": "physical",
                "input_schema": {
                    "type": "object",
                    "properties": {"max_steps": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
            "view_env_state": {
                "effect": "read",
                "input_schema": {
                    "type": "object",
                    "properties": {"step": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
        },
    }
    return examples, payload, contract


def test_all_examples_run_and_expose_behavioral_results():
    examples, payload, contract = _example_fixture()
    records = []
    asyncio.run(
        check_runtime_contracts(
            AgentBundle.model_validate(payload), contract, case_results=records
        )
    )
    assert all(record["validation_status"] == "passed" for record in records)
    context = next(record for record in records if record["scope"] == "C")
    case = examples["C"]["context"]["contract_cases"][0]
    ModelMessagesTypeAdapter.validate_python(case["input"]["messages"])
    ModelMessagesTypeAdapter.validate_python(context["result"]["messages"])
    for field in ("messages", "tools", "state"):
        assert context["result"][field] == case["input"][field]
    assert '"target_reached": false' in context["result"]["instructions"]
    loops = [record for record in records if record["scope"] == "L"]
    # Normal, unmet, repeated same state, new state, initial terminal, post-action terminal.
    assert [
        sum(
            call["name"] == "tool:execute"
            and call["arguments"]["call"]["name"] == "view_env_state"
            for call in record["calls"]
        )
        for record in loops
    ] == [0, 1, 1, 0, 0, 0]
    assert [
        sum(call["name"] == "model:plan" for call in record["calls"])
        for record in loops
    ] == [1, 2, 2, 2, 0, 1]
    workflows = [record for record in records if record["scope"] == "W"]
    assert len(workflows) == 2
    assert [record["result"]["status"] for record in workflows] == [
        "completed",
        "failed",
    ]


@pytest.mark.parametrize(
    "observation",
    [
        {"error": "state capture unavailable"},
        {
            "state": {"robot0_eef_pos": [0, 0, 1]},
            "observation_errors": ["camera capture failed"],
        },
    ],
)
def test_workflow_observation_failure_preserves_action_evidence(observation):
    _, payload, contract = _example_fixture()
    case = payload["workflows"][0]["contract_cases"][0]
    case["responses"][1]["result"] = observation
    case["expected_status"] = "failed"
    records = []
    asyncio.run(
        check_runtime_contracts(
            AgentBundle.model_validate(payload), contract, case_results=records
        )
    )
    result = next(record["result"] for record in records if record["scope"] == "W")
    assert result["status"] == "failed"
    assert result["output"] == {
        "action": case["responses"][0]["result"],
        "observation": observation,
    }
    assert "observation failed" in result["summary"]


def test_context_promotes_only_named_endpoints_with_complete_image_messages():
    import copy
    import json

    from tests.unit_tests.harness.planner.test_native_context import pixels

    encoded = base64.b64encode(pixels("red")).decode()

    _, payload, contract = _example_fixture()
    case = payload["context"]["contract_cases"][0]
    image = {
        "name": "agentview",
        "media_type": "image/png",
        "data_base64": encoded,
    }
    observation = {
        "step": 7,
        "terminated": False,
        "truncated": False,
        "capability_result": {
            "target_reached": False,
            "stop_reason": "max_steps",
            "final_dist_m": 0.123,
            "steps_used": 4,
            "max_steps": 4,
        },
        "_runtime_images": [image],
        "state": {"full_original": "retained in feedback"},
    }
    messages = [
        {
            "kind": "request",
            "parts": [
                {"part_kind": "user-prompt", "content": "Complete original feedback"},
                {
                    "part_kind": "user-prompt",
                    "content": [
                        {
                            "kind": "binary",
                            "data": encoded,
                            "media_type": "image/png",
                        }
                    ],
                },
            ],
        }
    ]
    case["input"].update(
        observation=observation,
        messages=messages,
        last_tool_result={
            "operation": "inspect_scene",
            "result": {"capability_result": {"target_reached": True}},
        },
    )
    endpoint = {
        "step": 7,
        "terminated": False,
        "truncated": False,
        **observation["capability_result"],
    }
    case["expected"]["instructions"] = (
        case["expected"]["instructions"].split("Current endpoint evidence: ")[0]
        + "Current endpoint evidence: "
        + json.dumps(endpoint)
    )
    case["expected"]["messages"] = copy.deepcopy(messages)
    records = []
    asyncio.run(
        check_runtime_contracts(
            AgentBundle.model_validate(payload), contract, case_results=records
        )
    )
    result = next(row["result"] for row in records if row["scope"] == "C")
    assert result["messages"] == messages
    assert "IMAGE_ORIGINAL" not in result["instructions"]
    assert "_runtime_images" not in result["instructions"]
    assert (
        json.loads(result["instructions"].split("Current endpoint evidence: ")[1])
        == endpoint
    )


def test_loop_refresh_publishes_images_and_keeps_physical_endpoint_after_read(tmp_path):
    import io

    from PIL import Image
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    from reweave.runtime.kernel import EpisodeLimits
    from reweave.tools.toolkit import readonly
    from tests.native_fixture import PhysicalRobot, ScriptedNativePlanner, rows

    class EndpointRobot(PhysicalRobot):
        def __init__(self, path):
            super().__init__(path)
            self.captures = []
            self.add_tool(
                "view_env_state",
                {
                    "name": "view_env_state",
                    "description": "Read current state",
                    "input_schema": {
                        "type": "object",
                        "properties": {"step": {"type": "integer"}},
                    },
                },
                self.view_env_state,
            )

        def move_once(self):
            self.moves += 1
            return {
                "target_reached": False,
                "stop_reason": "max_steps",
                "final_dist_m": 0.123,
                "steps_used": 1,
                "max_steps": 1,
            }

        def get_env_state(self, **kwargs):
            data = io.BytesIO()
            Image.new("RGB", (2, 2), "red" if self.captures else "blue").save(
                data, format="PNG"
            )
            self.captures.append(data.getvalue())
            return {
                "step": self.moves,
                "state": {"capture": len(self.captures)},
                "_image_bytes": data.getvalue(),
            }

        @readonly
        def view_env_state(self, step=-1):
            return self.get_env_state()

    examples = author_examples()
    payload = baseline().model_dump(mode="json")
    payload.update(context=examples["C"]["context"], loop=examples["L"]["loop"])
    robot = EndpointRobot(tmp_path)
    bodies = []

    def respond(messages, info):
        bodies.append(
            {
                "text": info.native_text,
                "guidance": info.context_guidance,
                "images": [path.read_bytes() for path in info.native_images],
            }
        )
        name, arguments = [
            ("move_once", {}),
            ("view_env_state", {"step": -1}),
            ("read_terminal_state", {}),
        ][len(bodies) - 1]
        return ModelResponse(parts=[ToolCallPart(name, arguments)])

    planner = ScriptedNativePlanner(respond, reasoning_effort="high")
    planner.bind_bundle(
        AgentBundle.model_validate(payload),
        episode_limits=EpisodeLimits(
            program_calls=100, gateway_calls=100, wall_time_s=30
        ),
    )
    result = planner.solve(
        system_prompt="Inspect the endpoint",
        user_message="Move once",
        toolkit=robot,
        max_turns=10,
    )
    assert result.error is None, result.error
    assert robot.moves == 1 and len(robot.captures) == 2
    events = rows(tmp_path / "harness_context/bundle_events.jsonl")
    contexts = [e for e in events if e["kind"] == "context_built"]
    assert len(contexts) == 3
    # The Loop first asks the Planner to confirm the read. Its result reaches the
    # following plan without replacing the last physical observation.
    assert contexts[1]["input"]["last_tool_result"]["operation"] == "move_once"
    assert contexts[2]["input"]["last_tool_result"]["operation"] == "view_env_state"
    for event in contexts[1:]:
        assert event["input"]["observation"]["state"]["capture"] == 1
        assert event["output"]["messages"] == event["input"]["messages"]
        instructions = event["output"]["instructions"]
        assert (
            '"target_reached": false' in instructions
            and '"final_dist_m": 0.123' in instructions
        )
        assert (
            "data_base64" not in instructions and "_runtime_images" not in instructions
        )
    for body in bodies[2:]:
        assert robot.captures[-1] in body["images"]
        assert '"target_reached": false' in body["guidance"]
        assert '"final_dist_m": 0.123' in body["guidance"]
        assert "view_env_state" in body["text"]
        assert "data_base64" not in body["text"]
