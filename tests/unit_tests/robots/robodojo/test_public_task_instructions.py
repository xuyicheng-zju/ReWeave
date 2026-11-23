# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Public task supplements reach authors and Planner without physical work."""

import ast
import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.benchmarks.robodojo.access import public_observation
from reweave.benchmarks.robodojo.robot_spec import get_robot_spec
from reweave.benchmarks.robodojo.tasks import (
    _COMPLETION_HOME_CHECK_TASKS,
    _COMPLETION_HOME_TRIGGER_TASKS,
    _NATIVE_STEP_LIMITS,
    _PROCESS_SCORE_TASKS,
    _QUERY_FAILURE_TASKS,
    _TRIGGER_COMPLETION_TASKS,
    TASK_RULE_REVIEWS,
    UPSTREAM_CLARIFICATION,
    native_step_limit,
    public_task_supplement,
    resolve_instruction,
    task_rule_review,
    validate_task_rule_coverage,
)
from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.runtime_manifest import RuntimeAuthoringContract
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.planner.providers.codex import CodexDecisionSession
from reweave.runtime.kernel import EpisodeLimits
from tests.native_fixture import ScriptedNativePlanner
from tests.unit_tests.harness.planner.test_native_provider_codex import FakeTransport
from tests.unit_tests.robots.robodojo import test_demo

completed_demo = test_demo.completed_demo
NATIVE_INSTRUCTION = (
    "Play tic-tac-toe as the first player and fill the board with the opponent."
)


def native_task(task_name, instruction):
    # CollectEnv and official EvalEnv set this field from their task config.
    return SimpleNamespace(
        task_name=task_name,
        obs_manager=SimpleNamespace(
            desc_manager=SimpleNamespace(get_one_description=lambda: [instruction])
        ),
    )


@pytest.mark.parametrize("limit, fraction", [(200, "25.00%"), (1050, "4.76%")])
def test_online_budget_prompt_uses_actual_task_limit_and_refreshes_after_action(
    tmp_path, limit, fraction
):
    rpc = SimulatedEnvironmentRPC(limit=limit, terminal_after=999)
    toolkit, _, _ = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        # Even an editable context that omits budget guidance receives the
        # fixed online reminder after C, without modifying its stored source.
        original = {"instructions": "Task-specific evolved guidance.", "messages": []}
        first = toolkit.finalize_model_context(original)
        assert original["messages"] == []
        assert f"{limit} environment control steps" in first["instructions"]
        assert fraction in first["instructions"]
        assert "used=0" in first["messages"][-1]["parts"][0]["content"]
        toolkit.execute_tool("set_gripper", {"arm": "left", "gripper": -1})
        second = toolkit.finalize_model_context(original)
        assert second["instructions"] == first["instructions"]
        reminder = second["messages"][-1]["parts"][0]["content"]
        assert f"total={limit}, used=1, remaining={limit - 1}" in reminder
        assert "two cost up to 100" in reminder
        ModelMessagesTypeAdapter.validate_python(second["messages"])
    finally:
        toolkit.close()


def test_native_task_language_retains_goal_and_labels_public_project_supplement():
    env = native_task("play_tic_tac_toe", NATIVE_INSTRUCTION)
    supplement = public_task_supplement(env.task_name)
    assert resolve_instruction(env) == NATIVE_INSTRUCTION + "\n\n" + supplement
    assert "Project task-rule supplement for play_tic_tac_toe" in supplement
    assert "https://github.com/RoboDojo-Benchmark/RoboDojo/issues/54" in supplement
    assert "2026-10-08" in supplement
    assert "return both controlled arms near their initial poses" in supplement
    assert "before acquiring the next player piece" in supplement
    assert "first four player placements" in supplement
    assert "At task completion" not in supplement
    assert "official failure condition" in supplement
    # The native public observation remains available verbatim as the source.
    observation = {"instruction": NATIVE_INSTRUCTION, "state": {}, "vision": {}}
    assert public_observation(observation)["instruction"] == NATIVE_INSTRUCTION


@pytest.mark.parametrize("task", ["general_pickup", "insert_key"])
def test_other_task_instructions_remain_byte_identical(task):
    original = "  Put the visible object away.  "
    assert resolve_instruction(native_task(task, original)) == original


def test_all_54_resolutions_preserve_official_bytes_and_select_one_review():
    for task_name, review in TASK_RULE_REVIEWS.items():
        official = f"  Official instruction for {task_name}.  "
        resolved = resolve_instruction(native_task(task_name, official))
        if review.supplement:
            assert resolved == official + "\n\n" + review.supplement
            assert resolved.count("Project task-rule supplement") == 1
        else:
            assert resolved == official


def test_every_official_task_has_one_explicit_review(tmp_path):
    config = tmp_path / "task/RoboDojo/config"
    config.mkdir(parents=True)
    for task_name in TASK_RULE_REVIEWS:
        (config / f"{task_name}.yml").touch()
    (config / "_task.yml").touch()
    validate_task_rule_coverage(tmp_path)
    assert len(TASK_RULE_REVIEWS) == 54
    assert {review.review_status for review in TASK_RULE_REVIEWS.values()} == {
        "supplemented",
        "reviewed_no_additional_rule",
    }
    assert all(review.task_name == name for name, review in TASK_RULE_REVIEWS.items())
    assert all(
        review.source_urls[1] == UPSTREAM_CLARIFICATION
        for review in TASK_RULE_REVIEWS.values()
    )
    assert (
        sum(
            review.completion_home != "not_required"
            for review in TASK_RULE_REVIEWS.values()
        )
        == 45
    )
    assert (
        sum(
            review.completion_home in {"check", "trigger_check"}
            for review in TASK_RULE_REVIEWS.values()
        )
        == 44
    )
    assert (
        TASK_RULE_REVIEWS["play_tic_tac_toe"].completion_home
        == "opponent_turn_gate_only"
    )
    assert TASK_RULE_REVIEWS["imitate_sorting_sequence"].completion_home == "check"
    assert sum(review.query_failure for review in TASK_RULE_REVIEWS.values()) == 12
    assert sum(review.process_scoring for review in TASK_RULE_REVIEWS.values()) == 34
    assert {review.native_step_limit for review in TASK_RULE_REVIEWS.values()} != {1100}
    assert native_step_limit("fasten_screws") == 1900
    assert native_step_limit("play_tic_tac_toe") == 1100


def test_missing_inventory_review_and_unknown_lookup_fail_visibly(tmp_path):
    config = tmp_path / "task/RoboDojo/config"
    config.mkdir(parents=True)
    for task_name in (*TASK_RULE_REVIEWS, "new_upstream_task"):
        (config / f"{task_name}.yml").touch()
    with pytest.raises(ValueError, match="missing=.*new_upstream_task"):
        validate_task_rule_coverage(tmp_path)
    with pytest.raises(ValueError, match="no public task-rule review"):
        task_rule_review("new_upstream_task")


def test_supplements_are_bounded_source_attributed_and_nonduplicated():
    supplements = [
        review.supplement for review in TASK_RULE_REVIEWS.values() if review.supplement
    ]
    assert len(supplements) == 51
    assert len(set(supplements)) == len(supplements)
    assert max(map(len, supplements)) < 1_200
    for supplement in supplements:
        assert supplement.count("Project task-rule supplement") == 1
        assert supplement.count("official task source") == 1
        assert supplement.count("issues/54") == 1


def _official_source_root() -> Path:
    value = os.environ.get("ROBODOJO_AUDIT_SOURCE_ROOT")
    if value is None:
        pytest.skip("ROBODOJO_AUDIT_SOURCE_ROOT is required for source-context audit")
    root = Path(value)
    if not (root / "task/RoboDojo/tasks").is_dir():
        pytest.fail(f"invalid ROBODOJO_AUDIT_SOURCE_ROOT: {root}")
    return root


def _enclosing_function(node, parents):
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.FunctionDef):
            return current
    raise AssertionError("call has no enclosing function")


def _ancestor_call_names(node, parents, function):
    names = []
    current = node
    while current in parents and current is not function:
        current = parents[current]
        if isinstance(current, ast.Call):
            names.append(ast.unparse(current.func).rsplit(".", 1)[-1])
    return names


def _home_contexts(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    contexts = set()
    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("all_robot_back_to_origin")
    ):
        function = _enclosing_function(call, parents)
        ancestors = _ancestor_call_names(call, parents, function)
        if function.name == "query_support_arm_traj":
            contexts.add("opponent_turn_gate_only")
            continue
        if "trigger_check" in ancestors:
            contexts.add("trigger_check")
            continue
        if {"check", "check_single_env"}.intersection(ancestors):
            contexts.add("check")
            continue
        parent = parents.get(call)
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Attribute)
            and parent.func.attr == "append"
            and isinstance(parent.func.value, ast.Name)
        ):
            variable = parent.func.value.id
            registered = any(
                isinstance(candidate, ast.Call)
                and ast.unparse(candidate.func).rsplit(".", 1)[-1]
                in {"check", "check_single_env"}
                and any(
                    isinstance(argument, ast.Name) and argument.id == variable
                    for argument in (
                        *candidate.args,
                        *(keyword.value for keyword in candidate.keywords),
                    )
                )
                for candidate in ast.walk(function)
            )
            if registered:
                contexts.add("check")
                continue
        contexts.add(f"unclassified:{function.name}:{call.lineno}")
    return contexts


def test_official_source_context_matches_all_54_review_entries():
    source = _official_source_root() / "task/RoboDojo/tasks"
    files = {path.stem: path for path in source.glob("*.py")}
    assert set(files) == set(TASK_RULE_REVIEWS)
    home_by_context = {
        context: {
            name for name, path in files.items() if context in _home_contexts(path)
        }
        for context in ("check", "trigger_check", "opponent_turn_gate_only")
    }
    assert home_by_context["check"] == set(_COMPLETION_HOME_CHECK_TASKS)
    assert home_by_context["trigger_check"] == set(_COMPLETION_HOME_TRIGGER_TASKS)
    assert home_by_context["opponent_turn_gate_only"] == {"play_tic_tac_toe"}
    assert not {
        context
        for path in files.values()
        for context in _home_contexts(path)
        if context.startswith("unclassified:")
    }

    def tasks_with(registrations):
        return {
            name
            for name, path in files.items()
            if any(
                isinstance(node, ast.Call)
                and ast.unparse(node.func).rsplit(".", 1)[-1] in registrations
                for node in ast.walk(ast.parse(path.read_text()))
            )
        }

    assert tasks_with({"query", "trigger_query"}) == set(_QUERY_FAILURE_TASKS)
    assert tasks_with(
        {"score", "score_single_env", "trigger_score", "final_score"}
    ) == set(_PROCESS_SCORE_TASKS)
    assert tasks_with({"final_check"}) == set()
    assert tasks_with({"trigger_check"}) == set(_TRIGGER_COMPLETION_TASKS)
    source_step_limits = {}
    for name, path in files.items():
        tree = ast.parse(path.read_text())
        values = [
            ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "step_lim"
                for target in node.targets
            )
        ]
        assert len(values) == 1
        source_step_limits[name] = values[0]
    assert source_step_limits == _NATIVE_STEP_LIMITS


@pytest.mark.parametrize("instruction", ["", " ", "Place <object> here."])
def test_supplement_does_not_hide_invalid_native_language(instruction):
    with pytest.raises(ValueError, match="instruction"):
        resolve_instruction(native_task("play_tic_tac_toe", instruction))


def test_real_robodojo_prompt_renderer_interpolates_every_declared_placeholder():
    prompts = get_robot_spec().prompts
    variables = {
        "task": "play_tic_tac_toe",
        "memory_dir": "/private/simulated-memory",
        "env_cfg_type": "arx_x5",
    }
    system = prompts.render("system", variables=variables)
    user = prompts.render("user", variables=variables)
    assert user == (
        "Task: play_tic_tac_toe. Read the initialized scene and its resolved "
        "instruction.\n"
    )
    assert "Reference text under /private/simulated-memory" in system
    assert "Robot configuration: arx_x5." in system
    assert (
        re.search(r"\{\{?\s*(?:task|memory_dir|env_cfg_type)\s*\}\}?", system + user)
        is None
    )
    # Strict rendering still diagnoses a missing required template variable.
    with pytest.raises(KeyError, match="task"):
        prompts.render("user", variables={})


def test_disabled_memory_prompt_does_not_advertise_a_corpus_path():
    system = get_robot_spec().prompts.render(
        "system",
        variables={"memory_profile": "none", "env_cfg_type": "arx_x5"},
    )
    assert "Published memory is disabled" in system
    assert "Reference text under" not in system
    assert "{{memory_dir}}" not in system


@pytest.mark.parametrize(
    "task_name,instruction",
    [
        ("play_tic_tac_toe", NATIVE_INSTRUCTION),
        ("make_toast", "Put two slices in the toaster and press the lever."),
        ("pick_from_conveyor_by_image", "Lift the basket and place the target in it."),
        ("play_Xylophone", "Strike all keys from left to right."),
    ],
)
def test_same_public_supplement_reaches_author_contract_and_planner(
    completed_demo, tmp_path, task_name, instruction
):
    captured, _, _, _, _ = completed_demo
    bundle = AgentBundle.model_validate_json(
        (captured / "baseline/bundle.json").read_text()
    )
    native = native_task(task_name, instruction)
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    rpc.meta["task"] = native.task_name
    rpc.observation["instruction"] = instruction
    original_call = rpc.call

    def call(method, args=(), kwargs=None, timeout_s=None):
        if method == "env.get_task_language":
            return resolve_instruction(native)
        return original_call(method, args, kwargs, timeout_s)

    rpc.call = call
    toolkit, _, policy = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    limits = EpisodeLimits(program_calls=20, gateway_calls=20, wall_time_s=30)
    task_message = get_robot_spec().prompts.render(
        "user", variables={"task": native.task_name}
    )
    runtime = BundleRuntime(bundle, toolkit, task_message, episode_limits=limits)
    contract = RuntimeAuthoringContract.from_runtime(runtime)
    tool = next(t for t in contract.tools if t["name"] == "view_env_state")
    capability = next(
        t for t in contract.capability_catalog if t["name"] == "view_env_state"
    )
    supplement = public_task_supplement(native.task_name)
    assert supplement not in tool["description"]
    assert capability["description"] == tool["description"]
    assert (
        next(t for t in runtime.specs() if t["name"] == "view_env_state")["description"]
        == tool["description"]
    )
    # This is the exact payload the existing NativeBundleAuthor writes to gateway.
    author_gateway = json.loads(contract.model_dump_json())
    assert (
        supplement
        not in next(
            t for t in author_gateway["tools"] if t["name"] == "view_env_state"
        )["description"]
    )
    requests = []

    def model(messages, info):
        requests.append(messages)
        actual_tool = next(t for t in info.function_tools if t.name == "view_env_state")
        assert actual_tool.description == tool["description"]
        assert "Official RoboDojo task instruction (verbatim):" in info.instructions
        assert instruction in info.instructions
        assert "Project task-rule supplement:" in info.instructions
        assert info.instructions.count(supplement) == 1
        assert public_task_supplement("make_kong") not in info.instructions
        assert info.developer_instructions == info.instructions
        assert (
            f"{rpc.meta['max_episode_steps']} environment control steps"
            in info.instructions
        )
        assert "One VLA chunk can execute up to 50 physical steps" in info.instructions
        if len(requests) == 1:
            return ModelResponse(parts=[ToolCallPart("view_env_state", {}, "scene")])
        text = ModelMessagesTypeAdapter.dump_json(messages).decode()
        assert "Current RoboDojo physical budget:" in text
        assert f"total={rpc.meta['max_episode_steps']}" in text
        assert instruction in text
        assert supplement in text
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "finish", {"status": "stuck", "summary": "CPU audit"}, "end"
                )
            ]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(model),
        dashboard_events=NullDashboardEventSink(),
        output_dir=tmp_path / "planner",
    )
    planner.bind_bundle(bundle, episode_limits=limits)
    try:
        result = planner.solve(
            system_prompt="SIMULATED public task context",
            user_message=task_message,
            toolkit=toolkit,
            max_turns=3,
        )
        assert result.error is None
        assert len(requests) == 2
        assert (
            planner.scripted_sessions[0].developer_instructions.count(supplement) == 1
        )
        assert rpc.actions == policy.calls == 0
        record = toolkit.state.get(-1)
        assert record.extras["task_language"] == resolve_instruction(native)
        assert record.extras["official_task_instruction"] == instruction
        assert record.extras["task_rule_supplement"] == supplement
        assert toolkit.state.get(-1).state["instruction"] == instruction

        async def verify_codex_delivery():
            workspace = planner.scripted_sessions[0].cwd
            developer_instructions = planner.scripted_sessions[0].developer_instructions
            provider_directory = tmp_path / "codex-provider"
            started_transport = FakeTransport(workspace)
            started = CodexDecisionSession(
                provider_directory, transport=started_transport
            )
            await started.start(developer_instructions, workspace)
            await started.close()
            resumed_transport = FakeTransport(workspace)
            resumed = CodexDecisionSession(
                provider_directory, transport=resumed_transport
            )
            await resumed.start(developer_instructions, workspace)
            await resumed.close()
            start = next(
                params
                for method, params in started_transport.calls
                if method == "thread/start"
            )
            resume = next(
                params
                for method, params in resumed_transport.calls
                if method == "thread/resume"
            )
            return start, resume

        start, resume = asyncio.run(verify_codex_delivery())
        assert start["developerInstructions"] == resume["developerInstructions"]
        assert start["developerInstructions"].count(supplement) == 1
        assert (
            "One VLA chunk can execute up to 50 physical steps"
            in start["developerInstructions"]
        )
        assert public_task_supplement("make_kong") not in start["developerInstructions"]
        assert start["config"]["project_doc_max_bytes"] == 0
        assert resume["config"]["project_doc_max_bytes"] == 0
    finally:
        toolkit.close()


def test_policy_is_prompted_with_the_official_sentence_not_the_supplement(tmp_path):
    native = native_task("play_tic_tac_toe", NATIVE_INSTRUCTION)
    assert public_task_supplement(native.task_name)
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    rpc.meta["task"] = native.task_name
    rpc.observation["instruction"] = NATIVE_INSTRUCTION
    original_call = rpc.call

    def call(method, args=(), kwargs=None, timeout_s=None):
        if method == "env.get_task_language":
            return resolve_instruction(native)
        return original_call(method, args, kwargs, timeout_s)

    rpc.call = call
    toolkit, _, policy = create_simulated_toolkit(tmp_path / "episode", rpc=rpc)
    try:
        toolkit.execute_tool("pi0_pick", {"arm": "right", "max_chunks": 1})
        assert policy.requests
        sent = json.dumps(policy.requests[0], default=str)
        assert NATIVE_INSTRUCTION in sent
        assert "Project task-rule supplement" not in sent
    finally:
        toolkit.close()
