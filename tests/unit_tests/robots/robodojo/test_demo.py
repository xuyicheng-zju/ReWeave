# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""SIMULATED full frozen execution and evaluator originals remain auditable."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from examples.robodojo_demo import run_demo
from reweave.evolution.contracts import (
    EvaluationRecord,
    EvaluationRequest,
    GraphPolicy,
    Instance,
)
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.evidence.trajectory import (
    build_graph,
    episode_features,
    episode_from_record,
)
from reweave.evolution.runtime_manifest import RuntimeManifest, execution_configuration
from reweave.evolution.search.rollout import RolloutConfiguration, RolloutEvaluator

REPOSITORY = Path(__file__).parents[4]


@pytest.fixture(scope="module")
def completed_demo(tmp_path_factory):
    directory = tmp_path_factory.mktemp("robodojo-demo") / "run"
    summary = run_demo(directory)
    manifest = RuntimeManifest.model_validate_json(
        (directory / "baseline/runtime.json").read_text()
    )
    runner = RolloutConfiguration(
        devices=(0,),
        arguments=("--robot", "robodojo"),
        memory_snapshot=str(directory / "memory"),
        runtime_manifest=manifest,
        episode_limits=manifest.configuration["limits"]["episode"],
    )
    request = EvaluationRequest(
        record_id="demo-request",
        bundle_id=summary["bundle_id"],
        instance=Instance(
            record_id="demo-instance",
            task="robodojo:put_bottles_into_dustbin",
            initialization={
                "task": "put_bottles_into_dustbin",
                "layout_set": 0,
                "layout": 0,
            },
        ),
        split="dev",
        manifest_id="demo-manifest",
        execution=manifest.execution_id,
        conditions=manifest.conditions,
    )
    return directory, summary, manifest, runner, request


def test_settled_native_telemetry_enters_graph_without_rewriting_receipt(
    completed_demo,
):
    directory, _, _, _, _ = completed_demo
    receipt = directory / "attempt/result.json"
    original = receipt.read_bytes()
    record = EvaluationRecord.model_validate_json(original)
    trace = directory / "attempt/episode/execution_trace.jsonl"
    trace_original = trace.read_bytes()
    episode = episode_from_record(record, receipt, ())
    graph = build_graph((episode,), GraphPolicy())
    observations = episode_features(graph.episodes[0])
    assert len(episode.frames) == record.usage.physical_steps + 1 == 3
    assert len(observations[0]) == 28
    assert observations[0]["robot.left_ee_pose.2"].value == 0.3
    assert observations[0]["robot.left_ee_pose.2"].unit == "m"
    assert observations[0]["robot.left_ee_pose.3"].unit == "dimensionless"
    assert observations[0]["robot.left_arm_joint_state.0"].unit == "rad"
    assert observations[0]["robot.left_ee_joint_state.0"].unit == "dimensionless"
    assert set(episode.feature_names) == set(observations[0])
    assert episode.receipt.artifact_id == artifact_record(receipt)["artifact_id"]
    assert all(frame.feature_source.path == str(trace) for frame in episode.frames)
    assert receipt.read_bytes() == original and trace.read_bytes() == trace_original
    with pytest.raises(ValidationError, match="PhysicalFeature"):
        episode_features(
            episode.model_copy(update={"feature_format": "physical_scalars"})
        )


def test_simulated_full_demo_runs_kernel_frozen_bundle_typed_vla_and_evaluator(
    completed_demo,
):
    directory, summary, manifest, _, _ = completed_demo
    assert summary["mode"] == "SIMULATED" and summary["physical_success_claim"] is False
    assert summary["simulated_environment_success"] is True
    assert summary["acknowledged_actions"] == summary["evaluated_actions"] == 2
    assert summary["planner_requests"] == 4
    assert summary["archived_note_revisions"] == 1
    assert summary["policy_forwards"] == 1 and summary["termination_class"] == "success"
    result = json.loads((directory / "attempt/result.json").read_text())
    assert (
        result["official_success"] is True
        and result["usage"]["physical_steps_complete"] is True
    )
    assert manifest.resources.routes["planner"].startswith("SIMULATED")
    assert not (directory / "baseline/source").exists()
    context = directory / "attempt/episode/harness_context"
    events = [
        json.loads(line)
        for line in (context / "bundle_events.jsonl").read_text().splitlines()
    ]
    assert any(event["kind"] == "context_built" for event in events)
    assert any(
        event["kind"] == "action_completed" and event["operation"] == "pi0_pick"
        for event in events
    )
    for operation in ("write_text_file", "read_text_file"):
        assert any(
            event["kind"] == "action_completed" and event["operation"] == operation
            for event in events
        )
    notes = directory / "attempt/episode/planner_memory"
    revisions = list((notes / "revisions").glob("*.json"))
    assert len(revisions) == 1
    revision = json.loads(revisions[0].read_text())
    assert revision["verification"] == "unverified" and revision["public_sources"]
    assert revision["text"] == (notes / "drafts/task-family/conditions.md").read_text()
    assert "SIMULATED" in revision["text"]
    assert (context.parent / "native/native_turns.jsonl").is_file()
    assert (context / "capability_catalog.json").is_file()
    assert all(event.get("bundle_id") == summary["bundle_id"] for event in events)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "fill_pen_holder"),
        ("layout", 1),
        ("success", False),
        ("success_source", "agent"),
    ],
)
def test_evaluator_rejects_result_identity_or_verdict_tampering(
    completed_demo, field, value
):
    directory, _, _, runner, request = completed_demo
    path = directory / "attempt/episode/result.json"
    original = path.read_bytes()
    content = json.loads(original)
    content[field] = value
    try:
        path.write_text(json.dumps(content))
        with pytest.raises(ValueError, match="RoboDojo evaluator"):
            RolloutEvaluator(
                runner, repository=REPOSITORY, workspace=directory / "search"
            )._close(request, directory / "attempt")
    finally:
        path.write_bytes(original)


def test_layout_is_instance_identity_not_frozen_execution_setting(completed_demo):
    _, _, manifest, runner, request = completed_demo
    arguments = manifest.configuration["arguments"]
    assert "layout" not in arguments
    task = request.instance.initialization["task"]
    assert execution_configuration(
        {**arguments, "task": task, "layout": 0}, {}
    ) == execution_configuration({**arguments, "task": task, "layout": 9}, {})
    other = request.instance.model_copy(
        update={
            "initialization": {
                "task": "put_bottles_into_dustbin",
                "layout_set": 0,
                "layout": 9,
            }
        }
    )
    argv = runner.build_arguments(
        other,
        directory=Path("/tmp/not-launched"),
        memory=Path(runner.memory_snapshot),
        device=0,
    )
    assert argv[argv.index("--layout") + 1] == "9"


@pytest.mark.parametrize(
    "module",
    [
        "reweave.cli.main",
        "reweave.benchmarks.robodojo.env_server",
        "reweave.robots.components.pi05_vla_server",
    ],
)
def test_cli_help_needs_no_isaac_or_model_initialization(module):
    command = [sys.executable, "-m", module]
    if module == "reweave.cli.main":
        command += ["--robot", "robodojo"]
    response = subprocess.run(
        [*command, "--help"], cwd=REPOSITORY, capture_output=True, text=True, timeout=30
    )
    assert response.returncode == 0, response.stderr
    assert "usage:" in response.stdout
    assert "Isaac" in response.stdout or "robodojo" in response.stdout


@pytest.fixture
def unavailable_verdict_episode(completed_demo, tmp_path, request):
    """Real aborted frozen Planner + real finalizer preserve unavailable verdict."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
    from reweave.benchmarks.robodojo.robot_spec import finalize_run
    from reweave.dashboard.events import NullDashboardEventSink
    from reweave.evaluation.result import RunFinalizationContext
    from reweave.evolution.bundle import AgentBundle
    from reweave.planner.base import ACTION_OUTCOME_UNKNOWN, API_ERROR
    from reweave.runtime.kernel import EpisodeLimits
    from reweave.runtime.model_budget import ModelServiceBudgetExceeded
    from reweave.utils.persistence import atomic_json
    from tests.native_fixture import ScriptedNativePlanner

    captured, _, _, runner, evaluation_request = completed_demo
    termination = request.param
    directory = tmp_path / "attempt"
    output = directory / "episode"
    rpc = SimulatedEnvironmentRPC(terminal_after=99, acknowledgement="missing")
    toolkit, _, _ = create_simulated_toolkit(output, rpc=rpc)
    toolkit.export_harness_context(str(output))

    def aborted_planner(messages, info):
        if termination == API_ERROR:
            raise ModelServiceBudgetExceeded(
                "controller service cost cap exhausted before an official verdict",
                retryable=False,
            )
        return ModelResponse(
            parts=[ToolCallPart("pi0_pick", {"max_chunks": 1}, "unknown-action")]
        )

    planner = ScriptedNativePlanner(
        FunctionModel(aborted_planner),
        max_tokens=123,
        dashboard_events=NullDashboardEventSink(),
    )
    planner.bind_bundle(
        AgentBundle.model_validate_json(
            (captured / "baseline/bundle.json").read_text()
        ),
        episode_limits=EpisodeLimits(
            program_calls=20, gateway_calls=20, wall_time_s=30
        ),
    )
    result = planner.solve(
        system_prompt="SIMULATED unavailable verdict",
        user_message="Inspect the fixture",
        toolkit=toolkit,
        max_turns=5,
    )
    assert result.termination_class == termination and result.error
    assert termination in {API_ERROR, ACTION_OUTCOME_UNKNOWN}
    finalize_run(
        RunFinalizationContext(
            output_dir=output,
            robot_name="robodojo",
            task_desc=evaluation_request.instance.initialization,
            environment_success=None,
            termination_class=termination,
            agent_error=result.error,
            elapsed_s=1.0,
            planner="SIMULATED",
            model="scripted",
            reasoning_effort="none",
            max_turns=5,
            planner_timeout_s=30,
            finish_result=None,
            stats=result.stats,
        )
    )
    atomic_json(
        output / "transcript_simulated.json",
        {
            "mode": "SIMULATED",
            "run_configuration": {
                "robot_name": "robodojo",
                **evaluation_request.instance.initialization,
            },
            "bundle_id": evaluation_request.bundle_id,
            "official_success": False,
            "termination_class": termination,
            "api_error_retryable": result.api_error_retryable,
            "stats": result.stats,
            "error": result.error,
        },
    )
    atomic_json(directory / "exit.json", {"returncode": 0, "elapsed_s": 1.0})
    toolkit.close()
    evaluator = RolloutEvaluator(
        runner, repository=REPOSITORY, workspace=tmp_path / "search"
    )
    return evaluator, evaluation_request, directory


@pytest.mark.parametrize(
    "unavailable_verdict_episode",
    ["api_error", "action_outcome_unknown"],
    indirect=True,
)
def test_unavailable_official_result_preserves_null_and_abort_origin(
    unavailable_verdict_episode,
):
    evaluator, request, directory = unavailable_verdict_episode
    path = directory / "episode/result.json"
    original = path.read_bytes()
    result = json.loads(original)
    assert result["valid"] is False and result["success"] is None
    record = evaluator._close(request, directory)
    assert path.read_bytes() == original
    assert record.official_success is False
    assert record.failure_origin == (
        "infrastructure" if record.infrastructure_retryable is not None else "unknown"
    )
    assert record.usage.physical_steps == 0
    assert record.usage.physical_steps_complete is (record.failure_origin != "unknown")
    assert "result.json" in record.episode.files


@pytest.mark.parametrize(
    "unavailable_verdict_episode",
    ["api_error", "action_outcome_unknown"],
    indirect=True,
)
@pytest.mark.parametrize("fabricated_success", [False, True])
def test_unavailable_official_result_rejects_fabricated_boolean(
    unavailable_verdict_episode, fabricated_success
):
    evaluator, request, directory = unavailable_verdict_episode
    path = directory / "episode/result.json"
    result = json.loads(path.read_text())
    result["success"] = fabricated_success
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="RoboDojo evaluator"):
        evaluator._close(request, directory)


@pytest.mark.parametrize("unavailable_verdict_episode", ["api_error"], indirect=True)
def test_unavailable_official_result_cannot_be_reclassified_as_agent_failure(
    unavailable_verdict_episode,
):
    evaluator, request, directory = unavailable_verdict_episode
    path = directory / "episode/transcript_simulated.json"
    transcript = json.loads(path.read_text())
    transcript["termination_class"] = "agent_execution_failure"
    transcript["api_error_retryable"] = None
    path.write_text(json.dumps(transcript))
    with pytest.raises(ValueError, match="RoboDojo evaluator"):
        evaluator._close(request, directory)


def test_simulated_rpc_returns_compact_snapshots_and_selected_lossless_rgb(tmp_path):
    import io

    import numpy as np
    from PIL import Image

    from examples.robodojo_demo import SimulatedEnvironmentRPC

    rpc = SimulatedEnvironmentRPC(directory=tmp_path, terminal_after=99)
    assert rpc.call("env.get_env_meta")["observation_transport"] == "server_snapshot"
    initial = rpc.call("env.reset")

    def no_arrays(value):
        assert not isinstance(value, np.ndarray)
        if isinstance(value, dict):
            for child in value.values():
                no_arrays(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                no_arrays(child)

    no_arrays(initial)
    assert initial["physical_step"] == 0
    assert "oracle_success" not in initial
    assert "object_pose" not in initial["state"]
    assert initial == rpc.call("env.get_obs")
    reference = initial["observation_ref"]
    receipt = rpc.call("env.retain_observation", kwargs={"observation_ref": reference})
    assert Path(receipt["archive_path"]).is_file()
    image = rpc.call(
        "env.read_rgb",
        kwargs={"observation_ref": reference, "cameras": ["cam_head"]},
    )
    assert set(image) == {"cam_head"}
    with Image.open(io.BytesIO(image["cam_head"])) as decoded:
        assert decoded.format == "PNG" and decoded.size == (5, 4)
    ack = rpc.call("env.step", args=(np.zeros(14),))
    no_arrays(ack[0])
    assert ack[0]["physical_step"] == ack[3]["status"]["step"] == 1
    assert ack[0]["observation_ref"] != reference
    # Historical RGB remains explicit retained evidence; current-only perception
    # cannot silently run on an older snapshot after the physical step.
    assert (
        rpc.call(
            "env.read_rgb",
            kwargs={"observation_ref": reference, "cameras": ["cam_head"]},
        )
        == image
    )
    with pytest.raises(ValueError, match="stale"):
        rpc.call(
            "env.back_project_ref",
            kwargs={"observation_ref": reference, "row": 0, "col": 0},
        )
    with pytest.raises(ValueError, match="raw-array"):
        rpc.call("env.render_camera", kwargs={"camera_name": "cam_head", "depth": True})


@pytest.mark.parametrize("acknowledgement", ["missing", "malformed"])
def test_simulated_unknown_action_does_not_revive_previous_snapshot(
    tmp_path, acknowledgement
):
    import numpy as np

    from examples.robodojo_demo import SimulatedEnvironmentRPC

    rpc = SimulatedEnvironmentRPC(directory=tmp_path, acknowledgement=acknowledgement)
    rpc.call("env.reset")
    if acknowledgement == "missing":
        with pytest.raises(TimeoutError, match="lost acknowledgement"):
            rpc.call("env.step", args=(np.zeros(14),))
    else:
        assert rpc.call("env.step", args=(np.zeros(14),)) == {
            "status": "not-an-acknowledgement"
        }
    assert rpc.actions == 1 and not rpc.observations.settled
    with pytest.raises(ValueError, match="unsettled"):
        rpc.call("env.get_obs")


def test_simulated_policy_runs_on_server_snapshot_without_action_execution(tmp_path):
    from examples.robodojo_demo import create_simulated_toolkit

    toolkit, rpc, policy = create_simulated_toolkit(tmp_path)
    current = toolkit._primitives.env.current_observation_ref
    prediction = toolkit._primitives.env.predict_ref(
        current, prompt=rpc.observation["instruction"]
    )
    assert policy.calls == 1 and prediction.actions.shape == (4, 14)
    assert rpc.actions == 0
    assert rpc.observations.vla is not None
    assert toolkit._primitives.env.current_observation_ref == current
