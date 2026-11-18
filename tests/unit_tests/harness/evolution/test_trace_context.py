# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Trace navigation preserves physical identity, evidence coverage and chronology."""

import json

import pytest
from PIL import Image

from reweave.evolution.bundle import new_record_id
from reweave.evolution.evidence.records import EpisodeEvidence, inventory_episode
from reweave.evolution.evidence.trace import (
    build_episode_context,
    query_trace,
    resolve_trace_reference,
)
from reweave.robots.execution_trace import ExecutionTrace


def _events():
    # Frozen retired-protocol originals, not a use of the current trace writer.
    events = [
        {
            "event_type": "active_execution_started",
            "execution_id": "workflow",
            "tool_name": "pull",
            "arguments": {},
        }
    ]
    for index in range(2):
        call = f"move-{index}"
        events.extend(
            [
                {
                    "event_type": "physical_tool_started",
                    "execution_ref": "workflow",
                    "physical_call_ref": call,
                    "tool_name": "move_to",
                    "arguments": {"xyz": [index, 0, 0]},
                },
                {
                    "event_type": "control_step",
                    "control_step_index": index,
                    "execution_ref": "workflow",
                    "physical_call_ref": call,
                    "tool_name": "move_to",
                    "forward_ref": None,
                    "action_index": 0,
                    "action": [0.1, 0, 0, 0, 0, 0, -1],
                    "terminated": index == 1,
                    "truncated": False,
                    "public_evidence": {"robot.eef.motion_m": 0.1}
                    if index == 0
                    else {},
                    "critic_private_evidence": {
                        "privileged.task.progress": bool(index)
                    },
                },
                {
                    "event_type": "physical_tool_finished",
                    "execution_ref": "workflow",
                    "physical_call_ref": call,
                },
                {
                    "event_type": "episode_tool_program_capability",
                    "execution_ref": "workflow",
                    "source_modification_ref": "generated",
                    "tool_program": "pull",
                    "capability": "move_to",
                    "arguments": {"xyz": [index, 0, 0]},
                    "effect": "physical",
                    "result": {"step": index + 1},
                    "environment_write": True,
                },
            ]
        )
    events.extend(
        [
            {
                "event_type": "active_execution_finished",
                "execution_ref": "workflow",
                "outcome": "completed",
            },
            {
                "event_type": "active_execution_started",
                "execution_id": "release",
                "tool_name": "release",
                "arguments": {},
            },
            {
                "event_type": "control_step",
                "control_step_index": 2,
                "execution_ref": "release",
                "physical_call_ref": "release",
                "tool_name": "release",
                "forward_ref": None,
                "action_index": 0,
                "action": [0, 0, 0, 0, 0, 0, -1],
                "terminated": False,
                "truncated": True,
                "public_evidence": {"robot.eef.motion_m": 0.0},
            },
            {
                "event_type": "active_execution_finished",
                "execution_ref": "release",
                "outcome": "completed",
            },
        ]
    )
    return [{**row, "event_index": index} for index, row in enumerate(events)]


def _messages():
    calls = [
        {"type": "tool_use", "id": identity, "name": name, "input": {}}
        for identity, name in (("call-pull", "pull"), ("call-release", "release"))
    ]
    # Parallel tool results need not arrive in the assistant's declaration order.
    return [{"role": "assistant", "content": calls}] + [
        {
            "role": "tool",
            "tool_call_id": identity,
            "name": name,
            "content": json.dumps({"active_execution_ref": execution, "step": step}),
        }
        for identity, name, execution, step in (
            ("call-release", "release", "release", 3),
            ("call-pull", "pull", "workflow", 2),
        )
    ]


def _episode(
    tmp_path,
    *,
    events=None,
    image_count=0,
    messages=None,
    requests=None,
    task_languages=None,
    task_rule_supplements=None,
    transcript_fields=None,
):
    source = tmp_path / "original"
    source.mkdir()
    (source / "execution_trace.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n" for row in (_events() if events is None else events)
        )
    )
    (source / "transcript_test.json").write_text(
        json.dumps(
            {
                "messages": _messages() if messages is None else messages,
                "official_success": False,
                "termination_class": "task_failure",
                "task_message": "open drawer",
                "run_configuration": {},
                **(transcript_fields or {}),
            }
        )
    )
    states = [
        {"step_idx": 0, "state": {}, "artifacts": [], "command": None},
        {
            "step_idx": 1,
            "state": {},
            "artifacts": [],
            "command": {"action": "move_to", "episode_generated_tool": "pull"},
        },
        {
            "step_idx": 2,
            "state": {},
            "artifacts": [],
            "command": {"action": "move_to", "episode_generated_tool": "pull"},
        },
        {
            "step_idx": 3,
            "state": {},
            "artifacts": [],
            "command": {"action": "release", "active_execution_ref": "release"},
        },
    ]
    if task_languages is not None:
        for state, language in zip(states, task_languages, strict=True):
            state["extras"] = {"task_language": language}
    if task_rule_supplements is not None:
        for state, supplement in zip(states, task_rule_supplements, strict=True):
            language = state["extras"]["task_language"]
            official = (
                language.removesuffix("\n\n" + supplement) if supplement else language
            )
            state["extras"].update(
                official_task_instruction=official,
                task_rule_supplement=supplement,
            )
    for index in range(image_count):
        name = f"camera_{index:02}.png"
        states[0]["artifacts"].append(name)
        directory = source / name
        directory.mkdir()
        Image.new("RGB", (24, 24), (index % 256, 0, 0)).save(directory / "00.png")
    (source / "states.json").write_text(json.dumps({"steps": states}))
    if requests is not None:
        context = source / "harness_context"
        context.mkdir()
        (context / "planner_requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in requests)
        )
    files = inventory_episode(source)
    evidence = EpisodeEvidence(evidence_id=new_record_id(), files=files)
    destination = tmp_path / ".archive/episodes" / evidence.evidence_id
    destination.parent.mkdir(parents=True)
    source.rename(destination)
    return evidence, destination


def _view(workspace, evidence, view, **filters):
    return query_trace(
        workspace, evidence_id=evidence.evidence_id, view=view, **filters
    )["data"]


def test_outcome_navigation_preserves_completion_disagreement_and_original_limits(
    tmp_path,
):
    fields = {
        "finish": {"status": "success", "summary": "All visible objects placed"},
        "stats": {"visible_turns": 100},
        "agent_run_contract": {"planner_max_turns": 100, "physical_step_budget": 300},
        "error": None,
    }
    evidence, original = _episode(tmp_path, transcript_fields=fields)
    before = (original / "transcript_test.json").read_bytes()
    reference = build_episode_context(tmp_path, evidence)
    index = tmp_path / reference["path"]
    value = json.loads(index.read_text())
    assert value["outcome"]["official_success"] is False
    assert all(value["outcome"][key] == val for key, val in fields.items())
    # Older cached indexes must not hide the missing finish/limit evidence.
    del value["outcome"]["finish"]
    index.write_text(json.dumps(value))
    build_episode_context(tmp_path, evidence)
    assert json.loads(index.read_text())["outcome"]["finish"] == fields["finish"]
    assert (original / "transcript_test.json").read_bytes() == before


@pytest.mark.parametrize(
    "defect",
    [
        "prediction_identity",
        "observation_identity",
        "traversal",
        "missing_image",
        "missing_observation",
    ],
)
def test_vla_input_index_preserves_missingness_and_rejects_mismatched_evidence(
    tmp_path, defect
):
    from reweave.evolution.evidence.trace import _Episode, _forward_inputs

    source = tmp_path / "original"
    prediction_dir = source / "predictions"
    observation_dir = source / "observations/observation"
    prediction_dir.mkdir(parents=True)
    observation_dir.mkdir(parents=True)
    prediction = {"forward_id": "forward", "observation_id": "observation"}
    observation = {"observation_id": "observation", "cameras": ["head"]}
    if defect == "prediction_identity":
        prediction["forward_id"] = "different"
    if defect == "observation_identity":
        observation["observation_id"] = "different"
    if defect == "traversal":
        prediction["observation_id"] = ".."
    (prediction_dir / "forward.json").write_text(json.dumps(prediction))
    if defect != "missing_observation":
        (observation_dir / "observation.json").write_text(json.dumps(observation))
    if defect != "missing_image":
        Image.new("RGB", (4, 3)).save(observation_dir / "head.png")
    files = inventory_episode(source)
    episode = _Episode(
        tmp_path,
        EpisodeEvidence(evidence_id=new_record_id(), files=files),
        source,
    )
    events = [
        {
            "event_type": "vla_forward",
            "forward_id": "forward",
            "execution_ref": "execution",
            "event_index": 0,
        }
    ]
    if defect in {"prediction_identity", "observation_identity", "traversal"}:
        with pytest.raises(ValueError, match="identity"):
            _forward_inputs(episode, events)
    else:
        inputs, gaps = _forward_inputs(episode, events)
        assert len(gaps) == 1
        if defect == "missing_image":
            assert gaps[0]["kind"] == "vla_input_image_missing"
            assert inputs[0]["images"][0]["source"] is None
        else:
            assert gaps[0]["kind"] == "vla_observation_missing"
            assert inputs == []
    assert inventory_episode(source) == files


def test_navigation_keeps_large_arguments_and_results_only_in_originals(tmp_path):
    body = {"payload": "complete body remains in its original " * 3000}
    events = _events()
    for event in events:
        if event["event_type"] in {"active_execution_started", "physical_tool_started"}:
            event["arguments"] = body
    messages = _messages()
    messages[0]["content"][0]["input"] = body
    messages[2]["content"] = json.dumps(
        {
            "active_execution_ref": "workflow",
            "step": 2,
            "log": {"result": body},
        }
    )
    evidence, original = _episode(tmp_path, events=events, messages=messages)
    before = inventory_episode(original)
    index = build_episode_context(tmp_path, evidence, include_visuals=False)
    timeline = _view(tmp_path, evidence, "timeline")
    assert "complete body remains" not in json.dumps(timeline)
    assert "reported_result" not in json.dumps(timeline)
    assert len(json.dumps(timeline)) < len(json.dumps(body)) / 2
    call = timeline["planner_calls"][0]
    assert resolve_trace_reference(tmp_path, call["arguments_reference"]) == body
    for execution in timeline["executions"]:
        assert (
            resolve_trace_reference(tmp_path, execution["arguments_reference"]) == body
        )
        for physical in execution["physical_calls"]:
            assert (
                resolve_trace_reference(tmp_path, physical["arguments_reference"])
                == body
            )
    result = json.loads(resolve_trace_reference(tmp_path, call["feedback"])["content"])
    assert result["log"]["result"] == body
    assert (
        json.loads((tmp_path / index["path"]).read_text())["format"]
        == "indexed_originals"
    )
    assert inventory_episode(original) == before


def test_repeated_request_diagnostic_uses_original_arguments_not_reference_locations(
    tmp_path,
):
    messages = _messages()
    messages[0]["content"][1].update(name="pull", input={})
    messages[1]["name"] = "pull"
    evidence, _ = _episode(tmp_path, messages=messages)
    build_episode_context(tmp_path, evidence)
    assert _view(tmp_path, evidence, "diagnostics")[
        "episode_identical_planner_requests"
    ] == [["call-pull", "call-release"]]


def test_telemetry_indexes_originals_and_decodes_only_selected_external_rows(
    tmp_path, monkeypatch
):
    from reweave.evolution.evidence import trace as trace_context

    store = tmp_path / "store"
    store.mkdir()
    evidence, original = _episode(store)
    workspace = tmp_path / "author"
    index = build_episode_context(workspace, evidence, original, include_visuals=False)
    navigation = json.loads(
        (workspace / index["path"]).with_name("telemetry.json").read_text()
    )
    assert "requested_actions" not in navigation
    assert "public_evidence" not in navigation
    assert "critic_private_evidence" not in navigation
    assert navigation["feature_names"]["public_evidence"] == ["robot.eef.motion_m"]
    decoded_indices = []
    original_loads = json.loads

    def selected_loads(value, *args, **kwargs):
        result = original_loads(value, *args, **kwargs)
        if isinstance(result, dict) and "event_type" in result:
            decoded_indices.append(result["event_index"])
        return result

    monkeypatch.setattr(trace_context.json, "loads", selected_loads)
    value = _view(
        workspace,
        evidence,
        "telemetry",
        start_step=1,
        end_step=1,
        features=["privileged.task.progress"],
        evidence_roots={evidence.evidence_id: original},
    )
    assert decoded_indices == [navigation["event_indices"][1]]
    assert value["requested_actions"] == [
        next(row["action"] for row in _events() if row.get("control_step_index") == 1)
    ]
    assert value["public_evidence"] == {}
    assert value["critic_private_evidence"] == {"privileged.task.progress": [True]}
    with pytest.raises(ValueError, match="outside this author request"):
        _view(
            workspace,
            evidence,
            "telemetry",
            start_step=1,
            end_step=1,
            evidence_roots={},
        )


def test_task_instructions_preserve_exact_public_text_changes_and_source(tmp_path):
    evidence, _ = _episode(
        tmp_path,
        task_languages=[
            "Open the middle drawer",
            None,
            "Open the middle drawer",
            "Close it",
        ],
    )
    index = build_episode_context(tmp_path, evidence)
    value = json.loads((tmp_path / index["path"]).read_text())["task_instructions"]
    assert value["missing_state_indices"] == [1]
    assert [(row["text"], row["state_indices"]) for row in value["values"]] == [
        ("Open the middle drawer", [0, 2]),
        ("Close it", [3]),
    ]
    for row in value["values"]:
        assert resolve_trace_reference(tmp_path, row["source"]) == row["text"]


def test_offline_task_context_keeps_official_text_and_supplement_distinct(tmp_path):
    official = "Play tic-tac-toe as the first player and fill the board."
    supplement = "Project task-rule supplement: wait for the opponent arm."
    resolved = official + "\n\n" + supplement
    evidence, _ = _episode(
        tmp_path,
        task_languages=[resolved] * 4,
        task_rule_supplements=[supplement] * 4,
    )
    index = build_episode_context(tmp_path, evidence)
    values = json.loads((tmp_path / index["path"]).read_text())["task_instructions"][
        "values"
    ]
    assert values == [
        {
            "text": resolved,
            "official_text": official,
            "task_rule_supplement": supplement,
            "source": values[0]["source"],
            "state_indices": [0, 1, 2, 3],
        }
    ]


@pytest.mark.parametrize(
    "defect",
    [
        "legacy",
        "official_text",
        "task_rule_supplement",
        "source",
        "state_indices",
        "inconsistent_text",
        "wrong_episode",
    ],
)
def test_stale_instruction_index_rebuilds_from_originals_and_preserves_old_bytes(
    tmp_path, defect
):
    import os

    evidence, original = _episode(
        tmp_path,
        task_languages=["Original task", None, "Original task", "Original task"],
    )
    original_inventory = inventory_episode(original)
    result = build_episode_context(tmp_path, evidence, include_visuals=False)
    path = tmp_path / result["path"]
    index = json.loads(path.read_text())
    if defect == "legacy":
        del index["schema_version"]
        for item in index["task_instructions"]["values"]:
            del item["official_text"]
            del item["task_rule_supplement"]
    elif defect == "wrong_episode":
        index["evidence_id"] = "a-different-episode"
    elif defect == "inconsistent_text":
        index["task_instructions"]["values"][0]["text"] = "Invented instruction"
    elif defect == "state_indices":
        index["task_instructions"]["values"][0]["state_indices"] = [True]
    else:
        del index["task_instructions"]["values"][0][defect]
    path.write_text(json.dumps(index))
    old_bytes = path.read_bytes()
    published = tmp_path / "published-index.json"
    os.link(path, published)
    rebuilt = build_episode_context(tmp_path, evidence, include_visuals=False)
    current = json.loads(path.read_text())
    assert rebuilt["path"] == result["path"]
    assert current["task_instructions"]["values"][0]["official_text"] == "Original task"
    assert current["task_instructions"]["values"][0]["task_rule_supplement"] == ""
    assert current["task_instructions"]["missing_state_indices"] == [1]
    assert current["task_instructions"]["values"][0]["state_indices"] == [0, 2, 3]
    assert (
        resolve_trace_reference(
            tmp_path, current["task_instructions"]["values"][0]["source"]
        )
        == "Original task"
    )
    archived = list(
        (tmp_path / "context/retired_trace_indices" / evidence.evidence_id).glob(
            "*/index.json"
        )
    )
    assert len(archived) == 1 and archived[0].read_bytes() == old_bytes
    assert published.read_bytes() == old_bytes
    assert inventory_episode(original) == original_inventory
    assert build_episode_context(tmp_path, evidence, include_visuals=False) == rebuilt


def test_failed_index_rebuild_preserves_previous_cache_and_originals(
    tmp_path, monkeypatch
):
    evidence, original = _episode(tmp_path, task_languages=["Recorded task"] * 4)
    result = build_episode_context(tmp_path, evidence, include_visuals=False)
    path = tmp_path / result["path"]
    index = json.loads(path.read_text())
    del index["schema_version"]
    path.write_text(json.dumps(index))
    before = path.read_bytes()
    originals = inventory_episode(original)

    def fail(_observations):
        raise ValueError("synthetic interrupted derivation")

    monkeypatch.setattr("reweave.evolution.evidence.trace._task_instructions", fail)
    with pytest.raises(ValueError, match="synthetic interrupted derivation"):
        build_episode_context(tmp_path, evidence, include_visuals=False)
    assert path.read_bytes() == before
    assert inventory_episode(original) == originals


def test_external_episode_root_is_referenced_without_copying(tmp_path):
    (tmp_path / "store").mkdir()
    evidence, episode_root = _episode(tmp_path / "store")
    workspace = tmp_path / "author"
    index = build_episode_context(workspace, evidence, episode_root)
    assert not (workspace / ".archive/episodes").exists()
    task = json.loads((workspace / index["path"]).read_text())["task_instructions"]
    reference = (
        task["values"][0]["source"]
        if task["values"]
        else json.loads(
            (
                workspace / "context/trace" / evidence.evidence_id / "timeline.json"
            ).read_text()
        )["source"]
    )
    assert (
        resolve_trace_reference(
            workspace, reference, {evidence.evidence_id: episode_root}
        )
        is not None
    )
    with pytest.raises(ValueError, match="outside this author request"):
        resolve_trace_reference(workspace, reference, {})


def test_missing_task_instructions_do_not_borrow_transcript_template(tmp_path):
    evidence, _ = _episode(tmp_path)
    index = build_episode_context(tmp_path, evidence)
    value = json.loads((tmp_path / index["path"]).read_text())["task_instructions"]
    assert value["values"] == []
    assert value["missing_state_indices"] == [0, 1, 2, 3]


@pytest.mark.parametrize("invalid", [False, "", ["open drawer"]])
def test_invalid_recorded_task_instruction_is_not_silently_reinterpreted(
    tmp_path, invalid
):
    evidence, _ = _episode(tmp_path, task_languages=[invalid, None, None, None])
    with pytest.raises(ValueError, match="recorded task_language"):
        build_episode_context(tmp_path, evidence)


def test_same_named_physical_calls_remain_distinct_and_intermediate_states_do_not_see_future(
    tmp_path,
):
    evidence, _ = _episode(tmp_path)
    build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    workflow = timeline["executions"][0]
    leaves = workflow["physical_calls"][1:]
    assert [call["tool_name"] for call in leaves] == ["move_to", "move_to"]
    assert len({call["physical_call_ref"] for call in leaves}) == 2
    assert [call["control_step_indices"] for call in leaves] == [[0], [1]]
    assert all(call["parent_physical_call_ref"] == "workflow" for call in leaves)
    states = {row["state_index"]: row for row in timeline["observations"]}
    assert states[0]["last_control_step_index"] is None
    assert states[1]["execution_ref"] == states[2]["execution_ref"] == "workflow"
    assert states[1]["last_control_step_index"] == 0
    assert states[2]["last_control_step_index"] == 1
    assert states[3]["last_control_step_index"] == 2


def test_reordered_planner_results_join_by_identity(tmp_path):
    evidence, _ = _episode(tmp_path)
    build_episode_context(tmp_path, evidence)
    calls = {
        row["tool_call_id"]: row
        for row in _view(tmp_path, evidence, "timeline")["planner_calls"]
    }
    assert calls["call-pull"]["execution_ref"] == "workflow"
    assert calls["call-pull"]["feedback"]["pointer"] == "/messages/2"
    assert calls["call-release"]["execution_ref"] == "release"
    assert calls["call-release"]["feedback"]["pointer"] == "/messages/1"


def test_multimodal_tool_feedback_preserves_unsuccessful_result_and_execution_link(
    tmp_path,
):
    messages = _messages()
    for message in messages[1:]:
        result = json.loads(message["content"])
        result["log"] = {"result": {"reached": False, "reason": "pose_not_reached"}}
        message["content"] = [
            {"type": "text", "text": json.dumps(result)},
            {
                "type": "image",
                "source": {"type": "base64", "data": "complete-image-original"},
            },
        ]
    evidence, source = _episode(tmp_path, messages=messages)
    original = (source / "transcript_test.json").read_bytes()
    build_episode_context(tmp_path, evidence)
    view = _view(tmp_path, evidence, "executions", execution_ref="workflow")
    assert view["executions"][0]["outcome"] == "completed"
    assert view["planner_feedback"] == [
        {
            "tool_call_id": "call-pull",
            "name": "pull",
            "request": _view(tmp_path, evidence, "timeline")["planner_calls"][0][
                "request"
            ],
            "source": _view(tmp_path, evidence, "timeline")["planner_calls"][0][
                "feedback"
            ],
        }
    ]
    feedback = resolve_trace_reference(tmp_path, view["planner_feedback"][0]["source"])
    assert json.loads(feedback["content"][0]["text"])["log"]["result"] == {
        "reached": False,
        "reason": "pose_not_reached",
    }
    assert feedback["content"][1]["source"]["data"] == "complete-image-original"
    assert (source / "transcript_test.json").read_bytes() == original


def test_ambiguous_feedback_blocks_remain_explicit_instead_of_picking_a_result(
    tmp_path,
):
    messages = _messages()
    messages[1]["content"] = [
        {"type": "text", "text": messages[1]["content"]},
        {"type": "text", "text": '{"active_execution_ref":"workflow"}'},
    ]
    evidence, _ = _episode(tmp_path, messages=messages)
    build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    call = next(
        c for c in timeline["planner_calls"] if c["tool_call_id"] == "call-release"
    )
    assert call["execution_ref"] is None
    assert any(
        gap["kind"] == "structured_feedback_not_uniquely_decoded"
        for gap in timeline["gaps"]
    )
    for execution in (None, "workflow", "release"):
        result = _view(tmp_path, evidence, "executions", execution_ref=execution)
        assert result["gaps"] == timeline["gaps"]


def test_telemetry_retains_every_step_and_does_not_convert_missing_to_zero(tmp_path):
    evidence, _ = _episode(tmp_path)
    build_episode_context(tmp_path, evidence)
    value = _view(tmp_path, evidence, "telemetry")
    assert value["control_step_indices"] == [0, 1, 2]
    assert value["public_evidence"]["robot.eef.motion_m"] == [0.1, None, 0.0]
    assert value["critic_private_evidence"]["privileged.task.progress"] == [
        False,
        True,
        None,
    ]
    assert value["terminated"] == [False, True, False]
    assert value["truncated"] == [False, False, True]
    assert value["action_indices"] == [0, 0, 0]
    assert value["tool_names"] == ["move_to", "move_to", "release"]


@pytest.mark.parametrize("corruption", ["event", "physical", "owner"])
def test_corrupt_execution_indices_fail_instead_of_guessing(tmp_path, corruption):
    events = _events()
    row = next(row for row in events if row["event_type"] == "control_step")
    if corruption == "event":
        row["event_index"] = 999
    elif corruption == "physical":
        row["control_step_index"] = 999
    else:
        row["physical_call_ref"] = "unknown"
    evidence, _ = _episode(tmp_path, events=events)
    with pytest.raises(ValueError):
        build_episode_context(tmp_path, evidence)
    assert not (tmp_path / "context/trace" / evidence.evidence_id).exists()


def test_all_images_are_paged_and_originals_remain_byte_identical(tmp_path):
    evidence, original = _episode(tmp_path, image_count=25)
    before = inventory_episode(original)
    result = build_episode_context(tmp_path, evidence)
    index = json.loads((tmp_path / result["path"]).read_text())
    pages = index["visual_pages"]
    frames = [frame for page in pages for frame in page["frames"]]
    assert len(pages) == 3
    assert len(frames) == index["coverage"]["images"] == 25
    assert len({frame["source"]["path"] for frame in frames}) == 25
    for page in pages:
        with Image.open((tmp_path / result["path"]).parent / page["file"]) as image:
            assert image.width <= 960 and image.height <= 1152
    assert inventory_episode(original) == before
    assert build_episode_context(tmp_path, evidence) == result


def test_cached_navigation_does_not_rehash_every_derived_file(tmp_path):
    evidence, _ = _episode(tmp_path)
    result = build_episode_context(tmp_path, evidence)
    telemetry = (tmp_path / result["path"]).parent / "telemetry.json"
    telemetry.write_text("{}")
    assert build_episode_context(tmp_path, evidence) == result
    assert telemetry.read_text() == "{}"


def test_query_filters_physical_interval_and_feature_planes_without_truncation(
    tmp_path,
):
    evidence, _ = _episode(tmp_path)
    build_episode_context(tmp_path, evidence)
    result = query_trace(
        tmp_path,
        evidence_id=evidence.evidence_id,
        view="telemetry",
        execution_ref="workflow",
        start_step=1,
        end_step=2,
        features=["privileged.task.progress"],
    )
    assert result["coverage"]["total_physical_steps"] == 3
    assert result["coverage"]["selected_physical_steps"] == 1
    assert result["data"]["control_step_indices"] == [1]
    assert result["data"]["public_evidence"] == {}
    assert result["data"]["critic_private_evidence"] == {
        "privileged.task.progress": [True]
    }
    timeline = _view(tmp_path, evidence, "timeline", execution_ref="workflow")
    assert [row["tool_call_id"] for row in timeline["planner_calls"]] == ["call-pull"]
    assert [row["state_index"] for row in timeline["observations"]] == [1, 2]
    assert len(timeline["executions"]) == 1


@pytest.mark.parametrize(
    "filters",
    [
        {"features": ["invented"]},
        {"execution_ref": "invented"},
        {"start_step": True},
        {"start_step": 2, "end_step": 1},
    ],
)
def test_invalid_query_never_silently_ignores_a_filter(tmp_path, filters):
    evidence, _ = _episode(tmp_path)
    build_episode_context(tmp_path, evidence)
    with pytest.raises(ValueError):
        query_trace(
            tmp_path, evidence_id=evidence.evidence_id, view="telemetry", **filters
        )


def test_later_read_of_old_state_does_not_move_its_capture_into_the_future(tmp_path):
    events = _events()
    # A generated program observes an earlier saved state after its second move.
    final = next(
        i
        for i, row in enumerate(events)
        if row["event_type"] == "active_execution_finished"
    )
    events.insert(
        final,
        {
            "event_type": "episode_tool_program_capability",
            "execution_ref": "workflow",
            "source_modification_ref": "generated",
            "tool_program": "pull",
            "capability": "view_env_state",
            "arguments": {"step": 0},
            "effect": "observe",
            "environment_write": False,
            "result": {"step": 0},
        },
    )
    for index, event in enumerate(events):
        event["event_index"] = index
    evidence, _ = _episode(tmp_path, events=events)
    build_episode_context(tmp_path, evidence)
    initial = _view(tmp_path, evidence, "timeline")["observations"][0]
    assert initial["last_control_step_index"] is None
    assert initial["execution_ref"] is None


def test_nested_same_named_tools_assign_only_the_executing_leaf(tmp_path):
    # Historical nested invocations are decoded without retaining a live producer.
    outer, inner = "outer", "inner"
    events = [
        {
            "event_type": "active_execution_started",
            "execution_id": "workflow",
            "tool_name": "pull",
            "arguments": {},
        },
        *[
            {
                "event_type": "physical_tool_started",
                "execution_ref": "workflow",
                "physical_call_ref": identity,
                "tool_name": "move_to",
                "arguments": {},
            }
            for identity in (outer, inner)
        ],
        {
            "event_type": "control_step",
            "control_step_index": 0,
            "execution_ref": "workflow",
            "physical_call_ref": inner,
            "tool_name": "move_to",
            "action": [0] * 7,
        },
        {
            "event_type": "physical_tool_finished",
            "execution_ref": "workflow",
            "physical_call_ref": inner,
        },
        {
            "event_type": "control_step",
            "control_step_index": 1,
            "execution_ref": "workflow",
            "physical_call_ref": outer,
            "tool_name": "move_to",
            "action": [0] * 7,
        },
        {
            "event_type": "physical_tool_finished",
            "execution_ref": "workflow",
            "physical_call_ref": outer,
        },
        {
            "event_type": "active_execution_finished",
            "execution_ref": "workflow",
            "outcome": "completed",
        },
    ]
    evidence, _ = _episode(
        tmp_path,
        events=[{**row, "event_index": index} for index, row in enumerate(events)],
        messages=[],
    )
    build_episode_context(tmp_path, evidence)
    execution = _view(tmp_path, evidence, "timeline")["executions"][0]
    calls = {call["physical_call_ref"]: call for call in execution["physical_calls"]}
    assert calls[inner]["parent_physical_call_ref"] == outer
    assert calls[inner]["control_step_indices"] == [0]
    assert calls[outer]["control_step_indices"] == [1]
    assert calls["workflow"]["control_step_indices"] == []
    assert execution["control_step_indices"] == [0, 1]


@pytest.mark.parametrize("role", ["claude_agent_sdk", "codex_sdk"])
def test_native_transcript_zero_indexed_calls_are_an_explicit_coverage_gap(
    tmp_path, role
):
    evidence, _ = _episode(
        tmp_path, messages=[{"role": role, "content": "Native tool execution log"}]
    )
    build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    assert timeline["planner_calls"] == []
    assert any(
        gap["kind"] == "native_planner_transcript_not_indexed"
        for gap in timeline["gaps"]
    )


@pytest.mark.parametrize("duplicate", [False, True])
def test_decision_inputs_match_exact_history_identity_and_preserve_ambiguity(
    tmp_path, duplicate
):
    initial = [
        {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": "open drawer"}],
        }
    ]
    response = {
        "kind": "response",
        "parts": [
            {
                "part_kind": "tool-call",
                "tool_call_id": "call-pull",
                "tool_name": "pull",
                "args": {},
            }
        ],
    }
    next_input = initial + [
        response,
        {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "tool-return",
                    "tool_call_id": "call-pull",
                    "content": "done",
                }
            ],
        },
    ]
    second_response = {
        "kind": "response",
        "parts": [
            {
                "part_kind": "tool-call",
                "tool_call_id": "call-release",
                "tool_name": "release",
                "args": {},
            }
        ],
    }
    # Deliberately reverse archive record order: adjacency must not determine identity.
    requests = [
        {"full_history": next_input},
        {"full_history": initial},
        {"full_history": next_input + [second_response]},
    ]
    if duplicate:
        requests.append({"full_history": initial})
    evidence, _ = _episode(tmp_path, requests=requests)
    build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    calls = {row["tool_call_id"]: row for row in timeline["planner_calls"]}
    assert [ref["pointer"] for ref in calls["call-release"]["decision_inputs"]] == [
        "/0"
    ]
    assert [ref["pointer"] for ref in calls["call-pull"]["decision_inputs"]] == (
        ["/1", "/3"] if duplicate else ["/1"]
    )
    ambiguities = [
        gap
        for gap in timeline["gaps"]
        if gap["kind"] == "decision_input_link_not_unique"
    ]
    assert [(gap["tool_call_id"], gap["matches"]) for gap in ambiguities] == (
        [("call-pull", 2)] if duplicate else []
    )


def test_queued_runtime_program_links_by_request_id_and_episode_start_is_explicit(
    tmp_path,
):
    trace = ExecutionTrace()
    trace.record_execution_started(
        execution_id="opening",
        tool_name="run_harness_program",
        arguments={"entry": "episode_start"},
    )
    trace.record_execution_finished(execution_ref="opening", outcome="completed")
    trace.record_execution_started(
        execution_id="requested",
        tool_name="run_harness_program",
        arguments={"request_id": "request-123"},
    )
    trace.record_execution_finished(execution_ref="requested", outcome="completed")
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "planner-call",
                    "name": "run_harness_program",
                    "input": {"program_id": "pull"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "planner-call",
            "name": "run_harness_program",
            "content": json.dumps(
                {
                    "runtime_request_accepted": True,
                    "request_id": "request-123",
                }
            ),
        },
    ]
    evidence, _ = _episode(tmp_path, events=list(trace.events), messages=messages)
    build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    assert timeline["planner_calls"][0]["execution_ref"] == "requested"
    opening = next(
        row for row in timeline["executions"] if row["execution_ref"] == "opening"
    )
    assert opening["origin"] == "episode_start"
    assert not any(
        gap["kind"] == "execution_without_planner_feedback_link"
        for gap in timeline["gaps"]
    )


def test_cached_index_is_reused_without_rehashing_metadata(tmp_path):
    evidence, _ = _episode(tmp_path)
    reference = build_episode_context(tmp_path, evidence)
    (tmp_path / "context/input.json").write_text(
        json.dumps(
            {
                "index": [
                    {"evidence_id": evidence.evidence_id, "trace_context": reference}
                ]
            }
        )
    )
    index_path = tmp_path / reference["path"]
    index = json.loads(index_path.read_text())
    index["coverage"]["physical_steps"] = 9000
    index_path.write_text(json.dumps(index))
    assert build_episode_context(tmp_path, evidence)["path"] == reference["path"]


def test_new_telemetry_sources_and_uncertainty_survive_query_filters(tmp_path):
    rows = _events()
    control = next(row for row in rows if row["event_type"] == "control_step")
    public = control.pop("public_evidence")
    private = control.pop("critic_private_evidence")
    rows.append(
        {
            "event_type": "control_telemetry",
            "event_index": len(rows),
            "control_step_index": control["control_step_index"],
            "control_event_index": control["event_index"],
            "execution_ref": control["execution_ref"],
            "public_evidence": public,
            "critic_private_evidence": private,
            "error": None,
        }
    )
    telemetry_index = rows[-1]["event_index"]
    rows.append(
        {
            "event_type": "action_outcome_unknown",
            "event_index": len(rows),
            "execution_ref": "release",
            "physical_call_ref": "release",
            "forward_ref": None,
            "action_index": 1,
            "requested_action": [0] * 7,
            "error": "lost acknowledgement",
        }
    )
    evidence, _ = _episode(tmp_path, events=rows)
    build_episode_context(tmp_path, evidence)
    value = _view(tmp_path, evidence, "telemetry", start_step=0, end_step=0)
    assert value["telemetry_event_indices"] == [telemetry_index]
    assert value["event_indices"] == [control["event_index"]]
    assert value["physical_steps_complete"] is False
    assert len(value["unconfirmed_actions"]) == 1
    assert value["public_evidence"]["robot.eef.motion_m"] == [0.1]


def test_cancelled_caller_keeps_settled_physical_execution_link(tmp_path):
    _, root = _episode(tmp_path)
    context = root / "harness_context"
    context.mkdir()
    identity = {
        "action_id": "action",
        "operation": "pull",
        "planner_call_id": "call-pull",
        "workflow_invocation_id": "w",
    }
    rows = [
        {**identity, "kind": "action_submitted", "effect": "physical", "arguments": {}},
        {
            **identity,
            "kind": "operation_settled",
            "cancellation_requested": True,
            "status": "completed",
            "result": {"active_execution_ref": "workflow"},
        },
        {**identity, "kind": "action_failed", "error": "CancelledError"},
    ]
    (context / "bundle_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    files = inventory_episode(root)
    evidence = EpisodeEvidence(evidence_id=new_record_id(), files=files)
    root.rename(root.parent / evidence.evidence_id)
    build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    execution = next(
        row for row in timeline["executions"] if row["execution_ref"] == "workflow"
    )
    assert len(execution["kernel_actions"]) == 1
    assert execution["workflow_invocation_ids"] == ["w"]
    assert "call-pull" in execution["planner_call_ids"]
    assert not any(
        gap["kind"] == "kernel_physical_execution_unlinked" for gap in timeline["gaps"]
    )


def test_navigation_references_complete_payloads_without_duplicating_history(tmp_path):
    from reweave.evolution.evidence.records import archive_record

    _, root = _episode(tmp_path)
    context = root / "harness_context"
    context.mkdir()
    image = "image-content-" * 100000
    events = [
        {
            "kind": "context_built",
            "program_call_id": "parent",
            "input": {
                "history": [{"data_base64": image, "media_type": "image/png"}],
                "a/b~c": [1, 2],
            },
            "output": {
                "messages": [
                    {"data": image, "media_type": "image/png", "kind": "binary"}
                ]
            },
        }
    ]
    original = context / "bundle_events.jsonl"
    original.write_text(
        "".join(json.dumps(archive_record(row, context)) + "\n" for row in events)
    )
    files = inventory_episode(root)
    evidence = EpisodeEvidence(evidence_id=new_record_id(), files=files)
    root.rename(root.parent / evidence.evidence_id)
    build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    assert len(json.dumps(timeline)) < 50000
    row = timeline["bundle_events"][0]
    assert row["kind"] == "context_built" and row["program_call_id"] == "parent"
    assert resolve_trace_reference(tmp_path, row["source"]) == events[0]
    ref = row["payloads"]["input"]
    assert resolve_trace_reference(tmp_path, ref) == events[0]["input"]
    nested = {**ref, "pointer": ref["pointer"] + "/a~1b~0c/1"}
    assert resolve_trace_reference(tmp_path, nested) == 2
    wrong = {**ref, "path": ".archive/episodes/unavailable/trace.jsonl"}
    with pytest.raises(ValueError, match="missing"):
        resolve_trace_reference(tmp_path, wrong)
    with pytest.raises(ValueError, match="archived episode"):
        resolve_trace_reference(tmp_path, {**ref, "path": "../sibling/secret.json"})


def test_telemetry_queries_do_not_open_timeline(tmp_path, monkeypatch):
    evidence, _ = _episode(tmp_path)
    build_episode_context(tmp_path, evidence)
    from reweave.evolution.evidence import trace as trace_context

    read = trace_context._read

    def reject_timeline(path):
        assert path.name != "timeline.json", "telemetry loaded unrelated payloads"
        return read(path)

    monkeypatch.setattr(trace_context, "_read", reject_timeline)
    result = _view(
        tmp_path,
        evidence,
        "telemetry",
        execution_ref="workflow",
        features=["robot.eef.motion_m"],
    )
    assert result["control_step_indices"] == [0, 1]
    assert result["public_evidence"]["robot.eef.motion_m"] == [0.1, None]


def test_execution_view_retains_the_full_sequence_and_its_references(tmp_path):
    evidence, _ = _episode(tmp_path)
    build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    result = query_trace(tmp_path, evidence_id=evidence.evidence_id, view="executions")
    assert result["data"]["executions"] == timeline["executions"]
    assert [row["tool_call_id"] for row in result["data"]["planner_feedback"]] == [
        "call-pull",
        "call-release",
    ]
    assert result["coverage"] == {
        "total_executions": 2,
        "selected_executions": 2,
        "total_planner_calls": 2,
        "selected_planner_calls": 2,
        "workflow_events_scope": "episode",
        "selection_complete": True,
    }
    selected = _view(tmp_path, evidence, "executions", execution_ref="workflow")
    assert selected["executions"] == [timeline["executions"][0]]


@pytest.mark.parametrize("physical", [False, True])
def test_zero_action_workflows_remain_visible_with_exact_lifecycle(tmp_path, physical):
    result = {"status": "failed", "output": {"stage": "localize", "found": False}}
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-w",
                    "name": "workflow_check",
                    "input": {"target": "fixture"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-w",
            "name": "workflow_check",
            "content": json.dumps(result),
        },
    ] + (_messages() if physical else [])
    _, root = _episode(tmp_path, messages=messages, events=None if physical else [])
    context = root / "harness_context"
    context.mkdir()
    identity = {
        "workflow_invocation_id": "w",
        "parent_workflow_invocation_id": None,
        "planner_call_id": "call-w",
        "workflow_id": "check",
    }
    nested = {
        **identity,
        "workflow_invocation_id": "child",
        "parent_workflow_invocation_id": "w",
    }
    events = [
        {**identity, "kind": "workflow_started", "arguments": {"target": "fixture"}},
        {**nested, "kind": "workflow_started", "arguments": {}},
        {**nested, "kind": "workflow_failed", "error": "sensor unavailable"},
        {**identity, "kind": "workflow_completed", "result": result},
        {
            **identity,
            "workflow_invocation_id": "unfinished",
            "kind": "workflow_started",
            "arguments": {},
        },
    ]
    (context / "bundle_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    files = inventory_episode(root)
    evidence = EpisodeEvidence(evidence_id=new_record_id(), files=files)
    root.rename(root.parent / evidence.evidence_id)
    build_episode_context(tmp_path, evidence)
    view = _view(tmp_path, evidence, "executions")
    feedback = view["planner_feedback"][0]
    assert feedback["tool_call_id"] == "call-w"
    assert "reported_result" not in feedback
    assert (
        json.loads(resolve_trace_reference(tmp_path, feedback["source"])["content"])
        == result
    )
    assert (
        resolve_trace_reference(tmp_path, feedback["request"])
        == messages[0]["content"][0]
    )
    assert resolve_trace_reference(tmp_path, feedback["source"]) == messages[1]
    assert view["program_feedback"] == []
    assert len(view["executions"]) == (2 if physical else 0)
    assert [
        resolve_trace_reference(tmp_path, row["source"])
        for row in view["workflow_events"]
    ] == events
    assert (
        resolve_trace_reference(
            tmp_path, view["workflow_events"][3]["payloads"]["result"]
        )
        == result
    )
    if physical:
        selected = query_trace(
            tmp_path,
            evidence_id=evidence.evidence_id,
            view="executions",
            execution_ref="workflow",
        )
        assert selected["data"]["workflow_events"] == view["workflow_events"]
        assert [
            row["tool_call_id"] for row in selected["data"]["planner_feedback"]
        ] == ["call-pull"]
        assert selected["coverage"]["total_planner_calls"] == 3
        assert selected["coverage"]["selected_planner_calls"] == 1
        assert selected["coverage"]["workflow_events_scope"] == "episode"


@pytest.mark.parametrize("identity", ["call-pull", "orphan"])
def test_duplicate_nonnull_feedback_identity_is_still_rejected(tmp_path, identity):
    message = {
        "role": "tool",
        "name": "pull",
        "tool_call_id": identity,
        "content": "{}",
    }
    evidence, _ = _episode(tmp_path, messages=[message, message])
    with pytest.raises(ValueError, match="duplicate Planner feedback id"):
        build_episode_context(tmp_path, evidence)


def test_program_feedback_keeps_each_original_identity_without_planner_calls(tmp_path):
    messages = (
        [
            {
                "role": "tool",
                "name": "move_to",
                "tool_call_id": None,
                "provider_tool_call_id": None,
                "program_call_id": "loop-7",
                "executed": {"name": "move_to", "arguments": {"xyz": [i, 0, 0]}},
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "active_execution_ref": "workflow",
                                "step": i,
                                "error": "not at target" if i == 1 else None,
                            }
                        ),
                    }
                ],
            }
            for i in (1, 2)
        ]
        + [
            {
                "role": "tool",
                "name": "read_image",
                "tool_call_id": None,
                "program_call_id": "loop-7",
                "content": [
                    {
                        "type": "text",
                        "text": '{"error":"unsupported artifact suffix: <none>"}',
                    }
                ],
            }
        ]
        + _messages()
    )
    evidence, original = _episode(tmp_path, messages=messages)
    before = inventory_episode(original)
    index = build_episode_context(tmp_path, evidence)
    timeline = _view(tmp_path, evidence, "timeline")
    program = timeline["program_feedback"]
    assert len(program) == 3
    assert [x["tool_call_id"] for x in program] == [None, None, None]
    assert [x["program_call_id"] for x in program] == ["loop-7"] * 3
    assert [x["feedback"]["pointer"] for x in program] == [
        f"/messages/{i}" for i in range(3)
    ]
    assert all(
        resolve_trace_reference(tmp_path, x["feedback"]) == messages[i]
        for i, x in enumerate(program)
    )
    assert "reported_result" not in program[2]
    assert (
        json.loads(
            resolve_trace_reference(tmp_path, program[2]["feedback"])["content"][0][
                "text"
            ]
        )["error"]
        == "unsupported artifact suffix: <none>"
    )
    assert (
        resolve_trace_reference(tmp_path, program[0]["executed_reference"])
        == messages[0]["executed"]
    )
    assert {x["tool_call_id"] for x in timeline["planner_calls"]} == {
        "call-pull",
        "call-release",
    }
    assert not any(x["kind"] == "feedback_without_request" for x in timeline["gaps"])
    selected = _view(tmp_path, evidence, "executions", execution_ref="workflow")
    assert len(selected["program_feedback"]) == 2
    assert [x["tool_call_id"] for x in selected["planner_feedback"]] == ["call-pull"]
    assert None not in selected["executions"][0]["planner_call_ids"]
    manifest = json.loads((tmp_path / index["path"]).read_text())
    assert manifest["coverage"]["program_feedback"] == 3
    assert manifest["coverage"]["planner_calls"] == 2
    assert inventory_episode(original) == before


def test_program_only_execution_links_by_explicit_execution_ref(tmp_path):
    message = {
        "role": "tool",
        "name": "pull",
        "tool_call_id": None,
        "program_call_id": "loop-1",
        "content": json.dumps({"active_execution_ref": "workflow", "step": 2}),
    }
    evidence, _ = _episode(
        tmp_path,
        messages=[
            message,
            {**message, "name": "read_image", "content": '{"error":"missing image"}'},
        ],
    )
    build_episode_context(tmp_path, evidence)
    view = _view(tmp_path, evidence, "executions", execution_ref="workflow")
    assert view["executions"][0]["origin"] == "program"
    assert view["executions"][0]["planner_call_ids"] == []
    assert view["planner_feedback"] == []
    assert len(view["program_feedback"]) == 1
    assert view["program_feedback"][0]["feedback"]["pointer"] == "/messages/0"
    assert len(_view(tmp_path, evidence, "timeline")["program_feedback"]) == 2


@pytest.mark.parametrize("field", ["program_feedback", "workflow_calls"])
def test_incomplete_navigation_is_rebuilt_from_originals(tmp_path, field):
    evidence, _ = _episode(tmp_path)
    result = build_episode_context(tmp_path, evidence)
    path = tmp_path / result["path"]
    index = json.loads(path.read_text())
    del index["coverage"][field]
    path.write_text(json.dumps(index))
    rebuilt = build_episode_context(tmp_path, evidence)
    assert rebuilt["path"] == result["path"]
    assert rebuilt["bytes"] == result["bytes"]
    assert rebuilt["artifact_id"] != result["artifact_id"]
    assert field in json.loads(path.read_text())["coverage"]


@pytest.mark.parametrize(
    "defect",
    [None, "foreign_reference", "wrong_step", "duplicate_forward", "unmatched_forward"],
)
def test_server_vla_forward_inputs_bind_exact_retained_snapshots(tmp_path, defect):
    from reweave.evolution.evidence.trace import _Episode, _forward_inputs

    source = tmp_path / "episode"
    source.mkdir()
    refs = ["scene:0:" + "a" * 32, "scene:30:" + "b" * 32]
    records = [
        {
            "forward_id": f"forward-{index}",
            "observation_ref": ref,
            "physical_step": step,
            "prompt": "Pick the piece",
            "mode": "eval",
            "prediction_horizon_steps": 50,
            "execution_horizon_steps": 30,
        }
        for index, (ref, step) in enumerate(zip(refs, (0, 30)))
    ]
    manifest = {
        "schema_version": 1,
        "transport_config_path": "/host/connection.json",
        "transport_python": "/host/python",
        "remote_observation_root": "/remote/server_observations",
        "snapshots": {
            ref: {
                "archive_path": f"/remote/server_observations/{ref.rsplit(':', 1)[1]}/rgbd.npz",
                "metadata_path": f"/remote/server_observations/{ref.rsplit(':', 1)[1]}/metadata.json",
                "archive_bytes": 128,
                "metadata_bytes": 64,
            }
            for ref in refs
        },
    }
    events = [
        {
            "event_type": "vla_forward",
            "prediction_horizon_steps": 50,
            "execution_horizon_steps": 30,
            "forward_id": "forward-0",
            "execution_ref": "pick",
            "event_index": 0,
        },
        {"event_type": "control_step", "control_step_index": 29, "event_index": 1},
        {
            "event_type": "vla_forward",
            "prediction_horizon_steps": 50,
            "execution_horizon_steps": 30,
            "forward_id": "forward-1",
            "execution_ref": "pick",
            "event_index": 2,
        },
    ]
    if defect == "foreign_reference":
        records[1]["observation_ref"] = "another:30:" + "c" * 32
    elif defect == "wrong_step":
        records[1]["physical_step"] = 29
    elif defect == "duplicate_forward":
        records[1]["forward_id"] = "forward-0"
    elif defect == "unmatched_forward":
        records[1]["forward_id"] = "never-received"
    (source / "vla_inputs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (source / "execution_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    (source / "server_evidence.json").write_text(json.dumps(manifest))
    files = inventory_episode(source)
    episode = _Episode(
        tmp_path, EpisodeEvidence(evidence_id=new_record_id(), files=files), source
    )
    if defect is not None:
        with pytest.raises(ValueError):
            _forward_inputs(episode, events)
    else:
        inputs, gaps = _forward_inputs(episode, events)
        assert gaps == []
        assert [row["observation_ref"] for row in inputs] == refs
        assert [row["forward_ref"] for row in inputs] == ["forward-0", "forward-1"]
        assert [row["physical_step"] for row in inputs] == [0, 30]
        assert [row["last_control_step_index"] for row in inputs] == [None, 29]
        assert all(
            row["sensor_snapshot"]["read_with"] == "read_sensor_snapshot"
            for row in inputs
        )
        assert all(
            row["sensor_snapshot"]["evidence_id"] == episode.evidence.evidence_id
            for row in inputs
        )
        assert all(image["source"] is None for row in inputs for image in row["images"])
        assert all(row["input_record"] is not None for row in inputs)
    assert inventory_episode(source) == files


def test_telemetry_query_preserves_exact_per_key_control_and_telemetry_sources(
    tmp_path,
):
    original = _events()
    rows = []
    expected = {}
    for row in original:
        row = {**row, "event_index": len(rows)}
        rows.append(row)
        if row["event_type"] != "control_step":
            continue
        step = row["control_step_index"]
        telemetry_index = len(rows)
        expected[step] = (row["event_index"], telemetry_index)
        rows.append(
            {
                "event_type": "control_telemetry",
                "event_index": telemetry_index,
                "control_step_index": step,
                "control_event_index": row["event_index"],
                "execution_ref": row["execution_ref"],
                "error": None,
                "public_evidence": {"rpc.total_s": step + 0.5},
                "critic_private_evidence": {"privileged.rpc.available": True},
                "physical_features": {},
            }
        )
    evidence, source = _episode(tmp_path, events=rows)
    build_episode_context(tmp_path, evidence)
    data = _view(tmp_path, evidence, "telemetry", start_step=0, end_step=0)
    assert data["public_evidence"]["robot.eef.motion_m"] == [0.1]
    assert data["public_evidence"]["rpc.total_s"] == [0.5]
    assert data["evidence_event_indices"]["public_evidence"]["robot.eef.motion_m"] == [
        expected[0][0]
    ]
    assert data["evidence_event_indices"]["public_evidence"]["rpc.total_s"] == [
        expected[0][1]
    ]
    from reweave.evolution.evidence.trajectory import resolve_original_pointer

    for plane in ("public_evidence", "critic_private_evidence"):
        for name, sources in data["evidence_event_indices"][plane].items():
            index = sources[0]
            if index is not None:
                pointer = f"/{index}/{plane}/" + name.replace("~", "~0").replace(
                    "/", "~1"
                )
                assert resolve_original_pointer(rows, pointer) == data[plane][name][0]
    assert (
        json.loads(
            (source / "execution_trace.jsonl").read_text().splitlines()[expected[0][0]]
        )["event_type"]
        == "control_step"
    )
