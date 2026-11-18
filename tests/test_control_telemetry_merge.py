"""Derived telemetry joins preserve every real scalar and original event owner."""

import copy

import pytest

from reweave.robots.execution_trace import merge_control_telemetry


def events():
    return [
        {
            "event_type": "control_step",
            "event_index": 0,
            "control_step_index": 0,
            "execution_ref": "move",
            "public_evidence": {"action_fields": "joints"},
            "critic_private_evidence": {"privileged.contact": True},
        },
        {
            "event_type": "control_telemetry",
            "event_index": 1,
            "control_step_index": 0,
            "control_event_index": 0,
            "execution_ref": "move",
            "error": None,
            "public_evidence": {"rpc.total_s": 0.7},
            "critic_private_evidence": {"privileged.progress": 0.2},
            "physical_features": {"measured": [1, 2]},
            "action_wall_time_s": 0.8,
        },
    ]


def test_disjoint_planes_merge_per_key_without_touching_originals():
    original = events()
    before = copy.deepcopy(original)
    control = merge_control_telemetry(original)[0]
    assert control["public_evidence"] == {"action_fields": "joints", "rpc.total_s": 0.7}
    assert control["critic_private_evidence"] == {
        "privileged.contact": True,
        "privileged.progress": 0.2,
    }
    assert control["evidence_event_indices"] == {
        "public_evidence": {"action_fields": 0, "rpc.total_s": 1},
        "critic_private_evidence": {"privileged.contact": 0, "privileged.progress": 1},
    }
    assert control["physical_features_event_index"] == 1
    assert original == before


@pytest.mark.parametrize(
    "plane,key",
    [
        ("public_evidence", "action_fields"),
        ("critic_private_evidence", "privileged.contact"),
    ],
)
def test_identical_overlap_keeps_original_control_source(plane, key):
    rows = events()
    rows[1][plane][key] = rows[0][plane][key]
    control = merge_control_telemetry(rows)[0]
    assert control["evidence_event_indices"][plane][key] == 0


@pytest.mark.parametrize(
    "plane,key",
    [
        ("public_evidence", "action_fields"),
        ("critic_private_evidence", "privileged.contact"),
    ],
)
def test_conflicting_overlap_rejects_instead_of_overwriting(plane, key):
    rows = events()
    rows[1][plane][key] = "different"
    with pytest.raises(ValueError, match="overwrite conflicting"):
        merge_control_telemetry(rows)


@pytest.mark.parametrize("telemetry_features", [{}, {"measured": [1, 2]}])
def test_existing_feature_plane_survives_empty_or_identical_telemetry(
    telemetry_features,
):
    rows = events()
    rows[0]["physical_features"] = {"measured": [1, 2]}
    rows[1]["physical_features"] = telemetry_features
    control = merge_control_telemetry(rows)[0]
    assert control["physical_features"] == {"measured": [1, 2]}
    assert control["physical_features_event_index"] == 0


def test_different_nonempty_feature_planes_remain_rejected():
    rows = events()
    rows[0]["physical_features"] = {"another": [1, 2]}
    with pytest.raises(ValueError, match="physical feature plane"):
        merge_control_telemetry(rows)


def test_duplicate_telemetry_remains_rejected_even_when_scalar_keys_are_disjoint():
    rows = events()
    rows.append({**rows[1], "event_index": 2, "public_evidence": {"another": 1}})
    with pytest.raises(ValueError, match="duplicate telemetry"):
        merge_control_telemetry(rows)


@pytest.mark.parametrize("left,right", [(True, 1), (1, 1.0)])
def test_different_scalar_types_are_conflicts_even_when_python_values_compare_equal(
    left, right
):
    rows = events()
    rows[0]["public_evidence"]["same"] = left
    rows[1]["public_evidence"]["same"] = right
    with pytest.raises(ValueError, match="overwrite conflicting"):
        merge_control_telemetry(rows)


def test_public_control_projection_does_not_expose_private_source_keys():
    from reweave.robots.execution_trace import ExecutionTrace

    trace = ExecutionTrace()
    trace._events = events()
    (control,) = trace.public_control_steps()
    assert "critic_private_evidence" not in control
    assert "physical_features" not in control
    assert set(control["evidence_event_indices"]) == {"public_evidence"}
    assert "privileged" not in repr(control["evidence_event_indices"])
    assert control["evidence_event_indices"]["public_evidence"] == {
        "action_fields": 0,
        "rpc.total_s": 1,
    }
