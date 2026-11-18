# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Lossless physical evidence scaffold, independent of semantic refinement."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    GraphEpisode,
    GraphPolicy,
    OriginalReference,
    PhysicalFeature,
    PhysicalFrame,
)
from reweave.evolution.evidence.trajectory import (
    build_graph,
    episode_features,
    frame_features,
    query_graph,
    read_original,
    read_originals,
)
from tests.unit_tests.harness.evolution.fixtures import artifact, usage


def physical(value, unit="m"):
    return PhysicalFeature(
        value=value,
        unit=unit,
        source="synthetic sensor",
        derivation="test observation",
        availability="observed" if value is not None else "unavailable",
    )


def episode(tmp_path, identity, distances=(0.2, 0.05, 0.2), *, errors=(), force=0.0):
    original = artifact(tmp_path, f"{identity}.json", {"samples": list(distances)})
    ref = OriginalReference(**original.model_dump(mode="json"), pointer="/samples")
    frames = tuple(
        PhysicalFrame(
            control_step=None if i == 0 else i - 1,
            execution_ref=None if i == 0 else "move" if i == 1 else "hold",
            tool=None if i == 0 else "move" if i == 1 else "hold",
            action=() if i == 0 else (0.1,),
            feature_source=None,
            references=(ref,),
            error="missing" if i in errors else None,
        )
        for i, distance in enumerate(distances)
    )
    observed = GraphEpisode(
        receipt_id=f"episode:{identity}",
        bundle_id=f"bundle:{identity % 2}",
        condition_id=f"condition:{(identity // 2) % 2}",
        version_ids=(),
        frames=frames,
        feature_names=(),
        official_success=False,
        exit_type="agent",
        usage=usage(),
        receipt=original,
    )
    return with_features(
        observed,
        [
            {
                "entity.object.distance_to_eef_m": physical(distance),
                "force": physical(force, "N"),
            }
            for distance in distances
        ],
    )


def with_features(observed, observations):
    values = [
        {name: value.model_dump(mode="json") for name, value in row.items()}
        for row in observations
    ]
    original = artifact(
        Path(observed.receipt.path).parent,
        f"features-{new_record_id()}.json",
        values,
    )
    return observed.model_copy(
        update={
            "frames": tuple(
                frame.model_copy(
                    update={
                        "feature_source": OriginalReference(
                            **original.model_dump(mode="json"),
                            pointer=f"/{index}",
                        )
                    }
                )
                for index, frame in enumerate(observed.frames)
            ),
            "feature_names": tuple(
                sorted({name for row in observations for name in row})
            ),
        }
    )


def test_graph_retains_revisits_and_dwell_without_inventing_missing_edges(tmp_path):
    observed = episode(tmp_path, 0, (0.2, 0.05, 0.05, 0.2, 0.1), errors=(4,))
    graph = build_graph([observed, observed], GraphPolicy())
    path = graph.paths[observed.receipt_id]
    assert len(graph.episodes) == 1
    assert path[0] == path[3] and path[1] == path[2] and path[4] is None
    assert len(graph.transitions) == 3
    assert graph.transitions[1].source == graph.transitions[1].target
    assert not graph.episodes[0].official_success
    assert read_original(observed.frames[0].references[0]) == [
        0.2,
        0.05,
        0.05,
        0.2,
        0.1,
    ]


def test_unknown_feature_is_not_false_or_zero(tmp_path):
    observed = episode(tmp_path, 1, (0.2, None, 0.05))
    graph = build_graph([observed], GraphPolicy())
    assert graph.paths[observed.receipt_id][1] is None
    assert not graph.transitions
    with pytest.raises(ValidationError):
        PhysicalFeature(
            value=0, unit="m", source="x", derivation="x", availability="unavailable"
        )


def test_graph_rejects_invalid_original_measurements_before_publication(tmp_path):
    observed = episode(tmp_path, 1)
    original = Path(observed.frames[0].feature_source.path)
    values = json.loads(original.read_text())
    values[0]["force"]["availability"] = "unavailable"
    original.write_text(json.dumps(values))
    with pytest.raises(ValidationError, match="unavailable"):
        build_graph([observed], GraphPolicy())


def test_graph_stores_measurement_references_and_rejects_copied_frame_values(
    tmp_path, monkeypatch
):
    from reweave.evolution.evidence import trajectory as task_graph

    observed = episode(tmp_path, 1)
    graph = build_graph([observed], GraphPolicy())
    frame = observed.frames[0].model_dump(mode="json")
    assert "features" not in frame and frame["feature_source"] is not None
    with pytest.raises(ValidationError, match="Extra inputs"):
        PhysicalFrame.model_validate({**frame, "features": {}})
    selected = query_graph(
        graph, view="episode", identity=observed.receipt_id, first_frame=0, last_frame=0
    )
    assert selected["frames"][0]["features"]["force"] == physical(0, "N").model_dump(
        mode="json"
    )
    monkeypatch.setattr(
        task_graph,
        "read_originals",
        lambda *args: pytest.fail(
            "summary must use navigation names without decoding measurements"
        ),
    )
    assert query_graph(graph)["feature_count"] == 2
    assert len(query_graph(graph, view="features")["feature_names"]) == 2


def test_original_batch_decodes_selected_rows_once_and_resolves_archived_images(
    tmp_path, monkeypatch
):
    from reweave.evolution.evidence.records import archive_record, artifact_record

    path = tmp_path / "journal.jsonl"
    row = {
        "image": {"data_base64": "exact\r\n", "media_type": "image/png"},
        "a/b": [1, 2],
    }
    encoded = json.dumps(archive_record(row, tmp_path))
    # Unselected rows may contain large image/history documents. Decoding an
    # invalid row here would demonstrate an unintended whole-file read.
    path.write_text("unselected\n" + encoded + "\nunselected\n")
    refs = [
        OriginalReference(**artifact_record(path), pointer=pointer)
        for pointer in ("/1/image/data_base64", "/1/a~1b/1")
    ]
    loads = json.loads
    decoded = []

    def tracked(value, *args, **kwargs):
        decoded.append(value)
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", tracked)
    assert read_originals(refs) == {refs[0]: "exact\r\n", refs[1]: 2}
    assert len(decoded) == 1
    with pytest.raises(json.JSONDecodeError):
        read_original(refs[0].model_copy(update={"pointer": "/0"}))
    with pytest.raises(ValueError, match="canonical"):
        read_original(refs[0].model_copy(update={"pointer": "/01"}))
    with pytest.raises(IndexError, match="absent"):
        read_original(refs[0].model_copy(update={"pointer": "/3"}))


def test_graph_summary_indexes_explicit_lossless_selections(tmp_path):
    observed = episode(tmp_path, 1, (0.2, 0.1, 0.05, 0.2))
    graph = build_graph([observed], GraphPolicy())
    summary = query_graph(graph)
    assert "nodes" not in summary
    assert summary["node_count"] == len(graph.nodes)
    assert summary["feature_count"] == 2
    assert summary["feature_groups"] == {"entity.object": 1, "force": 1}
    assert query_graph(graph, view="features", identity="entity.")["feature_names"] == [
        "entity.object.distance_to_eef_m"
    ]
    selected = query_graph(
        graph,
        view="episode",
        identity=observed.receipt_id,
        first_frame=1,
        last_frame=2,
        features=["force"],
    )
    assert selected["selection"] == {
        "first_frame": 1,
        "last_frame": 2,
        "total_frames": 4,
        "features": ["force"],
        "selection_complete": True,
    }
    assert len(selected["frames"]) == 2
    assert list(selected["frames"][0]["features"]) == ["force"]
    assert (
        selected["frames"][0]["references"]
        == observed.frames[1].model_dump(mode="json")["references"]
    )
    node = query_graph(graph, view="node", identity=selected["path"][0])
    assert node["signature"] == graph.nodes[node["node_id"]]
    for options in (
        {"view": "summary", "first_frame": 0},
        {"view": "episode", "identity": observed.receipt_id, "last_frame": 4},
        {"view": "episode", "identity": observed.receipt_id, "features": ["invented"]},
    ):
        with pytest.raises(ValueError):
            query_graph(graph, **options)
    assert len(graph.episodes[0].frames) == 4


@pytest.mark.parametrize("omitted", [False, True])
def test_partial_partition_gaps_break_edges_even_with_other_observations(
    tmp_path, omitted
):
    observed = episode(tmp_path, 1, (0.2, 0.05, 0.05, 0.2))
    observations = episode_features(observed)
    updated = []
    for index, frame in enumerate(observed.frames):
        features = dict(observations[index])
        if index in (1, 2):
            if omitted:
                features.pop("entity.object.distance_to_eef_m")
            else:
                features["entity.object.distance_to_eef_m"] = physical(None)
        features["entity.object.contact"] = physical(True, "boolean")
        updated.append(features)
    observed = with_features(observed, updated)
    graph = build_graph([observed], GraphPolicy())
    assert graph.paths[observed.receipt_id][1:3] == (None, None)
    assert not graph.transitions


def test_missing_nonpartition_measurement_keeps_observed_geometry(tmp_path):
    observed = episode(tmp_path, 1, force=None)
    graph = build_graph([observed], GraphPolicy())
    assert len(graph.transitions) == 2


def test_reset_relative_displacement_has_no_future_dependency(tmp_path):
    observed = episode(tmp_path, 1)
    observations = episode_features(observed)
    updated = []
    for i, frame in enumerate(observed.frames):
        features = dict(observations[i])
        features.update(
            {
                f"entity.object.position.{axis}": physical(
                    i * 0.03 if axis == "x" else 0
                )
                for axis in "xyz"
            }
        )
        updated.append(features)
    observed = with_features(observed, updated)
    assert frame_features(observed, 1)[
        "entity.object.displacement_m"
    ].value == pytest.approx(0.03)
    assert frame_features(
        observed.model_copy(update={"frames": observed.frames[:2]}), 1
    ) == frame_features(observed, 1)


def test_later_explicit_reset_restores_derivation_and_starts_a_new_interval(tmp_path):
    observed = episode(tmp_path, 1, (0.2, 0.2, 0.2, 0.2))
    frames = []
    observations = episode_features(observed)
    updated = []
    for index, frame in enumerate(observed.frames):
        features = dict(observations[index])
        features.update(
            {
                f"entity.object.position.{axis}": physical(index if axis == "x" else 0)
                for axis in "xyz"
            }
        )
        frames.append(
            frame.model_copy(
                update={
                    "control_step": None if index == 2 else index,
                }
            )
        )
        updated.append(features)
    observed = with_features(
        observed.model_copy(update={"frames": tuple(frames)}), updated
    )
    assert "entity.object.displacement_m" not in frame_features(observed, 1)
    assert frame_features(observed, 2)["entity.object.displacement_m"].value == 0
    assert frame_features(observed, 3)["entity.object.displacement_m"].value == 1
    graph = build_graph([observed], GraphPolicy())
    assert [t.frame_index for t in graph.transitions] == [3]


def test_original_is_read_without_hash_gate(tmp_path):
    observed = episode(tmp_path, 0)
    Path(observed.receipt.path).write_text(json.dumps({"samples": [0]}))
    assert read_original(observed.frames[0].references[0]) == [0]


def test_graph_reads_required_trace_without_unrelated_attachment_gate(tmp_path):
    from reweave.evolution.contracts import EvaluationRequest
    from reweave.evolution.evidence.trajectory import episode_from_record
    from reweave.utils.persistence import atomic_json
    from tests.unit_tests.harness.evolution.fixtures import baseline, protocol
    from tests.unit_tests.harness.evolution.test_graph_evolution import record

    configuration = protocol()
    request = EvaluationRequest(
        record_id=new_record_id(),
        bundle_id=baseline().record_id,
        instance=configuration.dev[0],
        split="dev",
        manifest_id=configuration.manifest_id,
        conditions=configuration.execution,
        execution=configuration.execution_id,
    )
    result = record(request, tmp_path, False)
    receipt = tmp_path / "result.json"
    atomic_json(receipt, result.model_dump(mode="json"))
    unrelated = next(f for f in result.evidence if Path(f.path).name == "trace.json")
    Path(unrelated.path).unlink()
    observed = episode_from_record(result, receipt, ())
    assert observed.frames
    trace = next(
        f for f in result.evidence if Path(f.path).name == "execution_trace.jsonl"
    )
    Path(trace.path).unlink()
    with pytest.raises(ValueError, match="not a regular absolute file"):
        episode_from_record(result, receipt, ())


@pytest.mark.parametrize(
    "telemetry_features", [{}, {"sensor": physical(0.25).model_dump(mode="json")}]
)
def test_embedded_measurement_keeps_control_original_when_telemetry_is_empty_or_equal(
    tmp_path, telemetry_features
):
    from reweave.evolution.contracts import (
        EvaluationRecord,
        EvaluationRequest,
        EvidenceFile,
        Usage,
    )
    from reweave.evolution.evidence.records import artifact_record
    from reweave.evolution.evidence.trajectory import episode_from_record
    from tests.unit_tests.harness.evolution.fixtures import protocol

    features = {"sensor": physical(0.25).model_dump(mode="json")}
    rows = [
        {
            "event_type": "control_step",
            "event_index": 0,
            "control_step_index": 0,
            "execution_ref": "move",
            "tool_name": "move",
            "action": [0.1],
            "physical_features": features,
        },
        {
            "event_type": "control_telemetry",
            "event_index": 1,
            "control_step_index": 0,
            "control_event_index": 0,
            "execution_ref": "move",
            "error": None,
            "public_evidence": {},
            "critic_private_evidence": {},
            "physical_features": telemetry_features,
        },
    ]
    trace = tmp_path / "execution_trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    frozen = protocol()
    request = EvaluationRequest(
        record_id="request",
        bundle_id="bundle",
        instance=frozen.dev[0],
        split="dev",
        manifest_id=frozen.manifest_id,
        execution=frozen.execution_id,
        conditions=frozen.execution,
    )
    record = EvaluationRecord(
        record_id="receipt",
        request=request,
        official_success=False,
        failure_origin="agent",
        infrastructure_retryable=None,
        usage=Usage(physical_steps=1, model_calls=0, wall_time_s=1, cost_usd=None),
        evidence=(EvidenceFile(**artifact_record(trace)),),
        episode=None,
        initialization=request.instance.initialization,
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(record.model_dump_json())
    observed = episode_from_record(record, receipt, ())
    assert observed.frames[0].feature_source.pointer == "/0/physical_features"
    assert frame_features(observed, 0)["sensor"].value == 0.25
    assert read_original(observed.frames[0].feature_source) == features
