# Modified for the ReWeave namespace.
"""Physical trace graph construction and exact original-evidence queries."""

from __future__ import annotations

import bisect
import json
import math
from collections import Counter
from pathlib import Path

from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import (
    EvidenceFile,
    GraphEpisode,
    GraphSnapshot,
    GraphTransition,
    OriginalReference,
    PhysicalFeature,
    PhysicalFrame,
)
from reweave.evolution.evidence.records import artifact_record, restore_record
from reweave.robots.execution_trace import merge_control_telemetry


def read_original(reference: OriginalReference):
    """Resolve a bound pointer, preserving JSONL line numbering."""
    return read_originals((reference,))[reference]


def read_originals(references):
    """Decode each selected JSONL row once; unrelated image/history rows stay on disk."""
    grouped = {}
    for reference in references:
        grouped.setdefault(
            (reference.path, reference.artifact_id, reference.bytes), set()
        ).add(reference)
    values = {}
    for (name, artifact_id, size), selected in grouped.items():
        EvidenceFile(path=name, artifact_id=artifact_id, bytes=size).verify()
        path = Path(name)
        if path.suffix == ".jsonl" and all(ref.pointer for ref in selected):
            rows = {}
            for ref in selected:
                if not ref.pointer.startswith("/"):
                    raise ValueError("original reference must be a JSON pointer")
                position = ref.pointer.split("/", 2)[1]
                if (
                    not position.isascii()
                    or not position.isdecimal()
                    or str(int(position)) != position
                ):
                    raise ValueError(
                        "JSONL pointer must start with a canonical record index"
                    )
                rows.setdefault(int(position), []).append(ref)
            with path.open() as stream:
                for index, line in enumerate(stream):
                    if index not in rows:
                        continue
                    value = restore_record(json.loads(line), path.parent)
                    for ref in rows.pop(index):
                        pointer = ref.pointer.removeprefix(f"/{index}")
                        values[ref] = resolve_original_pointer(value, pointer)
                    if not rows:
                        break
            if rows:
                raise IndexError("JSONL record is absent")
            continue
        if path.suffix not in {".json", ".jsonl"} and all(
            not ref.pointer for ref in selected
        ):
            try:
                value = path.read_text()
            except UnicodeError as error:
                raise ValueError(
                    "binary graph original: open the bound path with the native image/media reader"
                ) from error
        elif path.suffix == ".jsonl":
            with path.open() as stream:
                value = [
                    restore_record(json.loads(line), path.parent) for line in stream
                ]
        else:
            value = restore_record(json.loads(path.read_text()), path.parent)
        for ref in selected:
            values[ref] = resolve_original_pointer(value, ref.pointer)
    return values


def resolve_original_pointer(value, pointer):
    """Resolve one pointer in an already verified, decoded original document."""
    if pointer and not pointer.startswith("/"):
        raise ValueError("original reference must be a JSON pointer")
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if (
                not token.isascii()
                or not token.isdecimal()
                or (len(token) > 1 and token.startswith("0"))
            ):
                raise ValueError(
                    "JSON pointer array index must be a canonical nonnegative integer"
                )
            value = value[int(token)]
        else:
            value = value[token]
    return value


def episode_from_record(record, receipt: Path, versions) -> GraphEpisode:
    receipt_file = EvidenceFile(**artifact_record(receipt))
    if json.loads(receipt.read_text()) != record.model_dump(mode="json"):
        raise ValueError("graph receipt differs from registered evaluation")
    trace_files = [
        f for f in record.evidence if Path(f.path).name == "execution_trace.jsonl"
    ]
    if len(trace_files) > 1:
        raise ValueError("ambiguous physical execution trace")
    frames, starts, exits = [], [], []
    feature_names = set()
    feature_format = (
        "robodojo_proprioception"
        if record.request.instance.task.startswith("robodojo:")
        else "physical_scalars"
    )
    unknown_interval = False
    if trace_files:
        trace = trace_files[0]
        trace.verify()
        raw = [json.loads(line) for line in Path(trace.path).read_text().splitlines()]
        events = merge_control_telemetry(raw)
        line_by_event = {e["event_index"]: i for i, e in enumerate(raw)}
        for event in events:
            kind = event["event_type"]
            if kind == "active_execution_started":
                starts.append(
                    {
                        "event": event,
                        "reference": OriginalReference(
                            **trace.model_dump(mode="json"),
                            pointer=f"/{line_by_event[event['event_index']]}",
                        ).model_dump(mode="json"),
                    }
                )
            if kind in {"active_execution_finished", "action_outcome_unknown"}:
                exits.append(
                    {
                        "event": event,
                        "reference": OriginalReference(
                            **trace.model_dump(mode="json"),
                            pointer=f"/{line_by_event[event['event_index']]}",
                        ).model_dump(mode="json"),
                    }
                )
            if kind == "action_outcome_unknown":
                unknown_interval = True
            if kind in {"initial_state", "reset_state"}:
                unknown_interval = False
            if kind not in {"initial_state", "reset_state", "control_step"}:
                continue
            indices = [line_by_event[event["event_index"]]]
            if event.get("telemetry_event_index") is not None:
                indices.append(line_by_event[event["telemetry_event_index"]])
            features = event.get("physical_features", {})
            feature_event_index = event.get(
                "physical_features_event_index", event["event_index"]
            )
            feature_line = (
                line_by_event[feature_event_index]
                if feature_event_index is not None
                else None
            )
            feature_names.update(
                _physical_features(features, feature_format)
                if feature_format == "robodojo_proprioception"
                else features
            )
            frames.append(
                PhysicalFrame(
                    control_step=event.get("control_step_index"),
                    execution_ref=event.get("execution_ref"),
                    tool=event.get("tool_name"),
                    action=tuple(event.get("action", ())),
                    action_wall_time_s=event.get("action_wall_time_s"),
                    feature_source=OriginalReference(
                        **trace.model_dump(mode="json"),
                        pointer=f"/{feature_line}/physical_features",
                    )
                    if feature_line is not None
                    and "physical_features" in raw[feature_line]
                    else None,
                    references=tuple(
                        OriginalReference(
                            **trace.model_dump(mode="json"), pointer=f"/{i}"
                        )
                        for i in indices
                    ),
                    error="; ".join(
                        str(value)
                        for value in (
                            event.get("error"),
                            event.get("telemetry_error"),
                            "unacknowledged physical interval"
                            if unknown_interval
                            else None,
                            None if features else "physical channels unavailable",
                        )
                        if value
                    )
                    or None,
                )
            )
    return GraphEpisode(
        terminal_state_known=not unknown_interval,
        operation_starts=tuple(starts),
        operation_exits=tuple(exits),
        receipt_id=record.record_id,
        bundle_id=record.request.bundle_id,
        condition_id=record.request.instance.record_id,
        version_ids=tuple(
            v.version_id for v in versions if v.bundle_id == record.request.bundle_id
        ),
        frames=tuple(frames),
        feature_format=feature_format,
        feature_names=tuple(sorted(feature_names)),
        official_success=record.official_success,
        exit_type=record.failure_origin,
        usage=record.usage,
        receipt=receipt_file,
    )


def frame_features(episode: GraphEpisode, index: int):
    """Original scalars plus reset-relative displacement, without future windows."""
    return _frame_features(episode, index, episode_features(episode))


def episode_features(episode, indices=None):
    """Read requested measurements from their sole original authority."""
    indices = range(len(episode.frames)) if indices is None else indices
    sources = {index: episode.frames[index].feature_source for index in indices}
    values = read_originals(source for source in sources.values() if source is not None)
    return {
        index: _physical_features(values[source], episode.feature_format)
        if source is not None
        else {}
        for index, source in sources.items()
    }


def _physical_features(values, feature_format):
    if feature_format == "robodojo_proprioception" and values:
        from reweave.benchmarks.robodojo.access import graph_features

        values = graph_features(values)
    return {
        name: PhysicalFeature.model_validate(value) for name, value in values.items()
    }


def _frame_features(episode, index, observations):
    result = dict(observations[index])
    resets = [i for i in range(index + 1) if episode.frames[i].control_step is None]
    if not resets:
        return result
    initial = observations[resets[-1]]
    prefixes = [n[:-2] for n in result if n.endswith(".position.x") and "entity." in n]
    for prefix in prefixes:
        keys = [prefix + "." + a for a in "xyz"]
        if not all(
            k in initial
            and k in result
            and initial[k].value is not None
            and result[k].value is not None
            for k in keys
        ):
            continue
        distance = math.sqrt(
            sum((result[k].value - initial[k].value) ** 2 for k in keys)
        )
        result[prefix.removesuffix(".position") + ".displacement_m"] = PhysicalFeature(
            value=distance,
            unit="m",
            source="recorded initial and current entity positions",
            derivation="Euclidean displacement from latest explicit reset state",
            availability="observed",
        )
    return result


def region(features, policy):
    signature = {}
    for name, feature in sorted(features.items()):
        value = feature.value
        if name.endswith("distance_to_eef_m"):
            signature[name] = (
                None
                if value is None
                else bisect.bisect_right(policy.distance_bins_m, value)
            )
        elif name.endswith("fraction_from_lower_limit"):
            signature[name] = (
                None
                if value is None
                else bisect.bisect_right(policy.joint_fraction_bins, value)
            )
        elif name.endswith("displacement_m"):
            signature[name] = (
                None if value is None else value >= policy.displacement_bin_m
            )
        elif name == "contacts.geom_pairs" or (
            (value is None or isinstance(value, bool))
            and (
                (
                    name.startswith(("entity.", "privileged.entity."))
                    and name.endswith("contact")
                )
                or (".predicate." in name and name.endswith(".satisfied"))
            )
        ):
            signature[name] = value
    return signature


def episode_regions(episode, policy):
    """Partition observations only when all episode partition channels are present.

    Channel names observed elsewhere in this episode establish missingness, never
    an imputed value. This scaffold does not define semantic state membership.
    """
    observations = episode_features(episode)
    signatures = [
        region(_frame_features(episode, index, observations), policy)
        for index in range(len(episode.frames))
    ]
    channels = {name for signature in signatures for name in signature}
    return tuple(
        signature
        if signature
        and not frame.error
        and channels.issubset(signature)
        and all(value is not None for value in signature.values())
        else None
        for frame, signature in zip(episode.frames, signatures)
    )


def consecutive_actions(previous, frame):
    """A reset starts a fresh observed interval; it is never an action target."""
    return frame.control_step is not None and (
        previous.control_step is None or frame.control_step == previous.control_step + 1
    )


def build_graph(episodes, policy, *, record_id=None) -> GraphSnapshot:
    unique = {}
    for episode in episodes:
        if episode.receipt_id in unique:
            if episode != unique[episode.receipt_id]:
                raise ValueError("cached physical receipt has conflicting provenance")
            continue
        unique[episode.receipt_id] = episode
    nodes, paths, transitions = {}, {}, []
    # Signatures list channels in sorted order, so equal regions share one key.
    by_signature = {}
    for episode in unique.values():
        path = []
        signatures = episode_regions(episode, policy)
        for index, (frame, signature) in enumerate(zip(episode.frames, signatures)):
            node = None
            if signature is not None:
                node = by_signature.setdefault(
                    tuple(signature.items()), f"node:{len(nodes) + 1:03d}"
                )
            path.append(node)
            if node is not None:
                nodes[node] = signature
            if index == 0 or node is None or path[index - 1] is None:
                continue
            previous = episode.frames[index - 1]
            if consecutive_actions(previous, frame):
                transitions.append(
                    GraphTransition(
                        source=path[index - 1],
                        target=node,
                        episode_id=episode.receipt_id,
                        frame_index=index,
                    )
                )
        paths[episode.receipt_id] = tuple(path)
    return GraphSnapshot(
        record_id=record_id if record_id is not None else new_record_id(),
        policy=policy,
        episodes=tuple(unique.values()),
        nodes=nodes,
        paths=paths,
        transitions=tuple(transitions),
    )


def query_graph(
    graph: GraphSnapshot,
    *,
    view="summary",
    identity=None,
    first_frame=None,
    last_frame=None,
    features=None,
):
    """Request-bound navigation; full immutable frames and references stay available."""
    if view != "episode" and any(
        value is not None for value in (first_frame, last_frame, features)
    ):
        raise ValueError("frame and feature selection requires an execution episode")
    if view in {"summary", "features"}:
        names = sorted({key for e in graph.episodes for key in e.feature_names})
    if view == "features":
        return {
            "feature_names": [
                name for name in names if identity is None or name.startswith(identity)
            ],
            "prefix": identity,
            "total_features": len(names),
            "selection_complete": True,
        }
    if view == "summary":
        return {
            "graph_id": graph.record_id,
            "node_count": len(graph.nodes),
            "transition_count": len(graph.transitions),
            "feature_count": len(names),
            "feature_groups": dict(
                Counter(".".join(key.split(".")[:2]) for key in names)
            ),
            "navigation": "features lists exact names (identity selects a prefix); episode selects receipt_id, inclusive first_frame/last_frame and features; node selects an ID from an episode path. Measurements remain in the originals identified by each frame's feature_source.",
            "episodes": [
                {
                    "receipt_id": e.receipt_id,
                    "bundle_id": e.bundle_id,
                    "condition_id": e.condition_id,
                    "official_success": e.official_success,
                    "exit_type": e.exit_type,
                    "frames": len(e.frames),
                }
                for e in graph.episodes
            ],
        }
    if view == "node":
        return {"node_id": identity, "signature": graph.nodes[identity]}
    if view == "episode":
        episode = next(e for e in graph.episodes if e.receipt_id == identity)
        first = 0 if first_frame is None else first_frame
        last = len(episode.frames) - 1 if last_frame is None else last_frame
        if first < 0 or last < first or last >= len(episode.frames):
            raise ValueError("frame selection is outside the original episode")
        available = set(episode.feature_names)
        if features is not None and set(features) - available:
            raise ValueError("unknown physical feature in episode selection")
        value = episode.model_dump(mode="json", exclude={"frames"})
        value["frames"] = []
        observations = episode_features(episode, range(first, last + 1))
        for index in range(first, last + 1):
            frame = episode.frames[index]
            selected = frame.model_dump(mode="json")
            selected["features"] = {
                name: item.model_dump(mode="json")
                for name, item in observations[index].items()
                if features is None or name in features
            }
            value["frames"].append(selected)
        value["path"] = graph.paths[identity][first : last + 1]
        value["selection"] = {
            "first_frame": first,
            "last_frame": last,
            "total_frames": len(episode.frames),
            "features": features,
            "selection_complete": True,
        }
        return value
    if view == "transitions":
        return {
            "transitions": [
                t.model_dump(mode="json")
                for t in graph.transitions
                if identity is None or t.source == identity or t.target == identity
            ]
        }
    raise ValueError("unsupported graph view")
