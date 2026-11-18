"""First-observation history for the single-task reprojection ablation."""

import json
from pathlib import Path

from reweave.evolution.contracts import SemanticGraph
from reweave.evolution.graph.projection import project_map


def previous_projection(request):
    if request.previous_projection is None:
        return None
    request.previous_projection.verify()
    return json.loads(Path(request.previous_projection.path).read_text())


def projection_episode_ids(request, evidence):
    """Only receipts without an admitted interpretation may be projected anew."""
    previous = previous_projection(request)
    return {e.receipt_id for e in evidence.episodes} - (
        set(previous["episodes"]) if previous is not None else set()
    )


def _scoped_graph(graph, identity):
    """Node/edge/program IDs belong to an interpretation, not all future graphs."""

    def scoped(key):
        return f"{identity}::{key}" if key is not None else None

    raw = graph.model_dump(mode="json")
    # This view serves task-local diagnostics. The canonical analysis retains
    # matching; first-observation history is not a cross-task retrieval corpus.
    raw["matching"] = None
    raw["nodes"] = {
        scoped(key): {**node, "predecessors": [scoped(p) for p in node["predecessors"]]}
        for key, node in raw["nodes"].items()
    }
    raw["edges"] = {
        scoped(key): {
            **edge,
            "source": scoped(edge["source"]),
            "target": scoped(edge["target"]),
            "witnesses": [
                {**w, "program_id": scoped(w["program_id"])} for w in edge["witnesses"]
            ],
        }
        for key, edge in raw["edges"].items()
    }
    raw["programs"] = {scoped(key): value for key, value in raw["programs"].items()}
    raw["episodes"] = {
        key: [{**span, "node_id": scoped(span["node_id"])} for span in spans]
        for key, spans in raw["episodes"].items()
    }
    return SemanticGraph.model_validate(raw)


def project_analysis(request, evidence, graph, *, graph_id, cell_receipts):
    """One projection entry point for search and final archive selection."""
    cached = request.protocol.history_projection == "first_observation"
    scoped = _scoped_graph(graph, graph_id) if cached else graph
    value = project_map(
        evidence,
        scoped,
        request.versions,
        request.protocol.dev,
        graph_id=graph_id,
        cell_receipts=cell_receipts,
        validation_conditions=request.protocol.validation.conditions
        if request.protocol.validation
        else (),
        partial=cached,
    )
    if not cached:
        return value
    previous = previous_projection(request)
    for cell in value["episodes"].values():
        cell["interpretation_id"] = graph_id
    value["interpretations"] = {graph_id: scoped.model_dump(mode="json")}
    # All evidence attached to an old receipt stays byte-for-byte equivalent.
    # Namespaced semantic IDs prevent changed node meanings from being joined.
    if previous is not None:
        for key in (
            "episodes",
            "transitions",
            "edge_interfaces",
            "programs",
            "node_evidence",
            "edge_statistics",
            "interpretations",
        ):
            value[key] = previous[key] | value[key]
        value["milestone_nodes"] = (
            previous["milestone_nodes"] + value["milestone_nodes"]
        )
    # Interface numbers are local to a graph too; never group unrelated origins.
    for edge in scoped.edges:
        value["edge_interfaces"][edge] = f"{graph_id}::{value['edge_interfaces'][edge]}"
    value["cells"] = {
        version.version_id: {
            condition.record_id: cell_receipts.get(
                (version.bundle_id, condition.record_id)
            )
            for condition in request.protocol.dev
        }
        for version in request.versions
    }
    for row in value["cells"].values():
        for condition, receipt in row.items():
            if receipt not in value["episodes"]:
                row[condition] = None
    value["pending_episode_ids"] = sorted(
        {e.receipt_id for e in evidence.episodes} - value["episodes"].keys()
    )
    value["meaning"] = (
        "First-observation history: each receipt retains its original progress, frontier, "
        "state assignments and witnesses. Topology may change, but archived interpretations "
        "are never refreshed, including at final selection. Normalized cached progress is "
        "used after official success; nodes from different interpretation IDs are incomparable."
    )
    return value


def interpretation_graph(value, receipt):
    """Resolve a historical witness against its own immutable coordinate system."""
    identity = value["episodes"][receipt]["interpretation_id"]
    return SemanticGraph.model_validate(value["interpretations"][identity])
