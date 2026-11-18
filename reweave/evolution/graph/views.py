# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Targeted views over complete graph originals; summaries never replace evidence."""

from reweave.evolution.graph.projection import map_cell, map_row

_VERSION_KEYS = (
    "receipt_id",
    "bundle_id",
    "progress",
    "peak_progress",
    "official_success",
    "exit_type",
    "frontier",
    "lost_nodes",
    "unresolved_transition_ids",
    "remaining_frontier",
    "completed_nodes",
)
_VALIDATION_KEYS = tuple(
    key for key in _VERSION_KEYS if key not in {"bundle_id", "peak_progress"}
)
_COMPARISON_KEYS = (
    "receipt_id",
    "bundle_id",
    "progress",
    "peak_progress",
    "official_success",
    "frontier",
    "lost_nodes",
    "completed_nodes",
    "transition_ids",
    "unresolved_transition_ids",
    "usage",
    "exit_type",
)
_EPISODE_KEYS = ("bundle_id", "condition_id", "official_success", "exit_type")
_CONTINUATION_KEYS = ("edge_id", "program_id", "outcome", "cost")


def _select(value, keys):
    result = {key: value[key] for key in keys}
    if "interpretation_id" in value:
        result["interpretation_id"] = value["interpretation_id"]
    return result


def query_semantic(graph, *, view="summary", identity=None, graph_id=None):
    if view == "summary":
        return {
            "graph_id": graph_id,
            "milestone_nodes": [
                key for key, node in graph.nodes.items() if node.role == "milestone"
            ],
            "nodes": {
                key: {
                    **node.model_dump(mode="json", exclude={"references"}),
                    "reference_count": len(node.references),
                }
                for key, node in graph.nodes.items()
            },
            "edges": {
                key: {
                    "source": edge.source,
                    "target": edge.target,
                    "entry_conditions": edge.entry_conditions,
                    "exit_conditions": edge.exit_conditions,
                    "program_ids": sorted(
                        {
                            w.program_id
                            for w in edge.witnesses
                            if w.program_id is not None
                        }
                    ),
                    "witness_count": len(edge.witnesses),
                }
                for key, edge in graph.edges.items()
            },
            "programs": {
                key: {"mechanism": program.mechanism, "pointers": program.pointers}
                for key, program in graph.programs.items()
            },
            "episode_ids": list(graph.episodes),
            "navigation": "node, transitions, program and episode select identity; complete original is context/graph.json. Program attribution is historical evidence, not an extracted executable plugin.",
        }
    if view == "node":
        return {
            "node_id": identity,
            **graph.nodes[identity].model_dump(mode="json"),
            "incoming_edges": [
                key for key, edge in graph.edges.items() if edge.target == identity
            ],
            "outgoing_edges": [
                key for key, edge in graph.edges.items() if edge.source == identity
            ],
        }
    if view == "program":
        return {
            "program_id": identity,
            **graph.programs[identity].model_dump(mode="json"),
            "edges": {
                key: {
                    **edge.model_dump(mode="json", exclude={"witnesses"}),
                    "witnesses": [
                        w.model_dump(mode="json")
                        for w in edge.witnesses
                        if w.program_id == identity
                    ],
                }
                for key, edge in graph.edges.items()
                if any(w.program_id == identity for w in edge.witnesses)
            },
        }
    if view == "episode":
        return {
            "episode_id": identity,
            "states": [s.model_dump(mode="json") for s in graph.episodes[identity]],
            "transitions": [
                {"edge_id": key, **w.model_dump(mode="json")}
                for key, edge in graph.edges.items()
                for w in edge.witnesses
                if w.episode_id == identity
            ],
            "unclassified": [
                s.model_dump(mode="json")
                for s in graph.unclassified
                if s.episode_id == identity
            ],
        }
    if view == "transitions":
        selected = {
            key: edge
            for key, edge in graph.edges.items()
            if identity is None or identity in (key, edge.source, edge.target)
        }
        if identity is not None and not selected and identity not in graph.nodes:
            raise ValueError("unknown semantic transition or node")
        return {
            "edges": {
                key: edge.model_dump(mode="json") for key, edge in selected.items()
            }
        }
    raise ValueError("unsupported semantic graph view")


def _transition_views(value, identities, *, graph=None):
    """Join episode and endpoint facts on demand, instead of repeating them per attempt."""
    result = {}
    completed = graph.completed_nodes() if graph is not None else None
    for identity in identities:
        witness = value["transitions"][identity]
        episode = value["episodes"][witness["episode_id"]]
        if "interpretations" in value:
            from reweave.evolution.graph.reprojection import interpretation_graph

            graph = interpretation_graph(value, witness["episode_id"])
            completed = graph.completed_nodes()
        endpoints = {}
        for field, frame in (
            ("actual_source", witness["first_frame"]),
            ("actual_target", witness["last_frame"]),
        ):
            span = next(
                span
                for span in episode["state_spans"]
                if span["first_frame"] <= frame <= span["last_frame"]
            )
            endpoints[field] = {"status": span["status"], "node_id": span["node_id"]}
            if completed is not None:
                endpoints[field]["retained_nodes"] = (
                    None
                    if span["status"] == "unknown"
                    else sorted(completed[span["node_id"]])
                    if span["node_id"] is not None
                    else []
                )
        result[identity] = {
            **witness,
            **endpoints,
            **_select(episode, _EPISODE_KEYS),
        }
        if completed is not None:
            before = endpoints["actual_source"]["retained_nodes"]
            after = endpoints["actual_target"]["retained_nodes"]
            result[identity]["state_change"] = {
                "gained_nodes": None
                if before is None or after is None
                else sorted(set(after) - set(before)),
                "lost_nodes": None
                if before is None or after is None
                else sorted(set(before) - set(after)),
            }
            edge = graph.edges[witness["edge_id"]]
            result[identity]["interface"] = edge.model_dump(
                mode="json",
                include={
                    "source",
                    "target",
                    "roles",
                    "entry_conditions",
                    "exit_conditions",
                },
            )
            result[identity]["attribution_uncertainty"] = (
                graph.programs[witness["program_id"]].uncertainty
                if witness["program_id"] is not None
                else witness["uncertainty"]
            )
        following = witness["successor_id"]
        if following is not None and (
            value["transitions"][following]["episode_id"] != witness["episode_id"]
            or value["transitions"][following]["first_frame"] != witness["last_frame"]
        ):
            raise ValueError(
                "continuation requires the same episode and a shared boundary"
            )
        result[identity]["episode_usage"] = episode.get("usage")
        result[identity]["continuation"] = (
            {
                "transition_id": following,
                **_select(value["transitions"][following], _CONTINUATION_KEYS),
            }
            if following is not None
            else None
        )
    return result


def _program_interfaces(value, identity):
    """Compare witnessed implementations without ranking programs or inventing trials."""
    selected = value["programs"][identity]["transition_ids"]
    interfaces = {
        value["edge_interfaces"][value["transitions"][key]["edge_id"]]
        for key in selected
    }
    result = {}
    for interface in sorted(interfaces):
        witnesses = {
            key: witness
            for key, witness in value["transitions"].items()
            if value["edge_interfaces"][witness["edge_id"]] == interface
        }
        programs = {}
        for key, witness in witnesses.items():
            program = witness["program_id"]
            if program is not None:
                programs.setdefault(program, []).append(key)
        result[interface] = {
            "edge_ids": sorted({w["edge_id"] for w in witnesses.values()}),
            "programs": {
                program: {
                    "bundle_id": value["programs"][program]["bundle_id"],
                    "assets": value["programs"][program]["assets"],
                    "transition_ids": sorted(keys),
                    "episode_ids": sorted(
                        {witnesses[key]["episode_id"] for key in keys}
                    ),
                    "condition_ids": sorted(
                        {
                            value["episodes"][witnesses[key]["episode_id"]][
                                "condition_id"
                            ]
                            for key in keys
                        }
                    ),
                    "outcomes": {
                        outcome: sum(
                            witnesses[key]["outcome"] == outcome for key in keys
                        )
                        for outcome in ("achieved", "not_achieved", "unknown")
                    },
                    "costs": [
                        {"transition_id": key, "cost": witnesses[key]["cost"]}
                        for key in sorted(keys)
                    ],
                }
                for program, keys in sorted(programs.items())
            },
            "unattributed_transition_ids": sorted(
                key
                for key, witness in witnesses.items()
                if witness["program_id"] is None
            ),
        }
    return result


def query_map(value, *, view="summary", identity=None, graph=None):
    """Read one evidence unit or version without dumping the whole history matrix."""
    if value is None:
        return {
            "available": False,
            "reason": "No semantic projection is available yet.",
        }
    if view == "summary":
        return {
            "graph_id": value["graph_id"],
            "meaning": value["meaning"],
            "milestone_nodes": value["milestone_nodes"],
            "versions": {
                version: {
                    condition: _select(cell, _VERSION_KEYS)
                    for condition, cell in map_row(value, version).items()
                }
                for version in value["cells"]
            },
            "validation": {
                version: {
                    condition: _select(cell, _VALIDATION_KEYS)
                    for condition, cell in map_row(
                        value, version, split="validation"
                    ).items()
                }
                for version, row in value["validation_cells"].items()
                if row
            },
            "edge_statistics": value["edge_statistics"],
            "program_ids": list(value["programs"]),
            "transition_count": len(value["transitions"]),
            "unattributed_transition_count": sum(
                w["program_id"] is None for w in value["transitions"].values()
            ),
            "pending_episode_ids": value["pending_episode_ids"],
            "navigation": "version selects version_id; comparison pairs that version with historical observations on each same condition; episode selects receipt_id; transitions selects an edge ID; witness selects one physical transition ID; program selects program_id; node lists actual arrivals and same-episode completions. Invocation references expose exact tool arguments through an explicit original read. Full projection remains in context/progress_map.json.",
        }
    if view == "version":
        return {
            "version_id": identity,
            "cells": map_row(value, identity),
            "validation_cells": map_row(value, identity, split="validation"),
        }
    if view == "comparison":

        def observation_id(cell):
            # Null is an absence, not a shared receipt. Only aliases of the
            # same missing Bundle can collapse into one unmeasured contrast.
            return (
                ("receipt", cell["receipt_id"])
                if cell["receipt_id"] is not None
                else ("missing", cell["bundle_id"])
            )

        def entry(version, cell):
            return {"version_id": version, **_select(cell, _COMPARISON_KEYS)}

        def pool(split):
            rows = value["cells" if split == "dev" else "validation_cells"]
            selected = map_row(value, identity, split=split)
            result = {}
            for condition, cell in selected.items():
                contrasts = []
                seen = {observation_id(cell)}
                for version, row in rows.items():
                    if condition not in row:
                        continue
                    other = map_cell(value, version, condition, split=split)
                    observation = observation_id(other)
                    if observation in seen:
                        continue
                    seen.add(observation)
                    a, b = cell["progress"], other["progress"]
                    relation = (
                        "unscored"
                        if a is None or b is None
                        else (
                            "higher_progress"
                            if b > a
                            else "lower_progress"
                            if b < a
                            else "equal_progress"
                        )
                    )
                    contrasts.append(
                        {
                            **entry(version, other),
                            "observed_relation": relation,
                            "official_result_differs": None
                            if cell["official_success"] is None
                            or other["official_success"] is None
                            else cell["official_success"] != other["official_success"],
                        }
                    )
                result[condition] = {
                    "selected": entry(identity, cell),
                    "contrasts": contrasts,
                }
            return result

        return {
            "graph_id": value["graph_id"],
            "version_id": identity,
            "dev": pool("dev"),
            "validation": pool("validation"),
            "meaning": "Paired observations on this graph and the same condition; a single result is not a causal effect or a reliable success probability. Unknown stays null. Transition IDs select original interface, program, binding, cost and same-episode successor evidence. Historical wins cannot be combined into a candidate score.",
        }
    if view == "node":
        return {"node_id": identity, "evidence": value["node_evidence"][identity]}
    if view == "episode":
        cell = value["episodes"].get(identity)
        if cell is None:
            raise ValueError("episode is not projected in this map")
        return {
            "cell": cell,
            "transitions": _transition_views(
                value, cell["transition_ids"], graph=graph
            ),
        }
    if view == "transitions":
        if identity is None:
            return {"edge_statistics": value["edge_statistics"]}
        selected = {
            key: item
            for key, item in value["transitions"].items()
            if identity == item["edge_id"]
        }
        if not selected:
            raise ValueError("unknown projected transition")
        return {"transitions": _transition_views(value, selected, graph=graph)}
    if view == "witness":
        if identity not in value["transitions"]:
            raise ValueError("unknown projected transition witness")
        return {"transitions": _transition_views(value, (identity,), graph=graph)}
    if view == "program":
        item = value["programs"][identity]
        return {
            "program_id": identity,
            **item,
            "transitions": _transition_views(
                value, item["transition_ids"], graph=graph
            ),
            "conditional_interfaces": _program_interfaces(value, identity),
            "meaning": "All witnessed programs for the same declared conditional interfaces, including failures and unknowns. Shared interface labels do not prove matched entry states, isolated causality or transfer. Costs are observed intervals, not total decision cost or remaining task cost. Repeated witnesses in one episode are not independent trials. Source Bundles and semantic program views retain dependencies; no donor ranking or automatic composition is performed.",
        }
    raise ValueError("unsupported progress map view")
