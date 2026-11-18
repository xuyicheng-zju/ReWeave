"""Source-only preparation and offline admission for fragment experiments."""

from reweave.evolution.distillation.fragments import fragment_transfer


def source_graphs(plan):
    graphs = dict(plan.source_graphs)
    names = {t.name for t in plan.sources}
    if len(plan.teachers) != len(names) or {t.task for t in plan.teachers} != names:
        raise ValueError("fragment preparation requires one teacher per source task")
    for teacher in plan.teachers:
        if teacher.graph is not None:
            if teacher.task in graphs and graphs[teacher.task] != teacher.graph:
                raise ValueError("conflicting source graph provenance")
            graphs[teacher.task] = teacher.graph
    if set(graphs) != names or not names:
        raise ValueError("fragment preparation requires every source topology")
    return graphs


def shared_encoding(graphs):
    """A compatible union alone is not evidence of joint source encoding."""
    values = list(graphs.values())
    return bool(values) and all(
        g.matching is not None
        and values[0].matching is not None
        and g.matching.catalog == values[0].matching.catalog
        for g in values
    )


def check_fragments(plan):
    if plan.method != "fragments" or plan.mode != "consolidation":
        raise ValueError("expected fragment consolidation configuration")
    if plan.loss_threshold is None or plan.stop_on_source_success:
        raise ValueError("fragment experiment requires an explicit loss threshold")
    graphs = source_graphs(plan)
    if not shared_encoding(graphs):
        raise ValueError("run prepare-fragments to encode one shared source catalog")
    proposals = fragment_transfer(graphs)
    used = {}
    for name in graphs:
        used[name] = {
            f["fluent"]
            for row in proposals["inventory"].values()
            if row["task"] == name
            for f in (*row["entry"], *row["intended_result"])
        }
    report = {
        "configuration_ready": True,
        "method": "fragments",
        "sources": sorted(graphs),
        "catalog_size": len(next(iter(graphs.values())).matching.catalog),
        "direct_interfaces": len(proposals["inventory"]),
        "retained_h0_observations": len(plan.initial_observations),
        "reuse_proposals": len(proposals["adaptation_proposals"]),
        "strict_reuse_proposals": len(proposals["proposals"]),
        "guidance_metrics": proposals["guidance_metrics"],
        "pairs": proposals["pairs"],
        "shared_used_coordinates": {
            f"{a} <- {b}": sorted(used[a] & used[b])
            for a in used
            for b in used
            if a != b
        },
        "warnings": []
        if proposals["adaptation_proposals"]
        else [
            "No cross-source reuse proposals; inspect source annotations before launching."
        ],
        "physical_work_started": False,
        "live_runtime_verified": False,
    }
    return report, proposals
