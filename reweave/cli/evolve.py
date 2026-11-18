# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""CLI for physical-graph-guided W/C/L offline evolution."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvolutionConfiguration(BaseModel):
    """Exact persisted inputs for the sole offline evolution command."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["physical_graph_evolution"] = Field(
        description="Sole production evolution algorithm; old matrix configurations are rejected."
    )

    baseline: dict[str, Any] = Field(
        description="Frozen complete AgentBundle at run creation."
    )
    protocol: dict[str, Any] = Field(
        description="Validated EvolutionProtocol with development conditions, independent author count and termination contract."
    )
    runner: dict[str, Any] = Field(
        description="Validated rollout configuration and exact runtime manifest."
    )
    author: dict[str, Any] = Field(
        description="Native AuthorPolicy used for every independent diagnosis-and-mutation session; independent of the Planner."
    )


def select_runner_device(runner, device):
    """Narrow scheduling only; preserve the published evidence/runtime contract."""
    from reweave.evolution.search.rollout import RolloutConfiguration

    if device not in runner.devices:
        raise ValueError("leased GPU must belong to the published runner pool")
    raw = runner.model_dump(mode="json")
    raw["devices"] = [device]
    raw["shared_inference_services"] = [
        item for item in raw["shared_inference_services"] if item["device"] == device
    ]
    return RolloutConfiguration.model_validate(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run physical graph analysis and single-session W/C/L repair mutations."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("schema", help="Print schemas for the evolution inputs.")

    upgrade = subparsers.add_parser(
        "upgrade-context",
        help="Publish a new unmeasured Bundle with the current baseline C program; never modify its input or reuse its receipts.",
    )
    upgrade.add_argument("--bundle", type=Path, required=True)
    upgrade.add_argument("--output", type=Path, required=True)
    upgrade.add_argument(
        "--runtime-manifest",
        type=Path,
        required=True,
        help="Explicit target runtime capture supporting the current C input contract; historical runtime is not silently retained.",
    )

    initialize = subparsers.add_parser(
        "init", help="Initialize one physical-graph-guided evolution run."
    )
    initialize.add_argument("--baseline", type=Path, required=True)
    initialize.add_argument("--protocol", type=Path, required=True)
    initialize.add_argument("--runner", type=Path, required=True)
    initialize.add_argument("--workspace", type=Path, required=True)
    initialize.add_argument("--author-policy", type=Path)
    initialize.add_argument(
        "--runtime-source",
        type=Path,
        help="Explicit physical source for CLI preflight when it differs from the offline Host.",
    )
    run = subparsers.add_parser(
        "run", help="Run or resume the physical-graph-guided author mutations."
    )
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument(
        "--device",
        type=int,
        help="Lease one GPU from the published runner pool for this complete chain; recovery must retain it.",
    )
    run.add_argument(
        "--runtime-source",
        type=Path,
        required=True,
        help="Repository directory imported by rollout workers.",
    )
    run.add_argument(
        "--recover",
        action="store_true",
        help="Explicitly recover unfinished native/runner attempts using the same identities.",
    )
    status = subparsers.add_parser(
        "status",
        help="Read the last committed evolution snapshot without blocking its author.",
    )
    status.add_argument("--workspace", type=Path, required=True)

    replay = subparsers.add_parser(
        "replay-graph",
        help="Derive a fresh graph from an archive without model calls or physics.",
    )
    replay.add_argument("--workspace", type=Path, required=True)
    replay.add_argument("--graph-policy", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    index = subparsers.add_parser(
        "index-graphs",
        help="Cache local Qwen0.6B encodings for an explicitly configured final-graph corpus.",
    )
    index.add_argument("--config", type=Path, required=True)
    retrieve = subparsers.add_parser(
        "retrieve-graphs",
        help="Match a whole/local query graph to the same corpus, without physics or distillation.",
    )
    retrieve.add_argument("--config", type=Path, required=True)
    retrieve.add_argument(
        "--query",
        type=Path,
        required=True,
        help="GraphSearch JSON with optional query, top_k and exclude_tasks; an omitted query uses --graph.",
    )
    retrieve.add_argument(
        "--graph",
        type=Path,
        help="Current complete SemanticGraph used when the query JSON omits query. Author tools bind it automatically each round.",
    )
    return parser


def _evolution_command(args: argparse.Namespace) -> int:
    if args.command in {"index-graphs", "retrieve-graphs"}:
        from reweave.evolution.graph.retrieval import GraphRetrieval, GraphSearch

        retrieval = GraphRetrieval.from_file(args.config)
        default_query = None
        if args.command == "retrieve-graphs" and args.graph is not None:
            from reweave.evolution.contracts import SemanticGraph
            from reweave.evolution.graph.matching import semantic_query

            default_query, _ = semantic_query(
                SemanticGraph.model_validate_json(args.graph.read_text())
            )
        result = (
            retrieval.index()
            if args.command == "index-graphs"
            else retrieval.search(
                GraphSearch.model_validate_json(args.query.read_text()),
                default_query=default_query,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    from reweave.evolution.authoring.author import NativeBundleAuthor
    from reweave.evolution.authoring.config import AuthorPolicy
    from reweave.evolution.bundle import AgentBundle
    from reweave.evolution.contracts import EvolutionProtocol
    from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
    from reweave.evolution.search.rollout import RolloutConfiguration, RolloutEvaluator
    from reweave.utils.persistence import atomic_json

    if args.command == "schema":
        inputs = {
            "baseline": AgentBundle,
            "protocol": EvolutionProtocol,
            "runner": RolloutConfiguration,
            "author_policy": AuthorPolicy,
        }
        print(
            json.dumps(
                {name: model.model_json_schema() for name, model in inputs.items()},
                indent=2,
            )
        )
        return 0

    if args.command == "upgrade-context":
        import os
        import tempfile

        from reweave.evolution.authoring.candidate import write_bundle_source
        from reweave.evolution.bundle import upgrade_context_bundle
        from reweave.evolution.runtime_manifest import RuntimeManifest

        original = AgentBundle.model_validate_json(args.bundle.read_text())
        manifest = RuntimeManifest.model_validate_json(
            args.runtime_manifest.read_text()
        )
        upgraded = upgrade_context_bundle(original, manifest.runtime_configuration())
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError("context upgrade output already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="context-upgrade-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary) / "candidate"
            write_bundle_source(upgraded, staging)
            atomic_json(staging / "bundle.json", upgraded.model_dump(mode="json"))
            atomic_json(
                staging / "upgrade.json",
                {
                    "source_bundle": original.record_id,
                    "bundle_id": upgraded.record_id,
                    "scopes": ["C"],
                    "measured": False,
                    "evaluation_receipt": None,
                    "runtime_changed": upgraded.runtime != original.runtime,
                    "target_runtime_manifest": str(args.runtime_manifest.resolve()),
                },
            )
            os.rename(staging, destination)
        print(destination)
        return 0
    root = args.workspace.expanduser().resolve()
    configuration_path = root / "configuration.json"
    if args.command == "replay-graph":
        from reweave.evolution.graph.replay import replay_graph

        replay_graph(root, args.output.expanduser().resolve(), args.graph_policy)
        print(args.output)
        return 0
    if args.command == "init":
        bundle = AgentBundle.model_validate_json(args.baseline.read_text())
        protocol = EvolutionProtocol.model_validate_json(args.protocol.read_text())
        runner = RolloutConfiguration.model_validate_json(args.runner.read_text())
        if runner.runtime_manifest.conditions != protocol.execution:
            raise ValueError("runner manifest differs from baseline/protocol")
        for instance in protocol.instances:
            runner.validate_arguments(
                instance,
                directory=root / ".scratch/runtime/preflight" / instance.record_id,
                memory=Path(runner.memory_snapshot).expanduser().resolve(),
                device=runner.devices[0],
                repository=args.runtime_source.resolve()
                if args.runtime_source
                else None,
            )
        policy = (
            AuthorPolicy.model_validate_json(args.author_policy.read_text())
            if args.author_policy
            else AuthorPolicy(author="codex")
        )
        configuration = EvolutionConfiguration(
            method="physical_graph_evolution",
            baseline=bundle.model_dump(mode="json"),
            protocol=protocol.model_dump(mode="json"),
            runner=runner.model_dump(mode="json"),
            author=policy.model_dump(mode="json"),
        ).model_dump(mode="json")
        workspace = EvolutionWorkspace(root, bundle, protocol)
        with workspace.transaction():
            if (
                configuration_path.exists()
                and json.loads(configuration_path.read_text()) != configuration
            ):
                raise ValueError("evolution configuration is immutable")
            atomic_json(configuration_path, configuration)
            workspace.put_bundle(bundle)
        print(root)
        return 0
    configuration = EvolutionConfiguration.model_validate_json(
        configuration_path.read_text()
    )
    bundle = AgentBundle.model_validate(configuration.baseline)
    protocol = EvolutionProtocol.model_validate(configuration.protocol)
    runner = RolloutConfiguration.model_validate(configuration.runner)
    policy = AuthorPolicy.model_validate(configuration.author)
    workspace = EvolutionWorkspace(root, bundle, protocol)
    if args.command == "status":
        print(workspace.state_path.read_text())
        return 0
    if (root / "author_deployment.json").exists():
        raise ValueError(
            "This run requires its original local account Host; use remote_evolution run"
        )
    if args.device is not None:
        runner = select_runner_device(runner, args.device)
        with workspace.transaction():
            lease_path = root / "device_lease.json"
            lease = {"device": args.device}
            if lease_path.exists() and json.loads(lease_path.read_text()) != lease:
                raise ValueError("recovery cannot change the task's leased GPU")
            atomic_json(lease_path, lease)
    elif (root / "device_lease.json").exists():
        from reweave.evolution.infrastructure.campaign_slots import (
            campaign_root,
            enabled,
        )

        campaign = campaign_root(root)
        if campaign is None or not enabled(campaign):
            raise ValueError("this task requires its original explicit --device lease")
        # Keep the historical whole-chain lease; the explicit campaign revision
        # selects episode admission without rewriting published configuration.
    runtime_source = args.runtime_source.expanduser().resolve()
    evaluator = RolloutEvaluator(runner, repository=runtime_source, workspace=root)

    no_topology = protocol.evolution_mode == "no_topology"
    author = NativeBundleAuthor(
        bundle_loader=workspace.bundle,
        policy=policy,
        evidence_workspace=root / "evidence",
        runtime_manifest=runner.runtime_manifest,
    )
    from reweave.evolution.graph.analysis import NativeGraphAnalysis

    analyzer = (
        None
        if no_topology
        else NativeGraphAnalysis(
            bundle_loader=workspace.bundle,
            policy=policy,
            evidence_workspace=root / "evidence",
            native_runtime_root=root / "graph_session/runtime",
            runtime_manifest=runner.runtime_manifest,
        )
    )
    evolution = GraphEvolution(
        workspace, author, evaluator, analyzer=analyzer, recover=args.recover
    )
    state = asyncio.run(evolution.run())
    print(
        json.dumps(
            {
                "method": "full_without_topology"
                if no_topology
                else "physical_graph_evolution",
                "completed": state["completed"],
                "stop_reason": state["stop_reason"],
                "versions": len(state["versions"]),
                "usage": evolution.accounting(state),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    return _evolution_command(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
