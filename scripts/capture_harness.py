"""Capture the default LIBERO-PRO Harness and its pinned runtime."""

import argparse
import subprocess
import sys
from pathlib import Path

from reweave.cli.harness import Harness
from reweave.cli.main import parse_cli_arguments
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.contracts import EvolutionProtocol
from reweave.evolution.runtime_manifest import (
    RuntimeManifest,
    RuntimeResources,
    validate_budget_arguments,
)
from reweave.evolution.search.rollout import RolloutConfiguration
from reweave.runtime.kernel import EpisodeLimits
from reweave.utils.persistence import atomic_json


def main():
    examples = Path(__file__).resolve().parents[1] / "examples/graph_evolution"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A new Harness directory."
    )
    parser.add_argument(
        "--model", required=True, help="Online Codex model for rollouts."
    )
    parser.add_argument(
        "--resources", type=Path, required=True, help="Runtime resource JSON."
    )
    task = parser.add_mutually_exclusive_group()
    task.add_argument(
        "--protocol", type=Path, help="Custom LIBERO-PRO development protocol."
    )
    task.add_argument(
        "--task", type=int, help="Task index in the bundled LIBERO-PRO protocol."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--codex-auth-mode", choices=("auth", "api"))
    args = parser.parse_args()

    protocol_path = args.protocol or examples / "libero_protocol.json"
    protocol_input = EvolutionProtocol.model_validate_json(
        protocol_path.read_text()
    ).model_dump(mode="json")
    if args.task is not None:
        for instance in protocol_input["dev"]:
            instance["initialization"]["task"] = args.task
            suite = instance["initialization"]["suite"]
            instance["task"] = f"libero:{suite}:{args.task}"
    protocol = EvolutionProtocol.model_validate(protocol_input)
    first = protocol.dev[0]
    if not first.task.startswith("libero:"):
        parser.error("this example requires a LIBERO-PRO protocol")
    resources = RuntimeResources.model_validate_json(args.resources.read_text())
    conditions = protocol.execution
    limits = EpisodeLimits(
        program_calls=conditions.program_calls,
        gateway_calls=conditions.gateway_calls,
        wall_time_s=conditions.wall_time_s,
    )
    kernel = {"episode": limits.model_dump(mode="json")}
    fixed = [
        "--robot",
        "libero",
        "--libero-type",
        "pro",
        "--planner",
        "native",
        "--model",
        args.model,
        "--max-episode-steps",
        str(conditions.physical_steps),
        "--wait-steps",
        str(conditions.wait_steps),
        "--max-turns",
        str(conditions.model_requests),
        "--reasoning-effort",
        conditions.model_sampling["reasoning_effort"],
        "--planner-timeout-s",
        str(int(conditions.wall_time_s or 0)),
        "--adaptation-mode",
        conditions.adaptation_mode,
    ]
    if conditions.online_workflows:
        fixed.append("--online-workflows")
    if args.codex_auth_mode is not None:
        fixed.extend(("--codex-auth-mode", args.codex_auth_mode))
    root = args.output_dir.resolve()
    baseline = root / "baseline"
    memory = root / "capture_memory"
    kernel_path = root / "kernel.json"
    resources_path = root / "runtime_resources.json"
    capture = [
        *fixed,
        "--cuda-device",
        str(args.device),
        "--memory-profile",
        "local",
        "--memory-dir",
        str(memory),
        "--output-dir",
        str(root / "capture"),
        "--capture-bundle",
        str(baseline),
        "--runtime-resources",
        str(resources_path),
        "--kernel-limits",
        str(kernel_path),
    ]
    for key, value in first.initialization.items():
        capture.extend(("--" + key.replace("_", "-"), str(value)))
    _, capture_args, _ = parse_cli_arguments(capture)
    validate_budget_arguments(conditions, vars(capture_args), kernel)
    if conditions.model_sampling != {"reasoning_effort": capture_args.reasoning_effort}:
        parser.error("Codex protocol model_sampling must contain its reasoning_effort")

    root.mkdir(parents=True, exist_ok=False)
    memory.mkdir()
    atomic_json(resources_path, resources.model_dump(mode="json"))
    atomic_json(
        kernel_path, {**kernel, "conditions": conditions.model_dump(mode="json")}
    )
    subprocess.run([sys.executable, "-m", "reweave.cli", *capture], check=True)

    manifest = RuntimeManifest.model_validate_json(
        (baseline / "runtime.json").read_text()
    )
    runner = RolloutConfiguration(
        devices=(args.device,),
        arguments=tuple(fixed),
        memory_snapshot=str(baseline / "memory"),
        runtime_manifest=manifest,
        episode_limits=limits,
    )
    bundle = AgentBundle.model_validate_json((baseline / "bundle.json").read_text())
    print(
        Harness(
            suite=first.initialization["suite"], bundle=bundle, runner=runner
        ).publish(root)
    )


if __name__ == "__main__":
    main()
