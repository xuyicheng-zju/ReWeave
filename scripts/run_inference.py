"""Run one episode with the selected Harness."""

import argparse
import asyncio
from pathlib import Path

from reweave.cli.harness import Harness
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import EvaluationRequest
from reweave.evolution.search.rollout import RolloutEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-state-id", type=int, default=0)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.task, args.seed, args.reset_state_id) < 0:
        parser.error("task, seed and reset-state-id must be nonnegative")
    harness = Harness.read(args.harness)
    manifest = harness.runner.runtime_manifest
    request = EvaluationRequest(
        record_id=new_record_id(),
        bundle_id=harness.bundle.record_id,
        instance=harness.instance(args.task, args.seed, args.reset_state_id),
        split=args.split,
        manifest_id=new_record_id(),
        execution=manifest.execution_id,
        conditions=manifest.conditions,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evaluator = RolloutEvaluator(
        harness.runner,
        repository=Path(__file__).resolve().parents[1],
        workspace=output,
    )
    record = asyncio.run(
        evaluator(
            request, harness.bundle, output / "attempt", before_launch=lambda: None
        )
    )
    print(output / "attempt/result.json")
    print(
        f"official_success={record.official_success} failure_origin={record.failure_origin}"
    )
    return int(record.failure_origin in {"infrastructure", "unknown"})


if __name__ == "__main__":
    raise SystemExit(main())
