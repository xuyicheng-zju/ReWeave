# Usage and evaluation

The [README quick start](../README.md#quick-start) contains the complete
LIBERO-PRO workflow. Use the [installation guide](INSTALLATION.md) to prepare
the simulator and model dependencies.

## Harnesses and trajectories

Every command that produces a Harness publishes a `harness.json` containing
the executable C/W/L Bundle, the captured rollout runner, and any task
topologies and development evidence carried forward. Keep the initial memory
snapshot and original rollout directories at their recorded paths. The
artifact uses those paths to resolve the frozen runtime and evidence.

Inference writes `attempt/result.json` beside the exact `attempt/bundle.json`
and the episode data. The simulator supplies `official_success`. The
`failure_origin` field distinguishes a settled policy failure from an
infrastructure or unknown outcome. Inference defaults to development data.
Use `--split test` for independent evaluation.

Select development receipts from the same task for an optimization round.
Multiple selected trials on the same reset remain available to the analyst.
The last supplied receipt for a given Harness and reset supplies that
progress-map cell. Optimization and distillation publish executable
candidates with no new simulator verdict. Inference measures a candidate
on the specified reset.

## Continual learning

`--h0` accepts a captured baseline or any compatible Harness published by
optimization, distillation, or a previous stream. Each stage evaluates its
starting Harness, adds a topology for each new task, and evaluates updates
against the old and new tasks. Existing task topologies remain frozen
throughout the stream. The current LIBERO stream uses reset 0 for retention
and resets 11–13 for new-task development. The selected H0 determines the
suite and runner.

`--rounds` caps search rounds per stage. Use a dedicated GPU or an existing
ReWeave episode pool. `--test-reset-id` additionally evaluates all stage
checkpoints on the same held-out reset and writes
`stream/stage_test_report.json`. The held-out reset must differ from every
development reset. Read `stream/checkpoints.json` for the stage order and
`stream/updates/NNN/harness.json` for each deployable Harness.

Re-run optimization, distillation, or continual learning with the same
inputs and output directory to reuse completed work. For a different parent,
trajectory selection, source set, or stage schedule, use a new output path.

## Benchmark CLIs

The lower-level CLIs also support custom protocols and fixed-panel
experiments. See [example protocols](../examples/graph_evolution/README.md)
for task, budget, and author configurations.

```bash
reweave --robot libero --help
reweave --robot robodojo --help
reweave-evolve schema
reweave-distill schema
```

For a custom budgeted search, capture a baseline with `reweave --capture-bundle`
using a private `--memory-dir`, kernel limits, and a runtime resource manifest.
Pass the captured Bundle and matching rollout recipe to `reweave-evolve init`.
Use `reweave-evolve replay-graph` to recompute projections from existing receipts.
Use `reweave-distill` for full shared plans, per-task runners, and fixed panels.

For RoboDojo setup and asset preparation, see the
[RoboDojo guide](../reweave/benchmarks/robodojo/README.md).
