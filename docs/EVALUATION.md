# Evaluation

Run the CPU smoke demo before connecting external services.

```bash
python examples/robodojo_demo.py --output-dir runs/robodojo-demo
```

It exercises the production RoboDojo adapter, W/C/L runtime, trace recorder,
and episode finalizer with scripted inputs. Use a fresh output directory.

## Simulator episodes

Prepare the simulator and model dependencies in [Installation](INSTALLATION.md).
The following commands start one episode with the selected native model.
Pass an explicit `--model` for your account.

```bash
reweave --robot libero --libero-type pro --suite libero_goal_task --task 0 \
  --seed 0 --reset-state-id 0 --cuda-device 0 --output-dir runs/libero
reweave --robot robodojo --task put_bottles_into_dustbin --layout-set 1 \
  --layout 0 --cuda-device 0 --max-episode-steps 700 --output-dir runs/robodojo
```

The episode artifacts include planner termination, environment verdict and
action acknowledgements. Benchmark success comes from the environment evaluator.
Keep development conditions separate from held-out tests. In LIBERO, record the
suite, task, simulator reset-state ID and seed. In RoboDojo, record the task,
layout set and layout ID.

## Frozen baseline and evolution

Choose a task [protocol](../examples/graph_evolution/README.md). Its
`execution` object declares the physical horizon, settling steps, controller
budget and model sampling. Prepare a kernel limits JSON with the protocol's
`execution` object as `conditions` and an `episode` object containing
`program_calls`, `gateway_calls` and `wall_time_s` with the same values.
Prepare a runtime resources JSON with `models`, `environment` and `routes`
describing the exact installed models, benchmark inputs and provider model
revisions. Keep credentials in the environment.

Run `reweave` with the target backend, budget flags matching `conditions`,
and `--capture-bundle BASELINE_DIR --kernel-limits LIMITS_JSON
--runtime-resources RESOURCES_JSON`. This initializes the toolkit and writes
`bundle.json`, `runtime.json` and a frozen `memory/` snapshot. The capture
does not query the online model.

Use those outputs to prepare a rollout recipe. Its `arguments` array contains
the fixed CLI options. The protocol supplies task identity and reset or layout,
and the worker supplies GPU, output, memory and bundle flags. Set
`memory_snapshot` to the captured memory directory, `runtime_manifest` to the
contents of `runtime.json`, `episode_limits` to the limits JSON's `episode`
object, and `devices` to the GPU indices assigned to the run. The full schema
is available with `reweave-evolve schema`.

```bash
reweave-evolve init --baseline BASELINE_DIR/bundle.json --protocol PROTOCOL_JSON \
  --runner RUNNER_JSON --author-policy AUTHOR_JSON --workspace runs/evolution
reweave-evolve run --workspace runs/evolution --runtime-source "$PWD"
reweave-evolve status --workspace runs/evolution
```

`init` checks the fixed recipe and initializes the workspace. `run` starts model
authoring and evaluation. `replay-graph` recomputes graph projections from
archived evidence without contacting a model or simulator.

## Cross-task distillation

`reweave-distill schema` prints the configuration schema. Select source runs
whose graphs and receipts are complete and declare distinct development and
held-out panels. Use `validate` to check the plan, `encode` and
`prepare-fragments` to prepare source evidence, `run` to produce a shared
candidate, and `test` to evaluate the frozen candidate. Each mode except
`schema` takes `--config` and `--output`.

The repository packages the method, simulator adapters and synthetic examples.
Full benchmark reproduction requires the upstream GPU stack, matching model and
benchmark revisions, and experiment-specific task and asset manifests.
