# ReWeave

ReWeave jointly optimizes an embodied agent harness and a physical task topology
from interaction. As new observations revise the topology, ReWeave reprojects
earlier trials onto it to recover progress and diagnose bottlenecks. The same
topology organizes reusable mechanisms for cross-task distillation.

[Quick start](#quick-start) · [Installation](docs/INSTALLATION.md) ·
[Evaluation](docs/EVALUATION.md) · [Method](docs/METHOD.md) ·
[Upstream components](third_party/README.md) · [中文](README.zh-CN.md)

## Quick start

Install Python 3.11 or 3.12 and clone this repository. The simulated example
runs locally on CPU with scripted model and environment responses.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/robodojo_demo.py --output-dir runs/robodojo-demo
```

The example exercises the production harness runtime and records actions and
an episode receipt. GPU simulation, model inference, and benchmark evaluation
require the dependencies in the installation guide.

## Repository

| Path | Responsibility |
| --- | --- |
| `reweave/runtime/` | Execute and validate context, workflows, and persistent state. |
| `reweave/evolution/graph/` | Analyze physical task topology and reproject archived trials. |
| `reweave/evolution/search/` | Evaluate candidate harnesses on fixed development conditions. |
| `reweave/evolution/distillation/` | Build shared harnesses from source-task evidence. |
| `reweave/benchmarks/` | LIBERO and RoboDojo adapters, task interfaces, and scoring. |
| `examples/` | A CPU demo and example evolution inputs. |
| `tests/` | Runtime, search, distillation, and adapter tests. |

Run `reweave --help` for episodes, `reweave-evolve --help` for task-local
evolution, and `reweave-distill --help` for cross-task distillation.

## License

ReWeave code and documentation are available under [Apache-2.0](LICENSE).
[NOTICE](NOTICE) retains source attributions. Benchmark assets, model weights,
and external simulators are installed separately under their upstream terms.
