<h1 align="center">ReWeave</h1>

<p align="center"><strong>Topology-guided harness evolution for embodied agents</strong></p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#repository">Repository</a> ·
  <a href="#acknowledgments">Credits</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<h2 id="overview">Overview</h2>

ReWeave jointly optimizes an embodied agent harness and a physical task topology
from interaction. It uses new evidence to revise the topology, reprojects earlier
trials to recover progress and diagnose bottlenecks, and uses those findings to
guide harness search. Across tasks, the topology connects reusable mechanisms
for distillation into a shared harness. [Method](docs/METHOD.md)

<h2 id="quick-start">Quick Start</h2>

**Python 3.11 or 3.12.** From the repository root, install the package and run
the CPU demo.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/robodojo_demo.py --output-dir runs/robodojo-demo
```

The demo runs the harness with scripted model and environment responses and
writes an action trace and episode receipt. For live simulation, prepare the
benchmark runtime, scene assets, and model weights. [Installation](docs/INSTALLATION.md)

| Command | Use |
| --- | --- |
| `reweave --help` | Run one episode. |
| `reweave-evolve --help` | Search for a harness on fixed development conditions. |
| `reweave-distill --help` | Build and evaluate a shared harness. |

<h2 id="documentation">Documentation</h2>

| Guide | Coverage |
| --- | --- |
| [Installation](docs/INSTALLATION.md) | Python, model providers, and simulator dependencies |
| [Evaluation](docs/EVALUATION.md) | Episode commands, frozen baselines, and held-out tests |
| [Method](docs/METHOD.md) | The C/W/L runtime, topology revision, and cross-task distillation |
| [Example protocols](examples/graph_evolution/) | Development conditions, author policy, and graph inputs |

<h2 id="benchmarks">Benchmarks</h2>

| Backend | Episode identity |
| --- | --- |
| LIBERO-PRO | Task suite, task index, random seed, and reset-state ID |
| RoboDojo | Task name, layout set, and layout ID |

The environment evaluator supplies the episode verdict. [Evaluation](docs/EVALUATION.md)

<h2 id="repository">Repository</h2>

```text
reweave/
  runtime/                 C/W/L execution and validation
  evolution/
    graph/                 Topology revision and history reprojection
    search/                Candidate search and evaluation
    distillation/          Shared harness construction
  benchmarks/              LIBERO and RoboDojo adapters
examples/                  CPU demo and example protocols
tests/                     Runtime, method, and adapter tests
third_party/               Upstream links and source attribution
```

<h2 id="acknowledgments">Acknowledgments</h2>

ReWeave builds on RPent and RLinf for embodied execution, LIBERO-PRO and RoboDojo
for simulation benchmarks, and OpenPI, SAM3, and GraspGen-X for policy and
perception services. Source adaptations and upstream terms are documented in
[third-party components](third_party/README.md).

<h2 id="license">License</h2>

ReWeave code and documentation are licensed under [Apache-2.0](LICENSE).
