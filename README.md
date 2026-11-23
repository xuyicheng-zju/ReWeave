<div align="center">

# $\color{#8557A0}{\textsf{ReWeave}}$: Emergent Physical&nbsp;Task&nbsp;Topologies<br>for Embodied Harness Optimization<br>and Cross-Task Meta-Distillation

<p>
  <strong><a href="https://github.com/xuyicheng-zju">Yicheng&nbsp;Xu</a></strong> &nbsp;·&nbsp;
  <strong><a href="https://github.com/Rinke02">Wencan&nbsp;Jiang</a></strong> &nbsp;·&nbsp;
  <strong>Zihao&nbsp;Zhu</strong> &nbsp;·&nbsp;
  <strong>Keru&nbsp;Zhou</strong> &nbsp;·&nbsp;
  <strong>Chang&nbsp;Zhou</strong><br>
  <strong>Li&nbsp;Liu</strong> &nbsp;·&nbsp;
  <strong>Maoqing&nbsp;Yao</strong> &nbsp;·&nbsp;
  <strong>Yong&nbsp;Liu</strong> &nbsp;·&nbsp;
  <strong><a href="https://github.com/zhangzjn">Jiangning&nbsp;Zhang</a></strong>
</p>

<p>
  <a href="#overview"><strong>Highlights</strong></a> &nbsp;·&nbsp;
  <a href="#quick-start"><strong>Quick Start</strong></a> &nbsp;·&nbsp;
  <a href="#documentation"><strong>Docs</strong></a> &nbsp;·&nbsp;
  <a href="#benchmarks"><strong>Benchmarks</strong></a> &nbsp;·&nbsp;
  <a href="#repository"><strong>Code</strong></a> &nbsp;·&nbsp;
  <a href="#acknowledgments"><strong>Credits</strong></a> &nbsp;·&nbsp;
  <a href="README.zh-CN.md"><strong>中文</strong></a>
</p>

<p>
  <a href="docs/assets/teaser.png"><img src="docs/assets/teaser.png" width="100%" alt="ReWeave teaser showing progress in failed trials, physical task topology revision, and the harness optimization and cross-task results."></a>
</p>

</div>

---

<h2 id="overview">✦ Highlights</h2>

- **Explicit task meta-knowledge.** See which physical conditions a rollout met and where it stalled, then reassess past trials as new evidence comes in.
- **Efficient harness self-evolution.** Reuse archived trials to diagnose bottlenecks and decide which harness revisions to test next.
- **Shared harness distillation and continual learning.** Distill validated mechanisms from several tasks into one harness, reuse it on unseen tasks, and keep improving it with new trials.

[Read the method](docs/METHOD.md)

<h2 id="quick-start">⚡ Quick Start</h2>

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

<h2 id="documentation">📚 Documentation</h2>

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
