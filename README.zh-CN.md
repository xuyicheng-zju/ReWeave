<h1 align="center">ReWeave</h1>

<p align="center"><strong>面向具身智能体的拓扑引导 Harness 演化</strong></p>

<p align="center">
  <a href="#overview">概览</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="#documentation">文档</a> ·
  <a href="#benchmarks">基准</a> ·
  <a href="#repository">代码结构</a> ·
  <a href="#acknowledgments">致谢</a> ·
  <a href="README.md">English</a>
</p>

<h2 id="overview">概览</h2>

ReWeave 从交互中联合优化具身智能体的 Harness 和物理任务拓扑。新证据修订拓扑后，系统在更新后的拓扑上重新投影历史试验，识别进展和瓶颈，用于指导 Harness 搜索。跨任务时，拓扑连接可复用机制，将其蒸馏为共享 Harness。[方法说明](docs/METHOD.md)

<h2 id="quick-start">快速开始</h2>

**Python 3.11 或 3.12。** 在仓库根目录安装并运行 CPU 示例。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/robodojo_demo.py --output-dir runs/robodojo-demo
```

示例使用脚本化的模型与环境响应，输出动作轨迹和单局回执。运行真实仿真需要准备上游运行时、场景资产和模型权重。[安装指南](docs/INSTALLATION.md)

| 命令 | 用途 |
| --- | --- |
| `reweave --help` | 运行单局任务 |
| `reweave-evolve --help` | 在固定开发条件下搜索 Harness |
| `reweave-distill --help` | 构建和评估共享 Harness |

<h2 id="documentation">文档</h2>

| 文档 | 内容 |
| --- | --- |
| [安装](docs/INSTALLATION.md) | Python 环境、模型服务和仿真依赖 |
| [评测](docs/EVALUATION.md) | 单局命令、冻结基线和留出评测 |
| [方法](docs/METHOD.md) | C/W/L 运行时、拓扑更新和跨任务蒸馏 |
| [示例协议](examples/graph_evolution/) | 开发条件、作者配置和拓扑分析输入 |

<h2 id="benchmarks">基准</h2>

| 后端 | 单局条件 |
| --- | --- |
| LIBERO-PRO | 任务套件、任务编号、随机种子和复位状态 ID |
| RoboDojo | 任务名称、布局集合和布局 ID |

单局结果以环境评估器的判定为准。[评测说明](docs/EVALUATION.md)

<h2 id="repository">代码结构</h2>

```text
reweave/
  runtime/                 C/W/L 执行与校验
  evolution/
    graph/                 拓扑修订与历史重投影
    search/                候选 Harness 搜索与评估
    distillation/          共享 Harness 构建
  benchmarks/              LIBERO 和 RoboDojo 适配器
examples/                  CPU 示例与示例协议
tests/                     运行时、方法和适配器测试
third_party/               上游链接与源码归属
```

<h2 id="acknowledgments">致谢</h2>

ReWeave 使用 RPent 和 RLinf 的具身执行组件，使用 LIBERO-PRO 和 RoboDojo 作为仿真基准，并接入 OpenPI、SAM3 和 GraspGen-X 提供策略与感知服务。[上游组件与源码归属](third_party/README.md)

<h2 id="license">许可证</h2>

ReWeave 代码与文档采用 [Apache-2.0](LICENSE) 许可证。
