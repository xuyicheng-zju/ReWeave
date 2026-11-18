# ReWeave

ReWeave 从交互中联合优化具身智能体的 Harness 和物理任务拓扑。新证据修订拓扑后，系统在更新后的拓扑上对齐历史试验，用重新识别的进展和瓶颈指导 Harness 搜索。同一套拓扑表示也用于组织可复用机制，支持跨任务蒸馏。

[安装](docs/INSTALLATION.md) · [评测](docs/EVALUATION.md) · [方法与代码](docs/METHOD.md) · [上游组件](third_party/README.md) · [English](README.md)

## 快速开始

在仓库根目录用 Python 3.11 或 3.12 创建环境，运行 CPU 示例。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/robodojo_demo.py --output-dir runs/robodojo-demo
```

示例使用脚本化的模型和环境响应，验证 Harness 执行和轨迹记录接口。GPU 仿真和基准评测需要按照安装文档单独准备上游环境、场景资产和模型权重。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `reweave/runtime/` | Context、Workflow 和持久状态的执行与校验 |
| `reweave/evolution/graph/` | 物理任务拓扑分析、更新及历史重投影 |
| `reweave/evolution/search/` | 在固定开发条件下搜索与评估候选 Harness |
| `reweave/evolution/distillation/` | 利用源任务证据构建共享 Harness |
| `reweave/benchmarks/` | LIBERO 和 RoboDojo 适配器 |
| `examples/`、`tests/` | 可运行示例和自动化测试 |

三个公开入口为 `reweave`、`reweave-evolve` 和 `reweave-distill`。使用 `--help` 查看参数。

项目采用 [Apache-2.0](LICENSE) 许可证。上游代码归属见 [NOTICE](NOTICE)。
