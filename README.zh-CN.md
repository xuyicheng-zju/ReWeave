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
  <a href="#overview"><strong>亮点</strong></a> &nbsp;·&nbsp;
  <a href="#quick-start"><strong>快速开始</strong></a> &nbsp;·&nbsp;
  <a href="#documentation"><strong>文档</strong></a> &nbsp;·&nbsp;
  <a href="#benchmarks"><strong>基准</strong></a> &nbsp;·&nbsp;
  <a href="#repository"><strong>代码</strong></a> &nbsp;·&nbsp;
  <a href="#acknowledgments"><strong>致谢</strong></a> &nbsp;·&nbsp;
  <a href="README.md"><strong>English</strong></a>
</p>

<p>
  <a href="docs/assets/teaser.png"><img src="docs/assets/teaser.png" width="100%" alt="ReWeave 论文导览图，展示失败试验中的保留进展、物理任务拓扑的修订，以及 Harness 优化和跨任务结果。"></a>
</p>

</div>

---

<h2 id="overview">✦ 亮点</h2>

- **显化的 Meta。** 在任务拓扑上看清每次执行满足了哪些物理条件、卡在哪一步，并在获得新证据后重评旧试验。
- **高效的自进化。** 复用历史试验定位瓶颈，优先验证有证据支撑的 Harness 修改。
- **共享 Harness 的蒸馏与持续学习。** 将多个任务验证过的机制蒸馏成一个共享 Harness，在未见任务上复用，并从新试验中继续改进。

[查看方法说明](docs/METHOD.md)

<h2 id="quick-start">⚡ 快速开始</h2>

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

<h2 id="documentation">📚 文档</h2>

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
