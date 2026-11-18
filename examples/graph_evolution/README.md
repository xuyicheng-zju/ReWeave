# Graph evolution examples

- [LIBERO protocol](libero_protocol.json) and [RoboDojo protocol](robodojo_protocol.json)
  specify one development condition and a search budget.
- [Author policy](author_policy.json) selects a native Codex author and model.
- [Graph policy](graph_policy.json) defines observation bins and the analysis
  time limit.
- [Analysis result](analysis_result.json) illustrates an authored topology and
  its evidence references. Its artifact paths and IDs are synthetic.

Verify the simulator horizon, model and resources for your environment before
using a protocol. The [evaluation guide](../../docs/EVALUATION.md) covers
baseline capture, rollout recipes and replay. The physical verdict and
development traces must come from the selected benchmark runtime.
