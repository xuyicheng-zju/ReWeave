# Upstream components

Source files retain their upstream copyright. Dependencies, weights and scene
assets are obtained separately under their upstream terms.

| Component | Source and use |
| --- | --- |
| LIBERO-PRO | [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) provides tasks, perturbations and reset states. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) provides the standard environments. |
| RPent | [RPent](https://github.com/RLinf/RPent) contributed harness and robot adapter code. |
| RLinf | [RLinf](https://github.com/RLinf/RLinf) supplies simulation and policy interfaces. |
| RoboDojo | [RoboDojo-Benchmark/RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) provides the official tasks and layout definitions. [RLinf/RoboDojo](https://github.com/RLinf/RoboDojo) and the [RLinf bridge](https://github.com/littleZ05/RLinf/tree/rpent/robodojo) supply the interactive runtime used by the adapter. |
| OpenPI | [OpenPI](https://github.com/Physical-Intelligence/openpi) provides Pi0.5. The [RLinf OpenPI fork](https://github.com/RLinf/openpi/tree/rpent) supplies the matching integration. |
| SAM3 | [SAM3](https://github.com/facebookresearch/sam3) provides segmentation. The adapter works with the [RLinf SAM3 fork](https://github.com/RLinf/sam3) interface. |
| GraspGen-X | [GraspGen-X](https://github.com/NVlabs/GraspGenX) provides optional grasp proposals. |
| PyRUA-Lean and ASPIRE | [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean) and [ASPIRE](https://github.com/NVlabs/ASPIRE) contributed adapted prompt or control source. |

The RoboDojo adapter targets task and capture contracts from upstream revision
[`cb2a37d`](https://github.com/RoboDojo-Benchmark/RoboDojo/commit/cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c).
Use a compatible `CollectEnv` bridge and policy runtime for that revision.


## Source adaptations

Online workflow authoring adapts the program contract and image notes from
`src/pyrualean/arms.py` and the Program rules and Stage 1 from
`.claude/skills/iterative-debugging/subagent-prompt.md`. The adaptation maps
complete programs to `WorkflowSpec` and uses workflow results and continuations
with ReWeave's observation, budget and admission contracts. Copyright 2026
PyRUA-Lean Contributors. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. Apache-2.0.
[PyRUA-Lean source](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/20dbb21e823030077ff4ddd4463cb91e36277515/src/pyrualean/arms.py) ·
[ASPIRE source](https://github.com/NVlabs/ASPIRE/blob/fa00f0ef2bc1d91d594b9aa502c22133c9f538f7/.claude/skills/iterative-debugging/subagent-prompt.md)

Object geometry and gripper convergence adapt `src/pyrualean/hosts/robodojo_tools.py`
to the measured support plane and each backend's action interface.
Copyright 2026 PyRUA-Lean Contributors. Apache-2.0.
[PyRUA-Lean source](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/fdf6e870654254e2c0c891ee407313fa48bdd0f9/src/pyrualean/hosts/robodojo_tools.py)
