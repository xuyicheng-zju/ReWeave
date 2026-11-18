# Upstream components

ReWeave retains copyright and adaptation notices for imported source in
[NOTICE](../NOTICE). Dependencies, weights and scene assets are obtained
separately and remain subject to their upstream terms.

| Component | Source and use |
| --- | --- |
| LIBERO-PRO | [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) provides tasks, perturbations and reset states. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) provides the standard environments. |
| RLinf | [RLinf](https://github.com/RLinf/RLinf) supplies simulation and policy interfaces. |
| RoboDojo | [RoboDojo-Benchmark/RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) provides the official tasks and layout definitions. [RLinf/RoboDojo](https://github.com/RLinf/RoboDojo) and the [RLinf bridge](https://github.com/littleZ05/RLinf/tree/rpent/robodojo) supply the interactive runtime used by the adapter. |
| OpenPI | [OpenPI](https://github.com/Physical-Intelligence/openpi) provides Pi0.5. The [RLinf OpenPI fork](https://github.com/RLinf/openpi/tree/rpent) supplies the matching integration. |
| SAM3 | [SAM3](https://github.com/facebookresearch/sam3) provides segmentation. The adapter works with the [RLinf SAM3 fork](https://github.com/RLinf/sam3) interface. |
| GraspGen-X | [GraspGen-X](https://github.com/NVlabs/GraspGenX) provides optional grasp proposals. |
| PyRUA-Lean and ASPIRE | [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean) and [ASPIRE](https://github.com/NVlabs/ASPIRE) contributed adapted prompt or control source. |

The RoboDojo adapter targets task and capture contracts from upstream revision
[`cb2a37d`](https://github.com/RoboDojo-Benchmark/RoboDojo/commit/cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c).
Use a compatible `CollectEnv` bridge and policy runtime for that revision.
