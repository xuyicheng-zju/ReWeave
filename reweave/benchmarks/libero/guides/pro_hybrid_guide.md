# LIBERO-PRO scene adaptation

Follow the [baseline tool contract](strict_hybrid_guide.md) and
[calibration contract](env_calibration.md). PRO perturbations change the scene;
a task number, old seed or recipe does not identify current coordinates.

Use the current `task_language` to identify target objects, destinations and
spatial relations. Agentview supplies global context; wrist views help inspect
occluded contact geometry. Object names do not reveal which duplicate satisfies
a relation, and fixed fixtures may not appear in the object inventory.

For swaps, preserve semantic identity across views instead of letting each
segmentation query independently choose a look-alike. Confirm masks in original
RGB. If the target is identifiable but obstructed, identify the physical blocker
and choose a feasible approach or rearrangement supported by the scene. Further
queries of an unchanged visible handle do not clear its approach path.

Consult matching memory only for a relevant knowledge gap. Verify the reference's task language,
suite and evidence. Transfer a technique only when its prerequisites hold;
recompute coordinates and held-object offsets, and reject reset/oracle commands
or parameters for an older controller. Numerical settings that once worked are
observations about that run, not proofs of a universally reachable workspace.

A failure of one approach does not establish that the task is physically
impossible. Report what was attempted, the observed effect and the unresolved
alternative explanations. Do not relabel an execution or perception error as a
failure of learned skills. Baseline repairs require a new source version and
fresh baseline results before comparison with evolved harnesses.
