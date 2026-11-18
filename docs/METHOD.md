# Method and implementation

## Harness

A candidate bundle contains the context program **C**, workflow programs **W**,
and loop program **L**. The runtime binds them to the robot toolkit and applies
a fixed capability, action, and model-request contract. See
[bundle.py](../reweave/evolution/bundle.py) and
[kernel.py](../reweave/runtime/kernel.py).

## Topology-guided evolution

1. Run a frozen candidate on the development conditions and archive the
   terminal verdict, control trace, observations and program executions.
2. Update the physical task topology with the new evidence. Reproject the
   admitted history onto the revised topology, deriving progress on its
   required milestones and retaining the evidence for conditional transitions.
3. Select a complete parent and use the topology and archived programs to
   guide new author mutations. Admit each resulting bundle before evaluating
   it under the same development contract.

[analysis.py](../reweave/evolution/graph/analysis.py),
[reprojection.py](../reweave/evolution/graph/reprojection.py),
[selection.py](../reweave/evolution/graph/selection.py), and
[engine.py](../reweave/evolution/search/engine.py) implement these stages.

## Cross-task distillation

The shared-harness path uses a source-only pool of frozen graphs, bundles and
episode evidence. It groups reusable transition mechanisms by topology and
conditions, constructs a shared candidate, and evaluates the frozen candidate
on declared test panels. [The distillation package](../reweave/evolution/distillation/)
includes fragment construction, topology factoring, shared authoring and
continual updates.

The [example protocols](../examples/graph_evolution/) show a single task
with a fixed development condition. See [Evaluation](EVALUATION.md) to
create a baseline and run the search.
