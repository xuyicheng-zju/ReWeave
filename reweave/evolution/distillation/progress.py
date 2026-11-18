"""Score source-dev trajectories on fixed physical topology backbones."""

from __future__ import annotations

from pydantic import Field, TypeAdapter

from reweave.evolution.bundle import Asset
from reweave.evolution.distillation.assessment import measurement_request
from reweave.evolution.distillation.author import NativeSharedAuthor
from reweave.evolution.validation.admission import _read_workspace_file
from reweave.utils.persistence import atomic_json


class TopologyObservation(Asset):
    record_id: str
    node_id: str | None
    citations: tuple[str, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)


def progress_scores(plan, records, observations):
    """One official goal, necessary milestones, and no reward for retry loops."""
    observed = {o.record_id: o for o in observations}
    failed = {r.record_id for r in records if not r.official_success}
    if len(observed) != len(observations) or observed.keys() != failed:
        raise ValueError(
            "topology projection must cover exactly the failed dev records"
        )
    tasks = {t.dev[0].task: t.name for t in plan.sources}
    values, details = {t.name: [] for t in plan.sources}, {}
    for record in records:
        name = tasks[record.request.instance.task]
        graph = plan.source_graphs[name]
        milestones = {k for k, n in graph.nodes.items() if n.role == "milestone"}
        denominator = len(milestones) + 1
        if record.official_success:
            lower = upper = 1.0
            node, basis = None, "official_success"
            citations = ()
        else:
            observation = observed[record.record_id]
            if not set(observation.citations) <= {e.path for e in record.evidence}:
                raise ValueError("topology projection cites unrelated evidence")
            node = observation.node_id
            if node is not None and (
                node not in graph.nodes or graph.nodes[node].role == "goal"
            ):
                raise ValueError("failed rollout cannot claim an official goal")
            retained = graph.completed_nodes()[node] if node is not None else ()
            lower = len(set(retained) & milestones) / denominator
            upper = lower if node is not None else len(milestones) / denominator
            basis = observation.explanation
            citations = observation.citations
        values[name].append(lower)
        details[record.record_id] = {
            "task": name,
            "lower": lower,
            "upper": upper,
            "node_id": node,
            "basis": basis,
            "citations": list(citations),
        }
    means = {name: sum(v) / len(v) for name, v in values.items()}
    return {
        "topology_progress": means,
        "topology_loss": 1 - sum(means.values()) / len(means),
        "topology_records": details,
    }


class NativeTopologyProgress(NativeSharedAuthor):
    source_queries = False
    instructions = """Project original SOURCE DEVELOPMENT rollouts onto fixed
reference task topologies. Do not change topology, C/W/L, conditions or verdicts.
Read context/topology_projection.json and its original observations, images and
ordered actions. For each requested failed rollout, identify the latest retained
physical state supported at the end of the rollout. Earlier transient progress
that was lost does not count. Choose a node from that task's fixed graph, or null
when the retained state is unobserved/uncertain. Never infer progress from the
candidate code, its narrative, a workflow return alone, or an unexecuted plan.
Optional branches and repeated attempts do not add completion. Failed rollouts
cannot reach a goal node. Official successes are scored separately by the Host.
Write candidate/topology_projection.json, one observation per requested record,
using context/topology_projection.schema.json. Cite exact original evidence FILE
paths; put message/frame/trace locations in explanation. No extra experiments,
physical actions, model probes, test evidence, or cross-task matching.
"""
    prompt = "Project only the requested failed source-dev trajectories onto the fixed graphs."

    def prepare(self, request, directory):
        workspace, readable = super().prepare(request, directory)
        atomic_json(
            workspace / "context/topology_projection.json", request["projection"]
        )
        atomic_json(
            workspace / "context/topology_projection.schema.json",
            TypeAdapter(list[TopologyObservation]).json_schema(),
        )
        return workspace, readable

    async def admit(self, request, workspace, audit):
        observations = TypeAdapter(list[TopologyObservation]).validate_json(
            _read_workspace_file(workspace, "candidate/topology_projection.json")
        )
        expected = {r["record_id"] for r in request["feedback"]}
        if (
            len(observations) != len(expected)
            or {o.record_id for o in observations} != expected
        ):
            raise ValueError("topology projection omitted or duplicated a record")
        return {"observations": [o.model_dump(mode="json") for o in observations]}

    async def score(self, plan, bundle, records, directory):
        failed = [r for r in records if not r.official_success]
        observations = ()
        if failed:
            request = measurement_request(plan, bundle, failed)
            request["projection"] = {
                "records": [r.record_id for r in failed],
                "graphs": {
                    name: {k: n.model_dump(mode="json") for k, n in graph.nodes.items()}
                    for name, graph in plan.source_graphs.items()
                },
                "rule": "latest retained physical state; fixed necessary backbone",
            }
            result = await self(request, directory / "author")
            observations = TypeAdapter(list[TopologyObservation]).validate_python(
                result["observations"]
            )
        report = progress_scores(plan, records, observations)
        atomic_json(directory / "result.json", report)
        return report
