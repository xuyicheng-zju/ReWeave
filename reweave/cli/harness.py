"""The Harness artifact exchanged by the four user-facing workflows."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.bundle import AgentBundle, new_record_id
from reweave.evolution.contracts import EvaluationRecord, Instance, SemanticGraph
from reweave.evolution.distillation.conditions import (
    NativeSourceConditions,
    encoding_request,
)
from reweave.evolution.distillation.engine import (
    SharedPlan,
    SharedTask,
    SharedTeacher,
    bind_json,
)
from reweave.evolution.search.rollout import RolloutConfiguration


class TaskKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: SharedTask
    graph: SemanticGraph
    evidence: tuple[EvaluationRecord, ...] = ()


class Harness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str
    bundle: AgentBundle
    runner: RolloutConfiguration
    knowledge: dict[str, TaskKnowledge] = {}

    @model_validator(mode="after")
    def runtime_matches(self):
        runtime = self.runner.runtime_manifest.runtime_configuration()
        if self.bundle.runtime.model_dump(
            exclude={"revision_id"}
        ) != runtime.model_dump(exclude={"revision_id"}):
            raise ValueError(
                "Harness bundle and runner have different runtime settings"
            )
        for name, item in self.knowledge.items():
            if name != item.task.name or any(
                row.request.split != "dev"
                or row.request.instance.task != item.task.dev[0].task
                for row in item.evidence
            ):
                raise ValueError(
                    "Harness knowledge must cite its own task's development evidence"
                )
        return self

    @classmethod
    def read(cls, path: Path):
        return cls.model_validate_json(path.read_text())

    def publish(self, directory: Path):
        path = directory / "harness.json"
        bind_json(path, self.model_dump(mode="json"))
        return path

    def instance(self, task: int, seed: int, reset_state_id: int):
        return Instance(
            record_id=new_record_id(),
            task=f"libero:{self.suite}:{task}",
            initialization={
                "suite": self.suite,
                "task": task,
                "seed": seed,
                "reset_state_id": reset_state_id,
            },
        )


def author_policy(path):
    return (
        AuthorPolicy.model_validate_json(path.read_text())
        if path is not None
        else AuthorPolicy(author="codex")
    )


async def encode_knowledge(parent, sources, policy, directory):
    """Encode exactly the selected graph bank against one common fluent catalog."""
    manifest = parent.runner.runtime_manifest
    teachers = tuple(
        SharedTeacher(task=name, bundle=bundle, learned_tasks=(name,), graph=item.graph)
        for name, (bundle, item) in sources.items()
    )
    plan = SharedPlan(
        record_id="source-encoding",
        mode="consolidation",
        method="direct",
        sources=tuple(item.task for _, item in sources.values()),
        teachers=teachers,
        initial=parent.bundle,
        execution_id=manifest.execution_id,
        manifest_id="source-encoding",
        conditions=manifest.conditions,
        test_conditions=manifest.conditions,
        max_rounds=0,
        conditional_retention=False,
    )
    request = encoding_request(plan)
    request["feedback"] = [
        record.model_dump(mode="json")
        for _, item in sources.values()
        for record in item.evidence
    ]
    author = NativeSourceConditions(
        bundle_loader=None,
        policy=policy,
        evidence_workspace=directory,
        runtime_manifest=manifest,
    )
    result = await author(request, directory)
    return plan, {
        name: SemanticGraph.model_validate(graph)
        for name, graph in result["graphs"].items()
    }
