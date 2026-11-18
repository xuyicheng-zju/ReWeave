"""Mixed-family budgets retain exact physical requests and source-only feedback."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.cli.distill import SharedConfiguration, TaskRolloutEvaluator
from reweave.cli.main import parse_cli_arguments
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.distillation.conditions import encoding_request
from reweave.evolution.distillation.engine import (
    SharedEvolution,
    SharedTask,
    SharedTaskExecution,
    SharedTeacher,
)
from reweave.evolution.runtime_manifest import execution_configuration
from reweave.evolution.search.rollout import RolloutConfiguration
from tests.unit_tests.harness.evolution.test_rollout_recipe import (
    instance,
    recipe_fixture,
)
from tests.unit_tests.harness.evolution.test_shared import (
    Author,
    Evaluator,
    bundle,
    plan,
)


def bound_task(name, steps, *, test_steps=None):
    original = plan()
    return SharedTask.model_validate(
        {
            **original.sources[0].model_dump(mode="json"),
            "name": name,
            "family": name,
            "dev": [
                {
                    "record_id": name + "-dev",
                    "task": name,
                    "initialization": {"seed": 0},
                }
            ],
            "test": [
                {
                    "record_id": name + "-test",
                    "task": name,
                    "initialization": {"seed": 1},
                }
            ],
            "execution": {
                "execution_id": name + "-execution",
                "conditions": {
                    **original.conditions.model_dump(),
                    "physical_steps": steps,
                },
                "test_conditions": {
                    **original.test_conditions.model_dump(),
                    "physical_steps": test_steps or steps,
                },
            },
        }
    )


def mixed_plan():
    tasks = (bound_task("spatial", 230), bound_task("long", 530))
    return plan().model_copy(
        update={
            "sources": tasks,
            "unseen": (),
            "teachers": tuple(
                SharedTeacher(
                    task=t.name,
                    bundle=bundle(t.name + "-expert"),
                    learned_tasks=(t.name,),
                )
                for t in tasks
            ),
            "max_rounds": 1,
        }
    )


def test_every_candidate_and_frozen_panel_use_task_budget_without_target_leak(tmp_path):
    p = mixed_plan()
    author, evaluator = Author(), Evaluator()
    engine = SharedEvolution(tmp_path, author, evaluator)
    asyncio.run(engine.run(p))
    assert len(evaluator.requests) == 6  # two teachers, H0 and one all-source round
    assert all(
        r.conditions.physical_steps == {"spatial": 230, "long": 530}[r.instance.task]
        for r in evaluator.requests
    )
    assert all(
        r.execution == r.instance.task + "-execution" for r in evaluator.requests
    )
    assert [
        t["execution"]["conditions"]["physical_steps"]
        for t in author.requests[0]["tasks"]
    ] == [230, 530]
    assert all(
        "test_conditions" not in t["execution"] for t in encoding_request(p)["tasks"]
    )
    ood = bound_task("ood", 290)
    # Both source retention and a separate within-family OOD panel are frozen tests.
    asyncio.run(engine.test_panels(p, {"source": p.sources, "ood": (ood,)}))
    tests = [r for r in evaluator.requests if r.split == "test"]
    assert {r.instance.task: r.conditions.physical_steps for r in tests} == {
        "spatial": 230,
        "long": 530,
        "ood": 290,
    }
    assert "ood" not in json.dumps(author.requests)
    count = len(evaluator.requests)
    asyncio.run(engine.test_panels(p, {"source": p.sources, "ood": (ood,)}))
    assert len(evaluator.requests) == count
    changed = p.sources[0].model_copy(
        update={"execution": bound_task("spatial", 450).execution}
    )
    with pytest.raises(ValueError, match="recovery identity changed"):
        asyncio.run(engine.evaluate(p, p.initial, (changed,), tmp_path / "initial"))
    assert len(evaluator.requests) == count


def configuration(tmp_path):
    _, _, runner, _ = recipe_fixture(tmp_path)
    p = mixed_plan().model_copy(
        update={
            "conditions": runner.runtime_manifest.conditions,
            "test_conditions": runner.runtime_manifest.conditions,
        }
    )
    tasks, runners = [], {}
    for task in p.sources:
        conditions = runner.runtime_manifest.conditions.model_copy(
            update={"physical_steps": task.execution.conditions.physical_steps}
        )
        binding = task.execution.model_copy(
            update={"conditions": conditions, "test_conditions": conditions}
        )
        task = task.model_copy(update={"execution": binding})
        tasks.append(task)
        value = runner.model_dump(mode="json")
        args = value["arguments"]
        args[args.index("--max-episode-steps") + 1] = str(conditions.physical_steps)
        value["runtime_manifest"]["conditions"] = conditions.model_dump(mode="json")
        value["runtime_manifest"]["execution_id"] = binding.execution_id
        bound_runner = RolloutConfiguration.model_validate(value)
        _, actual, robot = parse_cli_arguments(
            bound_runner.build_arguments(
                instance(),
                directory=tmp_path / task.name,
                memory=Path(runner.memory_snapshot),
                device=0,
            )
        )
        robot.parse_config(actual)
        value["runtime_manifest"]["configuration"] = execution_configuration(
            vars(actual), {"episode": runner.episode_limits.model_dump(mode="json")}
        )
        runners[task.name] = value
    return SharedConfiguration.model_validate(
        {
            "plan": p.model_copy(update={"sources": tuple(tasks)}),
            "runner": runner,
            "test_runner": runner,
            "author": AuthorPolicy(author="codex", graph_retrieval=None),
            "resource_workspace": tmp_path,
            "task_runners": runners,
            "task_test_runners": runners,
        }
    )


def test_task_runner_binding_requires_exact_conditions_and_both_splits(tmp_path):
    cfg = configuration(tmp_path)
    value = cfg.model_dump(mode="json")
    value["task_test_runners"].pop("long")
    with pytest.raises(ValidationError, match="exact dev/test runners"):
        SharedConfiguration.model_validate(value)
    value = cfg.model_dump(mode="json")
    value["task_runners"]["long"]["runtime_manifest"]["conditions"][
        "physical_steps"
    ] = 230
    with pytest.raises(ValidationError, match="execution binding"):
        SharedConfiguration.model_validate(value)
    with pytest.raises(ValidationError, match="static episodes"):
        SharedTaskExecution.model_validate(
            {
                **cfg.plan.sources[0].execution.model_dump(),
                "conditions": cfg.plan.conditions.model_copy(
                    update={"adaptation_mode": "adaptive"}
                ),
            }
        )


def test_dispatch_and_recovery_preserve_bundle_and_exact_request(tmp_path, monkeypatch):
    from reweave.cli import distill as shared_harness

    cfg = configuration(tmp_path)
    calls = []

    class Runner(Evaluator):
        def __init__(self, configuration, **kwargs):
            super().__init__()
            self.configuration = configuration

        async def __call__(self, request, b, directory, *, before_launch):
            assert request.conditions == self.configuration.runtime_manifest.conditions
            calls.append(("start", request, b))
            return await super().__call__(
                request, b, directory, before_launch=before_launch
            )

        async def recover(self, request, b, directory, *, before_launch):
            calls.append(("recover", request, b))
            return await super().__call__(
                request, b, directory, before_launch=before_launch
            )

    monkeypatch.setattr(shared_harness, "RolloutEvaluator", Runner)
    adapter = TaskRolloutEvaluator(cfg, repository=Path.cwd())
    engine = SharedEvolution(tmp_path / "shared", Author(), adapter)
    records = asyncio.run(
        engine.evaluate(
            cfg.plan, cfg.plan.initial, cfg.plan.sources, tmp_path / "evaluation"
        )
    )
    request = records[-1].request
    recovered = asyncio.run(
        adapter.recover(
            request, cfg.plan.initial, tmp_path / "recovery", before_launch=lambda: None
        )
    )
    assert recovered.request == request
    assert [c[0] for c in calls] == ["start", "start", "recover"]
    assert all(c[2] == cfg.plan.initial for c in calls)
    invalid = request.model_copy(
        update={"conditions": cfg.plan.sources[0].execution.conditions}
    )
    with pytest.raises(ValueError, match="task execution binding"):
        adapter.runner_for(invalid)
    assert len(calls) == 3


def test_shared_panel_runs_cells_concurrently_in_declared_order(tmp_path):
    class Concurrent(Evaluator):
        def __init__(self):
            super().__init__()
            self.started = 0
            self.all_started = asyncio.Event()

        async def __call__(self, request, b, directory, *, before_launch):
            self.started += 1
            if self.started == 2:
                self.all_started.set()
            await asyncio.wait_for(self.all_started.wait(), timeout=1)
            return await super().__call__(
                request, b, directory, before_launch=before_launch
            )

    async def run():
        p = mixed_plan()
        evaluator = Concurrent()
        engine = SharedEvolution(tmp_path, Author(), evaluator)
        records = await engine.evaluate(p, p.initial, p.sources, tmp_path / "panel")
        assert [r.request.instance.task for r in records] == ["spatial", "long"]
        count = evaluator.started
        assert (
            await engine.evaluate(p, p.initial, p.sources, tmp_path / "panel")
            == records
        )
        assert evaluator.started == count

    asyncio.run(run())


def test_shared_panel_settles_other_cells_before_raising(tmp_path):
    class PartiallyFailed(Evaluator):
        async def __call__(self, request, b, directory, *, before_launch):
            if request.instance.task == "spatial":
                raise ValueError("one cell failed")
            await asyncio.sleep(0.01)
            return await super().__call__(
                request, b, directory, before_launch=before_launch
            )

    p = mixed_plan()
    engine = SharedEvolution(tmp_path, Author(), PartiallyFailed())
    with pytest.raises(ValueError, match="one cell failed"):
        asyncio.run(engine.evaluate(p, p.initial, p.sources, tmp_path / "panel"))
    assert (tmp_path / "panel/long/000/shared_result.json").is_file()


def test_dev_only_direct_measures_h0_once_and_never_tests_teachers(tmp_path):
    p = mixed_plan().model_copy(
        update={
            "teacher_retention": False,
            "retention_weight": 0,
            "conditional_retention": False,
        }
    )
    author, evaluator = Author(), Evaluator()
    engine = SharedEvolution(tmp_path, author, evaluator)
    frozen = asyncio.run(engine.run(p))
    assert len(evaluator.requests) == 4  # H0 and one candidate, two dev tasks each.
    assert all(
        r.bundle_id not in {t.bundle.record_id for t in p.teachers}
        for r in evaluator.requests
    )
    assert all(r.instance.initialization["seed"] == 0 for r in evaluator.requests)
    assert author.requests[0]["anchors"] == {}
    assert frozen["score"]["retention_evaluated"] is False
    assert frozen["score"]["mean_drop"] is None
    assert asyncio.run(engine.run(p)) == frozen
    assert len(evaluator.requests) == 4


def test_physical_cli_preflight_retains_each_bound_horizon(tmp_path):
    cfg = configuration(tmp_path)
    for name, runner in cfg.task_runners.items():
        argv = runner.validate_arguments(
            instance(),
            directory=tmp_path / name,
            memory=Path(runner.memory_snapshot),
            device=0,
        )
        assert (
            int(argv[argv.index("--max-episode-steps") + 1])
            == {
                "spatial": 230,
                "long": 530,
            }[name]
        )


def test_legacy_task_serialization_keeps_published_identity_unchanged():
    original = plan().sources[0].model_dump(mode="json")
    assert "execution" not in original
    assert SharedTask.model_validate(original).model_dump(mode="json") == original


def test_cold_start_graph_uses_the_task_execution_and_runtime(tmp_path):
    from reweave.evolution.distillation.graph_initialization import (
        NativeTaskGraphInitializer,
    )
    from tests.unit_tests.harness.evolution.test_shared_graph_initialization import (
        FakeAnalysis,
        cold_fixture,
    )

    p, task, h0, record = cold_fixture(tmp_path)
    binding = SharedTaskExecution(
        execution_id=record.request.execution,
        conditions=record.request.conditions,
        test_conditions=record.request.conditions,
    )
    task = task.model_copy(update={"execution": binding})
    p = p.model_copy(
        update={
            "sources": (task,),
            "execution_id": "another-default-execution",
            "conditions": record.request.conditions.model_copy(
                update={"physical_steps": 530}
            ),
        }
    )
    manifests = []

    def factory(**kwargs):
        manifests.append(kwargs["runtime_manifest"])
        return FakeAnalysis(**kwargs)

    initializer = NativeTaskGraphInitializer(
        policy=AuthorPolicy(),
        runtime_manifest="default-runtime",
        task_runtime_manifests={task.name: "task-runtime"},
        analyzer_factory=factory,
    )
    FakeAnalysis.calls = []
    asyncio.run(initializer(p, task, h0, [record], tmp_path / "graph"))
    request = FakeAnalysis.calls[0][1]
    assert request.protocol.execution == binding.conditions
    assert request.protocol.execution_id == binding.execution_id
    assert manifests == ["task-runtime"]
