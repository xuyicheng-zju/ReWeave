"""Zero-model replay of the paired shared-student method and native inputs."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.bundle import new_record_id
from reweave.evolution.contracts import SemanticGraph
from reweave.evolution.distillation.author import NativeSharedAuthor
from reweave.evolution.distillation.engine import (
    SharedEvolution,
    SharedPlan,
    SharedTeacher,
)
from reweave.evolution.distillation.stream import run_stream
from reweave.evolution.distillation.student_gaps import (
    build_gap_report,
    teacher_unit_bank,
)
from reweave.evolution.distillation.task_topology import NativeTaskTopology
from reweave.evolution.graph.conditions import GraphSemantics
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request
from tests.unit_tests.harness.evolution.test_graph_retrieval import (
    CATALOG,
    corpus_graph,
    fact,
)
from tests.unit_tests.harness.evolution.test_rollout_recipe import recipe_fixture
from tests.unit_tests.harness.evolution.test_shared import Author, Evaluator, plan, task
from tests.unit_tests.harness.evolution.test_shared_graph_initialization import (
    FakeAnalysis,
)


def paired_plan(tmp_path, *, old_count=1, new_count=2):
    graphs = {}
    for i in range(old_count):
        path = tmp_path / f"old{i}"
        path.mkdir(parents=True)
        graphs[f"old{i}"] = corpus_graph(path)
    p = plan()
    old = tuple(task(n) for n in graphs)
    new = tuple(task(f"new{i}") for i in range(new_count))
    values = p.model_dump(mode="json")
    values.update(
        mode="continual",
        method="aligned",
        sources=[t.model_dump(mode="json") for t in (*old, *new)],
        unseen=[],
        initial_learned_tasks=list(graphs),
        new_tasks=[t.name for t in new],
        source_graphs={n: g.model_dump(mode="json") for n, g in graphs.items()},
        teachers=[
            SharedTeacher(
                task=n, bundle=p.initial, learned_tasks=tuple(graphs), graph=g
            ).model_dump(mode="json")
            for n, g in graphs.items()
        ],
        sequential_updates=True,
        checkpoint_selection="min_loss",
        initialize_new_task_graphs=True,
    )
    return SharedPlan.model_validate(values)


class TopologySpy:
    """Synthetic topology response; production matching is tested separately below."""

    def __init__(self):
        self.calls = []

    async def __call__(self, p, history, directory):
        self.calls.append((p, history[-1]["bundle"].record_id, len(history)))
        graphs = {
            n: next(iter(p.source_graphs.values())).model_dump(mode="json")
            for n in p.new_tasks
        }
        for g in graphs.values():
            g["nodes"]["a"]["meaning"] += f" H{history[-1]['round']}"
        return {
            "student_bundle_id": history[-1]["bundle"].record_id,
            "graphs": graphs,
            "student_gaps": {"synthetic": history[-1]["round"]},
            "teacher_unit_bank": teacher_unit_bank(p.source_graphs),
            "current_record_ids": [
                r.record_id
                for r in history[-1]["records"]
                if r.request.instance.task in p.new_tasks
            ],
        }


def test_every_candidate_refreshes_topology_and_selected_checkpoint_owns_frozen_graphs(
    tmp_path,
):
    p = paired_plan(tmp_path)
    author, topology = Author(), TopologySpy()
    # H1 improves one of two new tasks; H2 regresses. No early all-task stopping.
    evaluator = Evaluator(
        lambda r, b: (
            r.instance.task.startswith("old")
            or (b.record_id == "compact" and r.instance.task == "new0")
        )
    )
    result = asyncio.run(
        SharedEvolution(
            tmp_path / "run", author, evaluator, task_topology=topology
        ).run(p)
    )
    assert len(author.requests) == 2 and [x[2] for x in topology.calls] == [1, 2, 3]
    assert [r["parent"]["record_id"] for r in author.requests] == ["initial", "compact"]
    assert [r["student_gaps"]["synthetic"] for r in author.requests] == [0, 1]
    assert all(
        r["topology_refresh"]["student_bundle_id"] == r["parent"]["record_id"]
        for r in author.requests
    )
    assert all(
        set(r["source_graphs"]) == {"old0", "new0", "new1"} for r in author.requests
    )
    assert result["round"] == 1 and result["last_evaluated_round"] == 2
    assert result["source_graphs"]["new0"]["nodes"]["a"]["meaning"].endswith("H1")
    assert result["source_graphs"]["old0"] == p.source_graphs["old0"].model_dump(
        mode="json"
    )
    assert (
        len(evaluator.requests) == 9
    )  # one full initial panel, two full candidates; no separate teacher rollout.
    count = len(evaluator.requests)
    assert (
        asyncio.run(
            SharedEvolution(
                tmp_path / "run", author, evaluator, task_topology=topology
            ).run(p)
        )
        == result
    )
    assert len(evaluator.requests) == count


def test_success_still_checks_old_tasks_and_initializes_new_graphs(tmp_path):
    p = paired_plan(tmp_path).model_copy(update={"loss_threshold": 0.1})
    author, topology, evaluator = Author(), TopologySpy(), Evaluator(lambda r, b: True)
    result = asyncio.run(
        SharedEvolution(
            tmp_path / "run", author, evaluator, task_topology=topology
        ).run(p)
    )
    assert result["round"] == 0 and result["termination"] == "loss_threshold_reached"
    assert not author.requests and len(topology.calls) == 1
    assert len(evaluator.requests) == 3
    assert set(result["source_graphs"]) == {"old0", "new0", "new1"}


@pytest.mark.parametrize("change", ["order", "membership", "old_stopping_rule"])
def test_resume_preserves_bound_request_and_checks_old_task_membership(
    tmp_path, change
):
    values = paired_plan(tmp_path, old_count=2).model_dump(mode="json")
    values["source_graphs"] = dict(reversed(list(values["source_graphs"].items())))
    original = SharedPlan.model_validate(values)
    root = tmp_path / "run"
    generated = []
    author = Author()

    class Interrupted(RuntimeError):
        pass

    class CachedAuthor:
        async def __call__(self, request, directory):
            path = directory / "result.json"
            if path.exists():
                return json.loads(path.read_text())
            result = await author(request, directory)
            atomic_json(path, result)
            generated.append(request["round"])
            if len(generated) == 1:
                raise Interrupted()
            return result

    class CachedTopology(TopologySpy):
        async def __call__(self, p, history, directory):
            path = directory / "result.json"
            if path.exists():
                return json.loads(path.read_text())
            result = await super().__call__(p, history, directory)
            atomic_json(path, result)
            return result

    cached, topology = CachedAuthor(), CachedTopology()
    evaluator = Evaluator(
        lambda r, b: (
            r.instance.task.startswith("old")
            or (b.record_id == "compact" and r.instance.task == "new0")
        )
    )
    with pytest.raises(Interrupted):
        asyncio.run(
            SharedEvolution(root, cached, evaluator, task_topology=topology).run(
                original
            )
        )
    path = root / "rounds/001/request.json"
    saved = json.loads(path.read_text())
    if change == "membership":
        saved["topology_refresh"]["old_tasks"] = ["old0", "foreign"]
        atomic_json(path, saved)
    elif change == "old_stopping_rule":
        saved["criteria"]["loss_threshold"] = None
        saved["criteria"]["stopping_rule"] = (
            "all_joint_development_tasks_reach_target_with_retention_or_round_limit"
        )
        atomic_json(path, saved)
    before = path.read_bytes()
    restored = SharedPlan.model_validate_json(json.dumps(values, sort_keys=True))
    run = SharedEvolution(root, cached, evaluator, task_topology=topology).run(restored)
    if change == "membership":
        with pytest.raises(ValueError):
            asyncio.run(run)
        assert generated == [1]
    else:
        result = asyncio.run(run)
        assert result["round"] == 1
        assert generated == [1, 2]
        assert len(evaluator.requests) == 12
    assert path.read_bytes() == before


def test_dynamic_query_graphs_retrieve_only_fixed_old_units_and_reach_native_tools(
    tmp_path,
):
    p = paired_plan(tmp_path)
    graphs = {n: next(iter(p.source_graphs.values())) for n in p.new_tasks}
    rows = asyncio.run(
        SharedEvolution(
            tmp_path / "data", None, Evaluator(lambda r, b: False)
        ).evaluate(p, p.initial, p.sources[1:], tmp_path / "data")
    )

    def report(node):
        projection = {
            "topology_records": {
                r.record_id: {
                    "task": r.request.instance.task,
                    "node_id": node,
                    "citations": [r.evidence[0].path],
                    "basis": "synthetic observed state",
                }
                for r in rows
            }
        }
        return build_gap_report(
            p,
            teacher_unit_bank(p.source_graphs),
            p.initial.record_id,
            rows,
            projection,
            query_graphs=graphs,
        )

    first, second = report("a"), report("b")
    assert {g["frontier_node"] for g in first["gaps"]} == {"b"}
    assert {g["frontier_node"] for g in second["gaps"]} == {"c"}
    assert {m["unit"] for ms in first["matches"].values() for m in ms} == {
        '["old0","edge0"]'
    }
    assert {m["unit"] for ms in second["matches"].values() for m in ms} == {
        '["old0","edge1"]'
    }
    from reweave.evolution.distillation.conditions import encoding_request

    request = encoding_request(p)
    request["method"] = "aligned"
    request.update(
        source_graphs={
            n: g.model_dump(mode="json")
            for n, g in {**p.source_graphs, **graphs}.items()
        },
        student_gaps=first,
        teacher_unit_bank=teacher_unit_bank(p.source_graphs),
        topology_refresh={"student_bundle_id": p.initial.record_id},
    )
    native = NativeSharedAuthor(
        bundle_loader=None,
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path,
        runtime_manifest=SimpleNamespace(
            authoring_contract=SimpleNamespace(
                operations={}, tools=[], native_image_contract=None
            ),
            configuration={"limits": {"episode": {}}},
        ),
    )
    workspace, _ = native.prepare(request, tmp_path / "author")
    from reweave.evolution.distillation.tools import SharedAuthorTools

    tools = SharedAuthorTools(workspace, tmp_path / "tools.jsonl")
    search = asyncio.run(
        tools.model_call("retrieve_graphs", {"query_task": "new0", "top_k": 5})
    )
    assert search["evaluated_tasks"] == 1
    assert {m["task"] for m in search["matches"]} == {"old0"}
    mechanism = first["mechanisms"][0]["id"]
    content = asyncio.run(tools.model_call("query_student_gap", {"gap_id": mechanism}))
    assert content["queries"][0]["task"].startswith("new")
    assert set(tools.teacher_units) == {'["old0","edge0"]', '["old0","edge1"]'}
    assert any(k.startswith('["new0"') for k in tools.query_units)
    assert (
        json.loads((workspace / "context/input.json").read_text())["topology_refresh"]
        == "context/topology_refresh.json"
    )
    assert (workspace / "candidate").is_dir() and (
        workspace / "context/teachers/000"
    ).is_dir()
    assert not (workspace / "context/target_graphs.json").exists()


class EncodingSpy:
    calls = []

    def __init__(self, **kwargs):
        pass

    async def __call__(self, request, directory):
        self.calls.append(request)
        result = {}
        for n, g in request["source_graphs"].items():
            value = SemanticGraph.model_validate(g)
            semantics = GraphSemantics(
                catalog=CATALOG,
                entities={"object": {"description": "fixture physical object"}},
                nodes={
                    k: [fact("phase", "released" if v.role == "goal" else "held")]
                    for k, v in value.nodes.items()
                },
                interfaces={
                    k: {"entry": [fact("phase", "held")] if e.source is None else []}
                    for k, e in value.edges.items()
                },
            )
            result[n] = value.model_copy(update={"matching": semantics}).model_dump(
                mode="json"
            )
        return {"graphs": result}


@pytest.mark.parametrize("interruption", [None, "encoding", "changed_scaffold"])
def test_real_topology_updater_uses_cumulative_development_and_immutable_catalog(
    tmp_path,
    interruption,
):
    p = paired_plan(tmp_path)
    fresh = task("new0")
    histories = []
    for round_index in range(2):
        folder = tmp_path / f"evidence{round_index}"
        folder.mkdir()
        student = (
            p.initial
            if round_index == 0
            else p.initial.model_copy(update={"record_id": new_record_id()})
        )
        req, _ = analysis_request(folder, bundle=student)
        row = req.evidence[0]
        if round_index == 0:
            fresh = fresh.model_copy(update={"dev": (row.request.instance,)})
            p = p.model_copy(
                update={
                    "sources": (p.sources[0], fresh),
                    "new_tasks": ("new0",),
                    "conditions": row.request.conditions,
                    "execution_id": row.request.execution,
                    "manifest_id": row.request.manifest_id,
                }
            )
        else:
            row = row.model_copy(
                update={
                    "request": row.request.model_copy(update={"instance": fresh.dev[0]})
                }
            )
        histories.append({"round": round_index, "bundle": student, "records": [row]})

    class Interrupted(Exception):
        pass

    interrupt_pending = interruption is not None

    class Encoder(EncodingSpy):
        async def __call__(self, request, directory):
            nonlocal interrupt_pending
            if directory.parent.name == "h1" and interrupt_pending:
                interrupt_pending = False
                raise Interrupted
            return await super().__call__(request, directory)

    updater = NativeTaskTopology(
        policy=AuthorPolicy(author="codex"),
        runtime_manifest=None,
        task_runtime_manifests={"new0": None},
        analyzer_factory=FakeAnalysis,
        encoder_factory=Encoder,
    )
    EncodingSpy.calls = []
    FakeAnalysis.calls = []
    histories[0]["task_topology"] = asyncio.run(
        updater(p, histories[:1], tmp_path / "h0")
    )
    if interruption is not None:
        with pytest.raises(Interrupted):
            asyncio.run(updater(p, histories, tmp_path / "h1"))
        scaffold = tmp_path / "h1/new0/execution_graph.json"
        published = scaffold.read_bytes()
        assert len(FakeAnalysis.calls) == 2
        if interruption == "changed_scaffold":
            changed = json.loads(published)
            changed["policy"]["distance_bins_m"] = [0.01, 0.02]
            atomic_json(scaffold, changed)
            with pytest.raises(ValueError, match="execution_graph.json"):
                asyncio.run(updater(p, histories, tmp_path / "h1"))
            assert len(FakeAnalysis.calls) == 2
            return
    result = asyncio.run(updater(p, histories, tmp_path / "h1"))
    if interruption is not None:
        assert scaffold.read_bytes() == published
    assert len(FakeAnalysis.calls) == 2
    request = FakeAnalysis.calls[-1][1]
    assert len(request.evidence) == 2 and len(request.versions) == 2
    assert request.previous_analysis is not None
    assert all(r.request.split == "dev" for r in request.evidence)
    assert result["fixed_graphs"] == {
        n: g.model_dump(mode="json") for n, g in p.source_graphs.items()
    }
    assert result["current_record_ids"] == [histories[-1]["records"][0].record_id]
    assert set(result["graphs"]["new0"]["episodes"]) == {
        h["records"][0].record_id for h in histories
    }
    assert EncodingSpy.calls[0]["fixed_catalog"]  # H0 must already use old catalog.
    assert asyncio.run(updater(p, histories, tmp_path / "h1")) == result
    stale = [*histories[:-1], {**histories[-1], "bundle": p.initial}]
    with pytest.raises(ValueError, match="cached topology"):
        asyncio.run(updater(p, stale, tmp_path / "h1"))


def test_four_joint_stages_have_declared_panels_and_one_parent(tmp_path):
    p = paired_plan(tmp_path, old_count=8)
    _, h0, runner, _ = recipe_fixture(tmp_path / "runtime")
    sources = tuple(task(f"old{i}") for i in range(8))
    old_graphs = p.source_graphs
    p = p.model_copy(
        update={
            "mode": "consolidation",
            "new_tasks": (),
            "sources": sources,
            "initial": h0,
            "size_reference": h0,
            "conditions": runner.runtime_manifest.conditions,
            "test_conditions": runner.runtime_manifest.conditions,
            "execution_id": runner.runtime_manifest.execution_id,
            "manifest_id": "fixture-manifest",
            "max_rounds": 1,
            "loss_threshold": 0.1,
        }
    )
    initial = tmp_path / "initial.json"
    atomic_json(
        initial,
        {
            "bundle": h0.model_dump(mode="json"),
            "source_graphs": {
                n: g.model_dump(mode="json") for n, g in old_graphs.items()
            },
        },
    )
    targets = tuple(task(f"new{i}") for i in range(8))
    stages = [[t.name for t in targets[i : i + 2]] for i in range(0, 8, 2)]

    class JointAuthor(Author):
        async def __call__(self, request, directory):
            self.requests.append(request)
            value = request["parent"].copy()
            value["record_id"] = new_record_id()
            value["context"] = {
                **value["context"],
                "instructions": value["context"]["instructions"] + " joint",
            }
            return {"bundle": value}

    author, topology = JointAuthor(), TopologySpy()

    def success(r, b):
        return not (r.instance.task.startswith("new") and b.record_id == h0.record_id)

    evaluator = Evaluator(success)
    checkpoints = asyncio.run(
        run_stream(
            tmp_path / "stream",
            p,
            (*sources, *targets),
            stages,
            author,
            evaluator,
            task_topology=topology,
            initial_checkpoint=initial,
            task_runtime_manifests={
                t.name: runner.runtime_manifest for t in (*sources, *targets)
            },
        )
    )
    assert [
        sum(len(t.dev) for t in call[0].sources)
        for call in topology.calls
        if call[2] == 1
    ] == [14, 16, 18, 20]
    assert [c["processed_targets"] for c in checkpoints] == [2, 4, 6, 8]
    assert (
        len(author.requests) == 1
    )  # later pairs already pass, no forced extra updates.
    for stage in range(1, 5):
        value = json.loads(
            (tmp_path / f"stream/updates/{stage:03}/plan.json").read_text()
        )
        assert {x["bundle"]["record_id"] for x in value["teachers"]} == {
            value["initial"]["record_id"]
        }
        for t in value["sources"]:
            assert [i["initialization"]["seed"] for i in t["dev"]] == (
                [11, 12, 13] if t["name"] in value["new_tasks"] else [0]
            )
    # Original8 seed0 measured once per distinct shared version. Later identical versions reuse originals.
    for name in ("old0", "old7"):
        assert len([r for r in evaluator.requests if r.instance.task == name]) == 2


@pytest.mark.parametrize("bad_definition", [False, True])
def test_condition_encoder_preserves_fixed_catalog_definitions(
    tmp_path, bad_definition
):
    from reweave.evolution.distillation.conditions import encoding_request
    from reweave.evolution.distillation.task_topology import NativeQueryConditions

    p = paired_plan(tmp_path)
    graph = next(iter(p.source_graphs.values()))
    request = encoding_request(p)
    request.update(
        mode="continual",
        new_tasks=["new0"],
        tasks=[{"name": "new0", "family": "new", "dev": []}],
        teachers=[
            {
                "task": "new0",
                "bundle": p.initial.model_dump(mode="json"),
                "learned_tasks": [],
                "graph": graph.model_dump(mode="json"),
                "evidence": [],
            }
        ],
        source_graphs={"new0": graph.model_dump(mode="json")},
        fixed_catalog={
            k: v.model_dump(mode="json") for k, v in graph.matching.catalog.items()
        },
    )
    encoder = NativeQueryConditions(
        bundle_loader=None,
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path,
        runtime_manifest=SimpleNamespace(
            authoring_contract=SimpleNamespace(
                operations={}, tools=[], native_image_contract=None
            ),
            configuration={"limits": {"episode": {}}},
        ),
    )
    workspace, _ = encoder.prepare(request, tmp_path / "native")
    conditions = graph.matching.model_dump(mode="json")
    if bad_definition:
        conditions["catalog"]["phase"]["definition"] = "A different physical coordinate"
    atomic_json(workspace / "candidate/conditions.json", {"new0": conditions})
    if bad_definition:
        with pytest.raises(ValueError, match="(conflict|incompatible)"):
            asyncio.run(encoder.admit(request, workspace, tmp_path / "audit"))
    else:
        result = asyncio.run(encoder.admit(request, workspace, tmp_path / "audit"))
        assert (
            result["graphs"]["new0"]["matching"]["catalog"] == request["fixed_catalog"]
        )
        assert (
            result["graphs"]["new0"]["nodes"] == graph.model_dump(mode="json")["nodes"]
        )


def test_continual_rejects_disabled_dynamic_matching_before_any_work(tmp_path):
    p = paired_plan(tmp_path)
    for change in (
        {"initialize_new_task_graphs": False},
        {"method": "direct"},
        {"source_graphs": {}},
    ):
        with pytest.raises(ValueError, match="(dynamic|fixed topology)"):
            SharedPlan.model_validate({**p.model_dump(mode="json"), **change})
