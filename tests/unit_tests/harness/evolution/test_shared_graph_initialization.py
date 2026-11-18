"""Initial topology uses one real receipt contract, even after an H0 failure."""

import asyncio
from pathlib import Path

import pytest

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.contracts import (
    EvolutionIncomplete,
    GraphAnalysisReceipt,
    GraphAnalysisResult,
    GraphSnapshot,
)
from reweave.evolution.distillation.engine import SharedTask
from reweave.evolution.distillation.graph_initialization import (
    NativeTaskGraphInitializer,
    prepare_initial_graph_request,
)
from tests.unit_tests.harness.evolution.fixtures import artifact, baseline, usage
from tests.unit_tests.harness.evolution.graph_fixtures import semantic_fixture
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request
from tests.unit_tests.harness.evolution.test_shared import plan


def cold_fixture(tmp_path):
    h0 = baseline()
    (tmp_path / "original").mkdir()
    original, _ = analysis_request(tmp_path / "original", bundle=h0)
    record = original.evidence[0]
    task = SharedTask(name="new", family="new", dev=(record.request.instance,))
    p = plan().model_copy(
        update={
            "mode": "continual",
            "sources": (task,),
            "unseen": (),
            "initial": h0,
            "new_tasks": ("new",),
            "teachers": (),
            "conditions": record.request.conditions,
            "execution_id": record.request.execution,
            "manifest_id": record.request.manifest_id,
        }
    )
    return p, task, h0, record


class FakeAnalysis:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __call__(self, request, directory, *, before_launch):
        before_launch()
        self.calls.append(("start", request))
        graph = GraphSnapshot.model_validate_json(Path(request.graph.path).read_text())
        return GraphAnalysisReceipt(
            record_id="graph-receipt",
            request_id=request.record_id,
            analysis=GraphAnalysisResult(
                graph=semantic_fixture(graph),
                changes="Synthetic initial graph",
                forecast={
                    "outcomes": [{"progress": 0.0, "probability": 1.0}],
                    "confidence": 0.0,
                    "rationale": "Unused synthetic forecast",
                },
            ),
            session_id="original-native-session",
            usage=usage(),
            artifacts=(artifact(directory, "fixture.json", {"synthetic": True}),),
        )

    async def recover(self, request, directory, *, before_launch):
        result = await self(request, directory, before_launch=before_launch)
        self.calls[-1] = ("recover", request)
        return result


def test_failed_episode_initializes_and_does_not_invent_success(tmp_path):
    p, task, h0, record = cold_fixture(tmp_path)
    assert not record.official_success
    initializer = NativeTaskGraphInitializer(
        policy=AuthorPolicy(), runtime_manifest=None, analyzer_factory=FakeAnalysis
    )
    FakeAnalysis.calls = []
    result = asyncio.run(initializer(p, task, h0, [record], tmp_path / "job"))
    assert not result["official_success"]
    assert result["h0_bundle_id"] == h0.record_id
    assert set(result["graph"]["episodes"]) == {record.record_id}
    assert not result["matching_ready"]  # Semantic topology does not imply encoding.
    assert not any(
        span["node_id"] == "complete"
        for spans in result["graph"]["episodes"].values()
        for span in spans
    )
    request = FakeAnalysis.calls[0][1]
    assert request.round_index == 0 and request.previous_analysis is None
    assert request.protocol.dev == (task.dev[0],)
    assert request.versions[0].bundle_id == h0.record_id
    assert asyncio.run(initializer(p, task, h0, [record], tmp_path / "job")) == result
    assert len(FakeAnalysis.calls) == 1


def test_initialization_excludes_test_and_foreign_harness_and_unknown(tmp_path):
    p, task, h0, record = cold_fixture(tmp_path)
    policy = AuthorPolicy()
    for field, value in [("bundle_id", "old-shared"), ("split", "test")]:
        bad = record.model_copy(
            update={"request": record.request.model_copy(update={field: value})}
        )
        with pytest.raises(ValueError, match="declared H0/task/dev"):
            prepare_initial_graph_request(p, task, h0, [bad], tmp_path / field, policy)
    unknown = record.model_copy(update={"failure_origin": "unknown"})
    with pytest.raises(EvolutionIncomplete, match="settled physical evidence"):
        prepare_initial_graph_request(
            p, task, h0, [unknown], tmp_path / "unknown", policy
        )
    with pytest.raises(ValueError, match="one H0"):
        prepare_initial_graph_request(
            p, task, h0, [record, record], tmp_path / "duplicate", policy
        )


def test_started_graph_analysis_recovers_same_request(tmp_path):
    p, task, h0, record = cold_fixture(tmp_path)
    directory = tmp_path / "job"
    request = prepare_initial_graph_request(
        p, task, h0, [record], directory, AuthorPolicy()
    )
    artifact(directory / "analysis", "started.json", {"request_id": request.record_id})
    FakeAnalysis.calls = []
    initializer = NativeTaskGraphInitializer(
        policy=AuthorPolicy(), runtime_manifest=None, analyzer_factory=FakeAnalysis
    )
    asyncio.run(initializer(p, task, h0, [record], directory))
    assert FakeAnalysis.calls[0][0] == "recover"
    assert FakeAnalysis.calls[0][1] == request


def test_existing_native_graph_workspace_accepts_cold_start_inputs(tmp_path):
    from reweave.evolution.graph.analysis import NativeGraphAnalysis

    p, task, h0, record = cold_fixture(tmp_path)
    request = prepare_initial_graph_request(
        p, task, h0, [record], tmp_path / "job", AuthorPolicy()
    )
    analyzer = NativeGraphAnalysis(
        bundle_loader=lambda key: h0,
        policy=AuthorPolicy(),
        evidence_workspace=tmp_path / "evidence",
    )
    workspace = tmp_path / "native/workspace"
    analyzer._prepare(request, workspace)
    assert (workspace / "context/execution_graph.json").exists()
    text = (workspace / "context/task_instructions.json").read_text()
    assert "Open the drawer" in text
    assert "unseen" not in text
