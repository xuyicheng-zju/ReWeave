"""One executable package split by ownership, plus success-only publication."""

import asyncio
import json

import pytest

from reweave.evolution.authoring.candidate import (
    capture_bundle,
    host_input,
    read_candidate,
    read_proposal,
    write_bundle_source,
)
from reweave.evolution.authoring.workspace import native_output
from reweave.evolution.bundle import PythonFile, PythonProgram
from reweave.evolution.contracts import Proposal
from reweave.evolution.validation.admission import admit_mutation
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import baseline, globalized, workflow
from tests.unit_tests.harness.evolution.test_bundle_author import workspace


def test_three_scope_source_roundtrip_uses_global_import_once(tmp_path):
    initial = globalized(baseline())
    source = initial.context.program.files[0]
    shared = source.model_copy(update={"path": "shared.py"})
    program = PythonProgram(
        entrypoint="binding:build",
        files=(
            PythonFile(
                path="binding.py",
                source="from shared import build as shared_build\ndef build(runtime, context):\n    return shared_build(runtime, context)\n",
            ),
            shared,
        ),
    )
    initial = initial.model_copy(
        update={
            "context": initial.context.model_copy(update={"program": program}),
            "workflows": (workflow().model_copy(update={"layer": "global"}),),
        }
    )
    write_bundle_source(initial, tmp_path)
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {
        "global",
        "task-family",
        "task-specific",
    }
    captured = capture_bundle(tmp_path, initial.runtime, record_id=initial.record_id)
    assert captured.bundle == initial
    assert (tmp_path / "global/context/src/shared.py").is_file()
    assert not (tmp_path / "task-family/context/src/shared.py").exists()
    assert (tmp_path / "task-family/context/src/binding.py").is_file()


def test_trials_reject_changes_to_shared_implementation(tmp_path):
    initial = globalized(baseline())
    changed = initial.model_copy(update={"loop": baseline().loop})
    with pytest.raises(ValueError, match="only task-family"):
        changed.assert_task_family_edit(initial)
    root = workspace(tmp_path)
    proposal = read_proposal(root)
    write_bundle_source(initial, root / "candidate")
    with pytest.raises(ValueError, match="only task-family"):
        read_candidate(root, require_plan=False)
    assert proposal.purpose == "mutation"


def test_success_consolidation_admits_three_layers_and_protects_host_facts(tmp_path):
    root = workspace(tmp_path)
    original = read_proposal(root)
    record = original.evidence[0].model_copy(
        update={"official_success": True, "failure_origin": "none"}
    )
    values = original.model_dump(mode="json")
    values.update(
        purpose="success_consolidation",
        graph=None,
        decision=None,
        execution_graph=None,
        progress_map=None,
        evidence=[record.model_dump(mode="json")],
    )
    proposal = Proposal.model_validate(values)
    atomic_json(host_input(root, "proposal.json"), proposal.model_dump(mode="json"))
    candidate = globalized(baseline())
    write_bundle_source(candidate, root / "candidate")
    atomic_json(
        root / "candidate/task-specific/evidence.json", [record.model_dump(mode="json")]
    )
    atomic_json(
        root / "candidate/plan.json",
        {
            "diagnosis": "Source factoring of exact success.",
            "summary": "Share C/L.",
            "validation": "Synthetic interfaces only.",
            "scopes": ["C", "L"],
            "cross_scope_reason": "Both programs have shared ownership.",
        },
    )
    from reweave.evolution.authoring.tools import AuthorDevelopmentTools

    input_path = root / "context/input.json"
    prepared = json.loads(input_path.read_text())
    prepared["kind"] = "success_consolidation"
    atomic_json(input_path, prepared)
    assert native_output(root) == root / "candidate"
    tools = AuthorDevelopmentTools(root, tmp_path / "audit/development.jsonl")
    checked = asyncio.run(
        tools._call("validate_candidate", {"phase": "code"}, case_results=[])
    )
    assert checked["structurally_valid"] and not checked["physical_execution_tested"]
    admitted = asyncio.run(admit_mutation(root))[0]
    admitted.bundle.require_success_layers()
    assert admitted.bundle.record_id != record.request.bundle_id
    atomic_json(root / "candidate/task-specific/evidence.json", [])
    with pytest.raises(ValueError, match="Host record"):
        read_candidate(root)


def test_success_consolidation_requires_exact_success_and_graphless_request(tmp_path):
    root = workspace(tmp_path)
    value = read_proposal(root).model_dump(mode="json")
    value.update(
        purpose="success_consolidation",
        graph=None,
        decision=None,
        execution_graph=None,
        progress_map=None,
    )
    with pytest.raises(ValueError, match="exact-parent evaluator success"):
        Proposal.model_validate(value)


def test_online_revisions_preserve_global_and_host_fact_files(tmp_path):
    from reweave.runtime.adaptation import HarnessAdaptation

    initial = globalized(baseline())
    manager = HarnessAdaptation(
        initial,
        candidate_directory=tmp_path / "candidate",
        audit_directory=tmp_path / "audit",
    )
    source = tmp_path / "candidate/global/context/src/context.py"
    before = manager.active
    source.write_text(source.read_text() + "\n# forbidden shared edit\n")
    with pytest.raises(ValueError, match="only task-family"):
        manager.stage(reason="should reject global write")
    assert manager.active is before and manager.pending is None
    write_bundle_source(initial, tmp_path / "candidate")
    (tmp_path / "candidate/task-specific/evidence.json").write_text("[]")
    with pytest.raises(ValueError, match="Host task-specific files"):
        manager.stage(reason="should reject invented facts")
    assert manager.active is before


def test_success_consolidation_resumes_without_repeating_evaluation(tmp_path):
    from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
    from tests.unit_tests.harness.evolution.fixtures import (
        ConsolidatingAuthor,
        protocol,
    )
    from tests.unit_tests.harness.evolution.test_graph_evolution import record

    config = protocol().model_copy(update={"max_rounds": 0})
    archive = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    evaluations, requests = [], []

    async def evaluate(request, bundle, directory, *, before_launch):
        before_launch()
        evaluations.append(request)
        return record(request, directory, True)

    class Interrupted(ConsolidatingAuthor):
        async def __call__(self, proposal, directory, *, before_launch):
            before_launch()
            requests.append(proposal)
            # Represents the existing native adapter's retained same-session state.
            atomic_json(directory / "session.json", {"session_id": "retained-session"})
            raise RuntimeError("settled author interruption")

        async def recover(self, proposal, directory, *, before_launch):
            assert (
                json.loads((directory / "session.json").read_text())["session_id"]
                == "retained-session"
            )
            requests.append(proposal)
            return await super().__call__(
                proposal, directory, before_launch=before_launch
            )

    author = Interrupted(archive, None)
    with pytest.raises(RuntimeError, match="settled author interruption"):
        asyncio.run(GraphEvolution(archive, author, evaluate).run())
    assert not json.loads(archive.state_path.read_text())["completed"]
    state = asyncio.run(GraphEvolution(archive, author, evaluate, recover=True).run())
    assert requests[0] == requests[1] and len(evaluations) == 1
    assert len(state["versions"]) == 1 and state["stop_reason"] == "target_reached"
    publication = json.loads((tmp_path / "publications/V000.json").read_text())
    assert publication["winning_bundle_id"] == baseline().record_id
    assert publication["organized_bundle_id"] != baseline().record_id
    assert publication["official_success"] is None
    assert asyncio.run(GraphEvolution(archive, author, evaluate).run()) == state
    assert len(requests) == 2 and len(evaluations) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["success_publications"]["V000"] == publication
    assert report["total_usage_including_failed_work"]["physical_steps"] == 10


def test_native_success_workspace_uses_real_episode_evidence_without_graph_search(
    tmp_path,
):
    from reweave.evolution.authoring.author import AuthorPolicy, NativeBundleAuthor
    from reweave.evolution.contracts import (
        EvaluationRecord,
        EvaluationRequest,
        EvidenceFile,
        VersionReference,
    )
    from reweave.evolution.evidence.records import inspect_episode
    from tests.unit_tests.harness.evolution.fixtures import (
        evolution_history,
        protocol,
        usage,
    )
    from tests.unit_tests.harness.evolution.test_trace_context import _episode

    evidence_root = tmp_path / "physical"
    evidence_root.mkdir()
    _, original = _episode(evidence_root, task_languages=["Open the drawer"] * 4)
    parent = baseline()
    atomic_json(
        original / "harness_context/bundle_contract.json",
        {
            "bundle_id": parent.record_id,
            "operations": {},
            "episode_limits": {"wall_time_s": 30},
        },
    )
    atomic_json(
        original / "harness_context/planner_context.json",
        {"system_prompt": "Task", "tools": []},
    )
    atomic_json(original / "harness_context/capability_catalog.json", {})
    source = original / "harness_context/source/reweave/benchmarks/executed.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Actual source fixture, not a physical experiment.\n")
    evidence = inspect_episode(original)
    config = protocol()
    request = EvaluationRequest(
        record_id="observed",
        conditions=config.execution,
        bundle_id=parent.record_id,
        instance=config.dev[0],
        split="dev",
        manifest_id=config.manifest_id,
        execution=config.execution_id,
    )
    result = EvaluationRecord(
        record_id="won",
        request=request,
        official_success=True,
        failure_origin="none",
        infrastructure_retryable=None,
        usage=usage(),
        episode=evidence,
        evidence=tuple(
            EvidenceFile(path=str(original / name), **metadata)
            for name, metadata in evidence.files.items()
        ),
        initialization=request.instance.initialization,
    )
    reference = VersionReference(version_id="baseline", bundle_id=parent.record_id)
    proposal = Proposal(
        record_id="consolidate",
        purpose="success_consolidation",
        parent=reference,
        references=(reference,),
        scopes=("W", "C", "L"),
        evidence=(result,),
        history=evolution_history(tmp_path / "history"),
        task_protocol=config,
        round_index=0,
        author_index=0,
    )
    author = NativeBundleAuthor(
        evidence_workspace=tmp_path / "evidence",
        bundle_loader=lambda _: parent,
        policy=AuthorPolicy(),
    )
    directory = tmp_path / "author"
    author._prepare_inputs(proposal, directory, resume=False)
    root = directory / "workspace"
    prepared = json.loads((root / "context/input.json").read_text())
    assert prepared["kind"] == "success_consolidation"
    assert prepared["search_review"] is None
    assert not (root / "context/search_review.json").exists()
    assert not (root / "context/graph.json").exists()
    assert json.loads((root / "candidate/task-specific/evidence.json").read_text()) == [
        result.model_dump(mode="json")
    ]
    assert {p.name for p in (root / "candidate").iterdir() if p.is_dir()} == {
        "global",
        "task-family",
        "task-specific",
    }
    author._prepare_inputs(proposal, directory, resume=True)
