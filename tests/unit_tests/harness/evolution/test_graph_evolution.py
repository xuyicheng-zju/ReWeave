# Modified for the ReWeave namespace.
"""Global coverage selection and real controller transitions with offline adapters."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from reweave.evolution.bundle import AgentBundle, Scope, new_record_id
from reweave.evolution.contracts import (
    AuthoredMutation,
    BudgetRevision,
    BundleVersion,
    Candidate,
    EvaluationRecord,
    EvolutionIncomplete,
    EvolutionProtocol,
    EvolutionState,
    ExecutionRevision,
    Proposal,
)
from reweave.evolution.graph.projection import map_row
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    mutated,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.graph_fixtures import physical_trace


def graph_search(workspace, author, evaluator, **kwargs):
    from tests.unit_tests.harness.evolution.fixtures import ConsolidatingAuthor
    from tests.unit_tests.harness.evolution.graph_fixtures import FixtureGraphAnalysis

    analyzer = kwargs.pop("analyzer", None) or FixtureGraphAnalysis()
    author = ConsolidatingAuthor(workspace, author)
    return GraphEvolution(workspace, author, evaluator, analyzer=analyzer, **kwargs)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def width_schedule(first, later):
    """A controlled width for the first decision, then for every later one."""
    calls = []

    def schedule(*args):
        calls.append(None)
        width = first if len(calls) == 1 else later
        return width, {"fixture_schedule": 1.0}, ({"width": float(width)},)

    return schedule


@pytest.fixture
def two_then_one_authors(monkeypatch):
    """Isolate lifecycle assertions from width allocation: two authors, then one."""
    monkeypatch.setattr(
        "reweave.evolution.graph.selection.compute_width", width_schedule(2, 1)
    )


def test_context_contract_rejects_removed_component_forms():
    payload = baseline().model_dump(mode="json")
    assert payload["schema_version"] == 3
    assert baseline().novel_scopes((baseline(),)) == frozenset()
    assert "component_hashes" not in type(baseline()).__dict__
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentBundle.model_validate({**payload, "harness": payload["context"]})
    with pytest.raises(ValidationError, match="literal_error"):
        TypeAdapter(Scope).validate_python("H")


@pytest.mark.parametrize("removed", ["authors_per_round", "max_hypotheses"])
def test_protocol_requires_maximum_branch_width_not_removed_fields(removed):
    value = protocol().model_dump(mode="json")
    value[removed] = value.pop("max_authors_per_round")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvolutionProtocol.model_validate(value)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_protocol_rejects_invalid_author_count(count):
    value = protocol().model_dump(mode="json")
    value["max_authors_per_round"] = count
    with pytest.raises(ValidationError):
        EvolutionProtocol.model_validate(value)


@pytest.mark.parametrize("seconds", [0, -1, True, float("inf")])
def test_protocol_rejects_invalid_author_deadline(seconds):
    value = protocol().model_dump(mode="json")
    value["author_wall_time_s"] = seconds
    with pytest.raises(ValidationError):
        EvolutionProtocol.model_validate(value)


@pytest.mark.parametrize(
    "change", ["same", "decrease", "author", "execution", "conditions"]
)
def test_graph_budget_revision_rejects_other_protocol_changes(change):
    previous = protocol()
    value = previous.model_dump(mode="json")
    value["graph"]["analysis_wall_time_s"] *= 2
    if change == "same":
        value["graph"]["analysis_wall_time_s"] /= 2
    elif change == "decrease":
        value["graph"]["analysis_wall_time_s"] /= 4
    elif change == "author":
        value["author_wall_time_s"] += 1
    elif change == "execution":
        value["execution"]["wall_time_s"] += 1
    else:
        value["dev"][0]["initialization"]["seed"] = 999
    with pytest.raises(
        ValidationError,
        match="budget revision may only",
    ):
        BudgetRevision(
            previous=previous, current=EvolutionProtocol.model_validate(value)
        )


@pytest.mark.anyio
async def test_explicit_graph_budget_revision_retains_requests_work_and_settled_h0(
    tmp_path,
):
    from reweave.evolution.search.engine import AuthorWallTimeExceeded
    from tests.unit_tests.harness.evolution.graph_fixtures import FixtureGraphAnalysis

    old_protocol = configured(1).model_copy(update={"max_authors_per_round": 1})
    old_protocol = old_protocol.model_copy(
        update={
            "graph": old_protocol.graph.model_copy(
                update={"analysis_wall_time_s": 0.05}
            )
        }
    )
    old = tmp_path / "original"
    workspace = EvolutionWorkspace(old, baseline(), old_protocol)
    old.mkdir()
    (old / "configuration.json").write_text("{}")
    evaluations, authors, requests = [], [], []

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        evaluations.append(request)
        return record(request, directory, False)

    async def author(proposal, directory, *, before_launch):
        before_launch()
        authors.append(proposal)
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic budget continuation; no physical efficacy claim.",
                    branch_id="repair",
                    bundle=child("a"),
                    summary="Synthetic candidate",
                ),
            ),
            summary="Synthetic independent author.",
            session_id="fixture-author",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    class InterruptedGraph(FixtureGraphAnalysis):
        async def __call__(self, request, directory, *, before_launch):
            if request.purpose == "final_selection":
                return await super().__call__(
                    request, directory, before_launch=before_launch
                )
            before_launch()
            requests.append(request)
            await asyncio.sleep(60)

        async def recover(self, request, directory, *, before_launch):
            requests.append(request)
            return await super().__call__(
                request, directory, before_launch=before_launch
            )

    analyzer = InterruptedGraph()
    with pytest.raises(AuthorWallTimeExceeded):
        await graph_search(
            workspace, verified_author(author), evaluator, analyzer=analyzer
        ).run()
    original = workspace.state_path.read_bytes()
    new_protocol = old_protocol.model_copy(
        update={
            "record_id": "protocol:extended-graph",
            "graph": old_protocol.graph.model_copy(
                update={"analysis_wall_time_s": 5.0}
            ),
        }
    )
    new = tmp_path / "continuation"
    shutil.copytree(old, new)
    state = json.loads(original)
    state["protocol"] = new_protocol.record_id
    with pytest.raises(ValidationError, match="frozen task protocol"):
        EvolutionState.model_validate(state)
    revision = BudgetRevision(previous=old_protocol, current=new_protocol)
    state["budget_revisions"] = [revision.model_dump(mode="json")]
    invalid = {**state, "protocol": old_protocol.record_id}
    with pytest.raises(ValidationError, match="revision chain"):
        EvolutionState.model_validate(invalid)
    for key, attempt in state["attempts"].items():
        if attempt["status"] == "completed":
            attempt["result_directory"] = str(old / "attempts" / key)
    key = "round-01/graph-analysis"
    state["attempts"][key]["status"] = "failed"
    revised = EvolutionWorkspace(new, baseline(), new_protocol)
    revised.save(state)
    completed = await graph_search(
        revised, verified_author(author), evaluator, analyzer=analyzer, recover=True
    ).run()
    assert completed["completed"] and completed["stop_reason"] == "round_limit"
    assert original == workspace.state_path.read_bytes()
    assert requests[0] == requests[1]
    assert len(evaluations) == 6 and len(authors) == 1
    assert authors[0].task_protocol == old_protocol
    recovery = next((new / "audit/attempt_recoveries" / key).glob("*.json"))
    debit = json.loads(recovery.read_text())
    assert debit["prior_work_s"] > 0
    assert debit["remaining_work_s"] == pytest.approx(5.0 - debit["prior_work_s"])


@pytest.mark.anyio
async def test_prelaunch_rejection_does_not_charge_phantom_rollout(tmp_path):
    graph = EvolutionWorkspace(tmp_path, baseline(), protocol())
    (tmp_path / "configuration.json").write_text("{}")

    async def evaluator(request, bundle, directory, *, before_launch):
        raise ValueError("invalid fixed recipe before process creation")

    search = graph_search(graph, None, evaluator)
    with pytest.raises(ValueError, match="invalid fixed recipe"):
        await search.run()
    state = json.loads(graph.state_path.read_text())
    assert search.accounting(state)["episode_launches"] == 0
    attempt = next(iter(state["attempts"].values()))
    assert attempt["status"] == "failed"
    attempt["key"] = "tampered"
    graph.state_path.write_text(json.dumps(state))
    with pytest.raises(ValidationError, match="exact request identity"):
        with graph.transaction():
            pass


@pytest.mark.anyio
async def test_missing_configuration_fails_before_any_launch(tmp_path):
    graph = EvolutionWorkspace(tmp_path, baseline(), protocol())

    async def forbidden(*args, **kwargs):
        pytest.fail("missing configuration must fail before an adapter runs")

    with pytest.raises(ValueError, match="frozen regular configuration"):
        await graph_search(graph, forbidden, forbidden).run()
    assert not graph.state_path.exists()


@pytest.mark.anyio
async def test_author_receipt_for_different_proposal_is_rejected(tmp_path):
    workspace = EvolutionWorkspace(tmp_path, baseline(), configured(6))
    (tmp_path / "configuration.json").write_text("{}")
    requests = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id="b" * 64,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id="a",
                    bundle=child("a"),
                    summary="a",
                ),
            ),
            summary="Wrong request binding.",
            session_id="wrong-proposal",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        requests.append(request)
        return record(request, directory, False)

    with pytest.raises(ValueError, match="different proposal"):
        await graph_search(workspace, verified_author(author), evaluator).run()
    assert len(requests) == 3


def configured(max_rounds=2):
    value = protocol().model_dump(mode="json")
    value["dev"] = [
        {
            "record_id": f"condition:{seed}",
            "task": "drawer",
            "initialization": {"seed": seed},
        }
        for seed in (1, 2, 3)
    ]
    value["max_rounds"] = max_rounds
    value["max_authors_per_round"] = 2
    return EvolutionProtocol.model_validate(value)


@pytest.mark.anyio
async def test_independent_sessions_share_evidence_without_assigned_diagnosis(
    tmp_path, two_then_one_authors
):
    config = configured(1).model_copy(update={"max_authors_per_round": 3})
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    mutations = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        mutations.append(proposal)
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id="repair",
                    bundle=child("a"),
                    summary="repair",
                ),
            ),
            summary="Independently diagnosed and implemented the repair.",
            session_id="mutation-session",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    adapter = verified_author(author)

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, False)

    state = await graph_search(workspace, adapter, evaluator).run()
    assert len(mutations) == 2
    assert all(
        "hypothesis" not in p.model_dump() and "frontier" not in p.model_dump()
        for p in mutations
    )
    assert len({p.history.summary.artifact_id for p in mutations}) == 1
    assert len(state["proposals"]) == 2
    assert all(a["kind"] != "diagnosis" for a in state["attempts"].values())
    assert "round-01/author-01" in state["attempts"]
    assert "evaluation/V000/condition-001/attempt-01" in state["attempts"]
    search_accounting = graph_search(workspace, adapter, evaluator).accounting(state)
    assert search_accounting["author_launches"] == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["semantic_projection"]["pending_episode_ids"]
    assert (
        report["semantic_projection"]["graph"]
        == state["final_projections"]["0"]["graph"]
    )
    assert report["semantic_projection"]["graph"] != mutations[0].graph.model_dump(
        mode="json"
    )
    assert len(state["rounds"]) == 1


def record(request, directory, success):
    return EvaluationRecord(
        record_id=new_record_id(),
        request=request,
        official_success=success,
        failure_origin="none" if success else "agent",
        infrastructure_retryable=None,
        usage=usage(),
        evidence=(
            physical_trace(directory),
            artifact(
                directory,
                "trace.json",
                {"seed": request.instance.initialization["seed"], "success": success},
            ),
        ),
        episode=None,
        initialization=request.instance.initialization,
    )


def child(name):
    value = baseline().model_dump(mode="json")
    value["record_id"] = f"fixture-child:{name}"
    value["context"]["instructions"] = name
    return AgentBundle.model_validate(value)


@pytest.mark.anyio
@pytest.mark.parametrize("baseline_success", [False, True])
async def test_recovery_evolves_partial_h0_without_replaying_it(
    tmp_path, baseline_success
):
    values = configured(1).model_dump(mode="json")
    values.update(
        dev=[
            {
                "record_id": f"condition:{i}",
                "task": "drawer",
                "initialization": {"seed": i},
            }
            for i in range(1, 6)
        ],
        validation={
            "conditions": [
                {
                    "record_id": f"condition:{i}",
                    "task": "drawer",
                    "initialization": {"seed": i},
                }
                for i in range(6, 11)
            ],
            "target_success_rate": 0.8,
        },
        max_authors_per_round=1,
    )
    workspace = EvolutionWorkspace(
        tmp_path, baseline(), EvolutionProtocol.model_validate(values)
    )
    (tmp_path / "configuration.json").write_text("{}")
    calls, proposals = [], []

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        calls.append(request)
        if bundle.record_id != baseline().record_id:
            return record(request, directory, True)
        result = record(request, directory, baseline_success)
        if request.instance.initialization["seed"] == 4:
            result = result.model_copy(
                update={
                    "failure_origin": "infrastructure",
                    "infrastructure_retryable": False,
                    "official_success": False,
                }
            )
        return result

    class InterruptedAfterH0(RuntimeError):
        pass

    initial = graph_search(workspace, None, evaluator)

    async def stop_before_analysis(*args):
        raise InterruptedAfterH0

    initial._proposals = stop_before_analysis
    with pytest.raises(InterruptedAfterH0):
        await initial.run()
    originals = {
        p: p.read_bytes() for p in (tmp_path / "attempts").rglob("result.json")
    }
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["matrix"][baseline().record_id] == [
        baseline_success,
        baseline_success,
        baseline_success,
        None,
        baseline_success,
    ]
    assert all(s["success_rate"] is None for s in progress["scores"].values())
    assert len(calls) == 5

    async def author(proposal, directory, *, before_launch):
        before_launch()
        proposals.append(proposal)
        progress_map = json.loads(Path(proposal.progress_map.path).read_text())
        row = map_row(progress_map, proposal.parent.version_id)
        assert len(row) == 5
        missing = [c for c in row.values() if c["receipt_id"] is None]
        assert len(missing) == 1
        assert missing[0]["progress"] is None and missing[0]["official_success"] is None
        assert missing[0]["usage"] is None and missing[0]["state_spans"] == []
        history = Path(proposal.history.summary.path).read_text()
        assert "| C000 | C001 | C002 | C003 | C004 |" in history
        outcome = "success" if baseline_success else "failure"
        assert f"| {outcome} | {outcome} | {outcome} | unknown | {outcome} |" in history
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic recovery check, no physical claim",
                    branch_id="repair",
                    bundle=child("whole harness repair"),
                    summary="Synthetic change",
                ),
            ),
            summary="Synthetic whole-harness mutation",
            session_id="mutation-session",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    state = await graph_search(
        workspace, verified_author(author), evaluator, recover=True
    ).run()
    assert state["stop_reason"] == "target_reached"
    assert len(calls) == 15 and len(proposals) == 1
    assert sum(r.bundle_id == baseline().record_id for r in calls) == 5
    assert sum(r.split == "validation" for r in calls) == 5
    assert all(p.read_bytes() == content for p, content in originals.items())
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["matrix"][baseline().record_id][3] is None
    assert report["total_usage_including_failed_work"]["episode_launches"] == 15
    assert report["total_usage_including_failed_work"]["cost_usd"] is None
    score = next(
        s for s in report["scores"].values() if s["bundle_id"] != baseline().record_id
    )
    assert score["parent_changes"] is None
    assert score["integration_missing"] is None
    assert score["beyond_references"] is None
    assert score["success_rate"] == 1 and score["validation_success_rate"] == 1


def verified_author(author):
    verification_calls = []

    def verify_result(proposal, directory):
        verification_calls.append((proposal.record_id, directory))
        result = AuthoredMutation.model_validate_json(
            (directory / "result.json").read_text()
        )
        if result.proposal_id != proposal.record_id:
            raise ValueError("author receipt answers a different proposal")
        for evidence in result.artifacts:
            evidence.verify()
        return result

    author.verify_result = verify_result

    author.verification_calls = verification_calls
    return author


def test_protocol_rejects_duplicates_cross_task_and_removed_contracts():
    value = configured().model_dump(mode="json")
    for dev in (
        [value["dev"][0]] * 2,
        [value["dev"][0], {"task": "other", "initialization": {"seed": 2}}],
    ):
        with pytest.raises(ValidationError):
            EvolutionProtocol.model_validate({**value, "dev": dev})
    for removed in (
        "budget",
        "search_method",
        "shared_diagnosis",
        "selection",
        "sealed_test",
    ):
        with pytest.raises(ValidationError):
            EvolutionProtocol.model_validate({**value, removed: True})
    assert (
        EvolutionProtocol.model_validate(
            {k: v for k, v in value.items() if k != "max_authors_per_round"}
        ).max_authors_per_round
        == 1
    )
    assert (
        EvolutionProtocol.model_validate(
            {k: v for k, v in value.items() if k != "author_wall_time_s"}
        ).author_wall_time_s
        is None
    )
    for required in (
        "target_success_rate",
        "max_rounds",
    ):
        with pytest.raises(ValidationError):
            EvolutionProtocol.model_validate(
                {k: v for k, v in value.items() if k != required}
            )


@pytest.mark.anyio
async def test_three_independent_authors_overlap_and_stop_after_five_rounds(
    tmp_path, monkeypatch
):
    # Exercise scheduler concurrency separately from the tested width calculation.
    monkeypatch.setattr(
        "reweave.evolution.graph.selection.compute_width",
        lambda *args: (3, {"fixture_width": 3.0}, ({"width": 3.0},)),
    )
    config = configured(5).model_copy(update={"max_authors_per_round": 3})
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    entered = {}
    barriers = {}
    directories = set()
    requests = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        assert directory not in directories
        directories.add(directory)
        members = entered.setdefault(proposal.round_index, [])
        members.append(proposal)
        barrier = barriers.setdefault(proposal.round_index, asyncio.Event())
        if len(members) == 3:
            barrier.set()
        # A serial implementation cannot release this barrier.
        await asyncio.wait_for(barrier.wait(), 5)
        assert len({p.history for p in members}) == 1
        assert len({p.references for p in members}) == 1
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id="repair",
                    bundle=child(f"{proposal.round_index}:{proposal.author_index}"),
                    summary="independent",
                ),
            ),
            summary="Independent diagnosis and one mutation.",
            session_id=proposal.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        requests.append(request)
        return record(request, directory, False)

    search = graph_search(workspace, verified_author(author), evaluator)
    state = await search.run()
    assert state["stop_reason"] == "round_limit"
    assert len(entered) == 5 and len(directories) == 15
    assert len(state["versions"]) == 16
    assert len(requests) == 48  # H0 plus fifteen distinct authored programs.
    assert await search.run() == state
    assert len(directories) == 15 and len(requests) == 48


@pytest.mark.anyio
async def test_completed_author_rolls_out_before_slower_sibling_finishes(
    tmp_path, two_then_one_authors
):
    config = configured(1)
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    fast_rollout_started = asyncio.Event()

    async def author(proposal, directory, *, before_launch):
        before_launch()
        if proposal.author_index == 0:
            await asyncio.wait_for(fast_rollout_started.wait(), 2)
        name = "a" if proposal.author_index == 0 else "b"
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id="repair",
                    bundle=child(name),
                    summary="repair",
                ),
            ),
            summary="Independent diagnosis and mutation.",
            session_id=proposal.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        if bundle.context.instructions == "b":
            fast_rollout_started.set()
        return record(request, directory, False)

    state = await asyncio.wait_for(
        graph_search(workspace, verified_author(author), evaluator).run(), 5
    )
    authored_versions = [state["versions"][identity] for identity in state["order"][1:]]
    assert [version["proposal_id"] for version in authored_versions] == [
        "round_000000_author_000",
        "round_000000_author_001",
    ]


@pytest.mark.anyio
async def test_author_timeout_is_recorded_and_does_not_abort_the_round(
    tmp_path, two_then_one_authors
):
    config = configured(1).model_copy(update={"author_wall_time_s": 0.05})
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    calls = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        calls.append(proposal.author_index)
        if proposal.author_index == 0:
            await asyncio.sleep(60)
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id="repair",
                    bundle=child("b"),
                    summary="repair",
                ),
            ),
            summary="Independent diagnosis and mutation.",
            session_id=proposal.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, False)

    search = graph_search(workspace, verified_author(author), evaluator)
    state = await asyncio.wait_for(search.run(), 5)
    timeout = state["attempts"]["round-01/author-01"]
    assert timeout["status"] == "timed_out"
    assert timeout["result_id"] is None
    assert len(state["versions"]) == 2
    assert search.accounting(state)["author_timeouts"] == 1
    failure = json.loads((tmp_path / timeout["last_failure"]["path"]).read_text())
    assert failure["wall_time_limit_s"] == 0.05
    assert failure["wall_time_s"] >= 0.05
    assert await search.run() == state
    assert sorted(calls) == [0, 1]

    # Failed work remains evidence; private native runtimes never become history.
    timed_out_directory = tmp_path / "attempts/round-01/author-01"
    for relative in (
        "workspace/.scratch/runtime/codex/auth.json",
        "audit/session_inquiry/workspace/.scratch/runtime/codex/auth.json",
    ):
        artifact(timed_out_directory, relative, {"private_fixture": True})
    audit = artifact(
        timed_out_directory,
        "audit/session_inquiry/result.json",
        {
            "completed": False,
            "target": "historical-author",
            "question": {"question": "Which observation supports transfer?"},
            "answer": None,
        },
    )
    history = await search._freeze_history(state, await search.matrix(state), 1)
    manifest = json.loads(Path(history.originals.path).read_text())
    files = manifest["files"]
    timeout_entry = next(
        entry
        for entry in manifest["navigation"]["attempts"].values()
        if entry["status"] == "timed_out"
    )
    assert timeout_entry["receipt"] is None
    assert timeout_entry["inquiry"]["path"] == audit.path
    assert audit.path in files
    assert not any("/runtime/" in path for path in files)
    assert (
        "Which observation supports transfer?" in Path(history.summary.path).read_text()
    )


@pytest.mark.anyio
async def test_partial_round_recovers_only_unfinished_author(
    tmp_path, two_then_one_authors
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), configured(1))
    (tmp_path / "configuration.json").write_text("{}")
    calls = []
    recoveries = []
    failure_text = (
        "interrupted independent author\n" + "e" * 64 + " /authored/error  \n"
    )

    def result(proposal, directory):
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id="repair",
                    bundle=child(f"{proposal.round_index}:{proposal.author_index}"),
                    summary="repair",
                ),
            ),
            summary="One mutation.",
            session_id=proposal.record_id,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def author(proposal, directory, *, before_launch):
        before_launch()
        calls.append(proposal)
        if proposal.author_index == 1:
            raise RuntimeError(failure_text)
        return result(proposal, directory)

    async def recover(proposal, directory, *, before_launch):
        before_launch()
        recoveries.append(proposal)
        return result(proposal, directory)

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, False)

    adapter = verified_author(author)
    adapter.recover = recover
    with pytest.raises(RuntimeError, match="interrupted independent author"):
        await graph_search(workspace, adapter, evaluator).run()
    # A long operator downtime does not spend the failed author's work budget.
    with workspace.transaction() as interrupted:
        interrupted["attempts"]["round-01/author-02"]["deadline_at"] = 1.0
        workspace.save(interrupted)
    state = await graph_search(workspace, adapter, evaluator, recover=True).run()
    recovery_file = next(
        (tmp_path / "audit/attempt_recoveries/round-01/author-02").glob("*.json")
    )
    budget = json.loads(recovery_file.read_text())
    assert budget["previous_deadline_at"] == 1.0
    assert budget["prior_work_s"] > 0
    assert budget["remaining_work_s"] == pytest.approx(
        workspace.protocol.author_wall_time_s - budget["prior_work_s"]
    )
    assert state["stop_reason"] == "round_limit"
    assert len(calls) == 2
    assert recoveries == [next(p for p in calls if p.author_index == 1)]
    assert len(state["versions"]) == 3
    search = graph_search(workspace, adapter, evaluator, recover=True)
    history = await search._freeze_history(state, await search.matrix(state), 1)
    assert failure_text.encode() in Path(history.summary.path).read_bytes()
    manifest = json.loads(Path(history.originals.path).read_text())
    assert str(recovery_file) in manifest["files"]
    entry = next(
        value
        for value in manifest["navigation"]["attempts"].values()
        if value["name"] == "round-01/author-02"
    )
    assert entry["launches"] == 2
    assert len(entry["failures"]) == len(entry["recoveries"]) == 1
    assert entry["recoveries"][0]["path"] == str(recovery_file)


@pytest.mark.anyio
async def test_independent_sessions_global_references_history_and_idempotent_resume(
    tmp_path, two_then_one_authors
):
    from tests.unit_tests.harness.evolution.graph_fixtures import FixtureGraphAnalysis

    class InquiringGraph(FixtureGraphAnalysis):
        async def __call__(self, request, directory, *, before_launch):
            result = await super().__call__(
                request, directory, before_launch=before_launch
            )
            answer = artifact(
                directory,
                "audit/session_inquiry/result.json",
                {
                    "completed": True,
                    "target": "historical-author",
                    "question": {"question": "Which success conditions transfer?"},
                    "answer": "Synthetic testimony, not a rollout result.",
                },
            )
            return result.model_copy(update={"artifacts": (*result.artifacts, answer)})

    workspace = EvolutionWorkspace(tmp_path, baseline(), configured())
    (tmp_path / "configuration.json").write_text("{}")
    jobs, launches = [], []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        jobs.append(proposal)
        if proposal.round_index == 0:
            assert [r.bundle_id for r in proposal.references] == [baseline().record_id]
            candidates = (
                Candidate(
                    experience=f"Lesson from {proposal.record_id}: this interpretation may be wrong.",
                    branch_id="repair",
                    bundle=child("a" if proposal.author_index == 0 else "b"),
                    summary="independent repair",
                ),
            )
        else:
            # Exact ties go to the newest version: the second sibling of round 0.
            parent = child("b")
            assert proposal.references == (proposal.parent,)
            assert proposal.parent.bundle_id == parent.record_id
            history = Path(proposal.history.summary.path).read_text()
            assert history.count("### V") == 3
            # Notes remain exact and readable, without being repeated in every summary.
            assert "author_notes.summary / author_notes.experience" in history
            originals = json.loads(Path(proposal.history.originals.path).read_text())
            from reweave.evolution.contracts import OriginalReference
            from reweave.evolution.evidence.trajectory import read_original

            notes = [
                v["author_notes"]
                for v in originals["navigation"]["versions"].values()
                if v.get("author_notes")
            ]
            assert len(notes) == 2
            experiences = [
                read_original(OriginalReference(**n["experience"])) for n in notes
            ]
            for prior in jobs[:2]:
                lesson = (
                    f"Lesson from {prior.record_id}: this interpretation may be wrong."
                )
                assert lesson not in history
                assert lesson in experiences
            assert "subsequent official outcomes" in history
            assert all(
                read_original(OriginalReference(**n["summary"])) == "independent repair"
                for n in notes
            )
            assert "| success | success | failure |" in history
            assert "complete bundles" in history
            assert "incomplete runner evidence" in history
            originals = json.loads(Path(proposal.history.originals.path).read_text())
            assert originals["files"]
            assert len(originals["navigation"]["versions"]) == 3
            assert len(originals["navigation"]["bundles"]) == 3
            assert "Which success conditions transfer?" in history
            assert "Why did the changed decision regress?" in history
            inquiry_files = [
                path
                for path in originals["files"]
                if path.endswith("session_inquiry/result.json")
            ]
            assert (
                len(inquiry_files) == 3
            )  # Previous graph and both authors, once each.
            assert any(
                path.endswith(f"bundles/{child('a').record_id}.json")
                for path in originals["files"]
            )
            candidates = (
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id="integrated",
                    bundle=child("all"),
                    summary="integrated",
                ),
            )
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=candidates,
            summary="Competing root causes.",
            session_id=f"session-{len(jobs)}",
            usage=usage(),
            artifacts=(
                artifact(directory, "native.json", {"session": len(jobs)}),
                artifact(
                    directory,
                    "audit/session_inquiry/result.json",
                    {
                        "completed": True,
                        "target": "historical-graph",
                        "question": {
                            "question": "Why did the changed decision regress?"
                        },
                        "answer": "Synthetic testimony; compare actual outcomes.",
                    },
                ),
            ),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        launches.append(request)
        solved = {"": {1}, "a": {1, 2}, "b": {2, 3}, "all": {1, 2, 3}}[
            bundle.context.instructions
        ]
        return record(
            request, directory, request.instance.initialization["seed"] in solved
        )

    search = graph_search(
        workspace, verified_author(author), evaluator, analyzer=InquiringGraph()
    )
    state = await search.run()
    assert state["stop_reason"] == "target_reached"
    assert len(jobs) == 3 and len(launches) == 12
    assert len(state["versions"]) == 4
    report = json.loads((tmp_path / "report.json").read_text())
    # Three mutations and one consolidation of the fully qualified final version.
    assert (
        report["total_usage_including_failed_work"]["author_launches"] == len(jobs) + 1
    )
    assert [key for key in state["attempts"] if key.startswith("consolidation/")] == [
        "consolidation/V003"
    ]
    assert report["scores"][report["best_version"]]["success_rate"] == 1
    assert report["scores"][report["best_version"]]["integration_missing"] == []
    final_score = report["scores"][report["best_version"]]
    assert set(final_score["reference_changes"]) == {
        r.version_id for r in jobs[-1].references
    }
    assert final_score["parent"] == jobs[-1].parent.model_dump(mode="json")
    assert (
        final_score["parent_changes"]
        == final_score["reference_changes"][jobs[-1].parent.version_id]
    )
    assert final_score["changed_scopes"] == ["C"]
    assert "Available reference set:" in Path(jobs[-1].history.summary.path).read_text()
    verified_before_resume = len(author.verification_calls)
    assert await search.run() == state
    assert len(author.verification_calls) == verified_before_resume
    assert len(jobs) == 3 and len(launches) == 12
    # A mutated completed receipt is rejected, not re-evaluated.
    receipt = next((tmp_path / "attempts").rglob("result.json"))
    receipt.write_text("{}")
    with pytest.raises(ValueError):
        await search.run()
    assert len(launches) == 12


@pytest.mark.anyio
async def test_historical_content_reuse_preserves_new_origin_without_another_rollout(
    tmp_path, two_then_one_authors
):
    workspace = EvolutionWorkspace(tmp_path, baseline(), configured())
    (tmp_path / "configuration.json").write_text("{}")
    proposals, requests = [], []
    # Round 0 siblings tie, so the newest (b) becomes the parent; later rounds
    # re-propose the older sibling's content, a historical non-parent version.
    repeated = child("a").context.instructions

    async def author(proposal, directory, *, before_launch):
        before_launch()
        proposals.append(proposal)
        names = (
            (("a" if proposal.author_index == 0 else "b"),)
            if proposal.round_index == 0
            else (repeated,)
        )
        if proposal.round_index > 0:
            assert child(repeated).record_id not in {
                r.bundle_id for r in proposal.references
            }
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=tuple(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id=name,
                    bundle=child(name),
                    summary=name,
                )
                for name in names
            ),
            summary="One session.",
            session_id=f"session-{len(proposals)}",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        requests.append(request)
        successes = {"": set(), "a": {1, 2}, "b": {1, 2}, "all": {1, 2, 3}}
        return record(
            request,
            directory,
            request.instance.initialization["seed"]
            in successes[bundle.context.instructions],
        )

    search = graph_search(workspace, verified_author(author), evaluator)
    state = await search.run()
    assert state["stop_reason"] == "round_limit"
    assert len(proposals) == 3 and len(requests) == 9
    assert len(state["versions"]) == 4
    origins = [
        v
        for v in state["versions"].values()
        if v["bundle_id"] == child(repeated).record_id
    ]
    assert len(origins) == 2 and len({v["proposal_id"] for v in origins}) == 2
    history = await search._freeze_history(state, await search.matrix(state), 2)
    navigation = json.loads(Path(history.originals.path).read_text())["navigation"]
    assert len(navigation["versions"]) == 4
    assert len(navigation["bundles"]) == 3
    alias_entries = [
        entry
        for entry in navigation["versions"].values()
        if entry["bundle_id"] == child(repeated).record_id
    ]
    assert len({entry["label"] for entry in alias_entries}) == 2
    assert len({entry["author_attempt"] for entry in alias_entries}) == 2
    assert (
        len(
            next(
                entry
                for entry in navigation["bundles"]
                if entry["bundle_id"] == child(repeated).record_id
            )["evaluation_attempts"]
        )
        == 3
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["matrix"]) == 3
    assert (
        report["total_usage_including_failed_work"]["by_kind"]["author"]["wall_time_s"]
        == 3
    )
    assert await search.run() == state
    assert len(proposals) == 3 and len(requests) == 9
    Path(proposals[0].history.summary.path).write_text("corrupt")
    assert await search.run() == state
    assert len(requests) == 9


@pytest.mark.anyio
async def test_mixed_cached_and_new_batch_continues_with_new_full_evaluation(
    tmp_path, monkeypatch
):
    # A controlled schedule isolates mixed cache/new evaluation ownership.
    monkeypatch.setattr(
        "reweave.evolution.graph.selection.compute_width", width_schedule(3, 2)
    )
    value = configured(15).model_dump(mode="json")
    value["dev"] = [
        {
            "record_id": f"condition:{seed}",
            "task": "drawer",
            "initialization": {"seed": seed},
        }
        for seed in (1, 2, 3, 4)
    ]
    value["max_authors_per_round"] = 3
    workspace = EvolutionWorkspace(
        tmp_path, baseline(), EvolutionProtocol.model_validate(value)
    )
    (tmp_path / "configuration.json").write_text("{}")
    proposals, requests = [], []
    # Round 0 authors a, b, a: tied, so the newest version (content a) is the
    # parent; later rounds re-propose b, a historical non-parent version.
    repeated = child("b").context.instructions

    async def author(proposal, directory, *, before_launch):
        before_launch()
        proposals.append(proposal)
        name = (("a", "b", "a") if proposal.round_index == 0 else (repeated, "all"))[
            proposal.author_index
        ]
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id=name,
                    bundle=child(name),
                    summary=name,
                ),
            ),
            summary="Mixed cached and new hypotheses.",
            session_id=f"session-{len(proposals)}",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        requests.append(request)
        successes = {"": set(), "a": {1, 2}, "b": {1, 2}, "all": {1, 2, 3, 4}}
        return record(
            request,
            directory,
            request.instance.initialization["seed"]
            in successes[bundle.context.instructions],
        )

    from tests.unit_tests.harness.evolution.graph_fixtures import FixtureGraphAnalysis

    state = await graph_search(
        workspace,
        verified_author(author),
        evaluator,
        analyzer=FixtureGraphAnalysis(),
    ).run()
    assert state["stop_reason"] == "target_reached"
    assert len(proposals) == 5 and len(requests) == 16
    assert len(state["versions"]) == 6


@pytest.mark.anyio
async def test_unselected_failed_harnesses_remain_in_history(tmp_path):
    workspace = EvolutionWorkspace(tmp_path, baseline(), configured(9))
    (tmp_path / "configuration.json").write_text("{}")
    proposals = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        proposals.append(proposal)
        assert len(proposal.references) == 1
        if proposal.round_index == 1:
            history = Path(proposal.history.summary.path).read_text()
            assert child("failed").record_id not in history
            originals = json.loads(Path(proposal.history.originals.path).read_text())
            assert proposals[0].history.summary.path in originals["files"]
            assert child("failed").record_id in {
                entry["bundle_id"]
                for entry in originals["navigation"]["versions"].values()
            }
            assert proposals[0].record_id in {
                entry["origin"]["proposal_id"]
                for entry in originals["navigation"]["versions"].values()
                if entry["origin"] is not None
            }
        name = "failed" if proposal.round_index == 0 else "all"
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id=name,
                    bundle=child(name),
                    summary=name,
                ),
            ),
            summary="Repair",
            session_id=name,
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, bundle.context.instructions == "all")

    search = graph_search(workspace, verified_author(author), evaluator)
    state = await search.run()
    assert state["stop_reason"] == "target_reached"
    original_report = (tmp_path / "report.json").read_bytes()
    original_receipts = {
        p: p.read_bytes() for p in (tmp_path / "attempts").rglob("result.json")
    }
    (tmp_path / "report.json").unlink()

    async def forbidden(*args, **kwargs):
        pytest.fail("rebuilding a derived report cannot launch work")

    resumed = graph_search(workspace, forbidden, forbidden, analyzer=forbidden)
    assert await resumed.run() == state
    assert (tmp_path / "report.json").read_bytes() == original_report
    assert all(p.read_bytes() == data for p, data in original_receipts.items())


@pytest.mark.anyio
@pytest.mark.parametrize("success", [False, True])
async def test_baseline_only_round_limit_or_direct_reuse(tmp_path, success):
    workspace = EvolutionWorkspace(tmp_path, baseline(), configured(0))
    (tmp_path / "configuration.json").write_text("{}")

    async def forbidden(*args, **kwargs):
        pytest.fail("baseline-only run must not call author")

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, success)

    state = await graph_search(workspace, forbidden, evaluator).run()
    assert state["stop_reason"] == ("target_reached" if success else "round_limit")
    assert not state["proposals"]


@pytest.mark.anyio
async def test_interrupted_attempt_needs_explicit_recovery(tmp_path):
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    (tmp_path / "configuration.json").write_text("{}")
    calls = []

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        calls.append(request)
        raise RuntimeError("worker interrupted")

    search = graph_search(workspace, None, evaluator)
    with pytest.raises(RuntimeError, match="worker interrupted"):
        await search.run()
    with pytest.raises(EvolutionIncomplete, match="recovery"):
        await search.run()
    assert len(calls) == 1


def test_old_state_and_diagnosis_proposals_are_rejected(tmp_path):
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    with workspace.transaction() as state:
        snapshot = dict(state)
    with pytest.raises(ValidationError):
        EvolutionState.model_validate({**snapshot, "h1": None})
    with pytest.raises(ValidationError):
        Proposal.model_validate({"purpose": "diagnosis", "proposal_id": "diagnosis"})


def test_scope_permissions_and_multiple_candidates_rejected(tmp_path):
    for scope in ("W", "C", "L"):
        candidate = mutated(baseline(), scope)
        candidate.assert_candidate((baseline(),), frozenset({scope}))
        with pytest.raises(ValueError, match="forbidden"):
            candidate.assert_candidate((baseline(),), frozenset())
    with pytest.raises(ValidationError, match="at most 1"):
        AuthoredMutation(
            record_id=new_record_id(),
            proposal_id="a" * 64,
            candidates=tuple(
                Candidate(
                    experience="Synthetic lesson: distinguish observations from hypotheses; physical efficacy remains untested.",
                    branch_id=b,
                    bundle=child("same"),
                    summary=b,
                )
                for b in ("a", "b")
            ),
            summary="distinct labels are insufficient",
            session_id="one",
            usage=usage(),
            artifacts=(artifact(tmp_path, "native.json", {}),),
        )


@pytest.mark.anyio
async def test_history_indexes_verbatim_testimony_without_repeating_it(
    tmp_path, monkeypatch
):
    from reweave.evolution.contracts import OriginalReference
    from reweave.evolution.evidence.trajectory import read_original

    monkeypatch.setattr(
        "reweave.evolution.graph.selection.compute_width",
        lambda *args: (1, {"fixture_schedule": 1.0}, ({"width": 1.0},)),
    )
    workspace = EvolutionWorkspace(tmp_path, baseline(), configured(1))
    (tmp_path / "configuration.json").write_text("{}")
    raw_identity = "f" * 64
    literal = f"  Author text {raw_identity} /authored/path/code.py\r\n\r\n    trailing spaces  \n"
    question = f"Keep exact question {raw_identity}\n/authored/path/question\n"
    source_files = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        source = artifact(
            directory,
            "workspace/candidate/task-family/context/custom.json",
            {"literal": literal},
        )
        source_files.append(source)
        inquiry = artifact(
            directory,
            "audit/session_inquiry/result.json",
            {
                "completed": False,
                "target": "historical-author",
                "question": {"question": question},
                "answer": None,
            },
        )
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="repair",
                    bundle=child("history"),
                    summary=literal,
                    experience=literal,
                ),
            ),
            summary="Original author receipt",
            session_id="history-session",
            usage=usage(),
            artifacts=(source, inquiry),
        )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, False)

    search = graph_search(workspace, verified_author(author), evaluator)
    state = await search.run()
    history = await search._freeze_history(state, await search.matrix(state), 1)
    rendered = Path(history.summary.path).read_bytes()
    assert literal.encode() not in rendered
    assert question.encode() in rendered
    generated = rendered.replace(literal.encode(), b"").replace(question.encode(), b"")
    assert b"/authored/path" not in generated
    assert str(tmp_path).encode() not in generated
    assert all(identity.encode() not in generated for identity in state["versions"])
    assert all(
        value["bundle_id"].encode() not in generated
        for value in state["versions"].values()
    )
    assert b"Results against parent:" not in generated
    assert b"Candidate code and validation originals" not in generated
    originals = json.loads(Path(history.originals.path).read_text())
    assert set(originals) == {"files", "navigation"}
    navigation = originals["navigation"]
    authored = next(
        entry for entry in navigation["attempts"].values() if entry["kind"] == "author"
    )
    assert list(authored["candidate_files"]) == ["task-family/context/custom.json"]
    reference = OriginalReference(
        **authored["candidate_files"]["task-family/context/custom.json"]
    )
    assert read_original(reference) == {"literal": literal}
    notes = next(
        v["author_notes"]
        for v in navigation["versions"].values()
        if v.get("author_notes")
    )
    # Issued file handles resolve the precise candidate fields, including CRLF,
    # original IDs and whitespace, without reading a complete Bundle receipt.
    from reweave.evolution.authoring.references import AuthorReferences

    refs = AuthorReferences.create(
        tmp_path / "reader",
        registry=tmp_path / "reference_registry.json",
        evidence_roots={},
        originals=originals["files"],
    )
    handles = refs.encode(notes)
    for field in ("summary", "experience"):
        assert read_original(refs.original(handles[field])) == literal

    assert read_original(OriginalReference(**authored["inquiry"]))["answer"] is None
    assert all(
        entry["evidence_id"] is None
        for entry in navigation["attempts"].values()
        if entry["kind"] == "evaluation"
    )
    assert (
        Path(source_files[0].path).read_bytes()
        == json.dumps({"literal": literal}).encode()
    )


def execution_revision_fixture(old_protocol, kind):
    if kind == "budget":
        current = old_protocol.model_copy(
            update={
                "record_id": "protocol:revised-budget",
                "execution_id": "execution:revised-budget",
                "execution": old_protocol.execution.model_copy(
                    update={"wall_time_s": None}
                ),
            }
        )
        return (
            current,
            "budget_revisions",
            BudgetRevision(previous=old_protocol, current=current),
        )
    current = old_protocol.model_copy(
        update={
            "record_id": "protocol:revised-runtime",
            "execution_id": "execution:revised-runtime",
            "execution_runtime": baseline().runtime.model_copy(
                update={
                    "revision_id": "repaired-runtime",
                    "tool_api": ({"name": "repaired_tool"},),
                }
            ),
        }
    )
    return (
        current,
        "execution_revisions",
        ExecutionRevision(
            previous=old_protocol,
            current=current,
            reason="Synthetic CPU source/tool repair; no physical success claim.",
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("revision_kind", ["budget", "runtime"])
async def test_online_budget_revision_reuses_settled_requests_and_labels_new_work(
    tmp_path,
    revision_kind,
):
    old_protocol = configured(1)
    old = tmp_path / "old"
    old.mkdir()
    (old / "configuration.json").write_text("{}")
    original = EvolutionWorkspace(old, baseline(), old_protocol)
    calls = []

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        calls.append(request)
        return record(request, directory, False)

    search = graph_search(original, None, evaluator)
    with original.transaction() as state:
        await search._evaluate_bundle(state, baseline(), "V000")
    original_bytes = original.state_path.read_bytes()
    revised_protocol, revision_field, revision = execution_revision_fixture(
        old_protocol, revision_kind
    )
    revised_root = tmp_path / "revised"
    shutil.copytree(old, revised_root)
    state = json.loads(original_bytes)
    state["protocol"] = revised_protocol.record_id
    state[revision_field] = [revision.model_dump(mode="json")]
    for key, attempt in state["attempts"].items():
        attempt["result_directory"] = str(old / "attempts" / key)
    revised = EvolutionWorkspace(revised_root, baseline(), revised_protocol)
    revised.save(state)
    continuation = graph_search(revised, None, evaluator, recover=True)
    receipts = await continuation._evaluate_bundle(state, baseline(), "V000")
    assert all(r.request.execution == old_protocol.execution_id for r in receipts)
    assert len(calls) == len(old_protocol.dev)
    fresh = await continuation._evaluate_bundle(state, child("budget"), "fresh")
    assert all(r.request.execution == revised_protocol.execution_id for r in fresh)
    assert len(calls) == 2 * len(old_protocol.dev)
    assert original.state_path.read_bytes() == original_bytes
    assert revised.execution_ids(state) == frozenset(
        {
            old_protocol.execution_id,
            revised_protocol.execution_id,
        }
    )


@pytest.mark.anyio
@pytest.mark.parametrize("origin", ["agent", "infrastructure", "none"])
@pytest.mark.parametrize("complete_episode", [False, True])
@pytest.mark.parametrize("revision_kind", ["budget", "runtime"])
async def test_explicit_budget_retest_flows_through_graph_and_report(
    tmp_path, origin, complete_episode, revision_kind
):
    from reweave.evolution.graph.replay import replay_graph
    from reweave.evolution.search.report import publish_report

    old_protocol = protocol()
    old = tmp_path / "old"
    old.mkdir()
    (old / "configuration.json").write_text("{}")
    original = EvolutionWorkspace(old, baseline(), old_protocol)

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        result = record(request, directory, True)
        if complete_episode:
            from reweave.evolution.evidence.records import EpisodeEvidence

            files = {
                Path(f.path).name: {"artifact_id": f.artifact_id, "bytes": f.bytes}
                for f in result.evidence
            }
            result = result.model_copy(
                update={
                    "episode": EpisodeEvidence(evidence_id=new_record_id(), files=files)
                }
            )
        if request.execution == old_protocol.execution_id:
            return result
        return result.model_copy(
            update={
                "official_success": origin == "none",
                "failure_origin": origin,
                "infrastructure_retryable": False
                if origin == "infrastructure"
                else None,
            }
        )

    search = graph_search(original, None, evaluator)
    with original.transaction() as state:
        original.put_bundle(baseline())
        original.add_version(
            state,
            BundleVersion(
                record_id=new_record_id(),
                bundle_id=baseline().record_id,
                proposal_id=None,
                branch_id=None,
            ),
        )
        await search._evaluate_bundle(state, baseline(), "V000")
    original_bytes = original.state_path.read_bytes()
    revised_protocol, revision_field, revision = execution_revision_fixture(
        old_protocol, revision_kind
    )
    revised_root = tmp_path / "revised"
    shutil.copytree(old, revised_root)
    state["protocol"] = revised_protocol.record_id
    state[revision_field] = [revision.model_dump(mode="json")]
    for key, attempt in state["attempts"].items():
        attempt["result_directory"] = str(old / "attempts" / key)
    revised = EvolutionWorkspace(revised_root, baseline(), revised_protocol)
    revised.save(state)
    continuation = graph_search(revised, None, evaluator, recover=True)
    await continuation._evaluate_bundle(state, baseline(), "explicit-budget-retest")
    rows = await continuation.matrix(state)
    assert rows[baseline().record_id] == (
        None if origin == "infrastructure" else origin == "none",
    )
    await continuation._proposals(state, rows, 0)
    graph_request = state["rounds"]["0"]["request"]
    assert len(graph_request["evidence"]) == 2
    decision = json.loads(Path(state["rounds"]["0"]["decision"]["path"]).read_text())
    progress = json.loads(Path(decision["progress_map"]["path"]).read_text())
    assert len(progress["episodes"]) == (
        1 if origin == "infrastructure" and not complete_episode else 2
    )
    cell = next(iter(map_row(progress, state["order"][0]).values()))
    if origin == "infrastructure":
        assert cell["official_success"] is None
        if complete_episode:
            assert 0 < cell["progress"] < 1
            assert cell["receipt_id"] is not None
            assert cell["usage"] == continuation._records(state)[-1].usage.model_dump(
                mode="json"
            )
        else:
            assert cell["progress"] is None
            assert cell["receipt_id"] is None
    else:
        assert cell["official_success"] is (origin == "none")
    records = continuation._records(state)
    publish_report(revised, state, records, continuation.accounting(state))
    report = json.loads((revised_root / "progress.json").read_text())
    score = report["scores"][state["order"][0]]
    assert score["success_rate"] == (
        None if origin == "infrastructure" else float(origin == "none")
    )
    if origin == "infrastructure":
        assert not score["complete"] and "baseline_delta" not in score
        assert decision["statistics"]["parent_success_rate"] is None
    assert score["historical_execution_ids"] == sorted(revised.execution_ids(state))
    assert score["execution_ids"] == [revised_protocol.execution_id]
    assert score["execution_by_condition"] == {
        revised_protocol.dev[0].record_id: revised_protocol.execution_id
    }
    replay_policy = tmp_path / "replay_policy.json"
    replay_policy.write_text(revised_protocol.graph.model_dump_json())
    replay = replay_graph(revised_root, tmp_path / "replayed.json", replay_policy)
    assert len(replay.episodes) == len(progress["episodes"])
    assert original.state_path.read_bytes() == original_bytes


@pytest.mark.parametrize("complete_episode", [False, True])
def test_report_lists_complete_infrastructure_episode_pending_graph_projection(
    tmp_path, complete_episode
):
    from reweave.evolution.contracts import EvaluationRequest
    from reweave.evolution.evidence.records import EpisodeEvidence
    from reweave.evolution.search.report import semantic_projection

    task_protocol = protocol()
    observation = record(
        EvaluationRequest(
            record_id=new_record_id(),
            bundle_id=baseline().record_id,
            instance=task_protocol.dev[0],
            split="dev",
            manifest_id=task_protocol.manifest_id,
            conditions=task_protocol.execution,
            execution=task_protocol.execution_id,
        ),
        tmp_path / "evaluation",
        False,
    )
    files = {
        Path(f.path).name: {"artifact_id": f.artifact_id, "bytes": f.bytes}
        for f in observation.evidence
    }
    observation = observation.model_copy(
        update={
            "failure_origin": "infrastructure",
            "infrastructure_retryable": False,
            "episode": EpisodeEvidence(evidence_id=new_record_id(), files=files)
            if complete_episode
            else None,
        }
    )
    projection = semantic_projection({"rounds": {}}, (observation,))
    assert projection["pending_episode_ids"] == (
        [observation.record_id] if complete_episode else []
    )


@pytest.mark.anyio
@pytest.mark.parametrize("revision_kind", ["budget", "runtime"])
async def test_report_keeps_cross_execution_condition_measurements_incomplete(
    tmp_path, revision_kind
):
    """Only current measurements can complete a row or support a comparison."""
    from reweave.evolution.search.report import publish_report

    old_protocol = configured(1).model_copy(update={"max_authors_per_round": 1})
    old = tmp_path / "old"
    old.mkdir()
    (old / "configuration.json").write_text("{}")
    workspace = EvolutionWorkspace(old, baseline(), old_protocol)

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        return record(request, directory, False)

    async def author(proposal, directory, *, before_launch):
        before_launch()
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    experience="Synthetic budget pairing fixture, no physical claim.",
                    branch_id="repair",
                    bundle=child("budget pairing"),
                    summary="Synthetic candidate",
                ),
            ),
            summary="Synthetic independent author",
            session_id="budget-pairing-fixture",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    search = graph_search(workspace, verified_author(author), evaluator)
    state = await search.run()
    revised_protocol, revision_field, revision = execution_revision_fixture(
        old_protocol, revision_kind
    )
    new = tmp_path / "revised"
    shutil.copytree(old, new)
    state.update(
        protocol=revised_protocol.record_id,
        completed=False,
        stop_reason=None,
    )
    state[revision_field] = [revision.model_dump(mode="json")]
    revised = EvolutionWorkspace(new, baseline(), revised_protocol)
    revised.save(state)
    observations = []
    for observation in search._records(state):
        request = observation.request
        is_child = request.bundle_id != baseline().record_id
        first_seed = request.instance == old_protocol.dev[0]
        execution = (
            revised_protocol.execution_id
            if is_child == first_seed
            else old_protocol.execution_id
        )
        observations.append(
            observation.model_copy(
                update={
                    "request": request.model_copy(
                        update={
                            "execution": execution,
                            "conditions": revised_protocol.execution
                            if execution == revised_protocol.execution_id
                            else old_protocol.execution,
                        }
                    ),
                    "official_success": is_child,
                    "failure_origin": "none" if is_child else "agent",
                }
            )
        )
    publish_report(revised, state, observations, search.accounting(state))
    report = json.loads((new / "progress.json").read_text())
    parent, candidate = (report["scores"][v] for v in state["order"])
    assert parent["execution_ids"] == candidate["execution_ids"]
    assert parent["execution_by_condition"] != candidate["execution_by_condition"]
    assert candidate["complete"] is False and candidate["success_rate"] is None
    assert candidate["comparable_to_baseline"] is False
    assert "baseline_delta" not in candidate
    assert {
        "parent_changes",
        "reference_changes",
        "integration_missing",
        "beyond_references",
    }.isdisjoint(candidate)


@pytest.mark.anyio
async def test_explicit_runtime_retest_resumes_through_original_candidate_group(
    tmp_path,
):
    old_protocol = configured(1)
    old = tmp_path / "old"
    old.mkdir()
    (old / "configuration.json").write_text("{}")
    original = EvolutionWorkspace(old, baseline(), old_protocol)
    candidate_bundle = child("wifi-retest")
    calls = []

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        calls.append((request, directory))
        return record(
            request, directory, request.execution != old_protocol.execution_id
        )

    search = graph_search(original, None, evaluator)
    with original.transaction() as state:
        await search._evaluate_bundle(state, baseline(), "V000")
        await search._evaluate_bundle(state, candidate_bundle, "round-01-author-01")
    old_bytes = {
        path: path.read_bytes() for path in (old / "attempts").rglob("result.json")
    }
    current, revision_field, revision = execution_revision_fixture(
        old_protocol, "runtime"
    )
    revised_root = tmp_path / "revised"
    shutil.copytree(old, revised_root)
    state["protocol"] = current.record_id
    state[revision_field] = [revision.model_dump(mode="json")]
    for key, attempt in state["attempts"].items():
        attempt["result_directory"] = str(old / "attempts" / key)
    revised = EvolutionWorkspace(revised_root, baseline(), current)
    revised.save(state)
    continuation = graph_search(revised, None, evaluator, recover=True)

    # A runtime publication alone never repeats historical physical work.
    retained = await continuation._evaluate_bundle(
        state, candidate_bundle, "round-01-author-01"
    )
    assert all(r.request.execution == old_protocol.execution_id for r in retained)
    assert len(calls) == 2 * len(old_protocol.dev)

    # The operator explicitly registers a new trial of the same content.
    fresh = await continuation._evaluate_bundle(
        state, candidate_bundle, "round-01-explicit-wifi-author-01"
    )
    assert all(r.request.execution == current.execution_id for r in fresh)
    assert all(r.official_success for r in fresh)
    assert len(calls) == 3 * len(old_protocol.dev)

    # _finish_round uses this original group. It must now find the exact
    # published current request and reuse the new receipt, without an adapter.
    resumed = await continuation._evaluate_bundle(
        state, candidate_bundle, "round-01-author-01"
    )
    assert resumed == fresh
    h0 = await continuation._evaluate_bundle(state, baseline(), "V000")
    assert all(r.request.execution == old_protocol.execution_id for r in h0)
    assert len(calls) == 3 * len(old_protocol.dev)
    assert all(path.read_bytes() == data for path, data in old_bytes.items())

    from reweave.evolution.search.matrix import result_matrix

    rows = result_matrix(
        current,
        [baseline().record_id, candidate_bundle.record_id],
        continuation._records(state),
        execution_ids=revised.execution_ids(state),
    )
    assert rows[candidate_bundle.record_id] == (True,) * len(current.dev)
    assert rows[baseline().record_id] == (None,) * len(current.dev)


@pytest.mark.anyio
async def test_exhausted_settled_infrastructure_stays_missing_and_search_continues(
    tmp_path,
):
    config = protocol().model_copy(update={"infrastructure_retries": 1})
    workspace = EvolutionWorkspace(tmp_path, baseline(), config)
    (tmp_path / "configuration.json").write_text("{}")
    calls = []

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        calls.append(directory)
        if bundle.record_id != baseline().record_id:
            return record(request, directory, False)
        return record(request, directory, False).model_copy(
            update={
                "failure_origin": "infrastructure",
                "infrastructure_retryable": True,
            }
        )

    authored = []

    async def author(proposal, directory, *, before_launch):
        before_launch()
        authored.append(proposal)
        return AuthoredMutation(
            record_id=new_record_id(),
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="repair",
                    bundle=child("after infra"),
                    summary="Synthetic finite-search candidate",
                    experience="Missing official outcomes remain censored.",
                ),
            ),
            summary="Synthetic continuation",
            session_id="fixture-infra",
            usage=usage(),
            artifacts=(artifact(directory, "native.json", {}),),
        )

    for recover in (False, True):
        search = graph_search(
            workspace, verified_author(author), evaluator, recover=recover
        )
        await search.run()
        state = json.loads(workspace.state_path.read_text())
        assert state["completed"] and state["stop_reason"] == "round_limit"
        assert len(state["proposals"]) == 1 and len(authored) == 1
        assert len(calls) == len(config.dev) * 3
        assert (await search.matrix(state))[baseline().record_id] == (None,)
