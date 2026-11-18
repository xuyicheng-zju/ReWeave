"""Publication IDs survive parsing and reuse without content identities."""

import copy
import json

import pytest

from reweave.evolution.bundle import AgentBundle
from reweave.evolution.contracts import (
    AuthoredMutation,
    BundleVersion,
    Candidate,
    EvaluationRecord,
    EvaluationRequest,
    EvolutionProtocol,
    Proposal,
)
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.search.engine import EvolutionWorkspace, GraphEvolution
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    evolution_history,
    graph_proposal_fields,
    mutated,
    protocol,
    usage,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def observation(request, directory, *, record_id):
    return EvaluationRecord(
        record_id=record_id,
        request=request,
        official_success=False,
        failure_origin="agent",
        infrastructure_retryable=None,
        usage=usage(),
        evidence=(artifact(directory, "fixture.json", {"synthetic": True}),),
        episode=None,
        initialization=request.instance.initialization,
    )


def test_parsing_published_records_never_allocates_another_identifier(monkeypatch):
    bundle = baseline()
    value = bundle.model_dump_json()

    def forbidden():
        raise AssertionError("reading must not publish a new version")

    monkeypatch.setattr("reweave.evolution.bundle.uuid.uuid4", forbidden)
    assert AgentBundle.model_validate_json(value) == bundle
    assert AgentBundle.model_validate_json(value).record_id == bundle.record_id
    without_identifier = bundle.model_dump(mode="json")
    without_identifier.pop("record_id")
    with pytest.raises(ValueError, match="record_id"):
        AgentBundle.model_validate(without_identifier)


def test_condition_uniqueness_compares_actual_reset_values():
    declared = protocol()
    original = declared.dev[0]
    republished = original.model_copy(update={"record_id": "another-condition-record"})
    with pytest.raises(ValueError, match="duplicate or overlapping"):
        EvolutionProtocol.model_validate(
            declared.model_copy(update={"dev": (original, republished)}).model_dump(
                mode="json"
            )
        )
    another_reset = republished.model_copy(update={"initialization": {"seed": 2}})
    distinct = EvolutionProtocol.model_validate(
        declared.model_copy(update={"dev": (original, another_reset)}).model_dump(
            mode="json"
        )
    )
    assert distinct.dev == (original, another_reset)


def test_unchanged_values_reuse_the_published_bundle(tmp_path):
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    original = baseline()
    assert workspace.put_bundle(original) == original.record_id
    artifact_record(tmp_path / "bundles" / f"{original.record_id}.json")
    republished = original.model_copy(
        update={
            "record_id": "another-publication",
            "runtime": original.runtime.model_copy(
                update={"revision_id": "another-runtime-record"}
            ),
        }
    )
    assert republished.same_harness(original)
    assert republished.novel_scopes((original,)) == frozenset()
    assert workspace.put_bundle(republished) == original.record_id
    assert (
        len(
            [
                path
                for path in (tmp_path / "bundles").glob("*.json")
                if not path.name.startswith(".")
            ]
        )
        == 1
    )
    changed = mutated(original, "C")
    assert workspace.put_bundle(changed) == changed.record_id
    assert not changed.same_harness(original)
    with pytest.raises(ValueError, match="archive was modified"):
        workspace.put_bundle(
            changed.model_copy(update={"record_id": original.record_id})
        )


def test_failed_version_publication_does_not_mutate_memory_or_disk(tmp_path):
    workspace = EvolutionWorkspace(tmp_path, baseline(), protocol())
    workspace.put_bundle(baseline())
    with workspace.transaction() as state:
        workspace.add_version(
            state,
            BundleVersion(
                record_id="baseline",
                bundle_id=baseline().record_id,
                proposal_id=None,
                branch_id=None,
            ),
        )
        original = copy.deepcopy(state)
        data = workspace.state_path.read_bytes()
        with pytest.raises(ValueError, match="invalid initial version"):
            workspace.add_version(
                state,
                BundleVersion(
                    record_id="second-root",
                    bundle_id=baseline().record_id,
                    proposal_id=None,
                    branch_id=None,
                ),
            )
        assert state == original
        assert workspace.state_path.read_bytes() == data


@pytest.mark.anyio
async def test_equivalent_author_outputs_share_one_rollout_and_resume_cache(tmp_path):
    workspace = EvolutionWorkspace(tmp_path / "search", baseline(), protocol())
    fields = graph_proposal_fields(tmp_path / "inputs")
    parent = {"version_id": "b" * 64, "bundle_id": baseline().record_id}
    request = EvaluationRequest(
        record_id="baseline-evaluation",
        bundle_id=baseline().record_id,
        instance=workspace.protocol.dev[0],
        split="dev",
        manifest_id=workspace.protocol.manifest_id,
        execution=workspace.protocol.execution_id,
        conditions=workspace.protocol.execution,
    )
    history = evolution_history(tmp_path / "inputs")
    evidence = (
        observation(
            request, tmp_path / "baseline-result", record_id="baseline-observation"
        ),
    )
    proposals = tuple(
        Proposal(
            **fields,
            record_id=f"author-{index}",
            references=(parent,),
            parent=parent,
            round_index=0,
            author_index=index,
            scopes=("W", "C", "L"),
            history=history,
            evidence=evidence,
        )
        for index in range(2)
    )
    calls = {"author": 0, "evaluation": 0}

    class Author:
        async def __call__(self, proposal, directory, *, before_launch):
            before_launch()
            calls["author"] += 1
            bundle = mutated(baseline(), "C").model_copy(
                update={"record_id": f"candidate-{proposal.author_index}"}
            )
            return AuthoredMutation(
                record_id=f"author-result-{proposal.author_index}",
                proposal_id=proposal.record_id,
                candidates=(
                    Candidate(
                        branch_id="repair",
                        bundle=bundle,
                        summary="Synthetic equivalent edit",
                        experience="Fixture only",
                    ),
                ),
                summary="Synthetic mutation",
                session_id=f"synthetic-author-{proposal.author_index}",
                usage=usage(),
                artifacts=(artifact(directory, "native.json", {}),),
            )

        def verify_result(self, proposal, directory):
            return AuthoredMutation.model_validate_json(
                (directory / "result.json").read_text()
            )

    async def evaluator(request, bundle, directory, *, before_launch):
        before_launch()
        calls["evaluation"] += 1
        return observation(request, directory, record_id="candidate-observation")

    search = GraphEvolution(workspace, Author(), evaluator)

    async def no_graph_work(*args):
        return None

    search._settle_round = no_graph_work
    with workspace.transaction() as state:
        workspace.put_bundle(baseline())
        workspace.add_version(
            state,
            BundleVersion(
                record_id=parent["version_id"],
                bundle_id=baseline().record_id,
                proposal_id=None,
                branch_id=None,
            ),
        )
        state["proposals"] = {
            proposal.record_id: proposal.model_dump(mode="json")
            for proposal in proposals
        }
        workspace.save(state)
        await search._finish_round(state, proposals)
        assert calls == {"author": 2, "evaluation": 1}
        authored_versions = [
            version
            for version in state["versions"].values()
            if version["proposal_id"] is not None
        ]
        assert len(authored_versions) == 2
        assert len({version["bundle_id"] for version in authored_versions}) == 1
        data = workspace.state_path.read_bytes()
        original = json.loads(data)
        await search._finish_round(state, proposals)
        assert calls == {"author": 2, "evaluation": 1}
        assert state == original and workspace.state_path.read_bytes() == data
