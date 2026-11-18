"""Reuse a published H0 condition set without changing identities or rerunning it."""

from pathlib import Path

from reweave.evolution.contracts import AttemptState, BundleVersion, EvaluationRecord
from reweave.evolution.evidence.records import artifact_record
from reweave.utils.persistence import atomic_json


def seed_shared_baseline(workspace, result_paths):
    """Import the complete original development condition set into a fresh workspace.

    All arms must deliberately retain the bank's bundle, execution, manifest and
    development identities. The original receipt stays at its original location.
    Costs in each standalone report include this common observation; campaign
    totals must deduplicate it by record ID.
    """
    protocol = workspace.protocol
    paths = [Path(p).resolve() for p in result_paths]
    if len(paths) != len(protocol.dev) or len(set(paths)) != len(paths):
        raise ValueError(
            "shared H0 must cover every development condition exactly once"
        )
    records, attempts, sources = [], {}, []
    for index, (instance, result_path) in enumerate(zip(protocol.dev, paths), 1):
        if result_path.name != "result.json":
            raise ValueError("shared H0 must reference the original result.json")
        record = EvaluationRecord.model_validate_json(result_path.read_text())
        request = record.request
        if (
            protocol.validation is not None
            or request.bundle_id != workspace.baseline.record_id
            or request.execution != protocol.execution_id
            or request.manifest_id != protocol.manifest_id
            or request.conditions != protocol.execution
            or request.instance != instance
            or request.split != "dev"
            or record.official_success is None
            or record.failure_origin not in {"none", "agent"}
            or record.episode is None
        ):
            raise ValueError(
                "shared H0 differs from the exact settled development cell"
            )
        record.verify()
        key = f"evaluation/V000/condition-{index:03d}/attempt-01"
        attempts[key] = AttemptState(
            kind="evaluation",
            key=key,
            request=request.model_dump(mode="json"),
            status="completed",
            launches=1,
            result_id=record.record_id,
            result_directory=str(result_path.parent),
        ).model_dump(mode="json")
        records.append(record)
        sources.append(
            {
                "source": artifact_record(result_path),
                "record_id": record.record_id,
                "request_id": request.record_id,
            }
        )
    with workspace.transaction() as state:
        if state["attempts"] or state["versions"] or state["proposals"]:
            raise ValueError("shared H0 can only seed a fresh evolution workspace")
        workspace.put_bundle(workspace.baseline)
        workspace.add_version(
            state,
            BundleVersion(
                record_id="baseline",
                bundle_id=workspace.baseline.record_id,
                proposal_id=None,
                branch_id=None,
            ),
        )
        state["attempts"].update(attempts)
        workspace.save(state)
        atomic_json(
            workspace.root / "shared_h0.json",
            {
                "sources": sources,
                "physical_replay": False,
                "cost_policy": "One shared observation; charge once in campaign totals by record ID.",
            },
        )
    return tuple(records)
