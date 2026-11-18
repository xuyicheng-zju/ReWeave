"""Native execution metadata and comparison boundaries; no provider or robot calls."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.tools import AuthorDevelopmentTools
from reweave.evolution.bundle import ContextPackage, new_record_id
from reweave.evolution.contracts import (
    BudgetRevision,
    BundleVersion,
    EvaluationConditions,
    EvaluationRecord,
    EvaluationRequest,
)
from reweave.evolution.runtime_manifest import (
    RuntimeManifest,
    capture_baseline,
    validate_conditions,
)
from reweave.evolution.search.matrix import evaluation_cells, result_matrix
from reweave.evolution.search.report import publish_report
from tests.unit_tests.harness.evolution.fixtures import (
    artifact,
    baseline,
    protocol,
    usage,
)
from tests.unit_tests.harness.evolution.test_bundle_author import workspace
from tests.unit_tests.harness.evolution.test_runtime_manifest import fixture


@pytest.mark.parametrize("pending_success", [False, True])
def test_partial_round_report_retains_uncommitted_receipts_without_qualification(
    tmp_path, pending_success
):
    current = protocol()
    parent = baseline().record_id
    records = []
    for bundle, success in [(parent, False), ("pending-candidate", pending_success)]:
        records.append(
            EvaluationRecord(
                record_id=new_record_id(),
                request=EvaluationRequest(
                    record_id=new_record_id(),
                    bundle_id=bundle,
                    instance=current.dev[0],
                    split="dev",
                    manifest_id=current.manifest_id,
                    execution=current.execution_id,
                    conditions=current.execution,
                ),
                official_success=success,
                failure_origin="none" if success else "agent",
                infrastructure_retryable=None,
                usage=usage(),
                evidence=(artifact(tmp_path, bundle + ".json", {}),),
                episode=None,
                initialization=current.dev[0].initialization,
            )
        )
    version = BundleVersion(
        record_id="baseline", bundle_id=parent, proposal_id=None, branch_id=None
    )
    state = {
        "baseline": parent,
        "order": ["baseline"],
        "versions": {"baseline": version.model_dump(mode="json")},
        "rounds": {},
        "proposals": {},
        "attempts": {},
        "completed": False,
        "stop_reason": None,
    }
    config = tmp_path / "configuration.json"
    config.write_text("{}")
    workspace = SimpleNamespace(
        protocol=current,
        root=tmp_path,
        execution_ids=lambda _: frozenset({current.execution_id}),
        require_configuration=lambda: config,
    )
    publish_report(workspace, state, records, {})
    report = json.loads((tmp_path / "progress.json").read_text())
    assert report["matrix"] == {parent: [False]}
    assert report["best_version"] == "baseline"
    assert report["target_reached"] is False
    assert report["uncommitted_evaluation_ids"] == [records[1].record_id]
    assert [r["record_id"] for r in report["evaluations"]] == [
        r.record_id for r in records
    ]


class NativePlannerMetadata:
    """The production metadata interface with no API model or native client."""

    def runtime_configuration(self):
        return {
            "provider": "codex",
            "model": "gpt-5.4",
            "effort": "high",
            "executable_version": "synthetic CPU fixture",
        }

    def sampling_settings(self):
        return {"model": "gpt-5.4", "effort": "high"}

    def baseline_context(self):
        return ContextPackage(instructions="Native baseline current-observation view")


def native_inputs(tmp_path, *, mode="static"):
    values = fixture(tmp_path)
    planner = NativePlannerMetadata()
    values.update(
        planner=planner,
        arguments={
            **values["arguments"],
            "planner": "native",
            "native_provider": "codex",
            "adaptation_mode": mode,
        },
        conditions=values["conditions"].model_copy(
            update={
                "controller_request_unit": "native_turn",
                "adaptation_mode": mode,
                "model_sampling": planner.sampling_settings(),
            }
        ),
    )
    return values


@pytest.mark.parametrize("field", ["controller_request_unit", "adaptation_mode"])
def test_execution_semantics_are_required_and_not_inferred(field):
    value = protocol().execution.model_dump(mode="json")
    value.pop(field)
    with pytest.raises(ValidationError, match="Field required"):
        EvaluationConditions.model_validate(value)


@pytest.mark.parametrize("mode", ["static", "adaptive"])
def test_native_capture_uses_actual_exposed_metadata_and_native_c(tmp_path, mode):
    values = native_inputs(tmp_path, mode=mode)
    destination = tmp_path / "baseline"
    bundle = capture_baseline(destination, **values)
    manifest = RuntimeManifest.model_validate_json(
        (destination / "runtime.json").read_text()
    )
    assert bundle.context == values["planner"].baseline_context()
    assert manifest.models["backend"] == "native"
    assert manifest.models["controller"] == values["planner"].runtime_configuration()
    assert manifest.models["settings"] == values["planner"].sampling_settings()
    assert "max_tokens" not in manifest.models["settings"]
    assert manifest.conditions.controller_request_unit == "native_turn"
    assert manifest.conditions.adaptation_mode == mode
    manifest.validate_conditions(
        arguments=values["arguments"],
        limits=values["limits"],
        planner=values["planner"],
    )
    assert values["toolkit"].calls == 0


@pytest.mark.parametrize(
    "update,match",
    [
        ({"controller_request_unit": "api_attempt"}, "request unit"),
        ({"adaptation_mode": "adaptive"}, "adaptation mode"),
        ({"model_sampling": {"max_tokens": 123}}, "model sampling"),
    ],
)
def test_native_capture_rejects_api_units_or_invented_settings(tmp_path, update, match):
    values = native_inputs(tmp_path)
    with pytest.raises(ValueError, match=match):
        validate_conditions(
            values["conditions"].model_copy(update=update),
            values["arguments"],
            values["limits"],
            values["planner"],
        )


@pytest.mark.parametrize(
    "field,value",
    [("controller_request_unit", "native_turn"), ("adaptation_mode", "adaptive")],
)
def test_request_unit_and_adaptation_partition_current_results_and_reports(
    tmp_path, field, value
):
    old = protocol()
    old = old.model_copy(
        update={
            "execution": old.execution.model_copy(
                update={
                    "controller_request_unit": "api_attempt",
                    "adaptation_mode": "static",
                }
            )
        }
    )
    current = old.model_copy(
        update={
            "record_id": "native-protocol",
            "execution_id": "native-execution",
            "execution": old.execution.model_copy(update={field: value}),
        }
    )
    assert old.execution_id != current.execution_id
    with pytest.raises(ValidationError, match="budget revision may only"):
        BudgetRevision(previous=old, current=current)
    left, right = baseline().record_id, "b" * 64

    def observation(bundle, execution, success, name):
        return EvaluationRecord(
            record_id=new_record_id(),
            request=EvaluationRequest(
                record_id=new_record_id(),
                bundle_id=bundle,
                instance=current.dev[0],
                split="dev",
                manifest_id=current.manifest_id,
                execution=execution,
                conditions=old.execution
                if execution == old.execution_id
                else current.execution,
            ),
            official_success=success,
            failure_origin="none" if success else "agent",
            infrastructure_retryable=None,
            usage=usage(),
            evidence=(artifact(tmp_path, name, {"synthetic": True}),),
            episode=None,
            initialization=current.dev[0].initialization,
        )

    records = (
        observation(left, old.execution_id, True, "old.json"),
        observation(right, current.execution_id, False, "current.json"),
    )
    executions = frozenset({old.execution_id, current.execution_id})
    cells = evaluation_cells(current, (left, right), records, execution_ids=executions)
    assert (left, current.dev[0].record_id) not in cells
    assert cells[(right, current.dev[0].record_id)] == records[1]
    assert result_matrix(current, (left, right), records, execution_ids=executions) == {
        left: (None,),
        right: (False,),
    }
    versions = [
        BundleVersion(
            record_id=new_record_id(),
            bundle_id=bundle,
            proposal_id=None,
            branch_id=None,
        )
        for bundle in (left, right)
    ]
    state = {
        "baseline": left,
        "order": [version.record_id for version in versions],
        "versions": {
            version.record_id: version.model_dump(mode="json") for version in versions
        },
        "rounds": {},
        "proposals": {},
        "attempts": {},
        "completed": False,
        "stop_reason": None,
    }
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}")
    context = SimpleNamespace(
        protocol=current,
        root=tmp_path,
        execution_ids=lambda _: executions,
        require_configuration=lambda: configuration,
    )
    publish_report(context, state, records, {})
    report = json.loads((tmp_path / "progress.json").read_text())
    assert report["execution"][field] == value
    assert report["execution_id"] == current.execution_id
    before, after = (report["scores"][v.record_id] for v in versions)
    assert before["success_rate"] is None
    assert after["success_rate"] == 0.0
    assert after["comparable_to_baseline"] is False
    assert "baseline_delta" not in after


def test_native_author_catalog_has_no_removed_api_counterfactual_probe(tmp_path):
    tools = AuthorDevelopmentTools(
        workspace(tmp_path), tmp_path / "audit/development.jsonl"
    )
    names = {tool.name for tool in asyncio.run(tools.catalog())}
    assert "probe_planner" not in names
