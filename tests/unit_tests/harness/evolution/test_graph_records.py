"""Published references stay stable; comparison uses actual observations."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.contracts import (
    BundleVersion,
    GraphAnalysisReceipt,
    GraphAnalysisResult,
    GraphSnapshot,
)
from reweave.evolution.evidence.trajectory import build_graph
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from reweave.evolution.graph.projection import project_map
from reweave.evolution.search.matrix import (
    best_version,
    evaluation_cells,
    result_matrix,
)
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.fixtures import protocol, usage
from tests.unit_tests.harness.evolution.test_semantic_graph import case
from tests.unit_tests.harness.evolution.test_task_graph import episode


def test_saved_graph_id_and_node_numbers_survive_reads_without_new_allocation(tmp_path):
    observed = episode(tmp_path, 1, (0.2, 0.05, 0.2))
    graph = build_graph([observed], protocol().graph, record_id="graph:published-1")
    encoded = graph.model_dump_json()
    first = GraphSnapshot.model_validate_json(encoded)
    second = GraphSnapshot.model_validate_json(encoded)
    assert first == second == graph
    assert first.record_id == "graph:published-1"
    assert first.paths[observed.receipt_id][0] == first.paths[observed.receipt_id][2]
    assert set(first.nodes) == {"node:001", "node:002"}
    payload = json.loads(encoded)
    del payload["record_id"]
    with pytest.raises(ValidationError, match="record_id"):
        GraphSnapshot.model_validate(payload)


def test_current_execution_reference_does_not_replace_actual_condition_equality(
    tmp_path,
):
    request, _, _ = case(tmp_path)
    original = request.evidence[0]
    different = original.model_copy(
        update={
            "request": original.request.model_copy(
                update={
                    "conditions": original.request.conditions.model_copy(
                        update={"wall_time_s": None}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="different actual execution conditions"):
        evaluation_cells(request.protocol, (original.request.bundle_id,), (different,))


def test_new_request_id_cannot_hide_a_duplicate_settled_matrix_cell(tmp_path):
    request, _, _ = case(tmp_path)
    first = request.evidence[0]
    second = first.model_copy(
        update={
            "record_id": "evaluation:second",
            "request": first.request.model_copy(update={"record_id": "request:second"}),
        }
    )
    with pytest.raises(ValueError, match="duplicate settled matrix cell"):
        evaluation_cells(request.protocol, (first.request.bundle_id,), (first, second))


@pytest.mark.parametrize("identities", [("a-first", "z-last"), ("z-first", "a-last")])
def test_infra_only_matrix_cell_uses_attempt_order_instead_of_random_ids(
    tmp_path, identities
):
    request, _, _ = case(tmp_path)
    original = request.evidence[0]
    attempts = tuple(
        original.model_copy(
            update={
                "record_id": identity,
                "official_success": False,
                "failure_origin": "infrastructure",
                "infrastructure_retryable": True,
            }
        )
        for identity in identities
    )
    cells = evaluation_cells(request.protocol, (original.request.bundle_id,), attempts)
    assert (
        cells[(original.request.bundle_id, original.request.instance.record_id)]
        is attempts[-1]
    )


def test_official_best_version_uses_confirmed_success_coverage_and_recency():
    versions = {
        name: BundleVersion(
            record_id=name, bundle_id=name, proposal_id=None, branch_id=None
        )
        for name in ("old", "new", "unknown")
    }
    rows = {
        "old": (True, False, False),
        "new": (True, False, False),
        "unknown": (True, False, None),
    }
    assert best_version(tuple(versions), versions, rows) == "new"
    rows["unknown"] = (True, True, None)
    assert best_version(tuple(versions), versions, rows) == "unknown"
    rows["old"] = (True, True, False)
    assert best_version(tuple(versions), versions, rows) == "old"


@pytest.mark.parametrize("origin", ["unknown", "infrastructure"])
def test_non_policy_outcomes_keep_official_matrix_cell_unknown(tmp_path, origin):
    request, _, _ = case(tmp_path)
    original = request.evidence[0]
    missing = original.model_copy(
        update={
            "official_success": False,
            "failure_origin": origin,
            "infrastructure_retryable": False if origin == "infrastructure" else None,
        }
    )
    row = result_matrix(request.protocol, (missing.request.bundle_id,), (missing,))
    assert row[missing.request.bundle_id] == (None,)


def test_pure_semantic_value_uses_its_published_receipt_reference(tmp_path):
    request, raw, graph = case(tmp_path)
    result = project_map(
        raw,
        graph,
        request.versions,
        request.protocol.dev,
        graph_id="analysis:published-1/analysis/graph",
    )
    assert result["graph_id"] == "analysis:published-1/analysis/graph"
    assert "record_id" not in graph.model_dump()
    assert "sha256" not in graph.model_dump()


def test_analysis_example_has_explicit_illustrative_original_metadata():
    path = (
        Path(__file__).resolve().parents[4]
        / "examples/graph_evolution/analysis_result.json"
    )
    result = GraphAnalysisResult.model_validate_json(path.read_text())
    originals = result.graph.original_references()
    assert originals and {ref.artifact_id for ref in originals} == {
        "illustrative-execution-trace"
    }
    assert all(
        ref.bytes == 0 and ref.path.startswith("/absolute/run/") for ref in originals
    )
    assert "Illustrative" in result.changes


def test_saved_analysis_request_binds_actual_cache_inputs(tmp_path):
    request, _, graph = case(tmp_path)
    request = request.model_copy(update={"policy": {"enabled": True}})
    receipt = GraphAnalysisReceipt(
        record_id="analysis-receipt:completed",
        request_id=request.record_id,
        analysis=GraphAnalysisResult(
            graph=graph,
            changes="Synthetic completed publication.",
            forecast={
                "outcomes": [{"progress": 50.0, "probability": 1.0}],
                "confidence": 0.8,
                "rationale": "Synthetic fixture.",
            },
        ),
        session_id="synthetic-session",
        usage=usage(),
        artifacts=(request.graph,),
    )
    directory = tmp_path / "completed"
    atomic_json(
        directory / "audit/inputs/analysis_request.json",
        request.model_dump(mode="json"),
    )
    atomic_json(directory / "result.json", receipt.model_dump(mode="json"))
    before = (directory / "result.json").read_bytes()
    analyzer = NativeGraphAnalysis(
        evidence_workspace=tmp_path / "navigation",
        bundle_loader=lambda _: pytest.fail("cache reads must not load a Bundle"),
        policy=AuthorPolicy(),
    )
    assert analyzer.verify_result(request, directory) == receipt
    changed = request.model_copy(
        update={
            "protocol": request.protocol.model_copy(
                update={
                    "execution": request.protocol.execution.model_copy(
                        update={"model_requests": 21}
                    )
                }
            )
        }
    )
    assert changed.record_id == request.record_id
    with pytest.raises(ValueError, match="differs from its saved request"):
        analyzer.verify_result(changed, directory)
    changed_boolean = request.model_copy(update={"policy": {"enabled": 1}})
    assert changed_boolean == request  # Python equality conflates True and 1.
    with pytest.raises(ValueError, match="differs from its saved request"):
        analyzer.verify_result(changed_boolean, directory)
    assert (directory / "result.json").read_bytes() == before


def test_independent_repetitions_keep_all_receipts_and_select_latest_frontier(tmp_path):
    request, _, _ = case(tmp_path)
    first = request.evidence[0]
    first = first.model_copy(
        update={"official_success": True, "failure_origin": "none"}
    )
    second = first.model_copy(
        update={
            "record_id": "trial-second",
            "official_success": False,
            "failure_origin": "agent",
            "request": first.request.model_copy(
                update={
                    "record_id": "request-second",
                    "conditions": first.request.conditions.model_copy(
                        update={"repetition": 1}
                    ),
                }
            ),
        }
    )
    cells = evaluation_cells(
        request.protocol, (first.request.bundle_id,), (first, second)
    )
    assert cells[(first.request.bundle_id, first.request.instance.record_id)] == second
    missing = second.model_copy(
        update={
            "record_id": "trial-third",
            "official_success": None,
            "failure_origin": "infrastructure",
            "infrastructure_retryable": False,
            "request": second.request.model_copy(
                update={
                    "record_id": "request-third",
                    "conditions": second.request.conditions.model_copy(
                        update={"repetition": 2}
                    ),
                }
            ),
        }
    )
    cells = evaluation_cells(
        request.protocol, (first.request.bundle_id,), (first, second, missing)
    )
    assert cells[(first.request.bundle_id, first.request.instance.record_id)] == missing
    duplicate = second.model_copy(
        update={
            "record_id": "duplicate",
            "request": second.request.model_copy(
                update={"record_id": "duplicate-request"}
            ),
        }
    )
    with pytest.raises(ValueError, match="duplicate settled matrix cell"):
        evaluation_cells(
            request.protocol, (first.request.bundle_id,), (first, second, duplicate)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("physical_steps", 99),
        ("wait_steps", 11),
        ("model_requests", 21),
        ("controller_request_unit", "api_attempt"),
        ("adaptation_mode", "adaptive"),
        ("wall_time_s", None),
        ("program_calls", 21),
        ("gateway_calls", 21),
        ("model_sampling", {"reasoning_effort": "low"}),
    ],
)
def test_repetition_identity_never_relaxes_any_execution_setting(field, value):
    from reweave.evolution.contracts import same_execution_settings
    from tests.unit_tests.harness.evolution.fixtures import protocol

    settings = protocol().execution
    repeated = settings.model_copy(update={"repetition": 8})
    assert same_execution_settings(repeated, settings)
    assert not same_execution_settings(
        repeated.model_copy(update={field: value}), settings
    )
