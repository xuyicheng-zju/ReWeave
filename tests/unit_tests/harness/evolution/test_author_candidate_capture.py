# Modified for the ReWeave namespace.
"""Development reports bind the candidate bytes actually parsed and executed."""

import asyncio
import json

import pytest

from reweave.evolution.authoring import candidate
from reweave.evolution.authoring.tools import (
    AuthorDevelopmentTools,
    DevelopmentToolError,
    call_development_tool,
)
from reweave.evolution.validation import admission
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.test_bundle_author import workspace


def mutation_version(root, marker):
    messages = [
        {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": "Inspect the scene."}],
        }
    ]
    (root / "candidate/task-family/context/src/context.py").write_text(
        "def build(runtime, context):\n"
        f'    return {{"instructions": "{marker}", '
        '"messages": context["messages"], "tools": context["tools"]}\n'
    )
    atomic_json(
        root / "candidate/task-family/context/context.json",
        {
            "contract_cases": [
                {
                    "input": {"messages": messages, "tools": []},
                    "responses": [],
                    "expected": {
                        "instructions": marker,
                        "messages": messages,
                        "tools": [],
                    },
                }
            ]
        },
    )
    atomic_json(
        root / "candidate/plan.json",
        {
            "diagnosis": f"Hypothesis {marker}; no physical evidence tested.",
            "summary": f"Set the model instructions to synthetic marker {marker}.",
            "validation": "Synthetic context and integration checks; no physical claim.",
            "scopes": ["C"],
            "cross_scope_reason": None,
        },
    )


def test_admission_parses_and_executes_only_the_captured_bytes(tmp_path, monkeypatch):
    root = workspace(tmp_path)
    mutation_version(root, "B")
    expected_inventory = candidate.output_inventory(root / "candidate")
    original_capture = candidate._capture_candidate

    def restore_after_capture(directory, inventory):
        captured = original_capture(directory, inventory)
        mutation_version(root, "A")
        # A later reread of the plan would fail even if source and fixtures both
        # accidentally switched to A and remained mutually consistent.
        atomic_json(root / "candidate/plan.json", {})
        return captured

    monkeypatch.setattr(candidate, "_capture_candidate", restore_after_capture)
    inventory, cases = {}, []
    admitted = asyncio.run(
        admission.admit_candidate(
            root, candidate_inventory=inventory, case_results=cases
        )
    )
    assert inventory == expected_inventory
    assert inventory != candidate.output_inventory(root / "candidate")
    assert '"B"' in admitted.context.program.files[0].source
    assert cases[0]["result"] == {
        "instructions": "B",
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {"part_kind": "user-prompt", "content": "Inspect the scene."}
                ],
            }
        ],
        "tools": [],
    }
    assert cases[0]["validation_status"] == "passed"


def test_development_rejects_a_b_a_inventory_binding(tmp_path, monkeypatch):
    root = workspace(tmp_path)

    def write_version(marker):
        mutation_version(root, marker)

    write_version("A")
    before = candidate.output_inventory(root / "candidate")
    original_capture = candidate._capture_candidate
    captured_inventories = []

    def capture_other_version(directory, inventory):
        write_version("B")
        try:
            captured = original_capture(directory, inventory)
            captured_inventories.append(dict(inventory))
            return captured
        finally:
            # Restore A before any captured file is parsed or any case runs.
            write_version("A")

    monkeypatch.setattr(candidate, "_capture_candidate", capture_other_version)
    service = AuthorDevelopmentTools(root, tmp_path / "audit/development.jsonl")
    service.server.start()
    try:
        with pytest.raises(DevelopmentToolError) as error:
            call_development_tool(
                service.server.url,
                "validate_candidate",
                {"phase": "code"},
            )
        receipt = error.value.result["structuredContent"]
        report = call_development_tool(
            service.server.url,
            "read_original",
            {"reference": receipt["complete_report"]},
        )["value"]
        assert receipt["ok"] is report["ok"] is False
        assert "candidate changed during validation" in report["error"]
        assert "structurally_valid" not in receipt
        assert len(captured_inventories) == 1
        assert captured_inventories[0] != before
        assert candidate.output_inventory(root / "candidate") == before
        assert report["contract_case_results"][0]["result"] == {
            "instructions": "B",
            "messages": [
                {
                    "kind": "request",
                    "parts": [
                        {
                            "part_kind": "user-prompt",
                            "content": "Inspect the scene.",
                        }
                    ],
                }
            ],
            "tools": [],
        }
        assert report["contract_case_results"][0]["validation_status"] == "passed"
        audit = next(
            row
            for row in map(json.loads, service.audit.read_text().splitlines())
            if row["tool"] == "validate_candidate"
        )
        assert audit["ok"] is False
    finally:
        service.server.stop(timeout_s=None)


def test_development_rejects_a_b_a_empty_workflow_directory(tmp_path, monkeypatch):
    root = workspace(tmp_path)
    extra = root / "candidate/task-family/workflows/empty_workflow"
    extra.mkdir()
    before = candidate.output_inventory(root / "candidate")
    assert before["task-family/workflows/empty_workflow"] == {"directory": True}
    original_capture = candidate._capture_candidate
    captured_inventories = []

    def capture_without_extra_directory(directory, inventory):
        extra.rmdir()
        try:
            captured = original_capture(directory, inventory)
            captured_inventories.append(dict(inventory))
            return captured
        finally:
            extra.mkdir()

    monkeypatch.setattr(
        candidate, "_capture_candidate", capture_without_extra_directory
    )
    service = AuthorDevelopmentTools(root, tmp_path / "audit/development.jsonl")
    service.server.start()
    try:
        with pytest.raises(DevelopmentToolError) as error:
            call_development_tool(
                service.server.url,
                "validate_candidate",
                {"phase": "code"},
            )
        receipt = error.value.result["structuredContent"]
        report = call_development_tool(
            service.server.url,
            "read_original",
            {"reference": receipt["complete_report"]},
        )["value"]
        assert "candidate changed during validation" in report["error"]
        assert len(captured_inventories) == 1
        assert "task-family/workflows/empty_workflow" not in captured_inventories[0]
        assert candidate.output_inventory(root / "candidate") == before
    finally:
        service.server.stop(timeout_s=None)
