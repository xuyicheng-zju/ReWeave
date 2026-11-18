import asyncio
import json

from reweave.evolution.authoring.tools import AuthorDevelopmentTools
from tests.unit_tests.harness.evolution.fixtures import file_metadata
from tests.unit_tests.harness.evolution.test_bundle_author import workspace


def test_author_reads_own_correction(tmp_path):
    root = workspace(tmp_path)
    service = AuthorDevelopmentTools(root, tmp_path / "audit/development.jsonl")
    report = root.parent / "audit/corrections/000001/feedback.json"
    report.parent.mkdir(parents=True)
    value = {"error_type": "ValueError", "error": "missing interaction scenarios"}
    report.write_text(json.dumps(value))
    ref = service.references.reference(str(report), **file_metadata(report))
    result = asyncio.run(
        service.model_call("read_original", {"reference": ref}, case_results=[])
    )
    assert result["value"] == value
