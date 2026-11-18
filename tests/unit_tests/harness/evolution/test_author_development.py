# Modified for the ReWeave namespace.
"""Real author boundary, authenticated development tools and production workers."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

from reweave.evolution.authoring.tools import (
    AuthorDevelopmentTools,
    DevelopmentToolError,
    call_development_tool,
)
from reweave.evolution.authoring.workspace import native_author_launcher
from reweave.evolution.evidence.records import artifact_record
from reweave.evolution.validation.admission import admit_candidate
from tests.unit_tests.harness.evolution.fixtures import file_metadata
from tests.unit_tests.harness.evolution.test_bundle_author import host_input, workspace


@pytest.fixture
def development(tmp_path):
    root = workspace(tmp_path)
    service = AuthorDevelopmentTools(root, tmp_path / "audit/development.jsonl")
    service.server.start()
    try:
        yield root, service
    finally:
        service.server.stop(timeout_s=None)


def test_development_cli_remains_bound_after_staging_directory_moves(development):
    from reweave.evolution.authoring.client import development_client

    root, service = development
    staging = root / ".scratch/staging"
    staging.mkdir()
    (staging / "dev.py").write_text(development_client())
    (staging / "host_client.py").write_bytes(
        (root / "context/host_client.py").read_bytes()
    )
    settled = root / ".scratch/settled"
    staging.rename(settled)
    result = subprocess.run(
        [sys.executable, str(settled / "dev.py"), "validate"],
        cwd=root,
        env={**os.environ, "REWEAVE_AUTHOR_TOOLS_URL": service.server.url},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["structurally_valid"]


def test_real_isolated_author_can_validate_and_compile_without_polluting_candidate(
    development, tmp_path
):
    root, service = development
    # Use the real launch boundary, including the Linux chmod restriction. The
    # standard-library-only CLI contacts the Host's unchanged program sandbox.
    source = root / "context/host_client.py"
    (root / "context/ENGINEERING.md").write_text("Read-only engineering reference")
    launcher, env = native_author_launcher(
        root,
        root / "candidate",
        tmp_path / "control",
        backend="claude",
        cli=str(Path(sys.executable).resolve()),
    )
    code = """
import os, pathlib, py_compile, runpy, sys
root = pathlib.Path(sys.argv[1])
py_compile.compile(str(root/'candidate/task-family/context/src/context.py'), doraise=True)
assert not list((root/'candidate').rglob('*.pyc'))
assert list((root/'.scratch/runtime/pycache').rglob('*.pyc'))
client = runpy.run_path(sys.argv[2])['call_development_tool']
assert client(sys.argv[3], 'validate_candidate', {'phase':'environment'})['environment_ready']
from context.dev import call, DevelopmentToolError
assert issubclass(DevelopmentToolError, RuntimeError)
result = call('validate_candidate', phase='code')
assert result['structurally_valid'] and not result['plan_checked']
assert not result['physical_execution_tested']
assert result['contract_case_results'] == []
try: (root/'context/input.json').write_text('overwrite')
except PermissionError: pass
else: raise AssertionError('input write escaped')
assert (root/'context/ENGINEERING.md').read_text() == 'Read-only engineering reference'
try: (root/'context/ENGINEERING.md').write_text('overwrite')
except PermissionError: pass
else: raise AssertionError('engineering reference write escaped')
print('validated')
"""
    script = root / ".scratch/analysis/client_probe.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(code)
    result = subprocess.run(
        [launcher, str(script), str(root), str(source), service.server.url],
        cwd=root,
        env={**os.environ, **env, "REWEAVE_AUTHOR_TOOLS_URL": service.server.url},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "validated"
    assert [
        row["ok"]
        for row in map(json.loads, service.audit.read_text().splitlines())
        if row["tool"] == "validate_candidate"
    ] == [True, True]


def test_native_shell_can_use_host_venv_for_numeric_snapshot_analysis(tmp_path):
    import numpy as np

    workspace_root = tmp_path / "workspace"
    output = workspace_root / "candidate"
    output.mkdir(parents=True)
    snapshot = tmp_path / "observations"
    snapshot.mkdir()
    np.save(snapshot / "depth.npy", np.array([[0.0, 0.7], [0.8, 0.9]]))
    hidden = tmp_path / "unrelated.txt"
    hidden.write_text("outside the experiment")
    roots = tuple(
        Path(path).resolve()
        for path in (
            "/usr",
            "/bin",
            "/etc",
            "/System",
            "/dev",
            sys.prefix,
            sys.base_prefix,
        )
        if Path(path).exists()
    )
    launcher, env = native_author_launcher(
        workspace_root,
        output,
        tmp_path / "launcher",
        backend="claude",
        cli=sys.executable,
        readable_roots=(*roots, snapshot),
    )
    code = """import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
z = np.load(root/'depth.npy', allow_pickle=False)
assert int((z > 0).sum()) == 3
for path, mode in [(root/'depth.npy', 'ab'), (Path(sys.argv[2]), 'r')]:
    try: path.open(mode)
    except PermissionError: pass
    else: raise AssertionError('native boundary escaped')
print('numeric snapshot analysis verified')
"""
    result = subprocess.run(
        [launcher, "-c", code, str(snapshot), str(hidden)],
        cwd=workspace_root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "numeric snapshot analysis verified"


def test_draft_check_does_not_require_a_finished_plan_but_host_admission_does(
    development,
):
    root, service = development
    (root / "candidate/plan.json").write_text("{}")
    result = call_development_tool(
        service.server.url,
        "validate_candidate",
        {},
    )
    assert result["structurally_valid"] and not result["plan_checked"]
    with pytest.raises(ValueError, match="diagnosis"):
        asyncio.run(admit_candidate(root))


def test_model_gets_compact_validation_receipt_and_complete_report(development):
    root, service = development
    marker = "complete-validation-evidence-" * 20000

    async def synthetic_validation(name, arguments, *, case_results):
        assert name == "validate_candidate"
        assert arguments == {"phase": "code"}
        case_results.append(
            {
                "scope": "L",
                "case_index": 0,
                "validation_status": "passed",
                "calls": [{"operation": "model:plan", "result": marker}],
            }
        )
        return {
            "phase": "code",
            "structurally_valid": True,
            "plan_checked": False,
            "contract_case_results": case_results,
        }

    service._call = synthetic_validation
    receipt = call_development_tool(
        service.server.url,
        "validate_candidate",
        {"phase": "code"},
    )
    assert receipt["structurally_valid"] and not receipt["plan_checked"]
    assert receipt["contract_case_results"] == [
        {
            "scope": "L",
            "case_index": 0,
            "validation_status": "passed",
        }
    ]
    report_receipt = receipt["complete_report"]
    original = service.references.original(report_receipt)
    report_path = Path(original.path)
    raw = report_path.read_bytes()
    assert len(raw) == original.bytes
    assert original.artifact_id
    assert len(raw) == receipt["report_bytes"]
    complete = json.loads(raw)
    assert (
        complete["result"]["contract_case_results"][0]["calls"][0]["result"] == marker
    )
    assert len(json.dumps(receipt)) * 100 < len(raw)


def test_model_cannot_request_a_large_complete_validation_response(development):
    _, service = development
    with pytest.raises(RuntimeError, match="Additional properties"):
        call_development_tool(
            service.server.url,
            "validate_candidate",
            {"phase": "code", "response_detail": "complete"},
        )
    assert not service.audit.exists()


def test_failed_validation_receipt_retains_complete_executed_evidence(development):
    _, service = development
    marker = "failed-case-evidence-" * 20000

    async def synthetic_failure(name, arguments, *, case_results):
        assert name == "validate_candidate"
        assert arguments == {"phase": "code"}
        case_results.append(
            {
                "scope": "interaction",
                "case_index": 3,
                "validation_status": "failed",
                "error": "semantic assertion failed",
                "calls": [{"operation": "model:plan", "result": marker}],
            }
        )
        raise ValueError("candidate interaction failed")

    service._call = synthetic_failure
    with pytest.raises(DevelopmentToolError) as failure:
        call_development_tool(
            service.server.url,
            "validate_candidate",
            {"phase": "code"},
        )
    receipt = failure.value.result["structuredContent"]
    assert receipt["error_type"] == "ValueError"
    assert receipt["error_bytes"] == len("candidate interaction failed")
    assert receipt["contract_case_results"] == [
        {
            "scope": "interaction",
            "case_index": 3,
            "validation_status": "failed",
            "error_bytes": len("semantic assertion failed"),
        }
    ]
    report_path = Path(service.references.original(receipt["complete_report"]).path)
    complete = json.loads(report_path.read_text())
    assert complete["contract_case_results"][0]["calls"][0]["result"] == marker
    assert complete["error"] == "candidate interaction failed"


def test_failed_validation_receipt_never_inlines_large_errors(development):
    _, service = development
    marker = "large-failure-body-" * 100000

    async def synthetic_failure(name, arguments, *, case_results):
        case_results.append(
            {
                "scope": "loop",
                "case_index": 7,
                "validation_status": "failed",
                "error": marker,
            }
        )
        raise ValueError(marker)

    service._call = synthetic_failure
    with pytest.raises(DevelopmentToolError) as failure:
        call_development_tool(
            service.server.url, "validate_candidate", {"phase": "code"}
        )
    receipt = failure.value.result["structuredContent"]
    encoded = json.dumps(receipt)
    assert marker not in encoded
    assert len(encoded) < 4096
    report = json.loads(
        Path(service.references.original(receipt["complete_report"]).path).read_text()
    )
    assert report["error"] == marker
    assert report["contract_case_results"][0]["error"] == marker


def test_cli_file_reports_preserve_failure_and_do_not_rerun_for_inspection(development):
    import re
    import shlex

    from reweave.evolution.authoring.prompts import engineering_documents
    from reweave.evolution.contracts import Proposal

    root, service = development
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    guide = engineering_documents(proposal, sys.executable)["VALIDATION.md"]
    output = root / ".scratch/tests/check.json"
    command = shlex.split(re.search(r"```sh\n(.*?)\n```", guide, re.DOTALL).group(1))
    env = {**os.environ, "REWEAVE_AUTHOR_TOOLS_URL": service.server.url}
    completed = subprocess.run(
        command, cwd=root, env=env, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    full = json.loads(output.read_text())
    assert summary["structurally_valid"] is True
    assert full["structuredContent"]["candidate_files"]
    assert json.loads(full["content"][0]["text"]) == full["structuredContent"]
    assert [
        row["tool"] for row in map(json.loads, service.audit.read_text().splitlines())
    ] == ["validate_candidate", "read_original"]
    (root / "candidate/task-family/context/src/extra.pyc").write_bytes(b"undeclared")
    failed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    assert failed.returncode == 1, failed.stderr
    summary = json.loads(failed.stdout)
    full = json.loads(output.read_text())
    assert summary["is_error"] is True
    assert summary["structurally_valid"] is False
    assert full["isError"] is True
    assert "undeclared/missing" in full["structuredContent"]["error"]
    assert json.loads(full["content"][0]["text"]) == full["structuredContent"]
    # Reading the saved report neither invokes the validator nor erases the
    # earlier success from the append-only Host audit.
    assert [
        row["ok"]
        for row in map(json.loads, service.audit.read_text().splitlines())
        if row["tool"] == "validate_candidate"
    ] == [True, False]


def test_no_workspace_override_or_unauthenticated_sibling_route(development):
    _, service = development
    with pytest.raises(RuntimeError, match="Additional properties"):
        call_development_tool(
            service.server.url,
            "validate_candidate",
            {"workspace": "/tmp", "phase": "code"},
        )
    wrong_url = service.server.url.rsplit("/", 2)[0] + "/wrong/"
    with pytest.raises(HTTPError) as error:
        call_development_tool(wrong_url, "validate_candidate", {})
    assert error.value.code == 404
    with pytest.raises(RuntimeError):
        call_development_tool(
            service.server.url, "validate_candidate", {"phase": "final"}
        )
    with pytest.raises(RuntimeError):
        call_development_tool(service.server.url, "simulate_candidate", {})


def test_historical_source_is_read_on_demand_without_hash_gate(tmp_path):
    root = workspace(tmp_path)
    source_root = tmp_path / "executed"
    source_root.mkdir()
    source = source_root / "tools.py"
    source.write_text("CAPABILITY = 'pull'\n")
    metadata = file_metadata(source)
    sources = {
        "a" * 64: {
            "path": str(source_root),
            "files": {"tools.py": metadata},
        }
    }
    host_input(root, "executed_sources.json").write_text(json.dumps(sources))
    service = AuthorDevelopmentTools(root, tmp_path / "audit/source.jsonl")
    result = asyncio.run(
        service._call(
            "read_source",
            {"source_id": "a" * 64, "path": "tools.py"},
            case_results=[],
        )
    )
    assert result["source"] == "CAPABILITY = 'pull'\n"
    assert result["bytes"] == len(source.read_bytes())
    source.write_text("changed\n")
    updated = asyncio.run(
        service._call(
            "read_source",
            {"source_id": "a" * 64, "path": "tools.py"},
            case_results=[],
        )
    )
    assert updated["source"] == "changed\n"


def test_edit_invalidates_code_check_without_deleting_unexpected_files(development):
    root, service = development
    passed = call_development_tool(
        service.server.url,
        "validate_candidate",
        {"phase": "code"},
    )
    extra = root / "candidate/task-family/context/src/extra.pyc"
    extra.write_bytes(b"unwanted")
    with pytest.raises(DevelopmentToolError) as failure:
        call_development_tool(
            service.server.url,
            "validate_candidate",
            {"phase": "code"},
        )
    receipt = failure.value.result["structuredContent"]
    complete = json.loads(
        Path(service.references.original(receipt["complete_report"]).path).read_text()
    )
    assert "undeclared/missing" in complete["error"]
    assert extra.read_bytes() == b"unwanted"
    assert "candidate_files" not in passed
    assert passed["bundle_id"].startswith("bundle:")


def test_author_may_draft_any_plugin_but_host_plan_must_match(development):
    root, service = development
    path = root / "candidate/task-family/loop/src/loop.py"
    path.write_text(path.read_text() + "\n# branch L edit\n")
    result = call_development_tool(
        service.server.url,
        "validate_candidate",
        {"phase": "code"},
    )
    assert result["structurally_valid"]
    with pytest.raises(ValueError, match="plan scopes"):
        asyncio.run(admit_candidate(root))


def test_original_handle_preserves_complete_business_values_and_rejects_raw_reference(
    development,
):
    root, service = development
    original = root.parent / "public_evidence.json"
    value = {"bundle_id": "a" * 64, "source": "file:001#", "path": "/recorded/value"}
    original.write_text(json.dumps(value))
    metadata = file_metadata(original)
    service.references.originals[str(original.resolve())] = metadata
    reference = service.references.reference(str(original), **metadata)
    assert (
        call_development_tool(
            service.server.url, "read_original", {"reference": reference}
        )["value"]
        == value
    )
    with pytest.raises(DevelopmentToolError):
        call_development_tool(
            service.server.url,
            "read_original",
            {"reference": {**artifact_record(original), "pointer": ""}},
        )
    with pytest.raises(DevelopmentToolError):
        call_development_tool(
            service.server.url, "read_original", {"reference": "file:999999#"}
        )


def test_image_original_handle_returns_original_bytes_without_resizing(development):
    import base64

    from PIL import Image

    from reweave.evolution.authoring.client import _request_development_tool

    root, service = development
    original = root.parent / "observation.png"
    Image.new("RGB", (7, 11), "red").save(original)
    raw = original.read_bytes()
    metadata = file_metadata(original)
    service.references.originals[str(original.resolve())] = metadata
    reference = service.references.reference(str(original), **metadata)
    result = _request_development_tool(
        service.server.url, "read_original", {"reference": reference}
    )
    media = next(item for item in result["content"] if item["type"] == "image")
    assert media["mimeType"] == "image/png"
    assert base64.b64decode(media["data"], validate=True) == raw
    assert result["structuredContent"] == {
        "reference": reference,
        "selection_complete": True,
    }


def test_preflight_failure_is_explicit_before_a_native_query(development, monkeypatch):
    import asyncio

    from reweave.evolution.authoring.author import NativeBundleAuthor
    from reweave.evolution.authoring.config import AuthorPolicy
    from reweave.evolution.bundle import AgentBundle
    from reweave.evolution.contracts import Proposal

    root, _ = development

    def denied(*args, **kwargs):
        raise PermissionError("probe cannot materialize read-only sources")

    monkeypatch.setattr(
        "reweave.runtime.sandbox.ProgramSandbox.validate_program_import", denied
    )
    proposal = Proposal.model_validate_json(
        host_input(root, "proposal.json").read_text()
    )
    parent = AgentBundle.model_validate(
        next(iter(json.loads(host_input(root, "references.json").read_text()).values()))
    )
    author = NativeBundleAuthor(
        evidence_workspace=root / "evidence",
        bundle_loader=lambda _: parent,
        policy=AuthorPolicy(),
    )
    launches = []
    with pytest.raises(RuntimeError, match="PermissionError"):
        asyncio.run(
            author._execute(
                proposal,
                root.parent,
                resume=True,
                before_launch=lambda: launches.append(1),
            )
        )
    assert launches == []
    assert not list((root.parent / "audit/queries").glob("*.json"))
    rows = [
        json.loads(line)
        for line in (root.parent / "audit/events.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["event"] == "author_preflight_failed"
    assert "PermissionError" in rows[-1]["error"]


def test_parallel_development_endpoints_keep_workspaces_and_failures_separate(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    services = []
    for index in range(4):
        branch = tmp_path / str(index)
        branch.mkdir()
        root = workspace(branch)
        services.append(
            AuthorDevelopmentTools(root, branch / "audit/development.jsonl")
        )
    (services[0].workspace / "candidate/unexpected.pyc").write_bytes(
        b"reject this branch"
    )

    def validate(service):
        try:
            return call_development_tool(
                service.server.url,
                "validate_candidate",
                {"phase": "code"},
            )
        except RuntimeError as error:
            return error

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda s: s.server.start(), services))
            assert len({s.server._port for s in services}) == 4
            results = list(pool.map(validate, services))
        assert isinstance(results[0], RuntimeError)
        assert all(r["structurally_valid"] for r in results[1:])
        assert [json.loads(s.audit.read_text())["ok"] for s in services] == [
            False,
            True,
            True,
            True,
        ]
    finally:
        for service in services:
            service.server.stop(timeout_s=None)
