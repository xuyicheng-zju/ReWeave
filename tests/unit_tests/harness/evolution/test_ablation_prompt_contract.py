"""Rendered prompts, schemas and the real native OS boundary share scope authority."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from reweave.evolution.authoring.author import NativeBundleAuthor
from reweave.evolution.authoring.candidate import (
    author_plan_schema,
    author_write_roots,
    authoring_scope_contract,
)
from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.prompts import (
    author_instructions,
    author_method_reference,
    development_guide,
)
from reweave.evolution.authoring.remote import RemoteWorkspaceTools
from reweave.evolution.authoring.workspace import native_author_launcher
from reweave.evolution.contracts import (
    EvolutionProtocol,
    GraphAnalysisRequest,
    Proposal,
    VersionReference,
)
from reweave.evolution.graph.analysis import NativeGraphAnalysis
from tests.unit_tests.harness.evolution.fixtures import baseline, graph_proposal_fields
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request


def prepared_author(tmp_path, scopes):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    request, _ = analysis_request(evidence)
    fields = graph_proposal_fields(tmp_path / "views")
    fields["task_protocol"] = EvolutionProtocol.model_validate(
        {
            **request.protocol.model_dump(mode="json"),
            "mutation_scopes": scopes,
        }
    )
    parent = VersionReference(version_id="b" * 64, bundle_id=baseline().record_id)
    proposal = Proposal(
        **fields,
        record_id="scoped-author",
        parent=parent,
        references=(parent,),
        round_index=0,
        author_index=0,
        scopes=scopes,
        history=request.history,
        evidence=request.evidence,
    )
    author = NativeBundleAuthor(
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy(author="codex"),
        evidence_workspace=tmp_path / "archive",
    )
    directory = tmp_path / "author"
    author._prepare_inputs(proposal, directory, resume=False)
    return proposal, author, directory


@pytest.mark.parametrize("scopes", [("C",), ("W", "C")])
def test_all_materialized_author_surfaces_have_the_same_permissions(tmp_path, scopes):
    proposal, author, directory = prepared_author(tmp_path, scopes)
    root = directory / "workspace"
    context = root / "context"
    expected = authoring_scope_contract(scopes)
    assert (
        json.loads((context / "input.json").read_text())["authoring_authority"]
        == expected
    )
    runtime = json.loads((context / "runtime.schema.json").read_text())
    assert runtime["authoring_authority"] == expected
    assert (context / "DEVELOPMENT.md").read_text() == development_guide(
        proposal, sys.executable
    )
    assert (
        "scopes"
        not in json.loads((context / "plan.schema.json").read_text())["properties"]
    )
    examples = json.loads((context / "examples.json").read_text())
    for scope in ("W", "C", "L"):
        assert examples[scope]["editable"] == (scope in scopes)
        assert runtime["interfaces"][scope].startswith(
            "EDITABLE" if scope in scopes else "READ-ONLY"
        )
    instructions, request = author._prompts(proposal, root)
    assert "Editable components:" in instructions
    assert "authoring_authority" in request
    assert "ablation" not in instructions.lower()
    assert "Improve the complete parent in candidate/ toward" not in request
    before = {
        path.relative_to(context): path.read_bytes()
        for path in context.rglob("*")
        if path.is_file()
    }
    author._prepare_inputs(proposal, directory, resume=True)
    assert {
        path.relative_to(context): path.read_bytes()
        for path in context.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize("scopes", [("C",), ("W", "C")])
def test_restricted_guidance_removes_unrestricted_edit_directions(tmp_path, scopes):
    proposal, _, _ = prepared_author(tmp_path, scopes)
    from reweave.evolution.authoring.prompts import author_reference_documents

    short = author_instructions(proposal)
    long = (
        development_guide(proposal, sys.executable)
        + "\n"
        + "\n".join(author_reference_documents(proposal, sys.executable).values())
    )
    method = author_method_reference(scopes)
    for stale in (
        "Make related W/C/L changes together",
        "Choose C, L and W from the behavior you need",
        "Choose components by the limiting mechanism, without a preferred component",
        "Choose C, W, L or a coherent combination",
        "Modify the inherited W/C/L candidate",
        "Retain, revise or move the mechanism\nbetween components",
        "a consolidation assignment extracts\nreusable W/C/L",
        "Improve task reasoning, evidence and guidance (C), reusable executable work (W), and orchestration (L)",
    ):
        assert stale not in short + long
    assert "- L is read-only" in long
    if "W" not in scopes:
        assert "- W is read-only" in long
        assert "Use it when inputs, local transitions" not in long
    assert "including source, metadata and contract cases" in short
    assert "Contract cases inside read-only components are frozen too" in long
    assert "Read-only interfaces remain available" in long
    assert "{{" not in method


def test_plan_schema_leaves_component_detection_to_the_host():
    validator = Draft202012Validator(author_plan_schema())
    plan = {
        "diagnosis": "Synthetic format example",
        "summary": "Synthetic C change",
        "validation": "Synthetic only",
    }
    validator.validate(plan)
    with pytest.raises(ValidationError):
        validator.validate({**plan, "scopes": ["L"]})
    with pytest.raises(ValidationError):
        validator.validate({**plan, "cross_scope_reason": "Unneeded second report"})


def test_full_allows_c_only_and_never_requires_loop_edits(tmp_path):
    from reweave.evolution.authoring.prompts import author_reference_documents
    from tests.unit_tests.harness.evolution.fixtures import mutated

    proposal, author, directory = prepared_author(tmp_path, ("W", "C", "L"))
    candidate = mutated(baseline(), "C")
    candidate.assert_candidate((baseline(),), frozenset(proposal.scopes))
    assert candidate.loop == baseline().loop and candidate.novel_scopes(
        (baseline(),)
    ) == {"C"}
    texts = "\n".join(
        (
            *author._prompts(proposal, directory / "workspace"),
            *author_reference_documents(proposal, sys.executable).values(),
        )
    )
    assert "a change to one component is sufficient" in texts
    assert "Both C and L need reusable global source" not in texts
    disabled = proposal.task_protocol.model_copy(
        update={"success_consolidation": False}
    )
    with pytest.raises(ValueError, match="consolidation is disabled"):
        Proposal.model_validate(
            {
                **proposal.model_dump(mode="json"),
                "purpose": "success_consolidation",
                "task_protocol": disabled.model_dump(mode="json"),
                **dict.fromkeys(
                    ("graph", "execution_graph", "progress_map", "decision")
                ),
            }
        )


@pytest.mark.parametrize("scopes", [("C",), ("W", "C")])
def test_native_os_boundary_denies_read_only_component_writes(tmp_path, scopes):
    if platform.system() not in {"Darwin", "Linux"}:
        pytest.skip("Native author boundary requires Seatbelt or Landlock")
    proposal, _, directory = prepared_author(tmp_path, scopes)
    root = directory / "workspace"
    launcher, environment = native_author_launcher(
        root,
        root / "candidate",
        directory / "os-check",
        backend="codex",
        cli=sys.executable,
        readable_roots=(
            Path("/usr"),
            Path("/bin"),
            Path("/System"),
            Path(sys.prefix),
            Path(sys.base_prefix),
        ),
        writable_roots=author_write_roots(proposal, root / "candidate"),
    )
    code = """
import pathlib,sys
root=pathlib.Path(sys.argv[1])/'candidate'
(root/'task-family/context/instructions.md').write_text('permitted C edit')
(root/'plan.json').write_text('{}')
(root/'SKILL.md').write_text('Synthetic only')
if sys.argv[2]=='cw':
    (root/'task-family/workflows/permitted.txt').write_text('permitted W edit')
denied=[root/'task-family/loop/loop.json',root/'global/forbidden.txt',root/'forbidden.txt']
if sys.argv[2]=='c': denied.append(root/'task-family/workflows/forbidden.txt')
for path in denied:
    try: path.write_text('forbidden edit')
    except PermissionError: pass
    else: raise AssertionError('write authority escaped: '+str(path))
print('component_authority_verified')
"""
    result = subprocess.run(
        [launcher, "-c", code, str(root), "cw" if "W" in scopes else "c"],
        cwd=root,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "component_authority_verified" in result.stdout


@pytest.mark.parametrize("scopes", [("C",), ("W", "C")])
def test_component_grants_allow_the_production_development_preflight(tmp_path, scopes):
    from reweave.evolution.authoring.tools import AuthorDevelopmentTools

    proposal, _, directory = prepared_author(tmp_path, scopes)
    root = directory / "workspace"
    tools = AuthorDevelopmentTools(
        root, directory / "audit/development.jsonl", evidence_roots={}
    )
    tools.server.start()
    try:
        tools.preflight(
            readable=(
                Path("/usr"),
                Path("/bin"),
                Path("/System"),
                Path(sys.prefix),
                Path(sys.base_prefix),
            ),
            writable=author_write_roots(proposal, root / "candidate"),
            control=directory / "launcher/preflight",
            backend="codex",
        )
    finally:
        tools.server.stop()


@pytest.mark.parametrize("surface", ["input.json", "runtime.schema.json"])
def test_resume_rejects_mismatched_model_visible_scope_contract(tmp_path, surface):
    proposal, author, directory = prepared_author(tmp_path, ("C",))
    path = directory / "workspace/context" / surface
    value = json.loads(path.read_text())
    value.pop("authoring_authority")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="authority"):
        author._prepare_inputs(proposal, directory, resume=True)


@pytest.mark.parametrize("scopes", [("C",), ("W", "C"), ("W", "C", "L")])
@pytest.mark.parametrize(
    "mode,width", [("dynamic", 1), ("dynamic", 3), ("fixed", 2), ("fixed", 3)]
)
def test_graph_forecast_prompt_and_inputs_match_search_authority(
    tmp_path, scopes, mode, width
):
    request, _ = analysis_request(tmp_path)
    value = request.model_dump(mode="json")
    value["protocol"].update(
        mutation_scopes=scopes, width_mode=mode, max_authors_per_round=width
    )
    request = GraphAnalysisRequest.model_validate(value)
    analyzer = NativeGraphAnalysis(
        bundle_loader=lambda _: baseline(),
        policy=AuthorPolicy.model_validate(request.policy),
        evidence_workspace=tmp_path / "archive",
    )
    directory = tmp_path / "analysis"
    analyzer._prepare_inputs(request, directory, resume=False)
    instructions, _ = analyzer._prompts(request, directory / "workspace")
    public = json.loads(
        (directory / "workspace/context/analysis_request.json").read_text()
    )
    if scopes != ("W", "C", "L"):
        assert public["authoring_authority"] == authoring_scope_contract(scopes)
        assert "Authors may change the complete W/C/L Harness" not in instructions
        assert (
            "forecast must describe that permitted candidate space" in instructions
        ) == request.needs_forecast
        assert "Keep the same physical goal, topology semantics" in instructions
        assert "Your own output remains analysis/ only" in instructions
    if mode == "fixed":
        assert public["width_policy"] == {
            "mode": "fixed",
            "authors_per_round": width,
            "forecast_use": "disabled",
        }
        assert "No probability forecast is needed" in instructions
        assert "planning costs to select width" not in instructions
    assert (
        "for ONE complete child candidate after ONE independent author" in instructions
    ) == request.needs_forecast


@pytest.mark.anyio
async def test_remote_tool_description_uses_actual_component_write_paths(
    tmp_path, monkeypatch
):
    proposal, _, directory = prepared_author(tmp_path, ("C",))
    root = directory / "workspace"
    tools = RemoteWorkspaceTools(
        root, directory, (), author_write_roots(proposal, root / "candidate"), "unused"
    )

    async def development(*_):
        return {"tools": []}

    monkeypatch.setattr(tools, "development", development)
    catalog = await tools.catalog()
    description = next(
        tool["description"] for tool in catalog if tool["name"] == "remote_shell"
    )
    assert "candidate/task-family/context" in description
    assert "candidate/task-family/loop" not in description
    assert "candidate/ for author" not in description


def test_workflow_api_description_is_owned_by_w_and_still_reaches_the_planner(tmp_path):
    from reweave.evolution.bundle import AgentBundle
    from reweave.planner.native_context import baseline_context
    from reweave.planner.native_loop import NativeBundlePlanner
    from reweave.runtime.kernel import EpisodeLimits
    from tests.native_robot import Robot
    from tests.unit_tests.harness.evolution.fixtures import mutated
    from tests.unit_tests.harness.planner.test_native_loop import FakeSession

    parent = mutated(baseline(), "W")
    raw = parent.model_dump(mode="json")
    raw["workflows"][0]["description"] = "Updated W API documentation"
    changed = AgentBundle.model_validate(raw)
    assert changed.novel_scopes((parent,)) == {"W"}
    with pytest.raises(ValueError, match="forbidden scopes"):
        changed.assert_candidate((parent,), frozenset({"C"}))
    changed.assert_candidate((parent,), frozenset({"C", "W"}))
    raw["context"] = baseline_context().model_dump(mode="json")
    sessions = []

    def factory(path):
        session = FakeSession(path)
        sessions.append(session)
        return session

    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=factory)
    planner.bind_bundle(
        AgentBundle.model_validate(raw),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Shared task contract",
        user_message="Inspect",
        toolkit=Robot(tmp_path),
        max_turns=3,
    )
    assert result.error is None, result.error
    assert "Updated W API documentation" in sessions[0].instructions
    assert "workflow_probe" in sessions[0].instructions


def test_cannot_rewrite_workflow_api_documentation_through_c_projection(tmp_path):
    from reweave.evolution.bundle import AgentBundle
    from reweave.planner.native_loop import NativeBundlePlanner
    from reweave.runtime.kernel import EpisodeLimits
    from tests.native_robot import Robot
    from tests.unit_tests.harness.evolution.fixtures import mutated
    from tests.unit_tests.harness.planner.test_native_loop import FakeSession

    raw = mutated(baseline(), "W").model_dump(mode="json")
    raw["context"]["program"] = {
        "entrypoint": "context:build",
        "files": [
            {
                "path": "context.py",
                "source": """def build(runtime, context):
    tools = [dict(tool) for tool in context["tools"]]
    for tool in tools:
        if tool["name"] == "workflow_probe":
            tool["description"] = "Unauthorized C rewrite of W API"
    return {"instructions": context["base_instructions"], "messages": context["messages"], "tools": tools}
""",
            }
        ],
    }
    sessions = []

    def factory(path):
        session = FakeSession(path)
        sessions.append(session)
        return session

    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=factory)
    planner.bind_bundle(
        AgentBundle.model_validate(raw),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    result = planner.solve(
        system_prompt="Shared task contract",
        user_message="Inspect",
        toolkit=Robot(tmp_path),
        max_turns=3,
    )
    assert result.error is not None
    assert "changed a tool interface" in str(result.error)
    assert not sessions[0].inputs


def test_inline_workflow_needs_no_registered_tool_and_counts_as_loop_change(tmp_path):
    from reweave.evolution.bundle import AgentBundle
    from reweave.planner.native_context import baseline_context
    from reweave.planner.native_loop import NativeBundlePlanner
    from reweave.runtime.kernel import EpisodeLimits
    from tests.native_robot import Robot
    from tests.unit_tests.harness.planner.test_native_loop import FakeSession

    parent = baseline()
    raw = parent.model_dump(mode="json")
    raw["loop"]["program"]["files"][0]["source"] = """def run(runtime, context):
    while not runtime.snapshot()["terminal"]:
        # A local workflow is inline in L; each effect retains Planner authority.
        response = runtime.plan()
        for call in response["tool_calls"]:
            runtime.execute(call)
            if runtime.snapshot()["terminal"]:
                return {"summary": "Inline workflow finished"}
"""
    inline = AgentBundle.model_validate(raw)
    assert not inline.workflows
    assert inline.novel_scopes((parent,)) == {"L"}
    with pytest.raises(ValueError, match="forbidden scopes"):
        inline.assert_candidate((parent,), frozenset({"C", "W"}))
    inline.assert_candidate((parent,), frozenset({"C", "W", "L"}))
    raw["context"] = baseline_context().model_dump(mode="json")
    planner = NativeBundlePlanner(output_dir=tmp_path, session_factory=FakeSession)
    planner.bind_bundle(
        AgentBundle.model_validate(raw),
        episode_limits=EpisodeLimits(
            wall_time_s=30, program_calls=100, gateway_calls=100
        ),
    )
    robot = Robot(tmp_path)
    result = planner.solve(
        system_prompt="Shared task contract",
        user_message="Inspect",
        toolkit=robot,
        max_turns=3,
    )
    assert result.error is None, result.error
    assert robot.calls == 1 and robot.terminal
