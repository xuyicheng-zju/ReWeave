# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""The sole editable W/C/L candidate: layout, captured bytes and declared changes."""

from __future__ import annotations

import base64
import json
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from reweave.evolution.bundle import (
    AgentBundle,
    Asset,
    ContextPackage,
    LoopPolicy,
    PythonProgram,
    RuntimeConfiguration,
    Scope,
    WorkflowSpec,
    new_record_id,
)
from reweave.evolution.contracts import OriginalReference, Proposal
from reweave.memory.experience import EXPERIENCE_SCOPES
from reweave.runtime.sandbox import ProgramSandbox
from reweave.runtime.source import validate_program_source
from reweave.utils.persistence import atomic_json

# Trial write roots; shared components are reused from the global scope.
SCOPE_DIRECTORIES = {
    "W": "task-family/workflows",
    "C": "task-family/context",
    "L": "task-family/loop",
}


def authoring_scope_contract(scopes: tuple[Scope, ...]) -> dict[str, Any]:
    """One model-facing description of the Host's component write authority."""
    return {
        "editable_components": list(scopes),
        "read_only_components": [
            scope for scope in ("W", "C", "L") if scope not in scopes
        ],
        "interface_ownership": {
            "workflow_api_and_documentation": "W",
            "inline_workflow_in_loop_source": "L",
        },
        "candidate_write_paths": [SCOPE_DIRECTORIES[scope] + "/" for scope in scopes]
        + ["plan.json", "SKILL.md"],
        "read_only_use": "Inspect all deployed interfaces and the admitted run history; implement changes only in the editable components. Read authority never expands write authority.",
    }


def author_write_roots(proposal: Proposal, output: Path) -> tuple[Path, ...]:
    """The native CLI, remote shell and development probes share these OS grants."""
    if proposal.scopes == ("W", "C", "L"):
        return (output / "task-family", output / "plan.json", output / "SKILL.md") + (
            (output / "global",) if proposal.purpose == "success_consolidation" else ()
        )
    return tuple(output / SCOPE_DIRECTORIES[scope] for scope in proposal.scopes) + (
        output / "plan.json",
        output / "SKILL.md",
    )


def host_input(workspace: Path, name: str) -> Path:
    """Return an author-inaccessible Host input bound to this workspace."""
    return workspace.parent / "audit/inputs" / name


def staged_host_input(workspace: Path, name: str) -> Path:
    """Return a private input staged before the workspace becomes visible."""
    return workspace / ".host_inputs" / name


def read_proposal(workspace: Path) -> Proposal:
    from reweave.evolution.evidence.archive_inputs import archived_proposal

    return archived_proposal(
        json.loads(host_input(workspace, "proposal.json").read_text())
    )


def read_references(workspace: Path) -> tuple[AgentBundle, ...]:
    """Read the complete reference set, including this author's inherited parent."""
    proposal = read_proposal(workspace)
    values = json.loads(host_input(workspace, "references.json").read_text())
    if set(values) != {r.bundle_id for r in proposal.references}:
        raise ValueError("reference inventory differs from the proposal")
    references = tuple(
        AgentBundle.model_validate(values[r.bundle_id]) for r in proposal.references
    )
    if any(
        bundle.record_id != ref.bundle_id
        for bundle, ref in zip(references, proposal.references)
    ):
        raise ValueError("reference Harness content changed")
    return references


def candidate_root(workspace: Path) -> Path:
    """Return the sole candidate owned by this mutation workspace."""
    root = workspace / "candidate"
    if root.is_symlink():
        raise ValueError("candidate path must not be a symlink")
    return root


class TraceExpectation(Asset):
    """An observable signal, not proof of an uninstrumented internal branch."""

    kind: str = Field(
        min_length=1,
        description="Exact public bundle event kind, e.g. workflow_started or operation_settled.",
    )
    fields: dict[str, JsonValue] = Field(
        min_length=1,
        description="Nonempty recursive subset of event fields that must match. Use public tool/Workflow names, arguments and results; no private evaluator facts.",
    )


class MutationProbe(Asset):
    """A falsifiable author claim with independently observed trace signals."""

    claim: str = Field(min_length=1)
    activation: TraceExpectation = Field(
        description="Observable trigger/invocation expected when this intervention is exposed. Explain limitations in claim; an invocation alone cannot prove an internal branch ran.",
    )
    effect: TraceExpectation | None = Field(
        default=None,
        description="Optional subsequent observable local effect in the same episode. Matching does not establish causal improvement or official success.",
    )


class TopologyFeedback(Asset):
    """An author's assessment of one claim in the supplied task model."""

    target: str = Field(
        min_length=1, description="Graph node/edge ID or precise claim being assessed."
    )
    stance: Literal["agree", "modify", "refuse"]
    reason: str = Field(
        min_length=1,
        description="Why this claim is supported, needs revision (state the replacement), or should not guide this candidate. Author assessment, not a measured verdict.",
    )
    evidence: tuple[OriginalReference, ...] = Field(
        default=(),
        description="Issued original references actually inspected; leave empty when unavailable and state that limitation or local check in reason.",
    )

    @field_validator("target", "reason")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("topology feedback target and reason must be nonblank")
        return value


class AuthorPlan(Asset):
    """Concise edit record, with optional evidence-backed task-model feedback."""

    diagnosis: str = Field(
        min_length=1,
        description="Concise explanation of the observed problem and the evidence motivating this edit; distinguish uncertainty from observed facts.",
    )
    summary: str = Field(
        min_length=1,
        description="Actual candidate modification linked to this session's diagnosis.",
    )
    validation: str = Field(
        min_length=1,
        description="Checks actually performed, observed outcomes, evidence paths and untested limitations. Plans are not completed tests.",
    )
    revisions: tuple[str, ...] = Field(
        default=(),
        description="Inherited explanations revised or withdrawn in light of actual results, with evidence and the replacement uncertainty/hypothesis. Empty if no claim needs revision. These are author assessments, not Host-certified causes.",
    )
    topology_feedback: tuple[TopologyFeedback, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
        description="Assess only material graph claims actually reviewed: agree / modify / refuse, each with its reason. Empty means unreviewed, never implicit agreement. No exhaustive graph review required.",
    )
    probes: tuple[MutationProbe, ...] = Field(
        default=(),
        description="Observable intervention probes for subsequent physical traces. Empty historical plans remain readable and explicitly unassessed. Synthetic fixture coverage never substitutes for these observations.",
    )
    scopes: tuple[Scope, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Historical optional declaration. The Host derives actual changed components from the complete candidate and enforces proposal authority.",
    )
    cross_scope_reason: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional historical explanation of coupled component changes.",
    )

    @field_validator("diagnosis", "summary", "validation")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "diagnosis, change summary and validation must be nonblank"
            )
        return value

    @model_validator(mode="after")
    def _unique_scopes(self) -> AuthorPlan:
        if self.scopes is not None and len(set(self.scopes)) != len(self.scopes):
            raise ValueError("plan scopes must be unique")
        return self


def author_plan_schema(*, topology: bool = True) -> dict[str, Any]:
    """Expose author reasoning; component authority is enforced on actual code."""
    from reweave.evolution.authoring.references import AuthorReferences

    schema = AuthorPlan.model_json_schema()
    for field in ("scopes", "cross_scope_reason"):
        del schema["properties"][field]
    if not topology:
        del schema["properties"]["topology_feedback"]
        del schema["$defs"]["TopologyFeedback"]
        del schema["$defs"]["OriginalReference"]
    else:
        # Only evidence references use issued handles. In particular, the plan's
        # prose `validation` field is not a graph table keyed by version handles.
        schema["$defs"]["OriginalReference"] = AuthorReferences.schema(
            {"$defs": {"OriginalReference": schema["$defs"]["OriginalReference"]}}
        )["$defs"]["OriginalReference"]
    return schema


def topology_feedback(summary: str) -> tuple[TopologyFeedback, ...]:
    """Read structured plan feedback without reinterpreting historical free text."""
    try:
        plan = json.loads(summary)
    except json.JSONDecodeError:
        return ()
    if not isinstance(plan, dict):
        return ()
    return tuple(
        TopologyFeedback.model_validate(item)
        for item in plan.get("topology_feedback", ())
    )


def read_author_plan(workspace: Path, data: str | bytes) -> AuthorPlan:
    """Resolve workspace handles before publishing feedback across round sessions."""
    value = json.loads(data)
    if isinstance(value, dict) and value.get("topology_feedback"):
        from reweave.evolution.authoring.references import AuthorReferences

        references = AuthorReferences.load(workspace)
        for item in value["topology_feedback"]:
            item["evidence"] = [
                references.original(ref).model_dump(mode="json")
                for ref in item.get("evidence", [])
            ]
    return AuthorPlan.model_validate(value)


def candidate_summary(workspace: Path) -> str:
    return read_author_plan(
        workspace, (candidate_root(workspace) / "plan.json").read_bytes()
    ).model_dump_json()


def candidate_experience(workspace: Path) -> str:
    """Read optional additional lessons, without inventing missing experience."""
    note = candidate_root(workspace) / "SKILL.md"
    return note.read_text() if note.exists() else ""


def _write_program(
    directory: Path, program: PythonProgram, global_directory: Path | None = None
) -> None:
    atomic_json(directory / "program.json", {"entrypoint": program.entrypoint})
    for source in program.files:
        owner = global_directory if source.layer == "global" else directory
        target = (owner or directory) / "src" / source.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.source)


def write_bundle_source(bundle: AgentBundle, directory: Path) -> None:
    """Materialize complete W/C/L source equally for any reference Harness."""
    for layer in EXPERIENCE_SCOPES:
        target = directory / layer
        if target.is_symlink():
            raise ValueError("source scope directory must not be a symlink")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        if layer != "task-specific":
            (directory / layer / "workflows").mkdir(exist_ok=True)
            (directory / layer / "context/skills").mkdir(parents=True, exist_ok=True)
    task = directory / "task-family"
    shared = directory / "global"
    for item in bundle.workflows:
        write_workflow(directory / item.layer / "workflows" / item.workflow_id, item)
    write_context(task / "context", bundle.context, shared / "context")
    write_loop(task / "loop", bundle.loop, shared / "loop")
    for skill in bundle.context.skills:
        target = (
            directory / skill.scope / "context/skills" / skill.skill_id / "SKILL.md"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(skill.text)


def initialize_candidate(directory: Path, parent: AgentBundle) -> None:
    """Materialize a complete parent copy and fresh plan/experience outputs."""
    write_bundle_source(parent, directory)
    (directory / "SKILL.md").write_text("")
    atomic_json(
        directory / "plan.json",
        {
            "diagnosis": "",
            "summary": "",
            "validation": "",
        },
    )


def write_workflow(root: Path, workflow: WorkflowSpec) -> None:
    payload = workflow.model_dump(mode="json", exclude={"program"})
    atomic_json(root / "workflow.json", payload)
    _write_program(root, workflow.program)


def write_context(root: Path, package: ContextPackage, global_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    context = package.model_dump(mode="json")
    context.pop("skills")
    instructions = context.pop("instructions")
    context.pop("program")
    atomic_json(root / "context.json", context)
    (root / "instructions.md").write_text(instructions)
    _write_program(root, package.program, global_root)
    (root / "skills").mkdir(exist_ok=True)


def write_loop(
    root: Path, package: LoopPolicy, global_root: Path | None = None
) -> None:
    _write_program(root, package.program, global_root)
    atomic_json(
        root / "loop.json",
        {
            "contract_cases": [
                case.model_dump(mode="json") for case in package.contract_cases
            ]
        },
    )


def _candidate_text(contents: dict[Path, bytes], path: Path) -> str:
    # Match read_text's universal-newline handling while parsing only captured bytes.
    return contents[path].decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _read_program(
    directory: Path,
    expected: set[Path],
    contents: dict[Path, bytes],
    global_directory: Path | None = None,
    *,
    layer: str = "task-family",
) -> dict[str, Any]:
    config = directory / "program.json"
    expected.add(config)
    manifest = json.loads(_candidate_text(contents, config))
    if not isinstance(manifest, dict) or set(manifest) != {"entrypoint"}:
        raise ValueError("program.json must contain only entrypoint")
    files = []
    for owner, owner_layer in ((directory, layer), (global_directory, "global")):
        if owner is None:
            continue
        for source in sorted(contents):
            if not source.is_relative_to(owner / "src") or source.suffix != ".py":
                continue
            expected.add(source)
            files.append(
                {
                    "path": source.relative_to(owner / "src").as_posix(),
                    "source": _candidate_text(contents, source),
                    "layer": owner_layer,
                }
            )
    files.sort(key=lambda item: item["path"])
    program = PythonProgram.model_validate({**manifest, "files": files})
    validate_program_source(program)
    return program.model_dump(mode="json")


def _capture_candidate(
    root: Path, candidate_inventory: dict[str, Any] | None
) -> tuple[dict[Path, bytes], set[Path]]:
    """Read each regular file once and bind its captured bytes, without links."""
    contents: dict[Path, bytes] = {}
    directories: set[Path] = set()
    captured_inventory = {}
    if candidate_inventory is not None:
        candidate_inventory.clear()
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            directories.add(path)
            if candidate_inventory is not None:
                captured_inventory[path.relative_to(root).as_posix()] = {
                    "directory": True
                }
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"candidate contains a link or special file: {path}")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
            ):
                raise ValueError(f"candidate changed while capturing file: {path}")
            contents[path] = stream.read()
        if candidate_inventory is not None:
            captured_inventory[path.relative_to(root).as_posix()] = {
                "bytes": len(contents[path]),
                "data_base64": base64.b64encode(contents[path]).decode("ascii"),
                "links": opened.st_nlink,
            }
    if candidate_inventory is not None:
        candidate_inventory.update(captured_inventory)
    return contents, directories


@dataclass(frozen=True)
class CandidateCapture:
    """One complete bundle and the exact candidate file bytes parsed to obtain it."""

    # Immutable semantic W/C/L with the caller's fixed runtime, never editable runtime files.
    bundle: AgentBundle
    # Candidate-relative regular files, including optional offline plan/experience outputs.
    files: Mapping[str, bytes]


def capture_bundle(
    directory: Path,
    runtime: RuntimeConfiguration,
    *,
    candidate_inventory: dict[str, Any] | None = None,
    captured_files: dict[str, bytes] | None = None,
    record_id: str | None = None,
    validated_capture: CandidateCapture | None = None,
) -> CandidateCapture:
    """Capture and validate complete W/C/L without requiring an offline Proposal.

    File links, undeclared assets and invalid programs are rejected using the
    same parser as offline author admission. The optional inventory binds the
    exact bytes parsed even when the live tree changes during validation.
    ``captured_files`` also receives these bytes before parsing, so a failed
    online edit can retain its original source without rereading the live tree.
    An episode may reuse successful import checks from its previous capture in
    this process. Complete program values (entrypoint and every source file)
    must match; file capture, layout and source validation still run every time.
    Offline admission leaves this unset and always validates imports afresh.
    """
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("candidate path must not be a symlink")
    contents, directories = _capture_candidate(root, candidate_inventory)
    if captured_files is not None:
        captured_files.clear()
        captured_files.update(
            {path.relative_to(root).as_posix(): data for path, data in contents.items()}
        )
    files = set(contents)
    if any(
        path.relative_to(root).parts[0] not in EXPERIENCE_SCOPES for path in directories
    ):
        raise ValueError(
            "candidate has only global, task-family and task-specific directories"
        )
    paths = files | directories
    for required in (
        root / "task-family/workflows",
        root / "task-family/context/skills",
        root / "global",
        root / "task-specific",
    ):
        if required not in directories:
            raise ValueError(f"candidate is missing required directory: {required}")
    expected = files & {root / "plan.json", root / "SKILL.md"}
    if root / "task-specific/evidence.json" in files:
        expected.add(root / "task-specific/evidence.json")
    workflows = []
    for layer in ("global", "task-family"):
        for directory in sorted(
            path for path in paths if path.parent == root / layer / "workflows"
        ):
            if directory not in directories:
                raise ValueError("workflows/ contains a non-workflow file")
            config = directory / "workflow.json"
            expected.add(config)
            payload = json.loads(_candidate_text(contents, config))
            if not isinstance(payload, dict) or "program" in payload:
                raise ValueError(
                    "workflow.json must contain metadata without duplicate program source"
                )
            if (
                payload["workflow_id"] != directory.name
                or payload.get("layer", layer) != layer
            ):
                raise ValueError("workflow identity/layer differs from its directory")
            workflows.append(
                {
                    **payload,
                    "layer": layer,
                    "program": _read_program(
                        directory,
                        expected,
                        contents,
                    ),
                }
            )
    workflows.sort(key=lambda item: item["workflow_id"])
    context_root = root / "task-family/context"
    expected.update(context_root / name for name in ("context.json", "instructions.md"))
    context = json.loads(_candidate_text(contents, context_root / "context.json"))
    if not isinstance(context, dict) or set(context) != {"contract_cases"}:
        raise ValueError("task-family/context/context.json must contain an object")
    skills = []
    for scope in EXPERIENCE_SCOPES:
        skill_root = root / scope / "context/skills"
        for directory in sorted(path for path in paths if path.parent == skill_root):
            if directory not in directories:
                raise ValueError("each scoped skill must have its own directory")
            path = directory / "SKILL.md"
            if path not in files:
                raise ValueError("each scoped skill must contain exactly SKILL.md")
            expected.add(path)
            skills.append(
                {
                    "skill_id": directory.name,
                    "scope": scope,
                    "text": contents[path].decode("utf-8"),
                }
            )
    skills.sort(key=lambda skill: skill["skill_id"])
    context.update(
        instructions=_candidate_text(contents, context_root / "instructions.md"),
        program=_read_program(
            context_root, expected, contents, root / "global/context"
        ),
        skills=skills,
    )
    loop_root = root / "task-family/loop"
    expected.add(loop_root / "loop.json")
    loop = json.loads(_candidate_text(contents, loop_root / "loop.json"))
    if not isinstance(loop, dict) or set(loop) != {"contract_cases"}:
        raise ValueError("loop.json must contain contract_cases")
    loop.update(
        program=_read_program(loop_root, expected, contents, root / "global/loop")
    )
    candidate = AgentBundle(
        record_id=record_id or new_record_id(),
        runtime=runtime,
        workflows=workflows,
        context=context,
        loop=loop,
    )
    if files != expected:
        raise ValueError(
            f"undeclared/missing candidate files: {sorted(str(p.relative_to(root)) for p in files ^ expected)}"
        )
    validated_programs = (
        (
            validated_capture.bundle.loop.program,
            validated_capture.bundle.context.program,
            *(item.program for item in validated_capture.bundle.workflows),
        )
        if validated_capture is not None
        else ()
    )
    sandbox = None
    for program in (
        candidate.loop.program,
        candidate.context.program,
        *(item.program for item in candidate.workflows),
    ):
        if program not in validated_programs:
            if sandbox is None:
                sandbox = ProgramSandbox()
            sandbox.validate_program_import(program, wall_time_s=30)
    return CandidateCapture(
        bundle=candidate,
        files=MappingProxyType(
            {path.relative_to(root).as_posix(): data for path, data in contents.items()}
        ),
    )


def read_candidate(
    workspace: Path,
    *,
    require_plan: bool = True,
    candidate_inventory: dict[str, Any] | None = None,
) -> AgentBundle:
    """Apply offline Proposal and plan authority to the one captured bundle."""
    references = read_references(workspace)
    proposal = read_proposal(workspace)
    captured = capture_bundle(
        candidate_root(workspace),
        references[0].runtime,
        candidate_inventory=candidate_inventory,
    )
    missing = {"plan.json"} - captured.files.keys()
    if missing:
        raise ValueError(f"undeclared/missing candidate files: {sorted(missing)}")
    candidate = captured.bundle
    candidate.assert_candidate(references, frozenset(proposal.scopes))
    parent = next(r for r in references if r.record_id == proposal.parent.bundle_id)
    if proposal.purpose != "success_consolidation":
        candidate.assert_task_family_edit(parent)
        if "task-specific/evidence.json" in captured.files:
            raise ValueError(
                "task-specific evidence is Host-owned and cannot be authored during evolution"
            )
    else:
        if not any(
            r.official_success is True and r.request.bundle_id == parent.record_id
            for r in proposal.evidence
        ):
            raise ValueError(
                "global extraction requires fixed-evaluator success of the exact parent"
            )
        if tuple(
            s for s in candidate.context.skills if s.scope == "task-specific"
        ) != tuple(s for s in parent.context.skills if s.scope == "task-specific"):
            raise ValueError("task-specific skills are Host-owned")
        candidate.require_success_layers()
        expected_evidence = [
            r.model_dump(mode="json")
            for r in proposal.evidence
            if r.official_success is True and r.request.bundle_id == parent.record_id
        ]
        if (
            json.loads(captured.files["task-specific/evidence.json"])
            != expected_evidence
        ):
            raise ValueError(
                "task-specific success evidence differs from the Host record"
            )
    if require_plan:
        plan = read_author_plan(workspace, captured.files["plan.json"])
        if plan.topology_feedback and proposal.graph is None:
            raise ValueError("topology feedback requires a supplied graph")
        if proposal.purpose != "success_consolidation" and any(
            not candidate.novel_scopes((reference,)) for reference in references
        ):
            raise ValueError("candidate must differ from every reference Harness")
        if plan.scopes is not None and candidate.novel_scopes((parent,)) != set(
            plan.scopes
        ):
            raise ValueError(
                "plan scopes do not match components changed from the parent"
            )
    return candidate


def output_inventory(root: Path) -> dict[str, Any]:
    """Bind even invalid native output without following authored links."""
    values = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        info = path.lstat()
        if path.is_symlink():
            values[name] = {"link": os.readlink(path)}
        elif path.is_file():
            data = path.read_bytes()
            values[name] = {
                "bytes": len(data),
                "data_base64": base64.b64encode(data).decode("ascii"),
                "links": info.st_nlink,
            }
        elif path.is_dir():
            values[name] = {"directory": True}
        else:
            values[name] = {"special_mode": info.st_mode}
    return values
