# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Host admission of captured candidates through contract and interaction fixtures."""

from __future__ import annotations

import json
import os
import stat
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from reweave.evolution.authoring.candidate import (
    candidate_experience,
    candidate_summary,
    read_candidate,
    read_proposal,
    read_references,
)
from reweave.evolution.bundle import AgentBundle, NativeModelContext, ProgramLimits
from reweave.evolution.contracts import Candidate
from reweave.evolution.infrastructure.async_work import run_blocking
from reweave.runtime.model_input import unique_image_count, validate_native_request
from reweave.runtime.sandbox import ProgramSandbox


async def admit_mutation(
    workspace: Path, *, case_results=None
) -> tuple[Candidate, ...]:
    proposal = read_proposal(workspace)
    local_cases = []
    try:
        bundle = await admit_candidate(workspace, case_results=local_cases)
    finally:
        if case_results is not None:
            case_results.extend(local_cases)
    return (
        Candidate(
            branch_id=f"author_{proposal.author_index:03d}",
            bundle=bundle,
            summary=candidate_summary(workspace),
            experience=candidate_experience(workspace),
        ),
    )


async def admit_candidate(
    workspace: Path,
    *,
    require_plan: bool = True,
    check_interactions: bool = False,
    case_results: list[dict[str, Any]] | None = None,
    candidate_inventory: dict[str, Any] | None = None,
) -> AgentBundle:
    candidate = await run_blocking(
        read_candidate,
        workspace,
        require_plan=require_plan,
        candidate_inventory=candidate_inventory,
    )
    contract = json.loads((workspace / "context/gateway.json").read_text())
    await check_runtime_contracts(candidate, contract, case_results=case_results)
    from reweave.evolution.validation.context_replay import check_recorded_contexts

    await check_recorded_contexts(
        workspace,
        candidate,
        contract,
        case_results=case_results,
    )
    if require_plan or check_interactions:
        await check_interaction_suite(
            workspace,
            candidate,
            scenario_path=".scratch/tests/interaction.json",
            case_results=case_results,
        )
    return candidate


def _read_interaction_file(workspace: Path, relative: str) -> bytes:
    """Capture a regular, unlinked scenario file within the author's tests directory."""
    name = Path(relative)
    if (
        name.is_absolute()
        or ".." in name.parts
        or name.parts[:2] != (".scratch", "tests")
    ):
        raise ValueError("interaction scenarios must be files inside .scratch/tests/")
    return _read_workspace_file(workspace, relative)


def _read_workspace_file(workspace: Path, relative: str) -> bytes:
    """Read one authored regular file without following links in any component."""
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError("file must stay inside its workspace")
    directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in name.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            name.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory,
        )
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"workspace file must be regular and unlinked: {relative}")
        return stream.read()


async def check_interaction_suite(
    workspace: Path,
    candidate: AgentBundle,
    *,
    scenario_path: str,
    case_results: list[dict[str, Any]] | None = None,
    references: tuple[AgentBundle, ...] | None = None,
    execution=None,
) -> list[dict[str, Any]]:
    """Re-execute full candidate interactions and require coverage of changed assets.

    No cached passed flag is trusted. Scenarios are captured once, execute through
    the production loop and retain their complete inputs and synthetic outcomes.
    Coverage proves invocation, never physical efficacy or a correct hypothesis.
    """
    from reweave.evolution.validation.interaction import (
        InteractionFailure,
        simulate_interaction,
    )

    try:
        captured = await run_blocking(_read_interaction_file, workspace, scenario_path)
    except FileNotFoundError as error:
        raise ValueError(
            f"changed candidates require {scenario_path}: a nonempty "
            "array of full-runtime interaction scenarios for Host final admission"
        ) from error
    scenarios = json.loads(captured)
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("interaction suite must contain a nonempty array of scenarios")
    contract = json.loads((workspace / "context/gateway.json").read_text())
    if execution is None:
        execution = read_proposal(workspace).task_protocol.execution
    reports = []

    def record(report, index):
        report.update(
            case_index=index, suite_path=scenario_path, suite_bytes=len(captured)
        )
        if case_results is not None:
            case_results.append(report)

    # The native author's writable roots are inside workspace. Never let the
    # privileged validator create files beneath an author-controlled directory.
    directory = workspace.parent / "audit/interactions" / uuid.uuid4().hex
    for index, scenario in enumerate(scenarios):
        try:
            report = await run_blocking(
                simulate_interaction,
                candidate,
                contract,
                scenario,
                directory / str(index),
                controller_request_unit=execution.controller_request_unit,
                adaptation_mode=execution.adaptation_mode,
                online_workflows=execution.online_workflows,
            )
        except InteractionFailure as error:
            record(error.report, index)
            raise
        record(report, index)
        reports.append(report)
    if captured != await run_blocking(_read_interaction_file, workspace, scenario_path):
        raise ValueError(
            "interaction scenarios changed during validation; rerun settled files"
        )
    # Camera tasks cannot establish input validity with text-only authored cases.
    # Exercise actual image input through C and the production validator, not
    # textual camera labels. C remains free to select a text-only output.
    camera_count = (contract.get("native_image_contract") or {}).get("observation")
    if camera_count and not any(
        event["kind"] == "context_built"
        and unique_image_count(event["input"]["messages"]) > 0
        for report in reports
        for event in report["events"]
    ):
        raise ValueError(
            "visual input coverage missing: the interaction suite must exercise "
            "real image payloads as C input and validate its resulting model request; "
            "text-only cases cannot validate a camera task"
        )
    if references is None:
        references = read_references(workspace)
    novel = candidate.novel_scopes(references)
    loop_workflows_known = any(
        r.workflows == candidate.workflows and r.loop == candidate.loop
        for r in references
    )
    context_loop_known = any(
        r.context == candidate.context and r.loop == candidate.loop for r in references
    )
    # C can hide failure feedback even when L is unchanged. W-only additions
    # retain the tested conduit; their own full interactions are still required.
    needs_recovery = bool(novel & {"C", "L"}) or (
        "W" not in novel and not loop_workflows_known
    )
    needs_context_loop_integration = not context_loop_known and not (novel & {"C", "L"})
    if (
        needs_recovery
        and candidate.workflows
        and not any(r["scenario"]["recovery"] for r in reports)
    ):
        raise ValueError(
            "changed Context/Loop combination with Workflows requires a recovery=true interaction fixture: "
            "nonterminal Workflow failure, feedback to a subsequent Planner request, "
            "then execution of its recovery action or Workflow"
        )
    missing = []
    for scope in ("C", "L"):
        if (scope in novel or needs_context_loop_integration) and not any(
            r["coverage"][scope] for r in reports
        ):
            missing.append(scope)
    known_workflows = [w for ref in references for w in ref.workflows]
    invoked = {name for r in reports for name in r["coverage"]["workflows"]}
    for workflow in candidate.workflows:
        if workflow not in known_workflows and workflow.workflow_id not in invoked:
            missing.append("workflow:" + workflow.workflow_id)
    if missing:
        raise ValueError(
            "interaction suite did not exercise changed components: "
            + ", ".join(missing)
        )
    return reports


async def check_runtime_contracts(
    candidate: AgentBundle,
    contract: dict[str, Any],
    *,
    case_results: list[dict[str, Any]] | None = None,
) -> None:
    """Execute candidate interfaces through the production worker, using synthetic I/O.

    ``case_results`` collects complete evidence from this same validation run,
    including earlier cases and the current case when validation raises. Each
    record identifies ``scope``, its zero-based ``case_index`` and, for W only,
    ``workflow_id``. ``calls`` retains ordered fixture dispatches: C/L use
    method/name/arguments, W uses operation/arguments. A call contains its actual
    fixture ``result`` or ``error`` (type/message). The case's ``result`` is the
    complete worker return when one exists; ``validation_status`` describes the
    validator (passed/failed/interrupted), never physical success. ``error`` and
    ``unconsumed_responses`` retain the failure and pending synthetic fixtures.
    Results are snapshots, not references that authored code can later mutate.
    """
    from reweave.runtime.kernel import EpisodeKernel, EpisodeLimits

    operations = contract["operations"]
    episode_limits = EpisodeLimits.model_validate(contract["episode_limits"])

    @contextmanager
    def record_case(scope, index, responses, **identity):
        record = {
            "scope": scope,
            "case_index": index,
            **identity,
            "calls": [],
        }
        if case_results is not None:
            case_results.append(record)
        try:
            yield record
        except BaseException as error:
            record["validation_status"] = (
                "failed" if isinstance(error, Exception) else "interrupted"
            )
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        else:
            record["validation_status"] = "passed"
        finally:
            record["unconsumed_responses"] = deepcopy(responses)

    @contextmanager
    def record_call(record, actual):
        call = deepcopy(actual)
        record["calls"].append(call)
        try:
            yield call
        except BaseException as error:
            call["error"] = {"type": type(error).__name__, "message": str(error)}
            raise

    async def cancel():
        return None

    async def forbidden(name, arguments):
        raise ValueError(
            "admission loop fixture cannot directly dispatch a physical operation"
        )

    def kernel(gateway):
        return EpisodeKernel(
            candidate,
            operations=operations,
            gateway=gateway,
            cancel=cancel,
            emit=lambda event: None,
            episode_limits=episode_limits,
        )

    # Initialization validates all programs and workflow permissions. No Host-chosen
    # initial decision is imposed on independently authored orchestration code.
    kernel(forbidden)
    for scope, plugin in (("C", candidate.context), ("L", candidate.loop)):
        for case_index, case in enumerate(plugin.contract_cases):
            responses = list(case.model_dump(mode="json")["responses"])
            violations = []

            async def plugin_fixture(method, name, arguments):
                actual = {"method": method, "name": name, "arguments": arguments}
                with record_call(record, actual) as call:
                    if (
                        scope == "C"
                        and operations.get(name, {}).get("effect") != "read"
                    ):
                        violations.append(actual)
                        raise ValueError(
                            "C fixtures may only retrieve read evidence, never call models recursively"
                        )
                    if not responses:
                        violations.append(actual)
                        raise ValueError(
                            f"{scope} case {case_index}: unexpected plugin fixture operation {actual!r}"
                        )
                    expected_response = responses.pop(0)
                    if {
                        key: expected_response.get(key) for key in actual
                    } != actual or set(expected_response) != {*actual, "result"}:
                        violations.append(
                            {
                                "expected": expected_response,
                                "actual": actual,
                            }
                        )
                        raise ValueError(
                            f"{scope} case {case_index}: plugin fixture operation mismatch; expected={expected_response!r}, actual={actual!r}"
                        )
                    call["result"] = deepcopy(expected_response["result"])
                    return expected_response["result"]

            with record_case(scope, case_index, responses) as record:
                result = await ProgramSandbox().run_program(
                    plugin.program,
                    ProgramLimits(
                        runtime_calls=contract["episode_limits"]["gateway_calls"],
                        physical_calls=0,
                        model_calls=0,
                        wall_time_s=contract["episode_limits"]["wall_time_s"],
                    ),
                    case.model_dump(mode="json")["input"],
                    plugin_fixture,
                    input_schema={"type": "object"},
                )
                record["result"] = deepcopy(result)
                if scope == "C":
                    prepared = NativeModelContext.model_validate(result).model_dump(
                        mode="json"
                    )
                    validate_native_request(prepared, episode_limits.request_limits)
                if (
                    violations
                    or responses
                    or result != case.model_dump(mode="json")["expected"]
                ):
                    raise ValueError(
                        f"{scope} case {case_index}: plugin contract fixture did not match; expected={case.expected!r}, actual={result!r}, violations={violations!r}, unconsumed={responses!r}"
                    )
    for workflow in candidate.workflows:
        for case_index, case in enumerate(workflow.contract_cases):
            responses = case.model_dump(mode="json")["responses"]
            fixture_violations = []

            async def fixture(name, arguments):
                actual = {"operation": name, "arguments": arguments}
                with record_call(record, actual) as call:
                    if not responses:
                        fixture_violations.append(
                            {
                                "unexpected_operation": name,
                                "arguments": arguments,
                            }
                        )
                        raise ValueError(
                            f"W {workflow.workflow_id} case {case_index}: workflow performed an undeclared fixture operation {name!r}, arguments={arguments!r}"
                        )
                    response = responses.pop(0)
                    if (
                        set(response) != {"operation", "arguments", "result"}
                        or response["operation"] != name
                        or response["arguments"] != arguments
                    ):
                        fixture_violations.append(
                            {
                                "expected": response,
                                "actual": actual,
                            }
                        )
                        raise ValueError(
                            f"W {workflow.workflow_id} case {case_index}: workflow dispatch differs from the declared contract fixture; expected={response!r}, actual_operation={name!r}, actual_arguments={arguments!r}"
                        )
                    call["result"] = deepcopy(response["result"])
                    return response["result"]

            with record_case(
                "W", case_index, responses, workflow_id=workflow.workflow_id
            ) as record:
                runtime = kernel(fixture)
                result = await runtime.workflow(
                    workflow.workflow_id,
                    case.arguments,
                    observation=case.observation,
                    task_language="Contract fixture",
                    epoch=0,
                )
                record["result"] = result.model_dump(mode="json")
                if (
                    fixture_violations
                    or result.status != case.expected_status
                    or responses
                ):
                    raise ValueError(
                        f"W {workflow.workflow_id} case {case_index}: workflow did not satisfy its declared contract fixture; expected_status={case.expected_status!r}, actual={result.model_dump(mode='json')!r}, violations={fixture_violations!r}, unconsumed={responses!r}"
                    )
