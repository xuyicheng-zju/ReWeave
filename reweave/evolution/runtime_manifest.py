# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Record actual execution configuration without source-identity admission gates."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field

from reweave.evolution.bundle import (
    AgentBundle,
    Asset,
    RuntimeConfiguration,
    baseline_loop,
    new_record_id,
)
from reweave.evolution.contracts import EvaluationConditions, ExecutionRevision
from reweave.utils.persistence import atomic_json

# Per-instance identities, device/service routes and runner-written locations.
INSTANCE_ARGUMENTS = frozenset(
    {
        "suite",
        "task",
        "seed",
        "reset_state_id",
        "layout",
        "layout_set",
        "output_dir",
        "cuda_device",
        "vla_endpoint",
        "sam3_endpoint",
        "memory_dir",
        "agent_bundle",
        "kernel_limits",
        "runtime_manifest",
        "capture_bundle",
        "runtime_resources",
    }
)
# Options the rollout runner sets itself; a fixed recipe may not declare them.
# These additions (environment route, memory profile, interaction modes) are
# still recorded configuration.
RUNNER_OWNED_ARGUMENTS = INSTANCE_ARGUMENTS | {
    "env_endpoint",
    "memory_profile",
    "random",
    "explore",
    "dashboard",
    "interactive",
    "help",
}
# Every other parsed option, including defaults, records the actual configuration.
UNRECORDED_ARGUMENTS = INSTANCE_ARGUMENTS | {"recipe_tag", "verbose"}


def execution_configuration(
    arguments: dict[str, Any], limits: dict[str, Any]
) -> dict[str, Any]:
    result = {
        key: value
        for key, value in arguments.items()
        if key not in UNRECORDED_ARGUMENTS
    }
    if not result.get("online_workflows", False):
        result.pop("online_workflows", None)
    configuration = {"arguments": result, "limits": limits}
    if arguments.get("graspgen_config") is not None:
        from reweave.robots.components.graspgen_client import load_graspgen_config

        graspgen = load_graspgen_config(arguments["graspgen_config"])
        # Each worker keeps its exact path/route in argv and shared services.
        # Compare actual sampling/calibration content across the device pool.
        result.pop("graspgen_config")
        configuration["graspgen"] = graspgen.model_dump(
            mode="json", exclude={"endpoint"}
        )
    if arguments.get("robot_name") == "robodojo":
        from reweave.benchmarks.robodojo.tasks import TASK_RULE_REVIEWS

        # Only the evaluated task's rules are published; the other 53 would be
        # cross-task noise in every bundle and episode context the author reads.
        review = TASK_RULE_REVIEWS[arguments["task"]]
        configuration["task_rule_supplements"] = {
            f"robodojo:{arguments['task']}": {
                "task": f"robodojo:{arguments['task']}",
                "task_rule_supplement": review.supplement,
                "source_urls": review.source_urls,
                "review_status": review.review_status,
            }
        }
    return json.loads(json.dumps(configuration, default=str))


def validate_configuration(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Compare actual settings directly; source locations are provenance only."""
    configurations = json.loads(json.dumps([expected, actual]))
    for configuration in configurations:
        if not configuration["arguments"].get("online_workflows", False):
            configuration["arguments"].pop("online_workflows", None)
        for name in ("source_root", "native_source_root"):
            configuration["arguments"].pop(name, None)
    if configurations[0] != configurations[1]:
        raise ValueError("actual model, environment or budget configuration differs")


def dependency_inventory() -> dict[str, Any]:
    installed = sorted(
        {
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
        }
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "installed": installed,
    }


class RuntimeResources(Asset):
    """Actual nonsecret asset locations and service routes."""

    models: dict[str, Any] = Field(
        description="Model asset locations and ordinary configuration metadata."
    )
    environment: dict[str, Any] = Field(
        min_length=1,
        description="Environment, task, calibration or controller asset locations and metadata.",
    )
    routes: dict[str, str] = Field(
        min_length=1,
        description="Explicit nonsecret model routes and requested revisions. Hosted aliases are not weight attestations.",
    )


class RuntimeAuthoringContract(Asset):
    """Actual public gateway metadata for authoring under the active environment."""

    operations: dict[str, dict[str, Any]] = Field(
        description="Actual Kernel operation effects and input schemas, including its non-Toolkit operations."
    )
    tools: tuple[dict[str, Any], ...] = Field(
        description="Actual fixed Planner tool descriptions and schemas, with symbolic output paths."
    )
    capability_catalog: tuple[dict[str, Any], ...] = Field(
        description="Actual Toolkit capability descriptions, effects and result schemas."
    )
    native_image_contract: dict[str, Any] | None = Field(
        default=None,
        description="Actual adapter image-return upper bounds for native batch admission.",
    )

    @classmethod
    def from_capture(
        cls, *, operations, tools, capability_catalog, native_image_contract=None
    ) -> RuntimeAuthoringContract:
        """Use public tool descriptions without changing captured constraints.

        Planner-facing descriptions can be translated after Kernel operations
        were recorded. Reconcile only those annotations; any validation change
        requires an explicit execution contract revision instead.
        """

        def constraints(schema):
            if not isinstance(schema, dict):
                return schema
            result = {}
            for key, value in schema.items():
                if key == "description":
                    continue
                if key in {
                    "properties",
                    "patternProperties",
                    "$defs",
                    "definitions",
                    "dependentSchemas",
                }:
                    value = {name: constraints(item) for name, item in value.items()}
                elif key in {"allOf", "anyOf", "oneOf", "prefixItems"}:
                    value = [constraints(item) for item in value]
                elif key in {
                    "items",
                    "additionalProperties",
                    "unevaluatedProperties",
                    "unevaluatedItems",
                    "contains",
                    "not",
                    "if",
                    "then",
                    "else",
                    "propertyNames",
                } and isinstance(value, dict):
                    value = constraints(value)
                result[key] = value
            return result

        operations = json.loads(json.dumps(operations))
        for tool in tools:
            name = tool["name"]
            schema = tool["input_schema"]
            if name not in operations or json.dumps(
                constraints(schema), sort_keys=True
            ) != json.dumps(
                constraints(operations[name]["input_schema"]), sort_keys=True
            ):
                raise ValueError(
                    f"fixed tool {name!r} changes captured gateway constraints"
                )
            operations[name]["input_schema"] = schema
        return cls(
            operations=operations,
            tools=tools,
            capability_catalog=capability_catalog,
            native_image_contract=native_image_contract,
        )

    @classmethod
    def from_runtime(cls, runtime) -> RuntimeAuthoringContract:
        """Export an initialized BundleRuntime without a model call or action.

        Call immediately after constructing the normal runtime in a fresh smoke
        directory. Construction publishes the operation catalog but does not
        execute the loop. Toolkit tool_api alone cannot supply operation effects,
        model/local-task permissions, memory bindings or result schemas.
        """
        root = runtime.toolkit.state.output_dir
        contract = json.loads(
            (root / "harness_context/bundle_contract.json").read_text()
        )
        operations = contract["operations"]
        tools = [spec for spec in runtime.specs() if spec["name"] in operations]
        return cls.from_capture(
            operations=operations,
            native_image_contract=contract.get("native_image_contract"),
            tools=json.loads(json.dumps(tools).replace(str(root), "{{output_dir}}")),
            capability_catalog=runtime.toolkit.get_runtime_capability_catalog().authoring_payload(),
        )


class RuntimeManifest(Asset):
    """Actual runner metadata, budgets and author-facing tool contract."""

    revision_id: str = Field(
        min_length=1, description="Saved ordinary runtime configuration revision."
    )
    execution_id: str = Field(
        min_length=1, description="Saved complete evaluation configuration record."
    )
    execution_revision: ExecutionRevision | None = Field(
        default=None,
        description="Published configuration change retained as provenance, never source attestation.",
    )
    authoring_contract: RuntimeAuthoringContract | None = Field(
        default=None,
        description="Current actual public gateway for new authors.",
    )
    dependencies: dict[str, Any] = Field(
        description="Actual dependency versions and platform."
    )
    tool_api: tuple[dict[str, Any], ...] = Field(
        description="Actual public Toolkit schemas."
    )
    models: dict[str, Any] = Field(
        description="Actual exposed controller and service configuration."
    )
    resources: RuntimeResources
    configuration: dict[str, Any] = Field(
        description="Parsed arguments and execution budgets."
    )
    conditions: EvaluationConditions
    memory_files: tuple[str, ...] = Field(
        description="Initial memory file names, possibly empty."
    )
    services: dict[str, Any] = Field(
        default_factory=dict, description="Actual nonsecret service metadata."
    )
    evaluator: dict[str, Any] = Field(description="Official evaluator configuration.")

    def runtime_configuration(self) -> RuntimeConfiguration:
        """Record ordinary configuration without calculating a runtime identity."""
        return RuntimeConfiguration(
            revision_id=self.revision_id,
            dependencies=self.dependencies,
            tool_api=self.tool_api,
            models={"controller": self.models, "services": self.services},
            memory=self.memory_files,
            evaluator=self.evaluator,
            environment=self.resources.environment,
            configuration=self.configuration,
            wait_steps=self.conditions.wait_steps,
            physical_steps=self.conditions.physical_steps,
        )

    def validate_conditions(self, *, arguments, limits, planner) -> None:
        """Compare actual budget and model semantics, not source versions."""
        validate_configuration(
            self.configuration, execution_configuration(arguments, limits)
        )
        validate_conditions(self.conditions, arguments, limits, planner)


def sampling_settings(planner, backend: str) -> dict[str, Any]:
    """Read settings from the configured native decision backend."""
    if backend != "native":
        raise ValueError(f"unsupported Bundle decision backend: {backend}")
    return planner.sampling_settings()


def model_configuration(
    planner, resources: RuntimeResources, *, backend: str
) -> dict[str, Any]:
    settings = sampling_settings(planner, backend)
    return {
        "backend": backend,
        "controller": planner.runtime_configuration(),
        "settings": settings,
        "resources": resources.models,
        "routes": resources.routes,
    }


def validate_budget_arguments(
    conditions: EvaluationConditions,
    arguments: dict[str, Any],
    limits: dict[str, Any],
) -> None:
    backend = arguments["planner"]
    if backend != "native":
        raise ValueError(f"unsupported Bundle decision backend: {backend}")
    if conditions.controller_request_unit != "native_turn":
        raise ValueError("controller request unit differs from configured backend")
    if arguments["adaptation_mode"] != conditions.adaptation_mode:
        raise ValueError("online adaptation mode differs from evaluation conditions")
    if arguments.get("online_workflows", False) != conditions.online_workflows:
        raise ValueError("online W authoring differs from evaluation conditions")
    if conditions.physical_steps is None:
        raise ValueError("this robot requires a finite physical step budget")
    expected = {
        "max_episode_steps": conditions.physical_steps,
        "wait_steps": conditions.wait_steps,
        "max_turns": conditions.model_requests,
        "planner_timeout_s": conditions.wall_time_s
        if conditions.wall_time_s is not None
        else 0,
    }
    if any(arguments.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "CLI horizon, model-request or time budget differs from evaluation conditions"
        )
    episode = limits["episode"]
    if any(
        episode.get(key) != getattr(conditions, key)
        for key in ("program_calls", "gateway_calls", "wall_time_s")
    ):
        raise ValueError("kernel total budget differs from evaluation conditions")


def validate_conditions(conditions, arguments, limits, planner) -> None:
    validate_budget_arguments(conditions, arguments, limits)
    if conditions.model_dump(mode="json")["model_sampling"] != sampling_settings(
        planner, arguments["planner"]
    ):
        raise ValueError("actual model sampling differs from evaluation conditions")


def capture_baseline(
    destination: Path,
    *,
    arguments: dict[str, Any],
    limits: dict[str, Any],
    conditions: EvaluationConditions,
    resources: RuntimeResources,
    memory: Path,
    toolkit,
    planner,
) -> AgentBundle:
    """Run after real Toolkit initialization, before any controller query or learning."""
    if destination.exists():
        raise ValueError(
            "baseline destination exists; immutable snapshots cannot be overwritten"
        )
    validate_conditions(conditions, arguments, limits, planner)
    manifest = RuntimeManifest(
        revision_id=new_record_id(),
        execution_id=new_record_id(),
        dependencies=dependency_inventory(),
        tool_api=toolkit.get_tools_contract_spec(),
        models=model_configuration(planner, resources, backend=arguments["planner"]),
        resources=resources,
        configuration=execution_configuration(arguments, limits),
        conditions=conditions,
        memory_files=tuple(
            sorted(
                path.relative_to(memory).as_posix()
                for path in memory.rglob("*")
                if path.is_file()
            )
        ),
        evaluator={"robot": arguments["robot_name"]},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.parent / ".scratch/runtime"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="baseline-", dir=scratch) as temporary:
        staging = Path(temporary)
        shutil.copytree(memory, staging / "memory")
        bundle = AgentBundle(
            record_id=new_record_id(),
            runtime=manifest.runtime_configuration(),
            loop=baseline_loop(),
            context=planner.baseline_context(),
        )
        from reweave.planner.bundle_runtime import BundleRuntime
        from reweave.runtime.kernel import EpisodeLimits

        # Construct the actual public gateway without running the loop, asking
        # the model, or issuing a robot operation. Offline authors need its
        # permissions and effects, which tool_api alone cannot describe.
        runtime = BundleRuntime(
            bundle,
            toolkit,
            str(arguments.get("task", "")),
            episode_limits=EpisodeLimits.model_validate(limits["episode"]),
        )
        manifest = manifest.model_copy(
            update={
                "authoring_contract": RuntimeAuthoringContract.from_runtime(runtime)
            }
        )
        atomic_json(staging / "runtime.json", manifest.model_dump(mode="json"))
        atomic_json(staging / "bundle.json", bundle.model_dump(mode="json"))
        os.rename(staging, destination)
    return bundle
