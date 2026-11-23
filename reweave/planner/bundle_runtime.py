# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Connect executable bundles to the native planner and physical Toolkit."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
import traceback
from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from jsonschema.exceptions import ValidationError as JsonSchemaError

from reweave.evolution.bundle import AgentBundle, WorkflowSpec
from reweave.evolution.evidence.records import archive_record, store_bytes
from reweave.runtime.adaptation import HarnessAdaptation
from reweave.runtime.capability import (
    CandidateExecutionFailure,
    EnvironmentInvalid,
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
    ProgramDeadlineExceeded,
)
from reweave.runtime.kernel import EpisodeKernel, EpisodeLimits
from reweave.runtime.model_budget import ModelServiceFailure
from reweave.session.base import resolve_image_artifact
from reweave.tools.toolkit import Toolkit, ToolResult, _runtime_json_object


def workflow_tool_specs(catalog):
    """The same kernel-issued Workflow interfaces for execution and probes."""
    return [
        {
            "name": f"workflow_{item['name']}",
            "description": item["description"]
            + " Applicability: "
            + item["applicability"],
            "input_schema": {
                "type": "object",
                "properties": {
                    "arguments": json.loads(json.dumps(item["input_schema"])),
                    "continuation_id": {
                        "type": ["string", "null"],
                        "description": "Resume only this workflow with identical yielded resume_arguments and caller ancestry. All budgets remain cumulative; changing any argument requires a fresh call without this token. Invalid continuation returns failed without executing actions.",
                    },
                },
                "required": ["arguments"],
                "additionalProperties": False,
            },
        }
        for item in catalog
    ]


class BundleRuntime:
    """Public context and gateway for one episode in the existing planner loop."""

    def __init__(
        self,
        bundle: AgentBundle,
        toolkit: Toolkit,
        task: str,
        *,
        episode_limits: EpisodeLimits,
        adaptation: HarnessAdaptation | None = None,
        online_workflows: bool = False,
    ):
        self.bundle, self.toolkit, self.task = bundle, toolkit, task
        self.adaptation = adaptation
        self.online_workflows = online_workflows
        self._online_writes: list[WorkflowSpec] = []
        if online_workflows and (adaptation is None or not adaptation.online_workflows):
            raise ValueError("online W authoring requires the episode revision manager")
        self.history: list[dict[str, Any]] = []
        # The actual gateway settlement index for this async call; None means
        # no operation was settled (for example, rejected before the gateway).
        self.settled_operation: ContextVar[int | None] = ContextVar(
            "settled_operation", default=None
        )
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        self._submitted: set[threading.Event] = set()
        self.observation: dict[str, Any] = {}
        self._trace = (
            toolkit.state.output_dir / "harness_context" / "bundle_events.jsonl"
        )
        self._trace.parent.mkdir(parents=True, exist_ok=True)
        (self._trace.parent / "bundle.json").write_text(
            bundle.model_dump_json(indent=2) + "\n"
        )
        capabilities = toolkit.get_runtime_capability_catalog().capabilities
        operations = {
            cap.name: {
                "effect": "read" if cap.effect == "observe" else cap.effect,
                "input_schema": cap.input_schema,
            }
            for cap in capabilities
            if cap.effect in {"observe", "physical"}
        }
        operations.update(
            {
                "model:plan": {
                    "effect": "model",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "objective": {"type": ["string", "null"]},
                        },
                        "additionalProperties": False,
                    },
                },
                "task:start": {
                    "effect": "task",
                    "input_schema": {
                        "type": "object",
                        "required": ["entrypoint", "payload"],
                        "properties": {
                            "entrypoint": {"type": "string"},
                            "payload": {"type": "object"},
                            "permissions": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "task:join": {
                    "effect": "task",
                    "input_schema": {
                        "type": "object",
                        "required": ["handle"],
                        "properties": {"handle": {"type": ["string", "object"]}},
                        "additionalProperties": False,
                    },
                },
                "task:cancel": {
                    "effect": "task",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "handle": {"type": ["string", "object", "null"]}
                        },
                        "additionalProperties": False,
                    },
                },
            }
        )
        fixed_specs = toolkit.get_tools_contract_spec()
        fixed_names = {spec["name"] for spec in fixed_specs}
        self._memory_bindings = {}
        if toolkit._memory is not None:
            self._memory_bindings = {
                name: binding
                for name, binding in toolkit.memory.get_common_tool_bindings(
                    notes=toolkit.episode_notes, confined=True
                ).items()
                if name in {"list_dir", "read_text_file", "write_text_file"}
                and name in fixed_names
                and (name != "write_text_file" or toolkit.episode_notes is not None)
            }
            for name, (spec, _) in self._memory_bindings.items():
                operations[name] = {
                    "effect": "write" if name == "write_text_file" else "read",
                    "input_schema": spec["input_schema"],
                }
        self._image_spec = None
        if getattr(toolkit, "allow_image_reader", True):
            self._image_spec = {
                "name": "read_image",
                "description": "Read a recorded episode image by artifact name and step; this is a read-only gateway operation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "step": {"type": "integer", "default": -1},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
            operations["read_image"] = {
                "effect": "read",
                "input_schema": self._image_spec["input_schema"],
            }
        names = set(operations)
        # Finish is a Planner control request, never an observation or workflow permission.
        finish = next((spec for spec in fixed_specs if spec["name"] == "finish"), None)
        if finish is not None:
            operations["finish"] = {
                "effect": "control",
                "input_schema": finish["input_schema"],
            }
            names.add("finish")
        self._specs = [
            spec for spec in toolkit.get_tools_spec() if spec["name"] in names
        ]
        self._specs = [
            spec for spec in self._specs if spec["name"] not in self._memory_bindings
        ]
        self._specs.extend(spec for spec, _ in self._memory_bindings.values())
        if self._image_spec is not None:
            self._specs.append(self._image_spec)
        self._fixed_specs = tuple(self._specs)
        if online_workflows:
            from reweave.runtime.online_workflow import write_workflow_tool_spec

            spec = write_workflow_tool_spec()
            if spec["name"] in names:
                raise ValueError("online W tool collides with a fixed tool")
            self._specs.append(spec)
            self._fixed_specs = tuple(self._specs)
        self.kernel = EpisodeKernel(
            bundle,
            operations=operations,
            gateway=self._dispatch,
            cancel=self._cancel,
            emit=self._emit,
            episode_limits=episode_limits,
            workflow_scope=self._workflow_scope,
            snapshot=lambda: self.execution.snapshot(),
            episode_ended=lambda: toolkit.solved() or toolkit.episode_ended(),
        )
        (self._trace.parent / "bundle_contract.json").write_text(
            json.dumps(
                {
                    "bundle_id": bundle.record_id,
                    "operations": operations,
                    "native_image_contract": toolkit.native_image_contract(),
                    "episode_limits": episode_limits.model_dump(mode="json"),
                    "workflow_catalog": self.kernel.catalog(),
                    "plugin_api": "executable-wcl-v3",
                    "model_reasoning_available": True,
                    "local_tasks_available": True,
                    "online_author_available": adaptation is not None
                    and adaptation.mode == "adaptive",
                    **({"online_workflows": True} if online_workflows else {}),
                },
                indent=2,
            )
            + "\n"
        )
        for spec in workflow_tool_specs(self.kernel.catalog()):
            if spec["name"] in names:
                raise ValueError("workflow callable collides with a fixed tool")
            self._specs.append(spec)

    def specs(self, revision_id: str | None = None) -> list[dict[str, Any]]:
        if self.adaptation is None:
            return json.loads(json.dumps(self._specs))
        revision = (
            self.adaptation.revision(revision_id)
            if revision_id is not None
            else self.adaptation.active
        )
        workflows = {
            spec["name"]: spec
            for spec in workflow_tool_specs(self.kernel.catalog(revision.bundle))
        }
        retained = {}
        for token, binding in self.adaptation.continuations:
            name = "workflow_" + binding.workflow.workflow_id
            spec = workflow_tool_specs(
                (
                    next(
                        item
                        for item in self.kernel.catalog(binding.revision.bundle)
                        if item["name"] == binding.workflow.workflow_id
                    ),
                )
            )[0]
            branch = spec["input_schema"]
            branch["properties"]["continuation_id"] = {"const": token}
            branch["required"].append("continuation_id")
            retained.setdefault(name, []).append((token, spec, branch))
        for name, bindings in retained.items():
            current = workflows.get(name)
            branches = [branch for _, _, branch in bindings]
            if current is not None:
                fresh = current["input_schema"]
                fresh["not"] = {
                    "properties": {
                        "continuation_id": {"enum": [token for token, _, _ in bindings]}
                    },
                    "required": ["continuation_id"],
                }
                branches.insert(0, fresh)
            else:
                current = bindings[0][1]
                workflows[name] = current
            current["input_schema"] = {"anyOf": branches}
        if {spec["name"] for spec in self._fixed_specs} & {
            spec["name"] for spec in workflows.values()
        }:
            raise PermissionError("workflow callable collides with a fixed tool")
        return json.loads(json.dumps([*self._fixed_specs, *workflows.values()]))

    def _text_view(self, value: Any) -> Any:
        if isinstance(value, dict):
            result = {
                key: self._text_view(item)
                for key, item in value.items()
                if key != "_runtime_images"
            }
            if "_runtime_images" in value:
                references = []
                for frame in value["_runtime_images"]:
                    data = base64.b64decode(frame["data_base64"], validate=True)
                    from PIL import Image

                    try:
                        with Image.open(io.BytesIO(data)) as decoded:
                            actual_media_type = Image.MIME.get(decoded.format)
                            decoded.verify()
                        if actual_media_type != frame["media_type"]:
                            raise ValueError(
                                "image payload and declared media type disagree"
                            )
                    except (OSError, SyntaxError) as error:
                        raise ValueError(
                            "workflow returned invalid image bytes"
                        ) from error
                    extension = {
                        "image/png": "png",
                        "image/jpeg": "jpg",
                        "image/webp": "webp",
                        "image/gif": "gif",
                    }.get(actual_media_type)
                    if extension is None:
                        raise ValueError("unsupported public image media type")
                    directory = self._trace.parent / "observations"
                    stored = store_bytes(data, directory, suffix=f".{extension}")
                    path = directory / stored["path"]
                    references.append(
                        {**stored, "name": frame["name"], "path": str(path)}
                    )
                result["images"] = references
            return result
        if isinstance(value, (tuple, list)):
            return [self._text_view(item) for item in value]
        return value

    def _tool_result(self, name: str, result: dict[str, Any]) -> ToolResult:
        text_view = self._text_view(result)
        converted = ToolResult(name=name, result=text_view)
        image_inputs = []

        def collect(value: Any, path="") -> None:
            if isinstance(value, dict):
                for index, frame in enumerate(value.get("_runtime_images", ())):
                    image_inputs.append(
                        {
                            "image_number": len(image_inputs) + 1,
                            "source_pointer": f"{path}/_runtime_images/{index}",
                            "description": value.get("_image_labels", {}).get(
                                frame["name"], frame["name"]
                            ),
                        }
                    )
                    converted.content_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": frame["media_type"],
                                "data": frame["data_base64"],
                            },
                        }
                    )
                for key, child in value.items():
                    if key != "_runtime_images":
                        collect(
                            child,
                            path + "/" + str(key).replace("~", "~0").replace("/", "~1"),
                        )
            elif isinstance(value, (tuple, list)):
                for index, child in enumerate(value):
                    collect(child, f"{path}/{index}")

        collect(result)
        if image_inputs:
            text_payload = dict(text_view)
            text_payload.pop("_image_labels", None)
            text_payload["image_inputs"] = image_inputs
            converted.content_blocks[0] = {
                "type": "text",
                "text": json.dumps(text_payload),
            }
        return converted

    def record_tool(self, call, result, requested):
        self.messages.append(
            {
                "role": "tool",
                "name": requested["name"] if requested is not None else call["name"],
                "executed": call,
                "tool_call_id": call.get("id"),
                "provider_tool_call_id": requested["provider_id"]
                if requested
                else call.get("provider_id"),
                "content": result.content_blocks,
                "program_call_id": self.execution.current.get().identity,
            }
        )

    def record_feedback(self, requested, result):
        self.messages.append(
            {
                "role": "tool",
                "name": requested["name"],
                "executed": None,
                "tool_call_id": requested["id"],
                "provider_tool_call_id": requested["provider_id"],
                "content": [{"type": "text", "text": json.dumps(result)}],
                "program_call_id": self.execution.current.get().identity,
            }
        )

    def _emit(self, event: dict[str, Any]) -> None:
        with self._trace.open("a") as stream:
            stream.write(
                json.dumps(
                    archive_record(event, self._trace.parent), ensure_ascii=False
                )
                + "\n"
            )

    async def _cancel(self) -> None:
        for token in tuple(self._submitted):
            token.set()
        await asyncio.to_thread(self.toolkit.cancel_active_and_wait)

    def _record_operation(
        self, name, arguments, *, result=None, error=None, cancelled=False
    ):
        """Record a settled operation independently of caller cancellation."""
        record = {
            "operation": name,
            "arguments": arguments,
            "status": "unknown"
            if isinstance(error, PhysicalOutcomeUnknown)
            else "failed"
            if error
            else "completed",
            "cancellation_requested": cancelled,
        }
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
        else:
            record["result"] = result
        try:
            if self.kernel._operations[name]["effect"] == "read":
                # Only a pending model tool call explicitly requests pixels for the
                # next decision. Reads made by W/L are control feedback by default.
                record["feedback_source"] = self.kernel._event_context.get().get(
                    "feedback_source", "program"
                )
                record["program_call_id"] = self.kernel._event_context.get().get(
                    "program_call_id"
                )
            record = json.loads(json.dumps(record))
            self.history.append(record)
            self.settled_operation.set(len(self.history) - 1)
            if error is None and self.kernel._operations[name]["effect"] == "physical":
                self.observation = json.loads(json.dumps(result))
            self.kernel._event("operation_settled", **record)
        except BaseException as evidence_error:
            if isinstance(error, PhysicalOutcomeUnknown):
                error.add_note(
                    f"Settlement evidence failed: {evidence_error}"
                )
                raise error from evidence_error
            raise

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "model:plan":
            return await self.execution.plan(**arguments)
        if name.startswith("task:"):
            return await {
                "task:start": self.execution.start,
                "task:join": self.execution.join,
                "task:cancel": self.execution.cancel,
            }[name](arguments)
        return await self._dispatch_tool(name, arguments)

    async def _dispatch_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        # Install cancellation before the worker can begin, just as the existing
        # runtime dispatcher does. Cancelling an await never abandons a robot thread.
        if name in self._memory_bindings:
            root = self.toolkit.memory.root
            path = Path(arguments.get("path") or root)
            repository = Path(__file__).resolve().parents[2]
            if not path.is_absolute():
                path = (
                    repository / path
                    if path.parts[:4] == ("reweave", "benchmarks", "libero", "guides")
                    else root / path
                )
            # Check the destination, but keep the supplied path for the common
            # handlers so EpisodeNotes can still reject symlink components.
            resolved = path.resolve()
            guides = repository / "reweave/benchmarks/libero/guides"
            notes = self.toolkit.episode_notes
            draft_path = notes is not None and resolved.is_relative_to(notes.root)
            allowed = (
                draft_path
                if name == "write_text_file"
                else draft_path
                or resolved.is_relative_to(root)
                or resolved.is_relative_to(guides)
            )
            if not allowed:
                result = {
                    "error": "file access is confined to published memory, fixed guides and this episode draft notes; only draft notes are writable",
                    "error_type": "PermissionError",
                    "execution_performed": False,
                }
            else:
                _, handler = self._memory_bindings[name]
                try:
                    result = handler(**{**arguments, "path": str(path)})
                except (OSError, ValueError) as error:
                    result = {"error": str(error), "error_type": type(error).__name__}
            self._record_operation(name, arguments, result=result)
            return result
        if name == "read_image":
            try:
                step, _, media_type = resolve_image_artifact(
                    self.toolkit.state, arguments["name"], arguments.get("step", -1)
                )
                result = {
                    "artifact": arguments["name"],
                    "step": step,
                    "_runtime_images": [
                        {
                            "name": arguments["name"],
                            "media_type": media_type,
                            "data_base64": base64.b64encode(
                                self.toolkit.state.load_bytes(
                                    arguments["name"], step=step
                                )
                            ).decode(),
                        }
                    ],
                }
            except (ValueError, OSError, KeyError, IndexError) as error:
                result = {"error": str(error)}
            self._record_operation(name, arguments, result=result)
            return result
        cancellation = threading.Event()
        self._submitted.add(cancellation)

        def dispatch() -> dict[str, Any]:
            operation = self.toolkit._acquire_operation(cancel_event=cancellation)
            if operation is None:
                raise RuntimeError("physical gateway already has an owner")
            try:
                self.toolkit.raise_if_cancelled()
                return _runtime_json_object(
                    self.toolkit.execute_tool(name, arguments).result
                )
            finally:
                self.toolkit._release_operation(operation)

        pending = asyncio.create_task(asyncio.to_thread(dispatch))
        try:
            result = await asyncio.shield(pending)
        except asyncio.CancelledError:
            cancellation.set()
            # Remote adapters must forward cancellation over their owned channel;
            # a local token alone cannot interrupt their executor.
            stopping = asyncio.create_task(
                asyncio.to_thread(self.toolkit.cancel_active_and_wait)
            )
            settled = asyncio.gather(pending, stopping, return_exceptions=True)
            while not settled.done():
                try:
                    await asyncio.shield(settled)
                except asyncio.CancelledError:
                    continue
            if pending.cancelled():
                self._record_operation(
                    name, arguments, error=asyncio.CancelledError(), cancelled=True
                )
            else:
                error = pending.exception()
                self._record_operation(
                    name,
                    arguments,
                    result=None if error else pending.result(),
                    error=error,
                    cancelled=True,
                )
                if isinstance(error, PhysicalOutcomeUnknown):
                    raise error
            stopping.result()
            raise
        except BaseException as error:
            self._record_operation(name, arguments, error=error)
            raise
        finally:
            self._submitted.discard(cancellation)
        self._record_operation(name, arguments, result=result)
        return result

    def _workflow_scope(self, workflow_id):
        return self.execution.workflow_scope(workflow_id)

    async def run(self, model_request, base_instructions, *, model_budget=None):
        from reweave.runtime.execution import ProgramExecution

        self.execution = ProgramExecution(
            self, model_request, base_instructions, model_budget=model_budget
        )
        try:
            result = await self.execution.run()
            self.kernel.raise_if_failed()
            return result
        except BaseException as error:
            # The kernel's confirmed physical outcome is authoritative even when
            # later evidence writes or task cleanup fail with another exception.
            unknown = self.kernel._fatal
            if unknown is not None and unknown is not error:
                unknown.add_note(
                    f"Episode cleanup or reporting failed: {type(error).__name__}: {error}",
                )
                raise unknown from error
            service_failure = self.kernel._service_failure
            if service_failure is not None and service_failure is not error:
                service_failure.add_note(
                    f"Later program termination: {type(error).__name__}: {error}",
                )
                raise service_failure from error
            if isinstance(
                error,
                (
                    PhysicalOutcomeUnknown,
                    EnvironmentInvalid,
                    EnvironmentServiceFailure,
                    ModelServiceFailure,
                    ProgramDeadlineExceeded,
                ),
            ) or (error is self.kernel._evidence_error):
                raise
            if isinstance(
                error,
                (
                    ValueError,
                    PermissionError,
                    RuntimeError,
                    TimeoutError,
                    JsonSchemaError,
                ),
            ):
                # The worker may rethrow its RPC error as RuntimeError. A known
                # Host evidence failure must not become a candidate-code failure.
                evidence_error = self.kernel._evidence_error
                if evidence_error is not None:
                    evidence_error.add_note(
                        f"Program also failed: {type(error).__name__}: {error}",
                    )
                    raise evidence_error from error
                raise CandidateExecutionFailure(str(error)) from error
            raise

    async def execute(self, name, arguments, *, revision_id=None):
        if name == "write_workflow":
            return await self._write_workflow(arguments)
        if name.startswith("workflow_"):
            binding = (
                self.adaptation.bind(
                    "W",
                    workflow_id=name.removeprefix("workflow_"),
                    continuation_id=arguments.get("continuation_id"),
                    revision_id=revision_id,
                )
                if self.adaptation is not None
                else nullcontext(None)
            )
            with binding as pinned:
                with self.kernel.event_scope(
                    **(
                        {"revision_id": pinned.revision.revision_id}
                        if pinned is not None
                        else {}
                    )
                ):
                    started = time.monotonic() if self.online_workflows else None
                    result = await self.kernel.workflow(
                        name.removeprefix("workflow_"),
                        arguments["arguments"],
                        continuation_id=arguments.get("continuation_id"),
                        observation=self.observation,
                        task_language=self.execution.current.get().task,
                        epoch=self.kernel.epoch,
                        workflows=pinned.revision.bundle.workflows
                        if pinned is not None
                        else None,
                        recover_program_errors=pinned is not None
                        and pinned.workflow in self._online_writes,
                    )
                    if self.online_workflows:
                        self.kernel._event(
                            "workflow_timing",
                            workflow_id=name.removeprefix("workflow_"),
                            elapsed_s=time.monotonic() - started,
                        )
                if pinned is not None:
                    if result.continuation_id is not None:
                        self.adaptation.retain_continuation(
                            result.continuation_id, pinned
                        )
                    elif (
                        arguments.get("continuation_id") is not None
                        and result.output.get("error") != "invalid_continuation"
                    ):
                        self.adaptation.consume_continuation(
                            arguments["continuation_id"]
                        )
            return self._tool_result(name=name, result=result.model_dump(mode="json"))
        settlement = (
            self.adaptation.settling() if self.adaptation is not None else nullcontext()
        )
        with settlement:
            return self._tool_result(
                name=name,
                result=await self.kernel.invoke(
                    name, arguments, epoch=self.kernel.epoch
                ),
            )

    async def _write_workflow(self, arguments):
        """Write and optionally call ordinary W, never a parallel execution engine."""
        if not self.online_workflows:
            raise PermissionError("online W authoring is disabled")
        started = time.monotonic()
        try:
            from reweave.evolution.infrastructure.async_work import run_blocking

            workflow = WorkflowSpec.model_validate(arguments["workflow"])
            if "workflow_" + workflow.workflow_id in {
                spec["name"] for spec in self._fixed_specs
            }:
                raise ValueError("workflow callable collides with a fixed tool")
            for permission in workflow.permissions:
                operation = self.kernel._operations.get(permission)
                if operation is None or operation["effect"] not in {
                    "read",
                    "physical",
                    "model",
                    "task",
                }:
                    raise PermissionError(
                        f"W requires unknown/non-public operation {permission!r}"
                    )
            if any(
                self.kernel._operations[name]["effect"] != "read"
                for name in workflow.observations
            ):
                raise ValueError("workflow observations include a non-read operation")
            if "arguments" in arguments:
                from reweave.runtime.kernel import validate_json

                validate_json(workflow.input_schema, arguments["arguments"])
            revision = await run_blocking(
                self.adaptation.write_workflow,
                workflow,
                reason="Online agent wrote W " + workflow.workflow_id,
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            SyntaxError,
            PermissionError,
            JsonSchemaError,
        ) as error:
            self.kernel._event(
                "workflow_write_rejected",
                error_type=type(error).__name__,
                error=str(error),
                execution_performed=False,
            )
            return self._tool_result(
                name="write_workflow",
                result={
                    "status": "failed",
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "execution_performed": False,
                },
            )
        if workflow not in self._online_writes:
            self._online_writes.append(workflow)
        self.execution.root.permissions |= {"workflow_" + workflow.workflow_id}
        metadata = {
            "workflow_id": workflow.workflow_id,
            "revision_id": revision.revision_id,
            "write_elapsed_s": time.monotonic() - started,
            "active": self.adaptation.active is revision,
        }
        self.kernel._event("workflow_written", **metadata)
        if "arguments" not in arguments or not metadata["active"]:
            return self._tool_result(
                name="write_workflow",
                result={
                    "status": "completed",
                    "summary": "W written; no invocation performed.",
                    "output": metadata,
                },
            )
        result = await self.execute(
            "workflow_" + workflow.workflow_id,
            {"arguments": arguments["arguments"]},
            revision_id=revision.revision_id,
        )
        return self._tool_result(
            name="write_workflow", result={**result.result, "online_workflow": metadata}
        )
