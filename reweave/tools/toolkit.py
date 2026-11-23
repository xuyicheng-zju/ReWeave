# Copyright 2026 The RPent Authors.
# Modified for the ReWeave namespace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for agent tools.

``Toolkit`` is the agent-facing tool container. Subclasses can register tools
during ``__init__`` via :meth:`Toolkit.add_tool`; the planner calls the tools through :meth:`Toolkit.get_tools_spec` and
:meth:`Toolkit.execute_tool`.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from reweave.dashboard.events import DashboardEventSink, StepRecordEvent
from reweave.planner.decisions import tool_failed
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.execution_trace import ExecutionOutcome
from reweave.runtime.capability import (
    PHYSICAL_FEEDBACK_SCHEMA,
    Capability,
    CapabilityCatalog,
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
)
from reweave.utils.templates import substitute

if TYPE_CHECKING:
    from reweave.memory.corpus import ReadOnlyCorpus
    from reweave.session.base import EnvState, StepRecord


@dataclass(slots=True)
class _ToolOperation:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    owner_thread_id: int = field(default_factory=threading.get_ident)
    nesting_depth: int = 1


class ToolCancelled(Exception):
    """Raised when an environment reaches a safe cancellation boundary."""


def readonly(func):
    """Mark a tool handler as not advancing environment state.

    Tool handlers capture a fresh observation (:meth:`Toolkit.get_env_state`)
    by default. Apply this marker to observational and file/IO tools that do
    not move the robot or otherwise change the environment.
    """
    func._readonly = True
    return func


def _is_readonly(handler: Callable[..., Any]) -> bool:
    """Whether ``handler`` was marked with :func:`readonly`."""
    target = handler
    while isinstance(target, partial):
        target = target.func
    target = getattr(target, "__func__", target)
    return bool(getattr(target, "_readonly", False))


def _runtime_json_value(value: Any) -> Any:
    """Convert host results to the JSON-only generated-program boundary."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return None
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        images: list[dict[str, str]] = []
        for key, item in value.items():
            name = str(key)
            if name.startswith("_image") and isinstance(item, bytes):
                images.append(
                    {
                        "name": name,
                        "media_type": "image/png",
                        "data_base64": base64.b64encode(item).decode("ascii"),
                    }
                )
            else:
                converted[name] = _runtime_json_value(item)
        if images:
            converted["_runtime_images"] = images
        return converted
    if isinstance(value, (list, tuple)):
        return [_runtime_json_value(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _runtime_json_value(scalar())
        except ValueError:
            pass
    array = getattr(value, "tolist", None)
    if callable(array):
        return _runtime_json_value(array())
    raise TypeError(
        f"runtime capability value is not JSON serializable: {type(value).__name__}"
    )


def _runtime_json_object(value: Any) -> dict[str, Any]:
    converted = _runtime_json_value(value)
    if not isinstance(converted, dict):
        raise TypeError("runtime capability payload must be a JSON object")
    return converted


@dataclass
class ToolResult:
    """Result of executing one tool call.

    Carries the raw result dict (for logging and finish-signal detection)
    alongside the Anthropic-shaped content blocks the LLM consumes.
    """

    name: str
    result: dict[str, Any]

    content_blocks: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.content_blocks = self._build_content_blocks()

    def _build_content_blocks(self) -> list[dict[str, Any]]:
        """Build Anthropic-shaped content blocks (text + optional images).

        Strips image byte payloads from the text block and emits them as
        separate base64 image blocks so the LLM receives the state images as
        multimodal content.
        """
        result = self.result
        if not isinstance(result, dict):
            return [{"type": "text", "text": str(result)}]

        result_for_text = dict(result)
        image = result_for_text.pop("_image_bytes", None)
        image_cam = result_for_text.pop("_image_cam_bytes", None)
        image_nav = result_for_text.pop("_image_nav_bytes", None)
        image_wrist = result_for_text.pop("_image_wrist_bytes", None)
        labels = result_for_text.pop("_image_labels", {})
        slots = [
            ("_image_bytes", image),
            ("_image_cam_bytes", image_cam),
            ("_image_nav_bytes", image_nav),
            ("_image_wrist_bytes", image_wrist),
        ]
        if labels:
            result_for_text["image_inputs"] = [
                {"image_number": index + 1, "description": labels[slot]}
                for index, (slot, _data) in enumerate(
                    (slot, data) for slot, data in slots if data
                )
            ]
        text = json.dumps(result_for_text, indent=2, default=str)
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]

        def _add_image_bytes(data_bytes: bytes) -> None:
            data = base64.b64encode(data_bytes).decode("utf-8")
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": data,
                    },
                }
            )

        if image:
            _add_image_bytes(image)
        if image_cam:
            _add_image_bytes(image_cam)
        if image_nav:
            _add_image_bytes(image_nav)
        if image_wrist:
            _add_image_bytes(image_wrist)
        return blocks


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first)
    and register additional tools with :meth:`add_tool`. Robot-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying env Primitives in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release robot-side primitives / servers at the end of the run.
    """

    def __init__(
        self,
        *,
        dashboard_events: DashboardEventSink,
        state: Any = None,
        memory: "ReadOnlyCorpus | None" = None,
        active_execution: ActiveExecution | None = None,
        filesystem_tools_enabled: bool = True,
    ) -> None:
        self._tools: dict[
            str,
            tuple[dict[str, Any], Callable[..., Any]],
        ] = {}
        self._dashboard_events = dashboard_events
        self._state = state
        self._memory = memory
        self.episode_notes = None
        if state is not None and memory is not None:
            from reweave.memory.notes import EpisodeNotes

            self.episode_notes = EpisodeNotes(state.output_dir, state)
        self._active_execution = active_execution
        self._physical_outcome_error: PhysicalOutcomeUnknown | None = None
        self._environment_service_error: EnvironmentServiceFailure | None = None
        self._operation_lock = threading.Lock()
        self._active_operation: _ToolOperation | None = None
        self._register_common_tools(
            filesystem_tools_enabled=filesystem_tools_enabled,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        spec: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register one tool under ``name`` with its schema and handler.

        Args:
            name: Tool name as the LLM sees it (e.g. ``"read_text_file"``).
            spec: Anthropic-shaped tool schema dict (``name``,
                ``description``, ``input_schema``).
            handler: Callable invoked with the tool's input kwargs; returns
                a result dict. Decorate read-only handlers with
                :func:`readonly`; all other handlers capture state.
        """
        self._tools[name] = (spec, handler)

    def _register_common_tools(self, *, filesystem_tools_enabled: bool) -> None:
        """Register the file/IO tools shared by every run."""
        from reweave.tools import common

        memory_bindings = (
            self._memory.get_common_tool_bindings(
                notes=self.episode_notes, confined=not filesystem_tools_enabled
            )
            if self._memory is not None
            else {}
        )
        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            if (
                not filesystem_tools_enabled
                and self.episode_notes is None
                and name
                in {
                    "list_dir",
                    "read_text_file",
                    "write_text_file",
                }
            ):
                continue
            binding = memory_bindings.get(name)
            if binding is None:
                binding = (spec, common.TOOL_HANDLERS[name])
            tool_spec, handler = binding
            self.add_tool(name, tool_spec, handler)

    @property
    def memory(self) -> "ReadOnlyCorpus":
        """Return the toolkit's read-only reference corpus."""
        if self._memory is None:
            raise RuntimeError("toolkit has no reference corpus")
        return self._memory

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    @property
    def state(self) -> EnvState:
        """Return the run's artifact and step store."""
        if self._state is None:
            raise RuntimeError("toolkit has no environment state")
        return self._state

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        return substitute([spec for spec, _ in self._tools.values()])

    def get_tools_contract_spec(self) -> list[dict[str, Any]]:
        """Return seed-invariant schemas for the experiment contract.

        Planner-facing schemas bind ``{{output_dir}}`` to the current rollout
        directory.  That path is run metadata, not part of the tool API, so it
        remain symbolic when comparing experiment-wide tool contracts.
        """
        return substitute(
            [spec for spec, _ in self._tools.values()],
            variables={"output_dir": "{{output_dir}}"},
        )

    def get_runtime_capability_catalog(self) -> CapabilityCatalog:
        """Expose Planner-visible Lego without filesystem or lifecycle authority."""
        capabilities = []
        excluded = {"finish", "list_dir", "read_text_file", "reset", "write_text_file"}
        for name, (spec, handler) in self._tools.items():
            if name in excluded:
                continue
            input_schema = spec.get("input_schema")
            if not isinstance(input_schema, dict):
                raise RuntimeError(f"registered tool {name!r} has no input schema")
            effect = "observe" if _is_readonly(handler) else "physical"
            capabilities.append(
                Capability(
                    name=name,
                    description=str(spec.get("description") or name),
                    effect=effect,
                    input_schema=input_schema,
                    output_schema=PHYSICAL_FEEDBACK_SCHEMA
                    if effect == "physical"
                    else {"type": "object"},
                )
            )
        return CapabilityCatalog(capabilities)

    def harness_source_files(self) -> dict[str, str]:
        """Current implementation references, separate from executable overlays."""
        from reweave.evolution.evidence.records import implementation_sources

        return implementation_sources()

    def export_harness_context(self, output_dir: str) -> None:
        """Persist exact runtime APIs and source references for offline authoring."""
        from pathlib import Path

        root = Path(output_dir) / "harness_context"
        root.mkdir(exist_ok=True)
        (root / "capability_catalog.json").write_text(
            json.dumps(
                self.get_runtime_capability_catalog().authoring_payload(), indent=2
            )
        )
        for name, source in self.harness_source_files().items():
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)

    def _acquire_operation(
        self, *, cancel_event: threading.Event | None = None
    ) -> _ToolOperation | None:
        thread_id = threading.get_ident()
        with self._operation_lock:
            operation = self._active_operation
            if operation is not None:
                if operation.owner_thread_id != thread_id:
                    return None
                operation.nesting_depth += 1
            else:
                operation = _ToolOperation(owner_thread_id=thread_id)
                if cancel_event is not None:
                    operation.cancel_event = cancel_event
                self._active_operation = operation
            return operation

    def _release_operation(self, operation: _ToolOperation) -> None:
        with self._operation_lock:
            if self._active_operation is not operation:
                raise RuntimeError("tool operation ownership changed during dispatch")
            operation.nesting_depth -= 1
            if operation.nesting_depth == 0:
                self._active_operation = None
                operation.done_event.set()

    def control_step_budget(self) -> dict[str, Any] | None:
        """Return adapter-owned counters, or None when this robot has none.

        Values follow PhysicalBudget; observation indices and tool-call counts
        are not substitutes for an environment's physical counter.
        """
        return None

    def prepare_model_context(self, context: dict) -> dict:
        """Prepare current native observations on Host before editable C runs.

        Set current_observation_in_messages when the returned messages already
        contain the current state, so shared base C does not append it again.
        Stateless API history remains under its own C window contract.
        """
        from reweave.planner.native_context import prepare_observation

        return prepare_observation(context)

    def native_image_contract(self) -> dict:
        """Adapter-declared upper bounds; unknown readers stay unspecified."""
        reads = {
            cap.name: None
            for cap in self.get_runtime_capability_catalog().capabilities
            if cap.effect == "observe"
        }
        return {"observation": None, "reads": {**reads, "read_image": 1}}

    def validate_model_context(self, context: dict) -> None:
        """Check any additional task-specific model-input constraints after C."""
        from reweave.runtime.model_input import validate_native_feedback

        validate_native_feedback(
            {
                "instructions": context["instructions"],
                "messages": context["messages"],
            }
        )

    @property
    def physical_outcome_error(self) -> PhysicalOutcomeUnknown | None:
        """Return the latched unacknowledged action, before transport wrapping."""
        return self._physical_outcome_error

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call to its registered handler."""
        if self._physical_outcome_error is not None:
            raise self._physical_outcome_error
        if self._environment_service_error is not None:
            raise self._environment_service_error
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        _, handler = entry
        state_changing = not _is_readonly(handler)

        operation = self._acquire_operation()
        if operation is None:
            return ToolResult(
                name=name, result={"error": "another tool operation is still active"}
            )

        try:
            if state_changing and self.episode_ended():
                return ToolResult(
                    name=name,
                    result={
                        "status": "not_executed",
                        "execution": "not_executed",
                        "reason": "environment episode ended",
                    },
                )
            execution_ref = None
            if state_changing and self._active_execution is not None:
                snapshot = self._active_execution.snapshot
                if snapshot.phase != "idle":
                    return ToolResult(
                        name=name,
                        result={"error": "another invocation is already running"},
                    )
                execution_ref = self._active_execution.begin(name, input_dict)
            started = time.perf_counter()
            failed = False
            cancelled = False
            try:
                result = handler(**input_dict)
            except PhysicalOutcomeUnknown as e:
                self._physical_outcome_error = e
                result = {"error": str(e), "physical_outcome_unknown": True}
                failed = True
            except EnvironmentServiceFailure as e:
                self._environment_service_error = e
                result = {"error": str(e)}
                failed = True
            except TypeError as e:
                result = {
                    "error": f"bad arguments for {name}: {e}",
                    "got": input_dict,
                }
                failed = True
            except ToolCancelled as e:
                result = {
                    "error": str(e),
                    "code": "tool_cancelled",
                    "interrupted": True,
                }
                failed = True
                cancelled = True
            except Exception as e:
                result = {"error": str(e), "traceback": traceback.format_exc()}
                failed = True

            result_dict = result if isinstance(result, dict) else {"value": result}
            if tool_failed(result_dict):
                failed = True

            if execution_ref is not None and self._active_execution is not None:
                outcome = self._execution_outcome(
                    result_dict,
                    failed=failed,
                    cancelled=cancelled,
                )
                if outcome in {"failed", "cancelled"}:
                    physical_discarded = self.discard_active_vla_suffix()
                    traced_discarded = self._active_execution.discard_open_suffix(
                        reason=(
                            "tool_cancelled"
                            if outcome == "cancelled"
                            else "tool_failed"
                        )
                    )
                    if (
                        physical_discarded is not None
                        and physical_discarded != traced_discarded
                    ):
                        raise RuntimeError(
                            "physical VLA buffer and execution trace disagree on "
                            f"failed-tool suffix: {physical_discarded} != "
                            f"{traced_discarded}"
                        )
                self._active_execution.finish(outcome)
                result_dict["active_execution_ref"] = execution_ref
                result_dict["active_execution_outcome"] = outcome
                result = result_dict

            if state_changing and self._environment_service_error is None:
                elapsed_s = round(time.perf_counter() - started, 2)
                command = {"action": name, **input_dict}
                if execution_ref is not None:
                    command["active_execution_ref"] = execution_ref
                record: StepRecord | None = None
                try:
                    captured = self.get_env_state(
                        command=command,
                        result=result_dict,
                        elapsed_s=elapsed_s,
                    )
                except Exception as e:
                    captured = dict(result_dict)
                    captured["state_capture_error"] = str(e)
                    captured.setdefault(
                        "error", f"failed to capture state after {name}: {e}"
                    )
                    captured.setdefault("traceback", traceback.format_exc())
                else:
                    record = self._state.latest_record()
                if failed:
                    for key, value in result_dict.items():
                        captured.setdefault(key, value)
                for key, value in result_dict.items():
                    if key.startswith("active_execution_"):
                        captured[key] = value
                # State capture cannot erase a handler failure or its evidence.
                # Programs and the Planner receive the same physical contract.
                captured["capability_result"] = dict(result_dict)
                captured["error"] = result_dict.get("error") or captured.get("error")
                captured["interrupted"] = bool(
                    result_dict.get("interrupted") or captured.get("interrupted")
                )
                for key in ("terminated", "truncated"):
                    captured[key] = bool(captured.get(key, result_dict.get(key, False)))
                result = captured
                if record is not None:
                    self._publish_step(record)

            return ToolResult(name=name, result=result)
        finally:
            try:
                self._release_operation(operation)
            finally:
                # A cleanup failure cannot downgrade an uncertain robot action
                # to an ordinary program failure, or permit another invocation.
                if self._physical_outcome_error is not None:
                    raise self._physical_outcome_error
                if self._environment_service_error is not None:
                    raise self._environment_service_error

    @staticmethod
    def _execution_outcome(
        result: Any,
        *,
        failed: bool,
        cancelled: bool,
    ) -> ExecutionOutcome:
        """Classify one completed handler against the runtime vocabulary."""
        if cancelled:
            return "cancelled"
        if failed:
            return "failed"
        if isinstance(result, dict) and result.get("interrupted"):
            return "interrupted"
        if isinstance(result, dict) and any(
            result.get(field) is True
            for field in ("succeeded", "success", "task_success")
        ):
            return "succeeded"
        return "completed"

    def discard_active_vla_suffix(self) -> int | None:
        """Discard a robot-owned VLA suffix after tool failure.

        Toolkits with an action-buffered policy override this hook. Returning
        an integer lets the runtime validate physical and traced ownership;
        ``None`` means the remote policy protocol does not expose a count.
        """
        return None

    def leftover_vla_suffix_steps(self) -> int:
        """Return leftover VLA actions the host failed to dump on interrupt.

        After a monitor trigger the leftover must already be gone. A positive
        count is a host invariant failure, not an author surface.
        """
        return 0

    def _publish_step(self, record: StepRecord) -> None:
        """Publish one recorded environment step to the dashboard sink."""
        self._dashboard_events.emit(
            StepRecordEvent(
                record=record,
                env_state=self._state,
            )
        )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        """Capture and return the observation produced by a stateful tool."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Server lifecycle hooks (overridden by robot toolkits)
    # ------------------------------------------------------------------

    def cancel_active_and_wait(self) -> None:
        """Request cancellation and wait for the active tool to return."""
        with self._operation_lock:
            operation = self._active_operation
            if operation is None:
                return
            operation.cancel_event.set()
        operation.done_event.wait()

    def raise_if_cancelled(self) -> None:
        """Raise at an environment-defined safe cancellation boundary."""
        with self._operation_lock:
            operation = self._active_operation
        if operation is not None and operation.cancel_event.is_set():
            raise ToolCancelled("tool operation interrupted")

    def close(self) -> None:
        """Join active physical work before releasing episode resources."""
        self.cancel_active_and_wait()

    def solved(self) -> bool:
        """Whether the env has reported the task complete.

        Ground truth for the session loop: an agent may call ``finish`` with
        ``status="success"`` on a cell it did not actually finish, so the
        handoff decision reads the environment, not the agent.
        """
        raise NotImplementedError

    def finalize_episode_evaluation(self) -> None:
        """Collect a fixed external evaluator verdict after online execution ends.

        Simulation evaluators already expose their verdict through ``solved``.
        Real robot toolkits may wait here for a bound physical evaluation receipt.
        This hook runs outside the Planner's permissions and always before close.
        """

    def episode_ended(self) -> bool:
        """Whether the last settled environment feedback ended this episode."""
        record = self.state.latest_record()
        return bool(record is not None and (record.terminated or record.truncated))

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this robot, if supported."""
        return None
