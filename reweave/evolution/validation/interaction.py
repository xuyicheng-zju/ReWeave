# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Replay declared responses through the production W/C/L execution path.

Only the model and public gateway responses are synthetic. The Planner adapter,
context construction, loop, workflow dispatch and isolated program workers are
real. These checks establish interaction behavior, never physical task success.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import uuid
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    model_validator,
)

from reweave.dashboard.events import NullDashboardEventSink
from reweave.evolution.bundle import AgentBundle
from reweave.evolution.evidence.records import store_bytes
from reweave.memory.corpus import ReadOnlyCorpus
from reweave.runtime.capability import Capability
from reweave.runtime.kernel import EpisodeLimits, validate_json
from reweave.tools.toolkit import Toolkit, ToolResult

_FILE_TOOLS = frozenset({"list_dir", "read_text_file", "write_text_file"})
_CLOSEOUT_TEXT = "Synthetic episode close-out: no physical outcome is established."


class _ScenarioItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class InteractionToolCall(_ScenarioItem):
    name: str = Field(min_length=1, description="Exact Planner-visible tool name.")
    arguments: dict[str, JsonValue] = Field(
        description="Complete arguments emitted by this synthetic model response."
    )


class InteractionModelResponse(_ScenarioItem):
    text: str = Field(default="", description="Synthetic model response text.")
    tool_calls: list[InteractionToolCall] = Field(
        default_factory=list, description="Ordered synthetic model tool proposals."
    )
    instructions_contains: list[str] = Field(
        default_factory=list,
        description="Every string must occur in this actual model request's instructions.",
    )
    required_tools: list[str] = Field(
        default_factory=list,
        description="Exact tool names that this actual model request must expose.",
    )
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="Exact tool names that this actual model request must not expose.",
    )
    required_images: list[str] = Field(
        default_factory=list,
        description="Declared artifact filenames whose actual binary bytes must reach this model request; labels and text do not count. Leave empty for text-only reasoning.",
    )

    @model_validator(mode="after")
    def _actionable(self):
        if not self.text.strip() and not self.tool_calls:
            raise ValueError("model response requires text or a tool call")
        if set(self.required_tools) & set(self.forbidden_tools):
            raise ValueError("a tool cannot be both required and forbidden")
        return self


class InteractionEnvironmentState(_ScenarioItem):
    official_success: bool = Field(
        description="Synthetic environment success flag exposed to runtime.solved; never a measured task outcome."
    )
    terminated: bool = Field(
        description="Synthetic environment terminal flag, independent of the official success verdict."
    )
    truncated: bool = Field(
        description="Synthetic environment horizon flag exposed to the actual runtime terminal check."
    )


class InteractionArtifact(_ScenarioItem):
    name: str = Field(
        min_length=1,
        description="Public image artifact filename; directory components are forbidden.",
    )
    step: int = Field(
        ge=0, description="Recorded public step at which this image is available."
    )
    media_type: Literal["image/png", "image/jpeg"] = Field(
        description="Declared image encoding; Host verifies the actual decoded format."
    )
    data_base64: str = Field(
        min_length=1, description="Complete declared image bytes, encoded as base64."
    )

    @model_validator(mode="after")
    def _filename(self):
        if self.name in {".", ".."} or any(
            char in self.name for char in ("/", "\\", "\0")
        ):
            raise ValueError(
                "artifact name must be a filename without directory components"
            )
        suffix = Path(self.name).suffix.lower()
        if suffix not in (
            {".png"} if self.media_type == "image/png" else {".jpg", ".jpeg"}
        ):
            raise ValueError("artifact filename extension must agree with media_type")
        return self


class InteractionOperation(_ScenarioItem):
    name: str = Field(min_length=1, description="Exact public gateway operation name.")
    arguments: dict[str, JsonValue] = Field(
        description="Exact arguments required at this position in the gateway sequence."
    )
    result: dict[str, JsonValue] = Field(
        description="Complete synthetic public gateway result, not an environment measurement."
    )

    environment_state: InteractionEnvironmentState | None = Field(
        default=None,
        description="Optional explicit synthetic environment flags committed when this gateway result returns; absent preserves prior flags.",
    )


class InteractionAssertion(_ScenarioItem):
    path: Annotated[list[StrictStr | StrictInt], Field(min_length=1)] = Field(
        description="Keys and zero-based indices selecting a value in the full interaction report."
    )
    equals: JsonValue = Field(
        description="Exact expected JSON value; missing paths and unequal values fail validation."
    )


class InteractionScenario(_ScenarioItem):
    """Executable interaction expectations, with no injectable runtime state."""

    name: str = Field(
        min_length=1, description="Name identifying this interaction case."
    )
    system_prompt: str = Field(
        min_length=1,
        description="Fixed base instructions supplied to the real Planner adapter.",
    )
    task_message: str = Field(
        min_length=1,
        description="Initial user task; runtime creates its own state and conversation.",
    )
    recovery: bool = Field(
        default=False,
        description="This fixture supplies a recoverable nonterminal Workflow failure and a subsequent Planner-selected recovery. Host verifies feedback and execution independently of authored assertions; it does not impose retries on real episodes.",
    )
    model_responses: list[InteractionModelResponse] = Field(
        default_factory=list,
        description="Responses consumed in actual plan() request order, including calls from Workflows.",
    )
    operations: list[InteractionOperation] = Field(
        default_factory=list,
        description="Public gateway responses consumed in exact dispatch order; no live tools run.",
    )
    artifacts: list[InteractionArtifact] = Field(
        default_factory=list,
        description="Image fixtures for the production read_image route. Every declared artifact must be read; reads are audited separately from operations.",
    )
    initial_artifact_step: int = Field(
        default=0,
        ge=0,
        description="Initial recorded step used by read_image(step=-1); subsequent public results with an integer step advance this simulated artifact cursor.",
    )
    assertions: Annotated[list[InteractionAssertion], Field(min_length=1)] = Field(
        description="Mandatory behavior assertions evaluated against the complete actual report."
    )

    @model_validator(mode="after")
    def _artifact_contract(self):
        identities = [(artifact.step, artifact.name) for artifact in self.artifacts]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate image artifact step/name")
        if any(operation.name == "read_image" for operation in self.operations):
            raise ValueError(
                "read_image uses declared artifacts, not the operations response queue"
            )
        return self


class InteractionFailure(ValueError):
    """A failed simulation carrying its full, untruncated execution evidence."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(
            "interaction validation failed: " + "; ".join(report["errors"])
        )


def _same_json(left, right):
    """Compare JSON values, with numeric equality distinct from boolean equality."""
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same_json(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, list):
        return (
            isinstance(right, list)
            and len(left) == len(right)
            and all(_same_json(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


class _FixtureState:
    """Only explicit fixture bytes are ever reachable through the real image reader."""

    def __init__(self, directory, scenario, reads, errors):
        from PIL import Image

        self.output_dir = directory
        self.current_step = scenario.initial_artifact_step
        self._errors = errors
        self._reads = reads
        self._artifacts = {}
        self._consumed = set()
        self._requested_step = None
        for artifact in scenario.artifacts:
            data = base64.b64decode(artifact.data_base64, validate=True)
            with Image.open(io.BytesIO(data)) as decoded:
                actual_media_type = Image.MIME.get(decoded.format)
                decoded.verify()
            if actual_media_type != artifact.media_type:
                raise ValueError("artifact bytes and declared media_type disagree")
            extension = ".png" if artifact.media_type == "image/png" else ".jpg"
            record = store_bytes(data, directory / "artifacts", suffix=extension)
            path = directory / "artifacts" / record["path"]
            self._artifacts[(artifact.step, artifact.name)] = (path, data, record)

    def _reject(self, message):
        self._errors.append(message)
        raise ValueError(message)

    def get(self, step):
        self._requested_step = step
        resolved = self.current_step if step == -1 else step
        if resolved > self.current_step:
            self._reject(
                f"read_image requested future step {resolved}; latest published step is {self.current_step}"
            )
        if not any(key[0] == resolved for key in self._artifacts):
            self._reject(
                f"read_image requested undeclared artifact step {step} (resolved {resolved})"
            )
        return SimpleNamespace(
            step_idx=resolved,
            artifacts={name for at, name in self._artifacts if at == resolved},
        )

    def artifact_path(self, name, *, step):
        artifact = self._artifacts.get((step, name))
        if artifact is None:
            self._reject(
                f"read_image requested undeclared artifact {name!r} at step {step}; no external file was read"
            )
        return artifact[0]

    def load_bytes(self, name, *, step):
        path = self.artifact_path(name, step=step)
        data = path.read_bytes()
        if data != self._artifacts[(step, name)][1]:
            self._reject("declared artifact changed during interaction simulation")
        self._reads.append(
            {
                "name": name,
                "requested_step": self._requested_step,
                "step": step,
                "artifact_id": self._artifacts[(step, name)][2]["artifact_id"],
                "bytes": len(data),
            }
        )
        self._consumed.add((step, name))
        return data

    def remaining(self):
        return [
            {"step": step, "name": name}
            for step, name in self._artifacts
            if (step, name) not in self._consumed
        ]


class _FixtureCorpus(ReadOnlyCorpus):
    """Use declared file responses inside the production confined memory route."""

    def __init__(self, directory, specs, toolkit):
        super().__init__(directory / "published")
        self._specs = [spec for spec in specs if spec["name"] in _FILE_TOOLS]
        self._toolkit = toolkit

    def get_common_tool_bindings(self, *, notes=None, confined=False):
        bindings = super().get_common_tool_bindings(notes=notes, confined=confined)
        return {
            spec["name"]: (
                deepcopy(spec),
                partial(self._dispatch, spec["name"], bindings[spec["name"]][1]),
            )
            for spec in self._specs
        }

    def _dispatch(self, name, handler, **arguments):
        if name == "write_text_file":
            # Real note validation/revisions stay inside this fixture's private
            # output. This also gives close-out the production has_writes fact.
            written = handler(**arguments)
            owner = self._toolkit
            if owner.host_closeout_pending:
                if arguments != {
                    "path": str(
                        owner.episode_notes.drafts / "task-family/episode-closeout.md"
                    ),
                    "content": _CLOSEOUT_TEXT,
                }:
                    raise ValueError(
                        "Host note close-out arguments differ from its fixture"
                    )
                owner.host_closeout_pending = False
                owner.host_closeout_write = written
                return written
        return self._toolkit.execute_tool(name, arguments).result


class _FixtureToolkit(Toolkit):
    """Only synthetic public responses; inherited ownership/cancellation stays real."""

    def native_image_contract(self):
        return getattr(self, "_native_image_contract", super().native_image_contract())

    def __init__(self, directory, operations, specs, scenario, calls, reads, errors):
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=_FixtureState(directory, scenario, reads, errors),
            memory=(
                _FixtureCorpus(directory, specs, self)
                if any(spec["name"] in _FILE_TOOLS for spec in specs)
                else None
            ),
            filesystem_tools_enabled=False,
        )
        self._specs = deepcopy(specs)
        self._operations = operations
        self._fixtures = scenario.operations
        self.environment_state = InteractionEnvironmentState(
            official_success=False, terminated=False, truncated=False
        )
        self._calls = calls
        self._errors = errors
        self._fixture_lock = threading.Lock()
        self._consumed = 0
        self.host_closeout_pending = False
        self.host_closeout_write = None
        self.allow_image_reader = "read_image" in operations

    def get_tools_spec(self):
        # BundleRuntime owns the standard read_image schema and adds it once.
        return deepcopy([spec for spec in self._specs if spec["name"] != "read_image"])

    def bind_model_budget(self, budget):
        self._model_budget = budget

    def get_tools_contract_spec(self):
        return self.get_tools_spec()

    def get_runtime_capability_catalog(self):
        specs = {spec["name"]: spec for spec in self._specs}
        return SimpleNamespace(
            capabilities=tuple(
                Capability(
                    name=name,
                    description=specs[name]["description"],
                    effect="observe" if operation["effect"] == "read" else "physical",
                    input_schema=operation["input_schema"],
                )
                for name, operation in self._operations.items()
                if operation["effect"] in {"read", "physical"}
                and name not in _FILE_TOOLS | {"read_image"}
            )
        )

    def execute_tool(self, name, input_dict):
        with self._fixture_lock:
            record = {"name": name, "arguments": deepcopy(input_dict)}
            self._calls.append(record)
            if self._consumed >= len(self._fixtures):
                message = f"unexpected gateway call: {record!r}"
            else:
                expected = self._fixtures[self._consumed]
                self._consumed += 1
                actual_args, expected_args = (
                    deepcopy(input_dict),
                    deepcopy(expected.arguments),
                )
                if name in _FILE_TOOLS:
                    for arguments in (actual_args, expected_args):
                        path = Path(arguments.get("path") or self.memory.root)
                        if not path.is_absolute():
                            root = (
                                Path(__file__).resolve().parents[3]
                                if path.parts[:4]
                                == ("reweave", "benchmarks", "libero", "guides")
                                else self.memory.root
                            )
                            path = root / path
                        arguments["path"] = str(path.resolve())
                if name == expected.name and _same_json(actual_args, expected_args):
                    record["result"] = deepcopy(expected.result)
                    if expected.environment_state is not None:
                        self.environment_state = expected.environment_state
                        record["environment_state"] = (
                            expected.environment_state.model_dump(mode="json")
                        )
                    step = expected.result.get("step")
                    if type(step) is int and step >= 0:
                        self.state.current_step = max(self.state.current_step, step)
                    return ToolResult(name=name, result=deepcopy(expected.result))
                message = (
                    f"gateway fixture {self._consumed - 1} mismatch: "
                    f"expected={expected.model_dump(mode='json')!r}, actual={record!r}"
                )
            self._errors.append(message)
            record["error"] = message
            raise ValueError(message)

    def solved(self):
        return self.environment_state.official_success

    def episode_ended(self):
        return self.environment_state.terminated or self.environment_state.truncated


def _contract_specs(contract):
    operations = contract["operations"]
    specs = contract.get("tools")
    if not isinstance(specs, list):
        raise ValueError(
            "interaction simulation requires captured fixed tools metadata"
        )
    names = []
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {
            "name",
            "description",
            "input_schema",
        }:
            raise ValueError(
                "fixed tool metadata requires name, description and input_schema"
            )
        name = spec["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("fixed tool names must be nonempty and unique")
        names.append(name)
        if not isinstance(spec["description"], str) or not spec["description"].strip():
            raise ValueError(f"fixed tool {name!r} lacks its captured description")
        operation = operations.get(name)
        if operation is None or not _same_json(
            spec["input_schema"], operation["input_schema"]
        ):
            raise ValueError(f"fixed tool {name!r} differs from its gateway contract")
    public = {
        name
        for name, operation in operations.items()
        if operation["effect"] in {"read", "physical", "control", "write"}
    }
    if set(names) != public:
        raise ValueError(
            "fixed tool metadata must cover the complete public gateway catalog"
        )
    for name, operation in operations.items():
        effect = operation["effect"]
        if effect in {"model", "task"} and name not in {
            "model:plan",
            "task:start",
            "task:join",
            "task:cancel",
        }:
            raise ValueError(f"unsupported runtime operation: {name}")
        if effect == "control" and name != "finish":
            raise ValueError(f"unsupported public control operation: {name}")
        if effect not in {"read", "physical", "control", "write", "model", "task"}:
            raise ValueError(f"unsupported operation effect: {name}: {effect}")
    return deepcopy(specs)


def _jsonl(directory, name):
    from reweave.evolution.evidence.records import restore_record

    path = directory / name
    return (
        [
            restore_record(json.loads(line), directory)
            for line in path.read_text().splitlines()
        ]
        if path.exists()
        else []
    )


def check_recovery(report: dict[str, Any], contract: dict[str, Any]) -> None:
    """Check a declared recoverable fixture, not arbitrary real-world failures.

    A queued call from the pre-failure plan is not recovery. The failed result
    must reach a later root Planner request before its chosen action executes.
    """

    call_ids = {}

    def carries_failure(output, call_id):
        messages = output["messages"]
        # Native C may contain only new requests: the provider owns prior
        # responses. Use the Host's original response event to bind IDs.
        native_ids = {
            native for native, canonical in call_ids.items() if canonical == call_id
        }
        for message in messages:
            for part in message["parts"]:
                if (
                    part.get("part_kind") == "tool-return"
                    and part["tool_call_id"] in native_ids
                ):
                    content = part["content"]
                    if isinstance(content, str):
                        content = json.loads(content)
                    if isinstance(content, dict) and content.get("status") == "failed":
                        return True
        return False

    main_id = next(
        e["program_call_id"]
        for e in report["events"]
        if e["kind"] == "program_started" and e["plugin"] == "main"
    )
    pending = None
    feedback_seen = False
    planned = set()
    for event in report["events"]:
        kind = event["kind"]
        if kind == "model_response":
            call_ids.update(event["response"]["metadata"]["tool_call_ids"])
        if (
            kind == "workflow_completed"
            and event.get("planner_call_id")
            and event.get("parent_workflow_invocation_id") is None
            and event["result"]["status"] == "failed"
        ):
            pending = event["planner_call_id"]
            planned.clear()
            feedback_seen = False
        if not pending or event.get("program_call_id") != main_id:
            continue
        if kind == "context_built":
            if event["input"]["terminal"]:
                feedback_seen = False
                planned.clear()
                continue
            if not carries_failure(event["output"], pending):
                raise ValueError(
                    "recovery fixture dropped the failed Workflow result from Planner context"
                )
            feedback_seen = True
        if kind == "model_response" and feedback_seen:
            planned.update(event["response"]["metadata"]["tool_call_ids"].values())
        if event.get("planner_call_id") not in planned:
            continue
        if kind == "workflow_started":
            return
        if (
            kind == "action_completed"
            and contract["operations"][event["operation"]]["effect"] == "physical"
        ):
            return
    raise ValueError(
        "recovery fixture must execute a new Planner-selected physical action or Workflow after a nonterminal Workflow failure"
    )


def simulate_interaction(
    bundle: AgentBundle,
    contract: dict[str, Any],
    scenario: InteractionScenario | dict[str, Any],
    output_dir: Path,
    *,
    controller_request_unit: Literal["native_turn"] = "native_turn",
    adaptation_mode: Literal["static", "adaptive"] = "static",
    online_workflows: bool = False,
) -> dict[str, Any]:
    """Run one isolated production interaction and persist all evidence, even on failure.

    Call off the async Host loop. Each invocation owns a fresh directory, fixture
    queues and adapter; it changes no global environment, logging or model settings.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(
            "interaction output directory must be empty; prior evidence is immutable"
        )
    report: dict[str, Any] = {
        "scope": "interaction",
        "validation_status": "failed",
        "physical_execution_tested": False,
        "model_responses_synthetic": True,
        "gateway_responses_synthetic": True,
        "environment_terminal_state_simulated": True,
        "planner_tool_exposure_verified": False,
        "bundle_id": bundle.record_id,
        "controller_request_unit": controller_request_unit,
        "adaptation_mode": adaptation_mode,
        **({"online_workflows": True} if online_workflows else {}),
        "scenario": scenario.model_dump(mode="json")
        if isinstance(scenario, InteractionScenario)
        else deepcopy(scenario),
        "events": [],
        "model_requests": [],
        "calls": [],
        "fixture_calls": [],
        "artifact_reads": [],
        "workflows": [],
        "result": None,
        "assertion_results": [],
        "errors": [],
        "coverage": {"C": False, "L": False, "workflows": []},
    }
    toolkit = None
    consumed_models = 0
    parsed = None
    try:
        parsed = InteractionScenario.model_validate(report["scenario"])
        report["scenario"] = parsed.model_dump(mode="json")
        specs = _contract_specs(contract)
        for fixture in parsed.operations:
            operation = contract["operations"].get(fixture.name)
            if operation is None or operation["effect"] not in {
                "read",
                "physical",
                "control",
                "write",
            }:
                raise ValueError(
                    f"fixture cannot replace an undeclared or internal operation: {fixture.name}"
                )
            validate_json(operation["input_schema"], fixture.arguments)
        toolkit = _FixtureToolkit(
            directory,
            contract["operations"],
            specs,
            parsed,
            report["fixture_calls"],
            report["artifact_reads"],
            report["errors"],
        )
        if "native_image_contract" in contract:
            toolkit._native_image_contract = deepcopy(contract["native_image_contract"])

        def respond(actual, image_bytes):
            nonlocal consumed_models
            actual["distinct_images"] = len(image_bytes)
            report["model_requests"].append(actual)
            if toolkit._model_budget.controller_kind.get() == "note_closeout":
                if "host_note_closeout" in report or actual["tools"]:
                    raise ValueError(
                        "Host note close-out requires one text-only request"
                    )
                report["host_note_closeout"] = {
                    "model_response": _CLOSEOUT_TEXT,
                    "request_index": len(report["model_requests"]) - 1,
                    "model_response_synthetic": True,
                }
                toolkit.host_closeout_pending = True
                return {"text": _CLOSEOUT_TEXT, "tool_calls": []}
            if consumed_models >= len(parsed.model_responses):
                message = "unexpected model request after all declared responses were consumed"
                report["errors"].append(message)
                raise ValueError(message)
            index = consumed_models
            expected = parsed.model_responses[index]
            consumed_models += 1
            names = {spec["name"] for spec in actual["tools"]}
            errors = [
                f"model request {index} instructions lack {text!r}"
                for text in expected.instructions_contains
                if text not in (actual["instructions"] or "")
            ]
            if missing := set(expected.required_tools) - names:
                errors.append(
                    f"model request {index} lacks required tools: {sorted(missing)}"
                )
            if forbidden := set(expected.forbidden_tools) & names:
                errors.append(
                    f"model request {index} exposes forbidden tools: {sorted(forbidden)}"
                )
            for name in expected.required_images:
                fixtures = [a for a in parsed.artifacts if a.name == name]
                if not fixtures or not any(
                    base64.b64decode(a.data_base64) in image_bytes for a in fixtures
                ):
                    errors.append(f"model request {index} lacks binary image {name!r}")
            if errors:
                report["errors"].extend(errors)
                raise ValueError("; ".join(errors))
            return {
                "text": expected.text,
                "tool_calls": [
                    {"name": call.name, "arguments": deepcopy(call.arguments)}
                    for call in expected.tool_calls
                ],
            }

        if controller_request_unit != "native_turn":
            raise ValueError("interaction execution requires native_turn")
        from reweave.planner.native_loop import NativeBundlePlanner

        lifecycle = {"starts": 0, "turns": 0, "closes": 0}
        report["native_session"] = lifecycle

        class FixtureNativeSession:
            def __init__(self, directory):
                self.session_id = "synthetic-native-" + uuid.uuid4().hex
                self.context_generation = 0
                self.instructions = ""
                self.catalog = None
                lifecycle["session_id"] = self.session_id

            async def start(self, instructions, candidate_directory):
                lifecycle["starts"] += 1
                self.cwd = candidate_directory

            async def decide(self, text, images, schema, request_id):
                lifecycle["turns"] += 1
                payload_text = text.rsplit("\n\n", 1)[-1]
                payload = json.loads(payload_text)
                self.instructions = (self.cwd / "AGENTS.md").read_text()
                try:
                    self.catalog = json.loads(
                        self.instructions.rsplit("\n# Tool catalog\n\n```json\n", 1)[
                            1
                        ].split("\n```", 1)[0]
                    )
                except (IndexError, ValueError) as error:
                    raise ValueError(
                        "native tool catalog is missing from AGENTS.md"
                    ) from error
                if set(payload) == {
                    "messages",
                    "available_tools",
                    "tool_catalog",
                }:
                    if (
                        payload["tool_catalog"] != "unchanged"
                        or self.catalog is None
                        or payload["available_tools"]
                        != [tool["name"] for tool in self.catalog]
                    ):
                        raise ValueError(
                            "native tool catalog reference has no matching delivered definitions"
                        )
                else:
                    raise ValueError("native tool catalog delivery has invalid fields")
                actual = {
                    "instructions": self.instructions,
                    "messages": payload["messages"],
                    "tools": deepcopy(self.catalog),
                    "native_input": text,
                    "image_files": [str(path) for path in images],
                    "request_id": request_id,
                    "session_id": self.session_id,
                }
                response = respond(actual, {Path(path).read_bytes() for path in images})
                decision = {
                    "text": response["text"],
                    "tool_calls": [
                        {
                            "name": call["name"],
                            "arguments_json": json.dumps(call["arguments"]),
                        }
                        for call in response["tool_calls"]
                    ],
                }
                validate_json(schema, decision)
                return decision

            async def close(self):
                lifecycle["closes"] += 1

        planner = NativeBundlePlanner(
            output_dir=directory,
            adaptation_mode=adaptation_mode,
            online_workflows=online_workflows,
            session_factory=FixtureNativeSession,
        )
        planner.bind_bundle(
            bundle,
            episode_limits=EpisodeLimits.model_validate(contract["episode_limits"]),
        )
        result = planner.solve(
            system_prompt=parsed.system_prompt,
            user_message=parsed.task_message,
            toolkit=toolkit,
            # Preserve one undeclared-request check and the deployment close-out slot.
            max_turns=max(1, len(parsed.model_responses) + 2),
        )
        report["result"] = {
            key: deepcopy(getattr(result, key)) for key in result.__slots__
        }
        if result.error is not None:
            report["errors"].append(
                f"Planner {result.termination_class}: {result.error}"
            )
    except BaseException as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        if not isinstance(error, Exception):
            raise
    finally:
        traces = directory / "harness_context"
        report["events"] = _jsonl(traces, "bundle_events.jsonl")
        report["calls"] = [
            {
                "name": event["operation"],
                "arguments": deepcopy(event["arguments"]),
                **{
                    key: deepcopy(event[key])
                    for key in ("status", "result", "error", "cancellation_requested")
                    if key in event
                },
            }
            for event in report["events"]
            if event["kind"] == "operation_settled"
        ]
        report["remaining_artifacts"] = (
            toolkit.state.remaining()
            if toolkit
            else [
                {"step": artifact.step, "name": artifact.name}
                for artifact in parsed.artifacts
            ]
            if parsed
            else []
        )
        report["synthetic_environment_state"] = (
            toolkit.environment_state.model_dump(mode="json") if toolkit else None
        )
        if report["remaining_artifacts"]:
            report["errors"].append("unconsumed fixtures: remaining_artifacts")
        report["native_turns"] = _jsonl(directory / "native", "native_turns.jsonl")
        if toolkit is not None and "host_note_closeout" in report:
            report["host_note_closeout"]["write"] = toolkit.host_closeout_write
            if toolkit.host_closeout_pending or toolkit.host_closeout_write is None:
                report["errors"].append(
                    "Host note close-out did not retain its private fixture draft"
                )
        report["remaining_model_responses"] = (
            []
            if parsed is None
            else [
                value.model_dump(mode="json")
                for value in parsed.model_responses[consumed_models:]
            ]
        )
        report["remaining_operations"] = (
            []
            if parsed is None
            else [
                value.model_dump(mode="json")
                for value in parsed.operations[toolkit._consumed if toolkit else 0 :]
            ]
        )
        for name in ("remaining_model_responses", "remaining_operations"):
            if report[name]:
                report["errors"].append(f"unconsumed fixtures: {name}")
        failed_events = {
            "action_failed",
            "workflow_failed",
            "program_failed",
            "program_rpc_failed",
            "model_call_rejected",
            "model_context_rejected",
            "model_request_failed",
        }
        for event in report["events"]:
            if event["kind"] in failed_events:
                if (
                    online_workflows
                    and event["kind"] == "workflow_failed"
                    and event.get("recoverable") is True
                ):
                    continue
                report["errors"].append(
                    f"runtime failure event: {event['kind']}: {event}"
                )
        report["workflows"] = [
            {"workflow_id": event["workflow_id"], "result": deepcopy(event["result"])}
            for event in report["events"]
            if event["kind"] == "workflow_completed"
        ]
        main_id = next(
            (
                event["program_call_id"]
                for event in report["events"]
                if event["kind"] == "program_started" and event["plugin"] == "main"
            ),
            None,
        )
        report["coverage"] = {
            "C": any(event["kind"] == "context_built" for event in report["events"]),
            "L": any(
                event["kind"] == "program_completed"
                and event.get("program_call_id") == main_id
                for event in report["events"]
            ),
            "workflows": sorted(
                {
                    event["workflow_id"]
                    for event in report["events"]
                    if event["kind"] == "workflow_completed"
                }
            ),
        }
        if parsed is not None and parsed.recovery and not report["errors"]:
            try:
                check_recovery(report, contract)
            except ValueError as error:
                report["errors"].append(str(error))
        context_path = traces / "planner_context.json"
        if context_path.exists():
            report["planner_context"] = json.loads(context_path.read_text())
            workflow_names = {
                "workflow_" + item.workflow_id for item in bundle.workflows
            }
            actual = [
                spec
                for spec in report["planner_context"]["tools"]
                if spec["name"] not in workflow_names
            ]
            captured = contract.get("tools")
            if online_workflows and isinstance(captured, list):
                from reweave.runtime.online_workflow import write_workflow_tool_spec

                captured = [*captured, write_workflow_tool_spec()]
            if isinstance(captured, list) and _same_json(
                sorted(actual, key=lambda spec: spec["name"]),
                sorted(captured, key=lambda spec: spec["name"]),
            ):
                report["planner_tool_exposure_verified"] = True
            else:
                report["errors"].append(
                    "actual fixed Planner tool exposure differs from captured metadata"
                )
        contract_path = traces / "bundle_contract.json"
        if contract_path.exists():
            report["runtime_contract"] = json.loads(contract_path.read_text())
            for name, operation in contract["operations"].items():
                if not _same_json(
                    operation, report["runtime_contract"]["operations"].get(name)
                ):
                    report["errors"].append(
                        f"actual runtime operation differs from captured contract: {name}"
                    )
        if parsed is not None:
            for assertion in parsed.assertions:
                checked = assertion.model_dump(mode="json")
                try:
                    actual = report
                    for key in assertion.path:
                        if isinstance(key, int) and key < 0:
                            raise ValueError("assertion indices must be nonnegative")
                        actual = actual[key]
                    checked["actual"] = deepcopy(actual)
                    if not _same_json(actual, assertion.equals):
                        raise ValueError(
                            f"expected {assertion.equals!r}, actual {actual!r}"
                        )
                    checked["validation_status"] = "passed"
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    checked["validation_status"] = "failed"
                    checked["error"] = str(error)
                    report["errors"].append(f"assertion {assertion.path!r}: {error}")
                report["assertion_results"].append(checked)
        report["validation_status"] = "failed" if report["errors"] else "passed"
        (directory / "interaction.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    if report["errors"]:
        raise InteractionFailure(report)
    return report
