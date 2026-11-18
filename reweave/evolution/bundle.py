# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Versioned W/C/L assets; provenance and evaluation never enter content identity."""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reweave.memory.experience import ExperienceId, ExperienceScope

RecordId = Annotated[str, Field(min_length=1)]
AssetId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
Scope = Literal["W", "C", "L"]
HarnessLayer = Literal["global", "task-family"]


def new_record_id() -> str:
    """Allocate once at publication; parsing a saved record never calls this."""
    return uuid.uuid4().hex


class FrozenDict(dict):
    """JSON mapping whose nested values are immutable too."""

    def _reject(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("asset mappings are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = (
        __ior__
    ) = _reject

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict) -> FrozenDict:
        return self


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


class Asset(BaseModel):
    """Strict serialized assets. Consumers take private copies at admission."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def _freeze_containers(self) -> Asset:
        for name in type(self).model_fields:
            object.__setattr__(self, name, _freeze_json(getattr(self, name)))
        return self


class PublishedRecord(Asset):
    """An owner-published value with an ordinary, persisted reference."""

    record_id: RecordId = Field(
        description="Identifier allocated by its publishing owner."
    )


class RuntimeConfiguration(Asset):
    """Ordinary execution metadata; source identity never controls dispatch."""

    revision_id: RecordId = Field(
        description="Saved configuration revision allocated by its writer."
    )
    dependencies: dict[str, Any] = Field(description="Resolved dependency metadata.")
    tool_api: tuple[dict[str, Any], ...] = Field(
        description="Actual public tool catalog."
    )
    models: dict[str, Any] = Field(description="Actual exposed model configuration.")
    memory: tuple[str, ...] = Field(description="Initial memory file names.")
    evaluator: dict[str, Any] = Field(
        description="Actual official evaluator configuration."
    )
    environment: dict[str, Any] = Field(description="Actual environment configuration.")
    configuration: dict[str, Any] = Field(
        description="Parsed arguments and hard budgets."
    )
    wait_steps: int = Field(
        ge=0,
        strict=True,
        description="Settling physical steps charged before agent control begins.",
    )
    physical_steps: int | None = Field(
        gt=0,
        strict=True,
        description="Total physical horizon including settling; null declares no additional action cap.",
    )

    @model_validator(mode="after")
    def _horizon(self) -> RuntimeConfiguration:
        if self.physical_steps is not None and self.wait_steps >= self.physical_steps:
            raise ValueError("settling steps leave no agent control budget")
        return self


class PythonFile(Asset):
    """Exact source bytes represented as UTF-8 text, with a package-relative path."""

    path: str = Field(
        description="Canonical relative Python module path; no traversal or hidden files."
    )
    source: str = Field(
        min_length=1,
        description="Complete source, never truncated by the author transport.",
    )
    layer: HarnessLayer = Field(
        default="task-family",
        exclude_if=lambda value: value == "task-family",
        description="C/L source ownership: task-family is editable during evolution; global is extracted after fixed-evaluator success and reused read-only during subsequent trials. Workflow files inherit WorkflowSpec.layer and leave this field at its default.",
    )

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or path.suffix != ".py"
            or any(part.startswith(".") for part in path.parts)
            or "\\" in value
        ):
            raise ValueError("source requires a canonical relative Python path")
        if any(not part.isidentifier() for part in (*path.parts[:-1], path.stem)):
            raise ValueError("source path must contain Python module names")
        return value


class ProgramLimits(Asset):
    """Local execution limits; these never limit the native offline author loop."""

    runtime_calls: int | None = Field(
        gt=0,
        strict=True,
        description="Maximum gateway calls in one invocation; null means no local cap.",
    )
    physical_calls: int | None = Field(
        ge=0,
        strict=True,
        description="Maximum physical tool dispatches in one invocation; null means no local cap.",
    )
    model_calls: int | None = Field(
        ge=0,
        strict=True,
        description="Maximum fixed model adapter calls in one invocation; null means no local cap.",
    )
    wall_time_s: float | None = Field(
        gt=0,
        strict=True,
        description="Local worker elapsed allowance. With an episode deadline it includes runtime waits; without one it excludes all Host RPC handling, including model/robot waits, while the fixed 60-second nonresponsive-worker watchdog still applies. Null imposes no additional local allowance.",
    )

    @model_validator(mode="after")
    def _call_limits(self) -> ProgramLimits:
        if self.runtime_calls is not None and any(
            value is not None and value > self.runtime_calls
            for value in (self.physical_calls, self.model_calls)
        ):
            raise ValueError("physical/model call limits exceed total runtime calls")
        return self


class PythonProgram(Asset):
    """Executable source shared by workflow and scheduling policy workers."""

    entrypoint: str = Field(
        pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$",
        description="Package module:function accepting (runtime, input).",
    )
    files: tuple[PythonFile, ...] = Field(
        min_length=1, description="Complete module tree for this program."
    )

    @model_validator(mode="after")
    def _modules(self) -> PythonProgram:
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("program files must have unique, sorted paths")
        module = self.entrypoint.split(":")[0].replace(".", "/")
        if module + ".py" not in paths and module + "/__init__.py" not in paths:
            raise ValueError("entrypoint module is absent")
        return self


class WorkflowDependency(Asset):
    """Exact callable workflow dependency, not a floating name."""

    workflow_id: AssetId = Field(description="ID resolved within this frozen registry.")


class WorkflowCase(Asset):
    """Synthetic contract fixture, explicitly separate from official task evaluation."""

    arguments: dict[str, Any] = Field(description="Schema-valid workflow input.")
    observation: dict[str, Any] = Field(description="Public observation at invocation.")
    responses: tuple[dict[str, Any], ...] = Field(
        description="Ordered fixed-gateway fixtures: operation, arguments and result; no physical execution."
    )
    expected_status: Literal["completed", "yielded", "failed", "cancelled"] = Field(
        description="Expected local boundary status, never official success."
    )


class WorkflowSpec(Asset):
    """Reusable feedback program admitted offline and discovered automatically."""

    workflow_id: AssetId = Field(
        description="Stable callable name within the W registry."
    )
    layer: HarnessLayer = Field(
        default="task-family",
        exclude_if=lambda value: value == "task-family",
        description="Owner of this workflow's interface and implementation; task wrappers may call global workflows through declared dependencies.",
    )
    description: str = Field(
        min_length=1, description="Planner-visible purpose and return semantics."
    )
    applicability: str = Field(
        min_length=1,
        description="Observable conditions under which calling the workflow is justified.",
    )
    input_schema: dict[str, Any] = Field(
        description="JSON Schema for caller arguments; runtime state is supplied separately."
    )
    output_schema: dict[str, Any] = Field(
        description="JSON Schema for WorkflowResult.output."
    )
    observations: tuple[str, ...] = Field(
        description="Public observation operations required by this workflow."
    )
    permissions: tuple[str, ...] = Field(
        description="Exact fixed gateway operation IDs; never arbitrary code or hardware access."
    )
    dependencies: tuple[WorkflowDependency, ...] = Field(
        default=(), description="Exact versions of other callable workflows."
    )
    limits: ProgramLimits = Field(
        description="Local limits subordinate to the fixed episode budget."
    )
    program: PythonProgram = Field(description="Offline-authored executable program.")
    contract_cases: tuple[WorkflowCase, ...] = Field(
        min_length=1,
        description="Offline executable interface fixtures; all are charged and reported as synthetic admission checks.",
    )

    @field_validator("input_schema", "output_schema")
    @classmethod
    def _schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        Draft202012Validator.check_schema(value)
        if value.get("type") != "object":
            raise ValueError("workflow schemas must describe objects")

        # Remote references must never turn validation into network or file I/O.
        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                    if keyword in node:
                        reference = node[keyword]
                        if not isinstance(reference, str) or (
                            reference != "#" and not reference.startswith("#/")
                        ):
                            raise ValueError(
                                "only local schema JSON-pointer references are allowed"
                            )
                        target = value
                        if reference != "#":
                            try:
                                for token in reference[2:].split("/"):
                                    target = target[
                                        token.replace("~1", "/").replace("~0", "~")
                                    ]
                            except (KeyError, TypeError) as error:
                                raise ValueError(
                                    "local schema reference has no target"
                                ) from error
                if "$id" in node:
                    raise ValueError("schema base URI changes are not allowed")
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)
        return value

    @model_validator(mode="after")
    def _permissions(self) -> WorkflowSpec:
        if any(source.layer != "task-family" for source in self.program.files):
            raise ValueError(
                "Workflow source ownership is declared only by WorkflowSpec.layer"
            )
        for values in (self.observations, self.permissions):
            if tuple(sorted(set(values))) != values:
                raise ValueError(
                    "permissions and observations must be unique and sorted"
                )
        if not set(self.observations) <= set(self.permissions):
            raise ValueError("required observations must be permitted")
        ids = [item.workflow_id for item in self.dependencies]
        if ids != sorted(set(ids)):
            raise ValueError("workflow dependencies must be unique and sorted")
        return self


class WorkflowResult(Asset):
    """Local completion/yield/failure is never an official task-success verdict."""

    status: Literal["completed", "yielded", "failed", "cancelled"] = Field(
        description="Invocation status decided at its local boundary."
    )
    summary: str = Field(
        description="Concise evidence-grounded feedback to the caller."
    )
    output: dict[str, Any] = Field(description="Declared workflow output.")
    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Episode-private continuation state returned on yield.",
    )

    continuation_id: str | None = Field(
        default=None,
        description="Opaque kernel-issued resume token. Resume the same workflow with identical arguments and caller ancestry; all limits remain cumulative. Changed arguments require a fresh invocation. Invalid tokens return a failed control result without executing the program.",
    )
    resume_arguments: dict[str, Any] | None = Field(
        default=None,
        description="Kernel-issued original arguments on yield, for exact continuation. This is not a grant of new physical/model/time budget.",
    )


class TextSkill(Asset):
    """Online-readable experience; not an executable program or author notebook."""

    skill_id: ExperienceId = Field(
        description="Stable ID corresponding to an individual SKILL.md; supports Harness kebab-case and underscore-separated memory IDs."
    )
    scope: ExperienceScope = Field(
        description="Author-declared applicability: global for reusable reasoning or interfaces, task-family for conditional task-pattern knowledge, task-specific for knowledge bounded to an exact task and observable conditions. Scope certifies neither correctness nor transfer and does not authorize action replay."
    )
    text: str = Field(
        min_length=1,
        description="Complete Markdown skill consumed by the main controller.",
    )


class PluginCase(Asset):
    """Synthetic program contract; the author chooses inputs and expected behavior."""

    input: dict[str, Any] = Field(description="Complete plugin input fixture.")
    responses: tuple[dict[str, Any], ...] = Field(
        default=(),
        description="Ordered method/name/arguments/result runtime responses; no live environment or model calls.",
    )
    expected: dict[str, Any] = Field(
        description="Exact expected program output for this fixture."
    )


class RequestLimits(Asset):
    """Deployment input budgets; images default to 50, null disables a budget."""

    max_images: int | None = Field(
        default=50,
        gt=0,
        strict=True,
        description="Maximum distinct image byte payloads in one controller request; chosen deployment default is 50, not a universal model limit. Duplicate references share one image.",
    )
    max_input_bytes: int | None = Field(
        default=None,
        gt=0,
        strict=True,
        description="Host budget for compact UTF-8 JSON of instructions, provider-neutral messages and tools, including encoded media. Not a tokenizer or provider wire-size estimate.",
    )


class ImageSelection(Asset):
    """Explicit visual evidence projection; all unselected sources stay archived."""

    paths: tuple[str, ...] = Field(
        description="Exact JSON pointers to image occurrences in C messages to send. Empty explicitly selects text only. Every selected image is required and never automatically dropped.",
    )
    reason: str = Field(
        min_length=1,
        description="Why these current and historical images suffice for this request; retained in evidence audit alongside all source pointers.",
    )

    @model_validator(mode="after")
    def _selection(self):
        if len(set(self.paths)) != len(self.paths) or not self.reason.strip():
            raise ValueError(
                "image selection requires unique paths and a nonblank reason"
            )
        return self


class ModelContext(Asset):
    """Complete model request assembled by the C plugin."""

    instructions: str = Field(
        description="Task-level system instructions; fixed execution constraints are enforced outside the model."
    )
    messages: tuple[dict[str, Any], ...] = Field(
        min_length=1,
        description="Complete provider-neutral PydanticAI message objects selected and organized by C; source history remains archived.",
    )
    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Builder state for this invocation's conversation; never replaces source history.",
    )
    tools: tuple[dict[str, Any], ...] = Field(
        description="Selected immutable tool definitions from the caller's catalog."
    )
    image_selection: ImageSelection | None = Field(
        default=None,
        description="Explicit selection from messages; null retains every image. Host rejects oversized input instead of silently clipping evidence.",
    )


class NativeModelContext(ModelContext):
    """Native C guidance around Host-owned incremental observation delivery."""

    messages: tuple[dict[str, Any], ...] = Field(
        min_length=1,
        description="C freely organizes evidence. Default: current observation and newly settled feedback only. Relevant keyframes may be deliberately re-emphasized; no mandatory camera set or automatic history replay. At most 10 distinct images in C output, including unselected images. Duplicate byte payloads count once. Provider memory stays in the same native session.",
    )
    image_selection: ImageSelection | None = Field(
        default=None,
        description="Optional C evidence selection with an explicit reason; null retains all images. C may select views and emphasize relevant keyframes, including historical evidence. The complete C output still has a 10-image ceiling before selection; deployment budgets also apply and Host never silently clips.",
    )


def baseline_context_program() -> PythonProgram:
    """Send current incremental input; the native provider owns episode history."""
    return PythonProgram(
        entrypoint="context:build",
        files=(
            PythonFile(
                path="context.py",
                source="""def build(runtime, context):
    assets = context["assets"]
    guidance = assets["instructions"] + "\\n" + "\\n".join(skill["text"] for skill in assets["skills"])
    return {"instructions": context["base_instructions"] + "\\n" + guidance,
            "messages": context["messages"], "state": context.get("state", {}), "tools": context["tools"]}
""",
            ),
        ),
    )


def upgrade_context_bundle(
    bundle: AgentBundle, runtime: RuntimeConfiguration
) -> AgentBundle:
    """Replace C and bind an explicitly captured target runtime; preserve W/L assets.

    The result is new executable content and has no inherited evaluation receipt.
    """
    program = baseline_context_program()
    if bundle.context.program == program:
        raise ValueError("Bundle already uses the current baseline context program")
    value = bundle.model_dump(mode="json")
    value["record_id"] = new_record_id()
    value["context"]["program"] = program.model_dump(mode="json")
    value["runtime"] = runtime.model_dump(mode="json")
    return AgentBundle.model_validate(value)


class ContextPackage(Asset):
    """Replaceable model-context builder and the text assets it can load."""

    program: PythonProgram = Field(
        default_factory=baseline_context_program,
        description="Context builder for every native plan() call: build(runtime, input) returns instructions, incremental messages and selected tools. Default input contains the current observation and newly settled feedback; provider session history and compaction remain native. May deliberately retrieve relevant evidence; cannot recursively call a model.",
    )
    contract_cases: tuple[PluginCase, ...] = Field(
        default=(),
        description="Explicit synthetic context-builder cases; no unstated initial-behavior requirement.",
    )
    instructions: str = Field(
        default="",
        description="Editable prompt guidance supplied to the context builder; durable experience lives once in skills, not copied here.",
    )
    skills: tuple[TextSkill, ...] = Field(
        default=(),
        description="Single durable text-experience store, classified as global/task-family/task-specific. Each skill has one body and a unique ID across scopes; the builder decides what to use on each request.",
    )

    @model_validator(mode="after")
    def _skills(self) -> ContextPackage:
        ids = [skill.skill_id for skill in self.skills]
        if ids != sorted(set(ids)):
            raise ValueError("skills must be sorted and unique")
        return self


class LoopPolicy(Asset):
    """Replaceable complete episode loop; internal composition is authored Python."""

    program: PythonProgram = Field(
        description=(
            "run(runtime, input) owns the entire execution loop. Its triggers, checks "
            "and scheduling are editable, while executable root decisions pass through "
            "runtime.plan. A root return is valid only after environment termination "
            "or an accepted finish operation. Local Workflow failure requires "
            "feedback-driven replanning; invocation alone never justifies a finish-only objective."
        )
    )
    contract_cases: tuple[PluginCase, ...] = Field(
        default=(),
        description="Synthetic whole-loop fixtures chosen by the author; physical effects need rollout validation.",
    )


class AgentBundle(PublishedRecord):
    """Complete semantic candidate, independent of its production history."""

    schema_version: Literal[3] = Field(
        default=3,
        description="Executable W/C/L bundle serialization contract; prior component names are rejected.",
    )
    runtime: RuntimeConfiguration = Field(
        description="Recorded execution configuration at capture; the active runner owns actual execution and budgets."
    )
    workflows: tuple[WorkflowSpec, ...] = Field(
        default=(), description="W component; automatically discovered at episode load."
    )
    context: ContextPackage = Field(
        default_factory=ContextPackage,
        description="C component; model-consumed text and context view.",
    )
    loop: LoopPolicy = Field(
        description="L component; global scheduler executed by the fixed kernel."
    )

    @model_validator(mode="after")
    def _registry(self) -> AgentBundle:
        ids = [item.workflow_id for item in self.workflows]
        if ids != sorted(set(ids)):
            raise ValueError("workflow registry must be sorted and unique")
        registry = {item.workflow_id: item for item in self.workflows}

        def visit(name: str, ancestors: frozenset[str]) -> None:
            if name in ancestors:
                raise ValueError("workflow dependencies contain a cycle")
            for dependency in registry[name].dependencies:
                target = registry.get(dependency.workflow_id)
                if target is None:
                    raise ValueError("workflow dependency is absent from this bundle")
                visit(target.workflow_id, ancestors | {name})

        for name in registry:
            visit(name, frozenset())
        return self

    def components(self) -> dict[Scope, Any]:
        """Actual editable values, independent of publication identifiers."""
        return {
            "W": self.workflows,
            "C": self.context,
            "L": self.loop,
        }

    def same_harness(self, other: AgentBundle) -> bool:
        return self.components() == other.components()

    def global_assets(self) -> dict[str, Any]:
        """The reusable implementation and Host-owned text that trials cannot edit."""
        return {
            "workflows": tuple(w for w in self.workflows if w.layer == "global"),
            "context": tuple(
                f for f in self.context.program.files if f.layer == "global"
            ),
            "loop": tuple(f for f in self.loop.program.files if f.layer == "global"),
            "skills": tuple(s for s in self.context.skills if s.scope != "task-family"),
        }

    def assert_task_family_edit(self, parent: AgentBundle) -> None:
        if self.global_assets() != parent.global_assets():
            raise ValueError(
                "evolution may edit only task-family; global and task-specific are read-only"
            )

    def require_success_layers(self) -> None:
        """Require executable general C/L, with task entrypoints/bindings retained."""
        if any(
            not any(f.layer == "global" for f in program.files)
            for program in (self.context.program, self.loop.program)
        ):
            raise ValueError(
                "successful harness requires global context and loop implementation"
            )
        if self.workflows and not any(w.layer == "global" for w in self.workflows):
            raise ValueError(
                "successful W registry requires a reusable global workflow"
            )

    def novel_scopes(self, references: tuple[AgentBundle, ...]) -> frozenset[Scope]:
        """Components whose content is absent from every supplied reference.

        A new complete composition may reuse each component unchanged and thus
        have no novel scopes. This classification does not assign an ancestor.
        """
        if not references:
            raise ValueError("candidate requires complete reference Harnesses")
        return frozenset(
            scope
            for scope, value in self.components().items()
            if all(r.components()[scope] != value for r in references)
        )

    def assert_candidate(
        self, references: tuple[AgentBundle, ...], scopes: frozenset[Scope]
    ) -> None:
        """Verify common runtime and write authority without a distinguished source."""
        changed = self.novel_scopes(references)
        if any(
            self.runtime.model_dump(exclude={"revision_id"})
            != reference.runtime.model_dump(exclude={"revision_id"})
            for reference in references
        ):
            raise ValueError("candidate changed frozen runtime dependencies")
        if not changed <= scopes:
            raise ValueError(
                f"candidate changed forbidden scopes: {sorted(changed - scopes)}"
            )


def baseline_loop() -> LoopPolicy:
    """Initial serial decision loop; freeze the shared failure contract with L."""
    import inspect

    from reweave.planner.decisions import tool_failed

    source = (
        inspect.getsource(tool_failed)
        + "\n\n"
        + """def run(runtime, context):
    objective = None
    while True:
        state = runtime.snapshot()
        if state["terminal"]:
            return {"summary": "Environment ended"}
        response = runtime.plan(objective)
        objective = None
        calls = response["tool_calls"]
        if not calls:
            objective = "The episode is still active. Use the available feedback to choose the next operation, recover or try another mechanism."
            continue
        for index, call in enumerate(calls):
            result = runtime.execute(call)
            if runtime.snapshot()["terminal"]:
                return {"summary": "Environment ended"}
            if tool_failed(result):
                for skipped in calls[index + 1:]:
                    runtime.feedback(skipped["id"], {
                        "status": "not_executed",
                        "reason": "A preceding tool failed or was interrupted; replan from its actual feedback before executing queued calls.",
                    })
                objective = "A tool failed while the episode remains active. Reassess its actual feedback and choose a justified recovery; do not finish merely because the tool failed. Tracking residuals alone are not tool failures."
                break
"""
    )
    return LoopPolicy(
        program=PythonProgram(
            entrypoint="loop:run",
            files=(PythonFile(path="loop.py", source=source),),
        )
    )
