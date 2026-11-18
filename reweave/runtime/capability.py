# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Trusted runtime operations exposed to generated Python packages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, TypeAlias

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CapabilityEffect: TypeAlias = Literal[
    "observe",
    "physical",
    "control",
    "model",
]


class PhysicalOutcomeUnknown(RuntimeError):
    """An action was submitted without acknowledgement; abort all further I/O."""


class EnvironmentInvalid(RuntimeError):
    """The simulator invalidated the scene; preserve work without retrying it."""


class EnvironmentServiceFailure(RuntimeError):
    """Environment transport failed without a new physical action submission."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class CandidateExecutionFailure(RuntimeError):
    """A frozen candidate violated an executable runtime contract."""


class ModelInputRejected(CandidateExecutionFailure):
    """Known invalid request, not submitted; native input can be corrected in-session."""


class ProgramDeadlineExceeded(CandidateExecutionFailure):
    """A caller's time allowance expired without an established service failure."""

    def __init__(self, message: str, *, scope: Literal["episode", "program"]):
        super().__init__(message)
        self.scope = scope


class PhysicalBudget(BaseModel):
    """Current counters from the robot adapter, never inferred from observations."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    unit: str = Field(min_length=1, description="The adapter's actual counted unit.")
    limit: int | None = Field(ge=0, description="Declared horizon; null if unknown.")
    used: int | None = Field(ge=0, description="Consumed units; null if unknown.")
    remaining: int | None = Field(
        ge=0, description="Remaining units; null if unknown, never estimated."
    )
    operation_costs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Adapter-declared cost formulas for operations in this unit.",
    )

    @model_validator(mode="after")
    def _consistent(self):
        if all(value is not None for value in (self.limit, self.used, self.remaining)):
            if self.remaining != max(0, self.limit - self.used):
                raise ValueError("physical budget counters disagree")
        return self


class Capability(BaseModel):
    """One operation a sandboxed program may request from the trusted host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        min_length=1,
        description="Exact operation name granted to generated Python code.",
    )
    description: str = Field(
        min_length=1,
        description="Robot or reasoning behavior provided by this capability.",
    )
    effect: CapabilityEffect = Field(
        description=(
            "observe has no environment write; physical may move the robot; control "
            "changes episode-local runtime structure; model invokes reasoning."
        ),
    )
    input_schema: dict[str, Any] = Field(
        description="JSON-object argument contract validated before dispatch.",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object"},
        description="JSON-object result contract validated after dispatch.",
    )

    @field_validator("input_schema", "output_schema")
    @classmethod
    def _validate_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        Draft202012Validator.check_schema(schema)
        if schema.get("type") != "object":
            raise ValueError("capability schemas must describe JSON objects")
        return schema


PHYSICAL_FEEDBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Host-normalized physical feedback. Tool-specific fields are optional; inspect capability_result and current observation before using them.",
    "properties": {
        "capability_result": {
            "type": "object",
            "description": "Unmodified return value of the physical handler.",
            "properties": {
                "motion_response": {
                    "type": "object",
                    "description": "Measured response over the reported number of acknowledged steps, not an inferred failure cause or recovery decision.",
                    "properties": {
                        "steps": {"type": "integer", "minimum": 0},
                        "start_eef_pos": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "World-frame XYZ in metres at the measured interval start.",
                        },
                        "end_eef_pos": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "World-frame XYZ in metres at the measured interval end.",
                        },
                        "path_length_m": {"type": "number", "minimum": 0},
                        "target_distance_reduction_m": {
                            "type": "number",
                            "description": "Initial minus final EEF-to-commanded-target distance; negative means farther away.",
                        },
                    },
                    "required": [
                        "steps",
                        "start_eef_pos",
                        "end_eef_pos",
                        "path_length_m",
                        "target_distance_reduction_m",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "terminated": {
            "type": "boolean",
            "description": "Environment-owned terminal flag; interpret success using the robot task contract.",
        },
        "truncated": {
            "type": "boolean",
            "description": "Physical episode exhausted its horizon.",
        },
        "interrupted": {
            "type": "boolean",
            "description": "An optional check returned this operation early. The caller decides the next step.",
        },
        "error": {
            "type": ["string", "null"],
            "description": "Physical handler error, or null when none was reported.",
        },
    },
    "required": [
        "capability_result",
        "terminated",
        "truncated",
        "interrupted",
        "error",
    ],
}


class CapabilityCatalog:
    """Single authority for operations available to a program invocation."""

    def __init__(self, capabilities: Iterable[Capability]) -> None:
        by_name: dict[str, Capability] = {}
        for capability in capabilities:
            if capability.name in by_name:
                raise ValueError(f"duplicate capability name: {capability.name!r}")
            by_name[capability.name] = capability
        if not by_name:
            raise ValueError("Capability Runtime requires at least one capability")
        self._by_name = by_name

    def get(self, name: str) -> Capability:
        """Return one declared capability or reject an invented name."""
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ValueError(
                f"capability is unavailable in this episode: {name!r}"
            ) from exc

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        """Stable, name-sorted capability view for authoring and audit."""
        return tuple(self._by_name[name] for name in sorted(self._by_name))

    def authoring_payload(self) -> list[dict[str, Any]]:
        """Return the exact catalog embedded in the coding-agent request."""
        return [capability.model_dump(mode="json") for capability in self.capabilities]


def validate_capability_arguments(
    capability: Capability,
    arguments: dict[str, Any],
) -> None:
    """Validate exact call arguments before host dispatch."""
    schema = dict(capability.input_schema)
    if isinstance(schema.get("properties"), dict):
        schema["additionalProperties"] = False
    errors = sorted(
        Draft202012Validator(schema).iter_errors(arguments),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
            f"{error.message}"
            for error in errors
        )
        raise ValueError(
            f"invalid arguments for capability {capability.name!r}: {detail}"
        )


def validate_capability_result(capability: Capability, result: dict[str, Any]) -> None:
    """Reject a host result that violates its advertised program contract."""
    errors = list(Draft202012Validator(capability.output_schema).iter_errors(result))
    if errors:
        raise RuntimeError(
            f"capability {capability.name!r} returned an invalid result: "
            f"{errors[0].message}"
        )
