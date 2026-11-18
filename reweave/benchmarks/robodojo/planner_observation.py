# Modified for the ReWeave namespace.
"""Default incremental observation input for editable native RoboDojo C."""

from __future__ import annotations

import base64
import copy
import json
from collections import Counter

from reweave.benchmarks.robodojo.tools import CAMERA_ARTIFACTS
from reweave.planner.native_context import project_native_feedback
from reweave.runtime.capability import CandidateExecutionFailure, ModelInputRejected
from reweave.runtime.model_input import explicit_text_page, feedback_json_values

# These are archived sensing payloads, not incremental task feedback. Original
# records retain them; a native session already owns earlier delivered cameras.
IMAGE_FIELDS = frozenset(
    {
        "_runtime_images",
        "_image_bytes",
        "_image_cam_bytes",
        "_image_wrist_bytes",
        "_image_labels",
        "image_inputs",
        "archived_image_references",
    }
)
# Default feedback and the final-C guard share the same representative budget.
MAX_GRASP_REJECTION_EXAMPLES = 3


def _grasp_provenance(value, *, preserve_images):
    """Project a large rejection pool; complete tool/audit originals stay intact."""
    rejections = value.get("support_rejections")
    if (
        not isinstance(rejections, list)
        or len(rejections) <= MAX_GRASP_REJECTION_EXAMPLES
    ):
        return feedback_text(value, preserve_images=preserve_images)
    projected = feedback_text(
        {key: item for key, item in value.items() if key != "support_rejections"},
        preserve_images=preserve_images,
    )
    reasons = Counter()
    examples, signatures = [], set()
    for index, candidate in enumerate(rejections):
        signature = tuple(candidate.get("rejection_reasons", ()))
        reasons.update(signature)
        if signature not in signatures and len(examples) < MAX_GRASP_REJECTION_EXAMPLES:
            examples.append(index)
            signatures.add(signature)
    for index in range(len(rejections)):
        if len(examples) == MAX_GRASP_REJECTION_EXAMPLES:
            break
        if index not in examples:
            examples.append(index)
    projected["support_rejections"] = [
        feedback_text(rejections[index], preserve_images=preserve_images)
        for index in examples
    ]
    projected["support_rejections_summary"] = {
        "total_records": len(rejections),
        "displayed_records": len(examples),
        "omitted_records": len(rejections) - len(examples),
        "reason_counts": dict(sorted(reasons.items())),
        "truncated": True,
        "selection": "First distinct reason combinations, then first remaining records; not a statistical sample.",
        "complete_original": "Host original tool result and execution audit retain every rejection.",
    }
    return projected


def feedback_text(value, *, preserve_images=False):
    """Remove camera payloads from new feedback, including serialized tool text."""
    if isinstance(value, dict):
        if explicit_text_page(value):
            return copy.deepcopy(value)
        if value.get("kind") in {"binary", "image-url"}:
            return copy.deepcopy(value) if preserve_images else None
        if "capability_result" in value and "state" in value:
            outcome = value.get("active_execution_outcome")
            return {
                **{
                    key: feedback_text(value[key], preserve_images=preserve_images)
                    for key in (
                        "step",
                        "terminated",
                        "truncated",
                        "active_execution_outcome",
                        "capability_result",
                        "error",
                        "interrupted",
                    )
                    if key in value
                },
                "outcome_known": False
                if value.get("unknown") is True or outcome == "unknown"
                else True
                if outcome in {"completed", "failed", "rejected"}
                else None,
            }
        if "state" in value and "artifacts" in value:
            return {
                **{
                    key: value[key]
                    for key in (
                        "step",
                        "terminated",
                        "truncated",
                        "error",
                        "task_language",
                        "observation_ref",
                    )
                    if key in value
                },
                "state": copy.deepcopy(value["state"]["state"]),
                **(
                    {"log": feedback_text(value["log"], preserve_images=True)}
                    if preserve_images and "log" in value
                    else {}
                ),
            }
        return {
            key: _grasp_provenance(item, preserve_images=preserve_images)
            if key == "provenance" and isinstance(item, dict)
            else feedback_text(item, preserve_images=preserve_images)
            for key, item in value.items()
            if key not in IMAGE_FIELDS
            and key != "active_execution_ref"
            and not (key == "traceback" and "error" in value)
        }
    if isinstance(value, list):
        return [
            feedback_text(item, preserve_images=preserve_images)
            for item in value
            if not (
                not preserve_images
                and isinstance(item, dict)
                and item.get("kind") in {"binary", "image-url"}
            )
        ]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, (dict, list)):
            return json.dumps(
                feedback_text(parsed, preserve_images=preserve_images),
                ensure_ascii=False,
            )
    return value


def _explicit_image_message(message):
    """Identify an actual newly settled image-read operation, not old feedback."""
    explicit = {"read_image", "view_env_state"}
    for part in message.get("parts", ()):
        if part.get("part_kind") == "tool-return" and part.get("tool_name") in explicit:
            return True
        if part.get("part_kind") != "user-prompt":
            continue
        content = part.get("content")
        for item in content if isinstance(content, list) else [content]:
            if not isinstance(item, str):
                continue
            try:
                payload = json.loads(item)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            executed = payload.get("executed")
            feedback = payload.get("feedback")
            if isinstance(executed, dict) and executed.get("name") in explicit:
                return True
            if isinstance(feedback, dict) and feedback.get("operation") in explicit:
                return True
    return False


def contains_text(value, text) -> bool:
    """Whether any model-visible string, including serialized tool JSON, holds *text*."""
    if isinstance(value, dict):
        return any(contains_text(item, text) for item in value.values())
    if isinstance(value, list):
        return any(contains_text(item, text) for item in value)
    if isinstance(value, str):
        return text in value or any(
            contains_text(parsed, text) for parsed in feedback_json_values(value)
        )
    return False


class PlannerObservation:
    """Supply latest cameras by default without replaying intermediate action images."""

    def __init__(self):
        self.images = ()

    def validate(self, context):
        """Editable C may select evidence, but cannot reinsert an expanded GG pool."""

        def visit(value):
            if isinstance(value, dict):
                if value.get("kind") in {"binary", "image-url"}:
                    return
                if "support_rejections" in value:
                    rejected = value["support_rejections"]
                    if (
                        isinstance(rejected, list)
                        and len(rejected) > MAX_GRASP_REJECTION_EXAMPLES
                    ):
                        raise ModelInputRejected(
                            "native C contains expanded GraspGen rejection diagnostics; "
                            "retain counts/reasons and at most three representative records; "
                            "the complete original remains in the Host tool audit"
                        )
                if explicit_text_page(value):
                    return
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, str):
                for parsed in feedback_json_values(value):
                    visit(parsed)

        visit(
            {
                "instructions": context["instructions"],
                "messages": context["messages"],
            }
        )

    def prepare(self, context, observation, images):
        if len(images) != len(CAMERA_ARTIFACTS) or any(not data for data in images):
            raise CandidateExecutionFailure(
                "RoboDojo native decisions require all three current camera artifacts"
            )
        self.images = tuple(images)
        context = project_native_feedback(context)
        projected = {}
        for key, value in context.items():
            if key == "observation":
                continue
            if key in {
                "assets",
                "tools",
                "base_instructions",
                "request_limits",
                "source_messages",
            }:
                projected[key] = copy.deepcopy(value)
            elif key == "messages":
                projected[key] = [
                    feedback_text(
                        message, preserve_images=_explicit_image_message(message)
                    )
                    for message in value
                ]
            else:
                projected[key] = feedback_text(value)
        projected["observation"] = copy.deepcopy(observation)
        projected["current_camera_observation"] = {
            "observation_id": f"robodojo-record-{observation['step']}",
            "record_step": observation["step"],
            "cameras": [{"name": camera} for camera, _ in CAMERA_ARTIFACTS],
        }
        projected["current_observation_in_messages"] = True
        projected["messages"].append(
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": [
                            json.dumps(
                                {
                                    "current_camera_observation": projected[
                                        "current_camera_observation"
                                    ],
                                    "observation": projected["observation"],
                                    "physical_budget": projected.get("physical_budget"),
                                    "remaining_wall_time_s": projected.get(
                                        "remaining_wall_time_s"
                                    ),
                                    "objective": projected.get("objective"),
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            *[
                                {
                                    "kind": "binary",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(data).decode(),
                                }
                                for data in self.images
                            ],
                        ],
                    }
                ],
            }
        )
        return projected
