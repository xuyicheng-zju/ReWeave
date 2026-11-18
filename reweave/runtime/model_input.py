# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Auditable image projection and one request-body budget for root and local models."""

from __future__ import annotations

import base64
import json
from typing import Any

from reweave.evolution.bundle import ImageSelection, RequestLimits
from reweave.runtime.capability import CandidateExecutionFailure, ModelInputRejected


def explicit_text_page(value):
    """Recognize a selected file page; its content is evidence, not an envelope."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and type(value.get("size")) is int
        and isinstance(value.get("content"), str)
        and type(value.get("offset")) is int
    )


def project_diagnostic_feedback(value):
    """Keep actionable errors; complete exception stacks remain in Host originals."""
    if isinstance(value, dict):
        page = explicit_text_page(value)
        return {
            key: item
            if page and key == "content"
            else project_diagnostic_feedback(item)
            for key, item in value.items()
            if not (key == "traceback" and "error" in value)
        }
    if isinstance(value, list):
        return [project_diagnostic_feedback(item) for item in value]
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        projected = project_diagnostic_feedback(parsed)
        return (
            value if projected == parsed else json.dumps(projected, ensure_ascii=False)
        )
    return value


def feedback_json_values(text):
    """Read JSON/JSONL or JSON embedded in explanatory tool text without guessing."""
    stripped = text.lstrip()
    if stripped.startswith(('"', "{", "[")):
        try:
            yield json.loads(text)
            return
        except ValueError:
            pass
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        starts = [
            index for token in ("{", "[") if (index := text.find(token, position)) >= 0
        ]
        if not starts:
            return
        position = min(starts)
        try:
            value, end = decoder.raw_decode(text, position)
        except ValueError:
            position += 1
        else:
            yield value
            position = end


def validate_native_feedback(messages):
    """Reject known audit envelopes after editable C; do not rewrite its selection."""

    def visit(value, path):
        if isinstance(value, dict):
            if value.get("kind") in {"binary", "image-url"}:
                return
            event = value.get("event_type")
            raw_control = (
                event == "control_step"
                and {
                    "control_step_index",
                    "execution_ref",
                    "action",
                    "terminated",
                    "truncated",
                }
                <= value.keys()
            )
            raw_telemetry = (
                event == "control_telemetry"
                and {
                    "control_step_index",
                    "control_event_index",
                    "execution_ref",
                    "public_evidence",
                    "critic_private_evidence",
                }
                <= value.keys()
            )
            raw_forward = (
                event == "vla_forward"
                and {
                    "forward_id",
                    "execution_ref",
                    "prediction_horizon_steps",
                    "execution_horizon_steps",
                    "inference_elapsed_s",
                }
                <= value.keys()
            )
            if raw_control or raw_telemetry or raw_forward:
                raise ModelInputRejected(
                    f"native C contains a complete execution audit record at {path}; "
                    "select needed measured facts and their references instead of raw trace/ACK/action payloads"
                )
            if (
                "error" in value
                and isinstance(value.get("traceback"), str)
                and value["traceback"]
            ):
                raise ModelInputRejected(
                    f"native C contains a full exception traceback at {path}; "
                    "keep the actionable error and known outcome; the Host retains the original stack"
                )
            page = explicit_text_page(value)
            for key, item in value.items():
                if not (page and key == "content"):
                    visit(item, _pointer(path, key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, _pointer(path, index))
        elif isinstance(value, str):
            for parsed in feedback_json_values(value):
                visit(parsed, path + "#json")

    visit(messages, "/messages")


def _pointer(path, key):
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def image_bytes(value):
    if not isinstance(value, dict):
        return None
    if value.get("kind") == "binary" and str(value.get("media_type", "")).startswith(
        "image/"
    ):
        encoded = value["data"]
    elif value.get("kind") == "image-url":
        url = value["url"]
        if not url.startswith("data:image/") or ";base64," not in url:
            raise ModelInputRejected(
                "Native images must be embedded bytes, not external URLs"
            )
        encoded = url.split(",", 1)[1]
    else:
        return None
    try:
        return base64.b64decode(encoded, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as error:
        raise ModelInputRejected(
            "Native image contains invalid embedded base64 bytes"
        ) from error


def image_inventory(value: Any) -> list[dict]:
    """List each occurrence, including duplicates, without copying image bytes."""
    result = []

    def visit(item, path):
        image = image_bytes(item)
        if image is not None:
            result.append({"path": path, "bytes": len(image)})
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, _pointer(path, key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, _pointer(path, index))

    visit(value, "")
    return result


def unique_image_count(value):
    images = set()

    def visit(item):
        data = image_bytes(item)
        if data is not None:
            images.add(data)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return len(images)


def validate_native_images(messages: Any) -> None:
    """Bound authored native C before selection; no fixed camera or time window."""
    count = unique_image_count(messages)
    if count > 10:
        raise ModelInputRejected(
            f"native C output exceeds image budget: {count} > 10; "
            "select relevant evidence in C rather than hiding extras with image_selection"
        )


def validate_native_request(context: dict, limits: RequestLimits) -> Any:
    """Validate the shared native input contract without files or a model call.

    Selection pointers use the messages list as their root. The pre-selection
    image ceiling and post-selection deployment limits are separate contracts.
    Runtime delivery and offline replay use this exact same check.
    """
    validate_native_images(context["messages"])
    try:
        selected, _ = select_images(context["messages"], context.get("image_selection"))
    except CandidateExecutionFailure as error:
        raise ModelInputRejected(str(error)) from error
    validate_native_feedback(
        {"instructions": context["instructions"], "messages": context["messages"]}
    )
    validate_request(
        request_measurements(context["instructions"], selected, context["tools"]),
        limits,
    )
    return selected


def select_images(value: Any, selection: dict | None) -> tuple[Any, dict]:
    """Keep all text and explicit image references; select only specified pixels."""
    inventory = image_inventory(value)
    paths = {item["path"] for item in inventory}
    policy = ImageSelection.model_validate(selection) if selection is not None else None
    selected = set(policy.paths) if policy is not None else paths
    if selected - paths:
        raise CandidateExecutionFailure(
            f"image selection references missing evidence: {sorted(selected - paths)}"
        )
    references = {
        item["path"]: {**item, "selected": item["path"] in selected}
        for item in inventory
    }

    def visit(item, path):
        if path in references:
            return (
                item
                if path in selected
                else "[omitted image "
                + json.dumps(references[path], sort_keys=True)
                + "]"
            )
        if isinstance(item, dict):
            return {
                key: visit(child, _pointer(path, key)) for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [
                visit(child, _pointer(path, index)) for index, child in enumerate(item)
            ]
        return item

    return visit(value, ""), {
        "reason": policy.reason
        if policy is not None
        else "All supplied images retained; no selection requested.",
        "images": list(references.values()),
        "source_images": len(inventory),
        "selected_images": len(selected),
    }


def request_measurements(instructions, messages, tools):
    payload = {"instructions": instructions, "messages": messages, "tools": tools}
    return {
        "images": unique_image_count(messages),
        "input_bytes": len(
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        ),
    }


def validate_request(measurements: dict, limits: RequestLimits):
    for quantity, maximum in (
        ("images", limits.max_images),
        ("input_bytes", limits.max_input_bytes),
    ):
        if maximum is not None and measurements[quantity] > maximum:
            raise ModelInputRejected(
                f"candidate model input exceeds deployment {quantity} budget: "
                f"{measurements[quantity]} > {maximum}; select necessary evidence explicitly; originals retained"
            )
