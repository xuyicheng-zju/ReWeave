"""Incremental native messages and durable, readable episode evidence."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

from reweave.evolution.bundle import ContextPackage
from reweave.evolution.evidence.records import store_bytes
from reweave.runtime.model_input import (
    image_bytes,
    project_diagnostic_feedback,
    validate_native_request,
)

NATIVE_INPUT_CONTRACT = "current-observation-and-explicit-reads-v2"


def project_native_feedback(context):
    """Default media view; complete incremental originals remain available to C."""
    from copy import deepcopy

    messages = deepcopy(context["messages"])
    effects = context.get("tool_effects", {})
    physical_pixels = set()

    def internal_read(feedback):
        owner = feedback.get("program_call_id")
        reader = context.get("program_call_id")
        # A program can deliberately read evidence before its own plan(). Do
        # not broadcast its control-loop images to a different program's plan.
        # Missing legacy ownership is not evidence that pixels may be omitted.
        return (
            feedback.get("feedback_source") == "program"
            and owner is not None
            and reader is not None
            and owner != reader
        )

    def collect_physical(value):
        if isinstance(value, dict):
            for frame in value.get("_runtime_images", ()):
                physical_pixels.add(
                    base64.b64decode(
                        frame["data_base64"], altchars=b"-_", validate=True
                    )
                )
            for key, item in value.items():
                if key != "_runtime_images":
                    collect_physical(item)
        elif isinstance(value, list):
            for item in value:
                collect_physical(item)

    for settled in context.get("history", []):
        if effects.get(settled.get("operation")) == "physical":
            collect_physical(settled.get("result"))

    def operation(part):
        if part.get("part_kind") == "tool-return":
            return part.get("tool_name"), False
        content = part.get("content")
        text = content[0] if isinstance(content, list) and content else content
        if not isinstance(text, str):
            return None, False
        try:
            value = json.loads(text)
        except ValueError:
            return None, False
        if isinstance(value, dict):
            if isinstance(value.get("feedback"), dict) and "history_index" in value:
                feedback = value["feedback"]
                return feedback.get("operation"), internal_read(feedback)
            if isinstance(value.get("executed"), dict):
                return value["executed"].get("name"), internal_read(value)
        return None, False

    def references(value, physical):
        data = image_bytes(value)
        if data is not None:
            if physical or data in physical_pixels:
                return "[Intermediate control pixels omitted from the default view; originals remain in source_messages and archived tool references. The current observation is supplied separately.]"
            return value
        if isinstance(value, dict):
            return {key: references(item, physical) for key, item in value.items()}
        if isinstance(value, list):
            return [references(item, physical) for item in value]
        return value

    for message in messages:
        physical = False
        workflow = False
        for part in message.get("parts", []):
            name, program_read = operation(part)
            if name is not None:
                physical = effects.get(name) == "physical" or (
                    effects.get(name) == "read" and program_read
                )
                workflow = name.startswith("workflow_")
            if physical or workflow:
                part["content"] = references(part.get("content"), physical)
    return {
        **context,
        "source_messages": context.get("source_messages", context["messages"]),
        "messages": messages,
    }


def prepare_observation(context: dict) -> dict:
    """Host binds current observation pixels before editable native C runs."""
    context = {
        key: project_diagnostic_feedback(value)
        if key in {"messages", "observation", "history", "last_tool_result"}
        else value
        for key, value in context.items()
    }
    if context.get("native_input_contract") == NATIVE_INPUT_CONTRACT:
        return context
    # Recorded v1 inputs already contain a Host observation part. Rebuild only
    # that known envelope so references cannot point into projected feedback.
    if context.get("current_observation_in_messages", False):
        messages = json.loads(json.dumps(context["messages"]))
        rebuilt = False
        for message in messages:
            retained = []
            for part in message.get("parts", []):
                content = part.get("content")
                text = content[0] if isinstance(content, list) and content else content
                try:
                    value = json.loads(text) if isinstance(text, str) else None
                except ValueError:
                    value = None
                if (
                    part.get("part_kind") == "user-prompt"
                    and isinstance(value, dict)
                    and set(value)
                    == {
                        "observation",
                        "physical_budget",
                        "remaining_wall_time_s",
                        "objective",
                    }
                ):
                    rebuilt = True
                else:
                    retained.append(part)
            message["parts"] = retained
        if not rebuilt:
            return context
        context = {
            **context,
            "messages": messages,
            "current_observation_in_messages": False,
        }
    context = project_native_feedback(context)
    selected = json.loads(
        json.dumps(
            [message for message in context["messages"] if message["kind"] == "request"]
        )
    ) or [{"kind": "request", "parts": []}]
    image_sources = []

    def pointer(path, key):
        return path + "/" + str(key).replace("~", "~0").replace("/", "~1")

    def collect(value, path):
        if isinstance(value, dict):
            data = image_bytes(value)
            if data is not None:
                image_sources.append((data, path))
            else:
                for key, item in value.items():
                    collect(item, pointer(path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect(item, pointer(path, index))

    collect(selected, "/messages")
    new_images = []
    content_path = (
        f"/messages/{len(selected) - 1}/parts/{len(selected[-1]['parts'])}/content"
    )

    def observation_view(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key != "_runtime_images":
                    result[key] = observation_view(item)
                    continue
                references = []
                for frame in item:
                    raw = base64.b64decode(
                        frame["data_base64"], altchars=b"-_", validate=True
                    )
                    match = next(
                        (path for data, path in image_sources if data == raw),
                        None,
                    )
                    if match is None:
                        new_images.append(
                            {
                                "kind": "binary",
                                "media_type": frame["media_type"],
                                "data": frame["data_base64"],
                            }
                        )
                        match = content_path + "/" + str(len(new_images))
                        image_sources.append(
                            (
                                raw,
                                match,
                            )
                        )
                    references.append(
                        {
                            **{
                                name: field
                                for name, field in frame.items()
                                if name != "data_base64"
                            },
                            "image_reference": match,
                        }
                    )
                result[key] = references
            return result
        if isinstance(value, list):
            return [observation_view(item) for item in value]
        return value

    text = json.dumps(
        {
            "observation": observation_view(context["observation"]),
            "physical_budget": context["physical_budget"],
            "remaining_wall_time_s": context["remaining_wall_time_s"],
            "objective": context["objective"],
        }
    )
    selected[-1]["parts"].append(
        {
            "part_kind": "user-prompt",
            "content": [text, *new_images] if new_images else text,
        }
    )
    return {
        **context,
        "messages": selected,
        "current_observation_in_messages": True,
        "native_input_contract": NATIVE_INPUT_CONTRACT,
    }


def baseline_context() -> ContextPackage:
    """Editable task guidance; Host and native provider own camera/history delivery."""
    return ContextPackage()


def prepare_native(
    context: dict, directory: Path, limits, *, include_tools: bool = True
) -> tuple[str, tuple[Path, ...]]:
    """Materialize selected pixels, retaining the complete C request budget.

    Omit tool definitions only after this session accepted the exact catalog
    and its native context has not compacted since that delivery.
    """
    selected = validate_native_request(context, limits)
    images = []
    materialized = []
    directory.mkdir(parents=True, exist_ok=True)

    def visit(value):
        if isinstance(value, dict):
            data = image_bytes(value)
            if data is None:
                return {key: visit(item) for key, item in value.items()}
            with Image.open(io.BytesIO(data)) as picture:
                extension = picture.format.lower()
                picture.verify()
            for previous_data, previous_path in materialized:
                if data == previous_data:
                    return {"image_file": str(previous_path)}
            stored = store_bytes(data, directory, suffix="." + extension)
            path = directory / stored["path"]
            # This Host-owned append-only pool is outside the native workspace.
            # Providers may link its immutable bytes instead of copying them for
            # every turn; caller-owned writable files still require snapshots.
            path.chmod(0o444)
            images.append(path.resolve())
            materialized.append((data, path.resolve()))
            return {"image_file": str(path.resolve())}
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        return value

    delivery = {"messages": visit(selected)}
    if include_tools:
        delivery["tools"] = context["tools"]
    else:
        # The session has already accepted these exact definitions. Keep the
        # current allowed names explicit while native history owns the catalog.
        delivery["available_tools"] = [tool["name"] for tool in context["tools"]]
        delivery["tool_catalog"] = "unchanged"
    return json.dumps(delivery, ensure_ascii=False, separators=(",", ":")), tuple(
        images
    )


class EpisodeRecord:
    """Append-only readable evidence, outside the agent's editable source tree."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "episode.md"

    def append(self, kind: str, request_id: str, content: str, images=()):
        with self.path.open("a") as stream:
            stream.write(f"\n## {kind} — {request_id}\n\n")
            stream.write(content + "\n")
            for path in images:
                stream.write(f"\n![observation]({path})\n")
