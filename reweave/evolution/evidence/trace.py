# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Deterministic offline navigation over complete, immutable episode evidence.

Unlike a sampled diagnostic trace, every recorded control step is retained here.
This module describes evidence; it never labels a root cause or predicts a changed
program's physical outcome. Generated references use a stable virtual namespace;
the Host resolves them against request-scoped, read-only evidence roots.
"""

from __future__ import annotations

import json
import math
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from reweave.evolution.evidence.records import (
    EpisodeEvidence,
    artifact_record,
    inventory_episode,
    restore_record,
)
from reweave.utils.persistence import atomic_text

TRACE_INDEX_SCHEMA_VERSION = 1


def _read(path: Path) -> Any:
    return restore_record(json.loads(path.read_text(encoding="utf-8")), path.parent)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            yield row


def _trace_reference_path(
    workspace: Path,
    reference: dict[str, str],
    evidence_roots: dict[str, Path] | None = None,
) -> Path:
    """Bind an original to this request's evidence root without rehashing it."""

    if set(reference) != {"artifact_id", "path", "bytes", "pointer"}:
        raise ValueError("reference requires artifact_id, path, bytes and pointer")
    relative = Path(reference["path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != (".archive", "episodes")
        or len(relative.parts) < 4
    ):
        raise ValueError("reference must point into the archived episode namespace")
    evidence_id = relative.parts[2]
    if evidence_roots is None:
        root = workspace / ".archive/episodes" / evidence_id
    else:
        root = evidence_roots.get(evidence_id)
        if root is None:
            raise ValueError("reference names evidence outside this author request")
    suffix = Path(*relative.parts[3:])
    path = root / suffix
    if path.resolve() != root.resolve() / suffix or not path.is_file():
        raise ValueError("reference is missing or follows a link")
    return path


def resolve_trace_reference(
    workspace: Path,
    reference: dict[str, str],
    evidence_roots: dict[str, Path] | None = None,
) -> Any:
    """Read one request-bound original and RFC6901 pointer without rehashing it.

    JSONL decodes only the selected row. Published artifact records identify originals;
    required files and pointers are checked when actually read.
    """
    import re

    path = _trace_reference_path(workspace, reference, evidence_roots)
    pointer = reference["pointer"]
    if pointer and not pointer.startswith("/"):
        raise ValueError("invalid JSON pointer")
    parts = pointer[1:].split("/") if pointer else []
    if any(re.search(r"~(?![01])", part) for part in parts):
        raise ValueError("invalid JSON pointer escape")
    parts = [part.replace("~1", "/").replace("~0", "~") for part in parts]
    if path.suffix == ".jsonl" and parts:
        position = parts.pop(0)
        if not position.isdecimal() or str(int(position)) != position:
            raise ValueError("JSONL pointer must start with a record index")
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index == int(position):
                    value = restore_record(json.loads(line), path.parent)
                    break
            else:
                raise ValueError("JSONL record is absent")
    else:
        value = (
            [restore_record(row, path.parent) for row in _jsonl(path)]
            if path.suffix == ".jsonl"
            else _read(path)
        )
    for part in parts:
        if isinstance(value, list):
            if not part.isdecimal() or str(int(part)) != part:
                raise ValueError("invalid array pointer")
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise ValueError("pointer descends into a scalar")
    return value


def _decode(content: Any) -> Any:
    # Frozen BundleRuntime records a JSON text block followed by images. The
    # ordinary API loop records the same result as a JSON string. Never choose
    # one of several text blocks and silently discard competing feedback.
    if isinstance(content, list):
        text = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if (
            len(text) == 1
            and isinstance(text[0].get("text"), str)
            and all(
                isinstance(block, dict) and block.get("type") in {"text", "image"}
                for block in content
            )
        ):
            return _decode(text[0]["text"])
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content  # A tool may legitimately return plain text.
    return content


class _Episode:
    def __init__(
        self, workspace: Path, evidence: EpisodeEvidence, root: Path | None = None
    ) -> None:
        self.workspace = workspace
        self.evidence = evidence
        self.root = root or workspace / ".archive/episodes" / evidence.evidence_id

    def ref(self, name: str, pointer: str = "") -> dict[str, Any] | None:
        """A published file record and pointer, or explicit absence for optional data."""
        if name not in self.evidence.files:
            return None
        return {
            "path": f".archive/episodes/{self.evidence.evidence_id}/{name}",
            **self.evidence.files[name],
            "pointer": pointer,
        }

    def resolve(self, reference: dict[str, str]) -> Path:
        relative = Path(reference["path"])
        if relative.parts[:3] != (
            ".archive",
            "episodes",
            self.evidence.evidence_id,
        ):
            raise ValueError("reference belongs to another episode")
        return self.root / Path(*relative.parts[3:])


def _executions(
    episode: _Episode, events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    executions: dict[str, dict[str, Any]] = {}
    stack: list[dict[str, Any]] = []
    physical_ids: set[str] = set()
    for index, event in enumerate(events):
        if event.get("event_index") != index:
            raise ValueError("execution event_index must match its JSONL record index")
        kind = event["event_type"]
        if kind == "active_execution_started":
            identity = event["execution_id"]
            if identity in executions or stack:
                raise ValueError("duplicate or overlapping top-level execution")
            call = {
                "physical_call_ref": identity,
                "parent_physical_call_ref": None,
                "tool_name": event["tool_name"],
                "arguments_reference": episode.ref(
                    "execution_trace.jsonl", f"/{index}/arguments"
                ),
                "start_event": index,
                "finish_event": None,
                "control_step_indices": [],
            }
            executions[identity] = {
                "execution_ref": identity,
                "tool_name": event["tool_name"],
                "arguments_reference": call["arguments_reference"],
                "start_event": index,
                "finish_event": None,
                "outcome": None,
                "planner_call_ids": [],
                "state_indices": [],
                "physical_calls": [call],
                "control_step_indices": [],
            }
            if identity in physical_ids:
                raise ValueError("duplicate physical call identity")
            physical_ids.add(identity)
            stack.append(call)
        elif kind in {
            "physical_tool_started",
            "physical_tool_finished",
            "control_step",
            "active_execution_finished",
        }:
            execution = executions.get(event["execution_ref"])
            if (
                execution is None
                or not stack
                or execution["finish_event"] is not None
                or stack[0]["physical_call_ref"] != event["execution_ref"]
            ):
                raise ValueError(f"{kind} references an inactive execution")
            if kind == "physical_tool_started":
                identity = event["physical_call_ref"]
                if identity in physical_ids:
                    raise ValueError("duplicate physical call identity")
                physical_ids.add(identity)
                call = {
                    "physical_call_ref": identity,
                    "parent_physical_call_ref": stack[-1]["physical_call_ref"],
                    "tool_name": event["tool_name"],
                    "arguments_reference": episode.ref(
                        "execution_trace.jsonl", f"/{index}/arguments"
                    ),
                    "start_event": index,
                    "finish_event": None,
                    "control_step_indices": [],
                }
                execution["physical_calls"].append(call)
                stack.append(call)
            elif kind == "control_step":
                if event["physical_call_ref"] != stack[-1]["physical_call_ref"]:
                    raise ValueError(
                        "control step does not belong to the active physical leaf"
                    )
                stack[-1]["control_step_indices"].append(event["control_step_index"])
                execution["control_step_indices"].append(event["control_step_index"])
            elif kind == "physical_tool_finished":
                if (
                    len(stack) < 2
                    or stack[-1]["physical_call_ref"] != event["physical_call_ref"]
                ):
                    raise ValueError("physical calls did not unwind in order")
                stack.pop()["finish_event"] = index
            else:
                if (
                    len(stack) != 1
                    or stack[0]["physical_call_ref"] != event["execution_ref"]
                ):
                    raise ValueError(
                        "execution finished with an active nested physical call"
                    )
                execution["finish_event"] = index
                execution["outcome"] = event["outcome"]
                stack.pop()["finish_event"] = index
    # A crashed episode can end inside an invocation. Preserve the unfinished call.
    return list(executions.values())


def _native_turns(episode: _Episode) -> dict[str, dict[str, Any]]:
    """Index Host-published native responses and their exact provider inputs.

    A prepared provider request or native event alone is not a Host decision.
    The Host response records accepted call IDs and parsed arguments; rejected
    proposals retain their own rejection event instead of becoming tool calls.
    """
    name = "harness_context/bundle_events.jsonl"
    if episode.ref(name) is None:
        return {}
    session_name = "native/provider/session.json"
    session = _read(episode.root / session_name) if episode.ref(session_name) else {}
    requests = session.get("requests", {})
    rejected = {}
    turns = {}
    for index, event in enumerate(_iter_jsonl(episode.root / name)):
        if event.get("kind") == "model_call_rejected":
            call = event["call"]
            rejected[(event["request_id"], call["id"])] = (
                call,
                episode.ref(name, f"/{index}/call"),
            )
            continue
        if (
            event.get("kind") != "model_response"
            or event.get("controller_request_unit") != "native_turn"
        ):
            continue
        identity = event["request_id"]
        response = event["response"]
        metadata = response.get("metadata") or {}
        if (
            identity in turns
            or metadata.get("request_id") != identity
            or metadata.get("program_call_id") != event.get("program_call_id")
        ):
            raise ValueError("native response has inconsistent request identity")
        request = requests.get(identity)
        inputs = []
        completion = None
        confirmed = True
        if request is not None:
            directory = Path(request["directory"])
            if (
                len(directory.parts) != 1
                or directory.name in {".", ".."}
                or directory.is_absolute()
            ):
                raise ValueError("native request directory must be episode-relative")
            prefix = f"native/provider/requests/{directory.name}"
            input_name = prefix + "/input.json"
            completion_name = prefix + "/native_completion.json"
            confirmed = request.get("phase") == "completed"
            if episode.ref(completion_name):
                actual_completion = _read(episode.root / completion_name)
                confirmed = confirmed and (
                    actual_completion.get("status") == "completed"
                    and actual_completion.get("id") == request.get("turn_id")
                )
                completion = episode.ref(completion_name)
            if confirmed and episode.ref(input_name):
                inputs.append(episode.ref(input_name))
        turns[identity] = {
            "event": event,
            "response": response,
            "source": episode.ref(name, f"/{index}/response"),
            "session_id": session.get("session_id"),
            "confirmed": confirmed,
            "inputs": inputs,
            "completion": completion,
            "accepted": {
                part["tool_call_id"]: (
                    part,
                    episode.ref(name, f"/{index}/response/parts/{position}"),
                )
                for position, part in enumerate(response.get("parts", []))
                if part.get("part_kind") == "tool-call"
            },
            "rejected": {
                call_id: value
                for (request_id, call_id), value in rejected.items()
                if request_id == identity
            },
        }
    return turns


def _planner_calls(
    episode: _Episode,
    transcript_name: str,
    transcript: dict[str, Any],
    executions: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]
]:
    calls: dict[str, dict[str, Any]] = {}
    request_signatures = {}
    results: list[tuple[str | None, int, dict[str, Any]]] = []
    planner_results: set[str] = set()
    program_feedback = []
    by_execution = {row["execution_ref"]: row for row in executions}
    requests = {}
    for execution in executions:
        request_id = (
            events[execution["start_event"]]["arguments"].get("request_id")
            if execution["tool_name"] in {"run_harness_program", "modify_harness"}
            else None
        )
        if request_id is not None:
            if request_id in requests:
                raise ValueError("duplicate runtime request identity")
            requests[request_id] = execution["execution_ref"]
    gaps = []
    request_contexts = _decision_contexts(episode)
    native_turns = _native_turns(episode)
    for index, message in enumerate(transcript.get("messages", [])):
        if (
            message.get("role") == "assistant"
            and isinstance(message.get("content"), dict)
            and isinstance(message["content"].get("tool_calls"), list)
            and "native_session_id" in message
        ):
            request_id = message.get("request_id")
            turn = native_turns.get(request_id)
            if turn is None or not turn["confirmed"]:
                gaps.append(
                    {
                        "kind": "native_decision_not_confirmed",
                        "source": episode.ref(transcript_name, f"/messages/{index}"),
                        "request_id": request_id,
                    }
                )
                continue
            if turn["event"].get("program_call_id") != message.get(
                "program_call_id"
            ) or turn["session_id"] not in {None, message["native_session_id"]}:
                raise ValueError(
                    "native transcript and Host response identities differ"
                )
            metadata = turn["response"].get("metadata") or {}
            for proposal_index, proposal in enumerate(message["content"]["tool_calls"]):
                # These are the published native_loop IDs, bound to the request
                # and proposal index rather than the order of arriving results.
                provider_id = f"{request_id}_{proposal_index}"
                identity = metadata.get("tool_call_ids", {}).get(provider_id)
                accepted = turn["accepted"].get(provider_id)
                rejected = turn["rejected"].get(identity)
                if identity is None or (accepted is None and rejected is None):
                    gaps.append(
                        {
                            "kind": "native_proposal_without_host_disposition",
                            "source": episode.ref(
                                transcript_name,
                                f"/messages/{index}/content/tool_calls/{proposal_index}",
                            ),
                        }
                    )
                    continue
                if identity in calls:
                    raise ValueError(f"duplicate Planner tool call id: {identity}")
                actual, reference = accepted if accepted is not None else rejected
                arguments_key = "args" if accepted is not None else "arguments"
                actual_name = actual.get("tool_name", actual.get("name"))
                if actual_name != proposal["name"]:
                    raise ValueError("native proposal and Host tool names disagree")
                request_signatures[identity] = {
                    "name": actual_name,
                    "arguments": actual[arguments_key],
                }
                calls[identity] = {
                    "tool_call_id": identity,
                    "name": actual_name,
                    "arguments_reference": {
                        **reference,
                        "pointer": reference["pointer"] + "/" + arguments_key,
                    },
                    "decision": episode.ref(transcript_name, f"/messages/{index}"),
                    "request": episode.ref(
                        transcript_name,
                        f"/messages/{index}/content/tool_calls/{proposal_index}",
                    ),
                    "feedback": None,
                    "execution_ref": None,
                    "state_index": None,
                    "decision_inputs": turn["inputs"],
                    "native_completion": turn["completion"],
                    "native_response": turn["source"],
                    "disposition": reference,
                    "proposal_status": "accepted"
                    if accepted is not None
                    else "rejected",
                    "program_call_id": message.get("program_call_id"),
                    "provider_tool_call_id": provider_id,
                    "request_id": request_id,
                }
        elif message.get("role") == "assistant" and isinstance(
            message.get("content"), list
        ):
            for block_index, block in enumerate(message["content"]):
                if block.get("type") != "tool_use":
                    continue
                identity = block["id"]
                if identity in calls:
                    raise ValueError(f"duplicate Planner tool call id: {identity}")
                request_signatures[identity] = {
                    "name": block["name"],
                    "arguments": block["input"],
                }
                calls[identity] = {
                    "tool_call_id": identity,
                    "name": block["name"],
                    "arguments_reference": episode.ref(
                        transcript_name,
                        f"/messages/{index}/content/{block_index}/input",
                    ),
                    "decision": episode.ref(transcript_name, f"/messages/{index}"),
                    "request": episode.ref(
                        transcript_name, f"/messages/{index}/content/{block_index}"
                    ),
                    "feedback": None,
                    "execution_ref": None,
                    "state_index": None,
                    "decision_inputs": request_contexts.get(identity, []),
                }
        elif message.get("role") == "tool":
            identity = message["tool_call_id"]
            # Program-origin execute/read feedback has no Planner tool-call ID.
            # Its archival message reference is its identity, never a synthetic
            # Planner ID; several such messages are legitimate in one episode.
            if identity is not None:
                if identity in planner_results:
                    raise ValueError(f"duplicate Planner feedback id: {identity}")
                planner_results.add(identity)
            results.append((identity, index, message))
        elif message.get("role") in {"claude_agent_sdk", "codex_sdk"}:
            gaps.append(
                {
                    "kind": "native_planner_transcript_not_indexed",
                    "source": episode.ref(transcript_name, f"/messages/{index}"),
                    "note": "Read the complete native transcript; zero indexed calls does not mean no decisions.",
                }
            )
    for identity, index, message in results:
        if identity is None:
            call = {
                "origin": "program",
                "tool_call_id": None,
                "name": message.get("name"),
                "execution_ref": None,
                "state_index": None,
            }
            program_feedback.append(call)
        elif identity not in calls:
            gaps.append(
                {
                    "kind": "feedback_without_request",
                    "tool_call_id": identity,
                    "source": episode.ref(transcript_name, f"/messages/{index}"),
                }
            )
            continue
        else:
            call = calls[identity]
        if message.get("name") != call["name"]:
            raise ValueError("Planner request and feedback tool names disagree")
        call["feedback"] = episode.ref(transcript_name, f"/messages/{index}")
        for field in ("program_call_id", "provider_tool_call_id", "executed"):
            if field in message:
                call["executed_reference" if field == "executed" else field] = (
                    episode.ref(transcript_name, f"/messages/{index}/{field}")
                    if field == "executed"
                    else message[field]
                )
        feedback = _decode(message.get("content"))
        if isinstance(feedback, list):
            gaps.append(
                {
                    "kind": "structured_feedback_not_uniquely_decoded",
                    "tool_call_id": identity,
                    "source": call["feedback"],
                    "note": "No unique JSON text payload was established. Full blocks remain in the original; do not infer missing result or execution identity.",
                }
            )
        if isinstance(feedback, dict):
            execution_ref = feedback.get("active_execution_ref")
            if feedback.get("runtime_request_accepted") is True:
                linked = requests.get(feedback.get("request_id"))
                if (
                    execution_ref is not None
                    and linked is not None
                    and execution_ref != linked
                ):
                    raise ValueError(
                        "runtime request and execution feedback identities disagree"
                    )
                execution_ref = linked
                if linked is None:
                    gaps.append(
                        {
                            "kind": "runtime_request_execution_unlinked",
                            "tool_call_id": identity,
                        }
                    )
            call["execution_ref"] = execution_ref
            call["state_index"] = feedback.get("step")
            if execution_ref is not None:
                if execution_ref not in by_execution:
                    gaps.append(
                        {
                            "kind": "feedback_execution_missing",
                            "tool_call_id": identity,
                            "execution_ref": execution_ref,
                        }
                    )
                elif identity is not None:
                    by_execution[execution_ref]["planner_call_ids"].append(identity)
                else:
                    by_execution[execution_ref].setdefault(
                        "program_feedback", []
                    ).append(call["feedback"])
    for call in calls.values():
        if len(call["decision_inputs"]) != 1:
            gaps.append(
                {
                    "kind": "decision_input_link_not_unique",
                    "tool_call_id": call["tool_call_id"],
                    "matches": len(call["decision_inputs"]),
                    "note": "Only explicit request IDs or exact archival history-prefix identities are linked; read planner_requests originals for unmatched decisions.",
                }
            )
        if call["feedback"] is None:
            gaps.append(
                {
                    "kind": "request_without_feedback",
                    "tool_call_id": call["tool_call_id"],
                }
            )
    return list(calls.values()), program_feedback, gaps, request_signatures


def _bundle_actions(
    episode: _Episode, calls: list[dict[str, Any]], executions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Link fixed-kernel actions by explicit identities, never by temporal proximity."""
    name = "harness_context/bundle_events.jsonl"
    if name not in episode.evidence.files:
        return [], []
    by_call = {row["tool_call_id"]: row for row in calls}
    by_execution = {row["execution_ref"]: row for row in executions}
    rows, gaps, submitted, settled = [], [], {}, {}
    for index, event in enumerate(_iter_jsonl(episode.root / name)):
        # Pixel bodies are needed only when an author opens an original, never
        # to index a call, its feedback, or its physical execution identity.
        event.pop("image_payloads", None)
        # Payloads already live in the immutable original. Copying context_built
        # here duplicates every historical image on every model request.
        row = {
            **{
                key: value
                for key, value in event.items()
                if not isinstance(value, (dict, list))
            },
            "payloads": {
                key: episode.ref(
                    name, f"/{index}/{key.replace('~', '~0').replace('/', '~1')}"
                )
                for key, value in event.items()
                if isinstance(value, (dict, list))
            },
            "source": episode.ref(name, f"/{index}"),
        }
        rows.append(row)
        if event["kind"] == "action_submitted":
            identity = event["action_id"]
            if identity in submitted:
                raise ValueError("duplicate kernel action identity")
            submitted[identity] = event
        elif event["kind"] == "operation_settled":
            identity = event["action_id"]
            started = submitted.get(identity)
            if (
                identity in settled
                or started is None
                or any(
                    started.get(key) != event.get(key)
                    for key in (
                        "operation",
                        "planner_call_id",
                        "workflow_invocation_id",
                    )
                )
            ):
                raise ValueError(
                    "operation settlement has no unique matching submission"
                )
            settled[identity] = {
                "source": row["source"],
                "execution_ref": event.get("result", {}).get("active_execution_ref"),
            }
        elif event["kind"] in {"action_completed", "action_failed"}:
            identity = event["action_id"]
            started = submitted.pop(identity, None)
            if started is None or any(
                started.get(key) != event.get(key)
                for key in ("operation", "planner_call_id", "workflow_invocation_id")
            ):
                raise ValueError("kernel action completion has no matching submission")
            actual = (
                {
                    "source": row["source"],
                    "execution_ref": event.get("result", {}).get(
                        "active_execution_ref"
                    ),
                }
                if event["kind"] == "action_completed"
                else settled.get(identity)
            )
            settled.pop(identity, None)
            if started["effect"] != "physical" or actual is None:
                continue
            execution_ref = actual["execution_ref"]
            if execution_ref is None or execution_ref not in by_execution:
                gaps.append(
                    {
                        "kind": "kernel_physical_execution_unlinked",
                        "action_id": identity,
                        "execution_ref": execution_ref,
                        "source": actual["source"],
                    }
                )
                continue
            execution = by_execution[execution_ref]
            execution.setdefault("kernel_actions", []).append(actual["source"])
            workflow_id = event.get("workflow_invocation_id")
            if workflow_id is not None:
                execution.setdefault("workflow_invocation_ids", []).append(workflow_id)
            planner_id = event.get("planner_call_id")
            if planner_id is None:
                execution["kernel_origin"] = "loop_policy"
            elif planner_id not in by_call:
                gaps.append(
                    {
                        "kind": "kernel_planner_call_missing",
                        "tool_call_id": planner_id,
                        "source": row["source"],
                    }
                )
            else:
                if planner_id not in execution["planner_call_ids"]:
                    execution["planner_call_ids"].append(planner_id)
                by_call[planner_id].setdefault("execution_refs", []).append(
                    execution_ref
                )
    for identity in submitted:
        gaps.append({"kind": "kernel_action_without_completion", "action_id": identity})
    return rows, gaps


def _decision_contexts(episode: _Episode) -> dict[str, list[dict[str, Any]]]:
    """Link explicit request IDs or archival content identities, never response order.

    A later full_history contains earlier responses with their tool-call IDs.
    Current responses carry explicit request and program-call identities. Archived
    responses without these identities require an exact request-history prefix.
    An archival final response without later history can remain unlinked.
    """
    name = "harness_context/planner_requests.jsonl"
    if episode.ref(name) is None:
        return {}
    requests = []
    for request in _iter_jsonl(episode.root / name):
        # Current requests are joined by their explicit identities. Retaining
        # or hashing their cumulative image history adds no linkage evidence.
        requests.append(
            {
                "request_id": request["request_id"],
                "program_call_id": request.get("program_call_id"),
            }
            if "request_id" in request
            else request  # Historical archives without IDs require exact prefixes.
        )
    by_history = [
        (request["full_history"], index)
        for index, request in enumerate(requests)
        if "full_history" in request
    ]
    identities: dict[str, set[int]] = {}
    seen: set[str] = set()
    # Executable plugins record direct request identities; parallel model calls
    # can share identical histories and provider-local tool IDs.
    by_request = {}
    for index, request in enumerate(requests):
        if "request_id" in request:
            identity = request["request_id"]
            if identity in by_request:
                raise ValueError("duplicate model request identity")
            by_request[identity] = index
    responses_name = "harness_context/planner_responses.jsonl"
    if episode.ref(responses_name) is not None:
        for line in (
            (episode.root / responses_name).read_text(encoding="utf-8").splitlines()
        ):
            response_row = json.loads(line)
            if not isinstance(response_row, list) or any(
                not isinstance(item, dict) for item in response_row
            ):
                raise ValueError("planner response line must contain a message array")
            for response in response_row:
                metadata = response.get("metadata") or {}
                request_id = metadata.get("request_id")
                if request_id is None:
                    continue
                if request_id not in by_request:
                    raise ValueError("model response references missing request")
                index = by_request[request_id]
                if metadata.get("program_call_id") != requests[index].get(
                    "program_call_id"
                ):
                    raise ValueError(
                        "model response execution identity disagrees with request"
                    )
                for identity in metadata.get("tool_call_ids", {}).values():
                    if identity in identities:
                        raise ValueError("duplicate global tool-call identity")
                    identities[identity] = {index}
                    seen.add(identity)
    for request in requests:
        if "request_id" in request:
            continue
        history = request.get("full_history", [])
        for position, response in enumerate(history):
            if response.get("kind") != "response":
                continue
            ids = [
                part["tool_call_id"]
                for part in response.get("parts", [])
                if part.get("part_kind") == "tool-call"
            ]
            if not ids or all(identity in seen for identity in ids):
                continue
            matches = [
                index for prior, index in by_history if prior == history[:position]
            ]
            for identity in ids:
                identities.setdefault(identity, set()).update(matches)
                seen.add(identity)
    return {
        identity: [episode.ref(name, f"/{index}") for index in sorted(indices)]
        for identity, indices in identities.items()
    }


def _telemetry(episode: _Episode, events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in events if row["event_type"] == "control_step"]
    indices = [row["control_step_index"] for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("physical control_step_index must be contiguous from zero")
    return {
        "source": episode.ref("execution_trace.jsonl"),
        "alignment": "Every array position refers to the same recorded post-action control step. null means unrecorded, never zero. No sampling.",
        "control_step_indices": indices,
        "event_indices": [row["event_index"] for row in rows],
        "telemetry_event_indices": [row.get("telemetry_event_index") for row in rows],
        "telemetry_errors": [row.get("telemetry_error") for row in rows],
        "physical_steps_complete": not any(
            row["event_type"] == "action_outcome_unknown" for row in events
        ),
        "unconfirmed_actions": [
            {
                "source": episode.ref(
                    "execution_trace.jsonl", f"/{row['event_index']}"
                ),
                **{
                    key: row.get(key)
                    for key in (
                        "event_index",
                        "execution_ref",
                        "physical_call_ref",
                        "tool_name",
                    )
                },
            }
            for row in events
            if row["event_type"] == "action_outcome_unknown"
        ],
        "execution_refs": [row["execution_ref"] for row in rows],
        "physical_call_refs": [row["physical_call_ref"] for row in rows],
        "forward_refs": [row.get("forward_ref") for row in rows],
        "action_indices": [row.get("action_index") for row in rows],
        "tool_names": [row["tool_name"] for row in rows],
        "terminated": [row.get("terminated") for row in rows],
        "truncated": [row.get("truncated") for row in rows],
        "feature_names": {
            plane: sorted({name for row in rows for name in row.get(plane, {})})
            for plane in ("public_evidence", "critic_private_evidence")
        },
    }


def _observations(
    episode: _Episode, executions: list[dict[str, Any]], events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = episode.ref("states.json")
    if source is None:
        return [], [{"kind": "states_not_recorded"}]
    by_execution = {row["execution_ref"]: row for row in executions}
    # Capability feedback names its captured state explicitly. The ordered trace
    # gives the physical position at that feedback boundary, not at parent return.
    captures: dict[int, list[dict[str, Any]]] = {}
    last_control = None
    for event in events:
        if event["event_type"] == "control_step":
            last_control = event
        elif (
            event["event_type"]
            in {"runtime_modification_operation", "episode_tool_program_capability"}
            and event.get("effect") == "physical"
        ):
            feedback = event.get("result", {})
            if isinstance(feedback, dict) and type(feedback.get("step")) is int:
                captures.setdefault(feedback["step"], []).append(
                    {
                        "execution_ref": event["execution_ref"],
                        "event_index": event["event_index"],
                        "last_control_step_index": last_control["control_step_index"]
                        if last_control
                        else None,
                        "physical_call_ref": last_control["physical_call_ref"]
                        if last_control
                        and last_control["execution_ref"] == event["execution_ref"]
                        else None,
                    }
                )
    observations, gaps, seen = [], [], set()
    server_manifest = (
        _read(episode.root / "server_evidence.json")
        if "server_evidence.json" in episode.evidence.files
        else {}
    )
    server_snapshots = server_manifest.get("snapshots", {})
    from reweave.evolution.evidence.sensor_retention import read_retention

    retention = read_retention(episode.root)
    for index, state in enumerate(_read(episode.root / "states.json")["steps"]):
        step = state["step_idx"]
        if type(step) is not int or step < 0 or step in seen:
            raise ValueError(
                "state boundary indices must be unique non-negative integers"
            )
        seen.add(step)
        execution_ref = (state.get("command") or {}).get("active_execution_ref")
        matches = captures.get(step, [])
        matched_refs = {match["execution_ref"] for match in matches}
        if execution_ref is not None:
            matched_refs.add(execution_ref)
        if len(matched_refs) > 1:
            gaps.append(
                {
                    "kind": "ambiguous_state_execution",
                    "state_index": step,
                    "execution_refs": sorted(matched_refs),
                }
            )
            execution_ref = None
        elif matched_refs:
            execution_ref = next(iter(matched_refs))
        execution = by_execution.get(execution_ref)
        if execution_ref is not None and execution is None:
            gaps.append(
                {
                    "kind": "state_execution_missing",
                    "state_index": step,
                    "execution_ref": execution_ref,
                }
            )
        if execution is not None:
            execution["state_indices"].append(step)
        if execution_ref is None and state.get("command"):
            gaps.append({"kind": "state_without_execution_link", "state_index": step})
        positions = {match["last_control_step_index"] for match in matches}
        # An explicit top-level command captures after the invocation finishes.
        if not matches and execution and execution["finish_event"] is not None:
            positions.add(
                execution["control_step_indices"][-1]
                if execution["control_step_indices"]
                else None
            )
        if len(positions) > 1:
            gaps.append(
                {
                    "kind": "ambiguous_state_physical_position",
                    "state_index": step,
                }
            )
        observation_ref = state.get("extras", {}).get("observation_ref")
        server_original = server_snapshots.get(observation_ref)
        sensor_navigation = (
            {
                "storage": "persistent_server_archive",
                "read_with": "read_sensor_snapshot",
                "evidence_id": episode.evidence.evidence_id,
                "observation_ref": observation_ref,
                "archive_bytes": server_original.get("archive_bytes"),
                "metadata_bytes": server_original.get("metadata_bytes"),
                "meaning": "Retained exact RGB-D and calibration; not yet downloaded. Request only this snapshot when needed.",
            }
            if server_original is not None
            else None
        )
        images = []
        for name in state.get("artifacts", []):
            if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            path = Path(name) / f"{step:02d}{Path(name).suffix}"
            ref = episode.ref(path.as_posix())
            retained_image = (
                retention["model_observations"]
                .get(observation_ref, {})
                .get("images", {})
                .get(Path(name).stem)
                if retention
                else None
            )
            if ref is None and retained_image is not None:
                ref = episode.ref(retained_image["path"])
                if ref is None:
                    raise ValueError(
                        "retained original model image is absent from evidence inventory"
                    )
            state_video_only = bool(
                retention and observation_ref in retention["state_video_observations"]
            )
            if ref is None and sensor_navigation is None and not state_video_only:
                gaps.append(
                    {
                        "kind": "declared_image_missing",
                        "state_index": step,
                        "artifact": name,
                    }
                )
            images.append(
                {
                    "artifact": name,
                    "source": ref,
                    **(
                        {
                            "storage": "state_and_video_only",
                            "meaning": "Intermediate observation: state and continuous video retained; no exact RGB-D or image-time pixel claim.",
                        }
                        if state_video_only
                        else {}
                    ),
                    **(
                        {
                            "original_size_wh": retained_image["original_size_wh"],
                            "archive_size_wh": retained_image["archive_size_wh"],
                            "pixel_mapping": retained_image["pixel_mapping"],
                        }
                        if retained_image
                        else {}
                    ),
                    **(
                        {
                            "storage": "server_original_not_materialized",
                            "sensor_snapshot": sensor_navigation,
                        }
                        if ref is None and sensor_navigation is not None
                        else {}
                    ),
                }
            )
        observations.append(
            {
                "state_index": step,
                "source": episode.ref("states.json", f"/steps/{index}"),
                "task_language": state.get("extras", {}).get("task_language"),
                "official_task_instruction": state.get("extras", {}).get(
                    "official_task_instruction"
                ),
                "task_rule_supplement": state.get("extras", {}).get(
                    "task_rule_supplement"
                ),
                "execution_ref": execution_ref,
                "last_control_step_index": next(iter(positions))
                if len(positions) == 1
                else None,
                "capture_events": matches,
                "images": images,
                **(
                    {
                        "observation_ref": observation_ref,
                        "sensor_snapshot": sensor_navigation,
                    }
                    if observation_ref is not None
                    else {}
                ),
            }
        )
    return observations, gaps


def _server_forward_inputs(episode, events):
    """Bind each received RoboDojo forward to an admitted retained sensor input."""
    from reweave.evolution.evidence.server import _descriptor, _reference_parts

    transfer_failure = None
    transfer_scene = None
    gap_kind = "vla_sensor_evidence_transfer_failed"
    if "server_evidence.json" in episode.evidence.files:
        manifest = _read(episode.root / "server_evidence.json")
    else:
        name = "environment_transfer_failure.json"
        if name not in episode.evidence.files:
            # A stopped episode can be inventoried before its remote
            # transfer finishes. Preserve its admitted text/trace as a gap;
            # never claim a sensor snapshot or fabricate a transfer receipt.
            name = "environment_stop_requested.json"
            if not {name, "result.json"} <= episode.evidence.files.keys():
                raise ValueError(
                    "VLA inputs require this episode's admitted server evidence"
                )
            stop = _read(episode.root / name)
            result = _read(episode.root / "result.json")
            interrupted = isinstance(result, dict) and (
                (
                    result.get("valid") is False
                    and result.get("environment_result_available") is False
                    and result.get("success") is None
                )
                or (
                    result.get("valid") is True
                    and result.get("environment_result_available") is True
                    and result.get("success") is False
                    and result.get("environment_success") is False
                    and result.get("success_source") == "environment.is_success"
                )
            )
            if (
                not isinstance(stop, dict)
                or stop.get("automatic_replay") is not False
                or not isinstance(stop.get("service_instance_id"), str)
                or not stop["service_instance_id"]
                or not isinstance(stop.get("remote_output"), str)
                or not Path(stop["remote_output"]).is_absolute()
                or not interrupted
            ):
                raise ValueError("Invalid admitted interrupted-transfer evidence")
            gap_kind = "vla_sensor_evidence_not_admitted_after_stop"
        else:
            failure = _read(episode.root / name)
            if (
                not isinstance(failure, dict)
                or failure.get("complete") is not False
                or failure.get("automatic_replay") is not False
                or not isinstance(failure.get("error_type"), str)
                or not failure["error_type"].strip()
                or not isinstance(failure.get("error"), str)
                or not failure["error"].strip()
                or not isinstance(failure.get("remote_output"), str)
                or not Path(failure["remote_output"]).is_absolute()
            ):
                raise ValueError("Invalid admitted environment transfer failure")
        transfer_failure = episode.ref(name)
        manifest = None
        if "states.json" not in episode.evidence.files:
            raise ValueError("Transferred VLA gap requires admitted scene identity")
        states = _read(episode.root / "states.json")
        scenes = {
            _reference_parts(state["extras"]["observation_ref"])[0]
            for state in states["steps"]
            if state.get("extras", {}).get("observation_ref") is not None
        }
        if len(scenes) != 1:
            raise ValueError(
                "Transferred VLA gap has missing or inconsistent scene identity"
            )
        transfer_scene = scenes.pop()
    records = {}
    for index, row in enumerate(_jsonl(episode.root / "vla_inputs.jsonl")):
        if set(row) != {
            "forward_id",
            "observation_ref",
            "physical_step",
            "prompt",
            "mode",
            "prediction_horizon_steps",
            "execution_horizon_steps",
        }:
            raise ValueError("VLA input record contains unknown or missing fields")
        identity = row["forward_id"]
        if not isinstance(identity, str) or not identity or identity in records:
            raise ValueError("VLA forward identity must be nonempty and unique")
        reference = row["observation_ref"]
        scene, step, _ = _reference_parts(reference)
        if transfer_scene is not None and scene != transfer_scene:
            raise ValueError("VLA input belongs to a different recorded scene")
        if type(row["physical_step"]) is not int or row["physical_step"] != int(step):
            raise ValueError("VLA physical step differs from its sensor reference")
        if (
            not isinstance(row["prompt"], str)
            or not row["prompt"].strip()
            or row["mode"] not in {"eval", "train"}
        ):
            raise ValueError("VLA input instruction or inference mode is invalid")
        prediction, execution = (
            row["prediction_horizon_steps"],
            row["execution_horizon_steps"],
        )
        if (
            type(prediction) is not int
            or type(execution) is not int
            or not 0 < execution <= prediction
        ):
            raise ValueError("VLA input horizons are inconsistent")
        descriptor = _descriptor(manifest, reference) if manifest is not None else None
        records[identity] = (index, row, descriptor)
    inputs, gaps, seen = [], [], set()
    last_control = None
    for event in events:
        if event["event_type"] == "control_step":
            last_control = event["control_step_index"]
        if event["event_type"] != "vla_forward":
            continue
        identity = event["forward_id"]
        if identity in seen:
            raise ValueError("Trace has a repeated VLA forward identity")
        seen.add(identity)
        if identity not in records:
            gaps.append({"kind": "vla_input_not_archived", "forward_ref": identity})
            continue
        index, row, descriptor = records[identity]
        if (
            transfer_failure is not None
            and "instruction" in event
            and row["prompt"] != event["instruction"]
        ):
            raise ValueError(
                "VLA input instruction disagrees with the received forward"
            )
        if any(
            row[field] != event[field]
            for field in ("prediction_horizon_steps", "execution_horizon_steps")
        ):
            raise ValueError("VLA input horizons disagree with the received forward")
        if row["physical_step"] != (0 if last_control is None else last_control + 1):
            raise ValueError(
                "VLA input physical step disagrees with the acknowledged trace position"
            )
        sensor = (
            None
            if descriptor is None
            else {
                "storage": "persistent_server_archive",
                "read_with": "read_sensor_snapshot",
                "evidence_id": episode.evidence.evidence_id,
                "observation_ref": row["observation_ref"],
                "archive_bytes": descriptor["archive_bytes"],
                "metadata_bytes": descriptor["metadata_bytes"],
                "meaning": "Exact retained forward input; fetch only this admitted RGB-D/calibration snapshot when needed.",
            }
        )
        if descriptor is None:
            gaps.append(
                {
                    "kind": gap_kind,
                    "forward_ref": identity,
                    "observation_ref": row["observation_ref"],
                    "input_record": episode.ref("vla_inputs.jsonl", f"/{index}"),
                    "transfer_failure": transfer_failure,
                    "meaning": "The recorded VLA input is retained, but its server sensor evidence was not retrieved or admitted; no sensor snapshot is available.",
                }
            )
        inputs.append(
            {
                "forward_ref": identity,
                "execution_ref": event["execution_ref"],
                "event_index": event["event_index"],
                "source": episode.ref(
                    "execution_trace.jsonl", f"/{event['event_index']}"
                ),
                "input_record": episode.ref("vla_inputs.jsonl", f"/{index}"),
                "observation_ref": row["observation_ref"],
                "physical_step": row["physical_step"],
                "last_control_step_index": last_control,
                "prompt": row["prompt"],
                "mode": row["mode"],
                "prediction_horizon_steps": row["prediction_horizon_steps"],
                "execution_horizon_steps": row["execution_horizon_steps"],
                "sensor_snapshot": sensor,
                "images": []
                if sensor is None
                else [
                    {
                        "artifact": camera + ".png",
                        "source": None,
                        "storage": "server_original_not_materialized",
                        "sensor_snapshot": sensor,
                    }
                    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")
                ],
            }
        )
    if set(records) - seen:
        raise ValueError(
            "VLA input record has no matching received forward in the trace"
        )
    return inputs, gaps


def _forward_inputs(episode: _Episode, events: list[dict[str, Any]]):
    """Index archived VLA inputs by forward identity, never by image timestamps.

    The G2 prediction original owns observation_id. Each referenced observation
    owns its camera list and timestamps; the trace supplies only the preceding
    acknowledged control position, not an image for every physical step.
    """
    if "vla_inputs.jsonl" in episode.evidence.files:
        return _server_forward_inputs(episode, events)
    inputs, gaps = [], []
    archived_observations = {}
    for name in sorted(episode.evidence.files):
        parts = Path(name).parts
        if len(parts) >= 3 and parts[0] == "observations":
            archived_observations.setdefault(parts[1], []).append(name)
    if not any(name.startswith("predictions/") for name in episode.evidence.files):
        return inputs, gaps
    last_control = None
    for event in events:
        if event["event_type"] == "control_step":
            last_control = event["control_step_index"]
        if event["event_type"] != "vla_forward":
            continue
        forward = event["forward_id"]
        prediction_name = f"predictions/{forward}.json"
        prediction_ref = episode.ref(prediction_name)
        if prediction_ref is None:
            gaps.append({"kind": "vla_input_not_archived", "forward_ref": forward})
            continue
        prediction = _read(episode.root / prediction_name)
        if prediction["forward_id"] != forward:
            raise ValueError("archived prediction identity disagrees with VLA trace")
        identifier = prediction["observation_id"]
        if (
            not isinstance(identifier, str)
            or identifier in {"", ".", ".."}
            or Path(identifier).name != identifier
        ):
            raise ValueError("archived observation identity must be one path component")
        observation_name = f"observations/{identifier}/observation.json"
        observation_ref = episode.ref(observation_name)
        if observation_ref is None:
            gaps.append(
                {
                    "kind": "vla_observation_missing",
                    "forward_ref": forward,
                    "observation_id": identifier,
                }
            )
            continue
        observation = _read(episode.root / observation_name)
        if observation["observation_id"] != identifier:
            raise ValueError("archived observation identity disagrees with prediction")
        images = []
        for camera in observation["cameras"]:
            if (
                not isinstance(camera, str)
                or camera in {"", ".", ".."}
                or Path(camera).name != camera
            ):
                raise ValueError("archived camera name must be one path component")
            name = f"observations/{identifier}/{camera}.png"
            source = episode.ref(name)
            if source is None:
                gaps.append(
                    {
                        "kind": "vla_input_image_missing",
                        "forward_ref": forward,
                        "artifact": name,
                    }
                )
            images.append({"artifact": f"{camera}.png", "source": source})
        inputs.append(
            {
                "forward_ref": forward,
                "execution_ref": event["execution_ref"],
                "event_index": event["event_index"],
                "source": episode.ref(
                    "execution_trace.jsonl", f"/{event['event_index']}"
                ),
                "prediction": prediction_ref,
                "observation_id": identifier,
                "observation": observation_ref,
                "last_control_step_index": last_control,
                "images": images,
                "artifacts": {
                    name: episode.ref(name)
                    for name in archived_observations.get(identifier, [])
                    + [
                        f"predictions/{forward}.npy",
                        f"predictions/{forward}.request.msgpack",
                        f"predictions/{forward}.response.msgpack",
                    ]
                    if name in episode.evidence.files
                },
            }
        )
    return inputs, gaps


def _diagnostics(
    events: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    request_signatures: dict[str, dict],
) -> dict[str, Any]:
    controls = {
        row["control_step_index"]: row
        for row in events
        if row["event_type"] == "control_step"
    }
    measurements = []
    for execution in executions:
        for call in execution["physical_calls"]:
            rows = [controls[index] for index in call["control_step_indices"]]
            motion = [
                row.get("public_evidence", {}).get("robot.eef.motion_m") for row in rows
            ]
            available = [
                value
                for value in motion
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ]
            measurements.append(
                {
                    "execution_ref": execution["execution_ref"],
                    "physical_call_ref": call["physical_call_ref"],
                    "physical_steps": len(rows),
                    "eef_motion_observed_steps": len(available),
                    "eef_path_length_m": math.fsum(available)
                    if rows and len(available) == len(rows)
                    else None,
                }
            )
    transitions = []
    previous: dict[tuple[str, str], tuple[int, Any]] = {}
    for row in controls.values():
        step = row["control_step_index"]
        for plane in ("public_evidence", "critic_private_evidence"):
            for name, value in row.get(plane, {}).items():
                if not isinstance(value, (bool, str)):
                    continue
                key = (plane, name)
                old = previous.get(key)
                if old is not None and old[1] != value:
                    transitions.append(
                        {
                            "plane": plane,
                            "feature": name,
                            "previous_observed_step": old[0],
                            "control_step_index": step,
                            "before": old[1],
                            "after": value,
                        }
                    )
                previous[key] = (step, value)
    # Windows refer to the nearest *recorded* boundaries bracketing a change.
    # They do not invent dense images or equate video frame numbers with steps.
    boundaries = [
        row for row in observations if row["last_control_step_index"] is not None
    ]
    for transition in transitions:
        before = [
            row
            for row in boundaries
            if row["last_control_step_index"] <= transition["previous_observed_step"]
        ]
        after = [
            row
            for row in boundaries
            if row["last_control_step_index"] >= transition["control_step_index"]
        ]
        lower = max((row["last_control_step_index"] for row in before), default=None)
        upper = min((row["last_control_step_index"] for row in after), default=None)
        transition["image_window"] = {
            "before_state_indices": [
                row["state_index"]
                for row in before
                if row["last_control_step_index"] == lower
            ],
            "after_state_indices": [
                row["state_index"]
                for row in after
                if row["last_control_step_index"] == upper
            ],
            "meaning": "Bracketing archived boundaries only; absent sides remain empty. See timeline observations for original images. These are not images taken at the transition itself.",
        }
    repeated = []
    for call in calls:
        signature = request_signatures[call["tool_call_id"]]
        group = next(
            (group for group in repeated if group["signature"] == signature), None
        )
        if group is None:
            group = {"signature": signature, "ids": []}
            repeated.append(group)
        group["ids"].append(call["tool_call_id"])
    return {
        "meaning": "Descriptive measurements only. Path length sums recorded robot.eef.motion_m on leaf-owned steps; unavailable samples make the total unknown. Discrete changes bracket observations, not inferred contact onset. No stall or root-cause classifier.",
        "physical_calls": measurements,
        "discrete_changes": transitions,
        "episode_event_counts": dict(Counter(row["event_type"] for row in events)),
        "episode_identical_planner_requests": [
            group["ids"] for group in repeated if len(group["ids"]) > 1
        ],
    }


def _visuals(
    episode: _Episode,
    destination: Path,
    observations: list[dict[str, Any]],
    forward_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Page every archived boundary image; never imply physical-frame coverage."""
    from PIL import Image, ImageDraw

    frames = [
        {
            "state_index": row["state_index"],
            "forward_ref": None,
            "last_control_step_index": row["last_control_step_index"],
            **frame,
        }
        for row in observations
        for frame in row["images"]
        if frame["source"] is not None
    ]
    frames.extend(
        {
            "state_index": None,
            "forward_ref": row["forward_ref"],
            "last_control_step_index": row["last_control_step_index"],
            **frame,
        }
        for row in forward_inputs
        for frame in row["images"]
        if frame["source"] is not None
    )
    frames.sort(
        key=lambda frame: (
            frame["artifact"],
            -1
            if frame["last_control_step_index"] is None
            else frame["last_control_step_index"],
            math.inf if frame["state_index"] is None else frame["state_index"],
        )
    )
    pages = []
    # Group the same camera/artifact across the episode before its variants;
    # otherwise many segmentation overlays hide the temporal overview.
    # Fixed tile/page geometry bounds the image size, not the number of pages.
    for offset in range(0, len(frames), 12):
        members = frames[offset : offset + 12]
        sheet = Image.new("RGB", (960, math.ceil(len(members) / 3) * 288), "#151a21")
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(members):
            x, y = index % 3 * 320, index // 3 * 288
            with Image.open(episode.resolve(frame["source"])) as original:
                picture = original.convert("RGB")
                picture.thumbnail((316, 238))
                sheet.paste(picture, (x + (320 - picture.width) // 2, y))
            position = frame["last_control_step_index"]
            boundary = (
                f"VLA input {frame['forward_ref'][:10]}"
                if frame["forward_ref"] is not None
                else f"state {frame['state_index']}"
            )
            label = f"{boundary} | prior control {position if position is not None else 'none'}\n{frame['artifact']}"
            draw.text((x + 4, y + 240), label, fill="white")
        name = f"overview-{len(pages):04d}.jpg"
        sheet.save(destination / name, quality=90)
        pages.append({"file": name, "frames": members})
    return pages


def _task_instructions(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Index exact official text and its distinct public task-rule supplement."""
    values: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for observation in observations:
        language = observation["task_language"]
        step = observation["state_index"]
        if language is None:
            missing.append(step)
            continue
        if not isinstance(language, str) or not language.strip():
            raise ValueError("recorded task_language must be a nonempty string or null")
        official = observation.get("official_task_instruction")
        supplement = observation.get("task_rule_supplement")
        if official is None and supplement is None:
            official, supplement = language, ""
        if not isinstance(official, str) or not official.strip():
            raise ValueError(
                "recorded official_task_instruction must be a nonempty string"
            )
        if not isinstance(supplement, str):
            raise ValueError("recorded task_rule_supplement must be a string")
        expected = official + ("\n\n" + supplement if supplement else "")
        if language != expected:
            raise ValueError(
                "recorded task_language differs from its official instruction "
                "and task-rule supplement"
            )
        key = (official, supplement)
        if key not in values:
            source = dict(observation["source"])
            source["pointer"] += "/extras/task_language"
            values[key] = {
                "text": language,
                "official_text": official,
                "task_rule_supplement": supplement,
                "source": source,
                "state_indices": [],
            }
        values[key]["state_indices"].append(step)
    return {
        "semantics": "The official task instruction is preserved verbatim. A public, source-attributed project task-rule supplement is a distinct fixed field, separate from learned author hypotheses. text is their derived model-facing concatenation. Missing instructions are not inferred from task IDs or scene objects.",
        "values": list(values.values()),
        "missing_state_indices": missing,
    }


def _current_task_instruction_index(value: Any) -> bool:
    """Reject obsolete or incomplete derived instructions, never invent fields."""
    if not isinstance(value, dict) or not isinstance(value.get("values"), list):
        return False
    if not isinstance(value.get("missing_state_indices"), list):
        return False
    for item in value["values"]:
        if (
            not isinstance(item, dict)
            or not {
                "text",
                "official_text",
                "task_rule_supplement",
                "source",
                "state_indices",
            }
            <= item.keys()
        ):
            return False
        official, supplement = item["official_text"], item["task_rule_supplement"]
        if (
            not isinstance(official, str)
            or not official.strip()
            or not isinstance(supplement, str)
        ):
            return False
        if item["text"] != official + ("\n\n" + supplement if supplement else ""):
            return False
        source, states = item["source"], item["state_indices"]
        if (
            not isinstance(source, dict)
            or not {"path", "pointer", "artifact_id", "bytes"} <= source.keys()
        ):
            return False
        if not isinstance(states, list) or any(
            type(step) is not int or step < 0 for step in states
        ):
            return False
    return True


def build_episode_context(
    workspace: Path,
    evidence: EpisodeEvidence,
    episode_root: Path | None = None,
    *,
    include_visuals: bool = True,
) -> dict[str, Any]:
    """Build navigation once per episode and reuse its recorded index."""
    episode = _Episode(workspace, evidence, episode_root)
    from reweave.evolution.evidence.sensor_retention import read_retention

    read_retention(episode.root)
    retention_reference = episode.ref("sensor_retention.json")
    destination = workspace / "context/trace" / evidence.evidence_id
    manifest_path = destination / "index.json"
    if destination.exists():
        try:
            manifest = _read(manifest_path)
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}
        if (
            isinstance(manifest, dict)
            and manifest.get("schema_version") == TRACE_INDEX_SCHEMA_VERSION
            and manifest.get("format") == "indexed_originals"
            and manifest.get("evidence_id") == evidence.evidence_id
            and manifest.get("visuals_included") is include_visuals
            and manifest.get("sensor_retention") == retention_reference
            and _current_task_instruction_index(manifest.get("task_instructions"))
            and "media" in manifest
            and "planner_memory" in manifest
            and "native" in manifest
            and {"finish", "stats", "agent_run_contract", "error"}
            <= manifest.get("outcome", {}).keys()
            and {"program_feedback", "workflow_calls"}
            <= manifest.get("coverage", {}).keys()
            and "forward_inputs" in manifest.get("coverage", {})
        ):
            return {
                "path": manifest_path.relative_to(workspace).as_posix(),
                **{
                    key: value
                    for key, value in artifact_record(manifest_path).items()
                    if key != "path"
                },
            }
        # Rebuild from originals. Keep the old directory until the replacement
        # is complete, then retain it separately without changing published links.
    temporary = destination.with_name(f".{evidence.evidence_id}.building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        transcript_name = next(
            name
            for name in evidence.files
            if Path(name).parent == Path(".")
            and name.startswith("transcript_")
            and name.endswith(".json")
        )
        transcript = _read(episode.root / transcript_name)
        from reweave.robots.execution_trace import merge_control_telemetry

        events = merge_control_telemetry(_jsonl(episode.root / "execution_trace.jsonl"))
        executions = _executions(episode, events)
        calls, program_feedback, gaps, request_signatures = _planner_calls(
            episode, transcript_name, transcript, executions, events
        )
        bundle_events, bundle_gaps = _bundle_actions(episode, calls, executions)
        gaps.extend(bundle_gaps)
        observations, observation_gaps = _observations(episode, executions, events)
        gaps.extend(observation_gaps)
        forward_inputs, input_gaps = _forward_inputs(episode, events)
        gaps.extend(input_gaps)
        for execution in executions:
            execution["operation_feedback"] = [
                {
                    "event_index": row["event_index"],
                    "capability": row["capability"],
                    "effect": row["effect"],
                    "source": episode.ref(
                        "execution_trace.jsonl", f"/{row['event_index']}"
                    ),
                }
                for row in events
                if row["event_type"]
                in {"runtime_modification_operation", "episode_tool_program_capability"}
                and row.get("execution_ref") == execution["execution_ref"]
            ]
            execution["origin"] = (
                "planner_call"
                if execution["planner_call_ids"]
                else execution.get(
                    "kernel_origin",
                    "program" if execution.get("program_feedback") else "unknown",
                )
            )
            if (
                execution["tool_name"] == "run_harness_program"
                and events[execution["start_event"]]["arguments"].get("entry")
                == "episode_start"
            ):
                execution["origin"] = "episode_start"
            if execution["origin"] == "unknown":
                gaps.append(
                    {
                        "kind": "execution_without_planner_feedback_link",
                        "execution_ref": execution["execution_ref"],
                        "note": "May be automatic episode_start, nested supervision, or missing evidence. No temporal matching.",
                    }
                )
        atomic_text(
            temporary / "timeline.json",
            json.dumps(
                {
                    "source": episode.ref("execution_trace.jsonl"),
                    "observation_semantics": "Final archived state artifacts may include later segmentation overlays. They do not establish what a previous Planner decision saw. decision_inputs point to exact recorded requests when uniquely linkable.",
                    "planner_calls": calls,
                    "program_feedback": program_feedback,
                    "bundle_events": bundle_events,
                    "executions": executions,
                    "observations": observations,
                    "forward_inputs": forward_inputs,
                    "forward_input_semantics": "Exact archived model inputs. Prior control indices locate capture between recorded actions; they do not turn chunk inputs into continuous physical-step images or establish what the high-level Planner saw.",
                    "gaps": gaps,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )
        telemetry = _telemetry(episode, events)
        atomic_text(
            temporary / "telemetry.json",
            json.dumps(telemetry, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        atomic_text(
            temporary / "diagnostics.json",
            json.dumps(
                _diagnostics(
                    events, executions, observations, calls, request_signatures
                ),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )
        pages = (
            _visuals(episode, temporary, observations, forward_inputs)
            if include_visuals
            else []
        )
        manifest = {
            "schema_version": TRACE_INDEX_SCHEMA_VERSION,
            "sensor_retention": retention_reference,
            "format": "indexed_originals",
            "evidence_id": evidence.evidence_id,
            "visuals_included": include_visuals,
            "task_instructions": _task_instructions(observations),
            "authority": "Derived navigation. Originals and recorded contracts are authoritative. Original references are resolved by the Host against this request's immutable evidence roots; view files are relative to this index. No physical counterfactual evaluation.",
            "outcome": {
                key: transcript.get(key)
                for key in (
                    "official_success",
                    "termination_class",
                    "task_message",
                    "bundle_id",
                    "finish",
                    "stats",
                    "agent_run_contract",
                    "error",
                )
            },
            "contracts": {
                "transcript": episode.ref(transcript_name),
                "planner_context": episode.ref("harness_context/planner_context.json"),
                "planner_requests": episode.ref(
                    "harness_context/planner_requests.jsonl"
                ),
                "run_configuration": episode.ref(transcript_name, "/run_configuration")
                if "run_configuration" in transcript
                else None,
                "agent_run_contract": episode.ref(
                    transcript_name, "/agent_run_contract"
                )
                if "agent_run_contract" in transcript
                else None,
                "executed_source": f".archive/episodes/{evidence.evidence_id}/harness_context/source",
                "capabilities": episode.ref("harness_context/capability_catalog.json"),
                "bundle_gateway": episode.ref("harness_context/bundle_contract.json"),
            },
            "planner_memory": {
                "meaning": "Episode-local Planner task-family drafts and every original revision retaining its path. Historical drafts in older scopes remain evidence, not current write authority. These are author-declared applicability, not proved transfer, evaluator outcomes or automatically accepted C skills. Review them against source decisions and observed feedback before editing candidate C.",
                "closeout": episode.ref("planner_memory/closeout.json")
                if "planner_memory/closeout.json" in evidence.files
                else None,
                "drafts": [
                    episode.ref(name)
                    for name in sorted(evidence.files)
                    if name.startswith("planner_memory/drafts/")
                ],
                "revisions": [
                    episode.ref(name)
                    for name in sorted(evidence.files)
                    if name.startswith("planner_memory/revisions/")
                ],
            },
            "task_specific": {
                "meaning": "Host-recorded facts and the matched successful control sequence; these are not agent-written skills or replay authorization.",
                "files": [
                    episode.ref(name)
                    for name in sorted(evidence.files)
                    if name.startswith("task-specific/")
                ],
            },
            "native": {
                "meaning": "Native episode records and online revisions are retained originals, not automatically inherited skills or independently measured final bundles. Inspect completed decision input/completion references in the timeline and compare edits or notes with the observed execution before revising candidate W/C/L.",
                "episode_record": episode.ref("native/records/episode.md"),
                "guidance": episode.ref("native/prompt.md"),
                "provider_session": episode.ref("native/provider/session.json"),
                "provider_events": episode.ref("native/provider/events.jsonl"),
                "adaptation_receipt": episode.ref("native/adaptation/receipt.json"),
                "revision_records": [
                    episode.ref(name)
                    for name in sorted(evidence.files)
                    if name.startswith("native/adaptation/revisions/")
                    and name.endswith("/revision.json")
                ],
                "skills": [
                    episode.ref(name)
                    for name in sorted(evidence.files)
                    if name.startswith("native/")
                    and "/skills/" in name
                    and name.endswith("/SKILL.md")
                ],
            },
            "views": {
                "timeline": "timeline.json",
                "telemetry": "telemetry.json",
                "diagnostics": "diagnostics.json",
            },
            "coverage": {
                "event_count": len(events),
                "physical_steps": sum(
                    row["event_type"] == "control_step" for row in events
                ),
                "planner_calls": len(calls),
                "workflow_calls": sum(
                    call["name"].startswith("workflow_") for call in calls
                ),
                "program_feedback": len(program_feedback),
                "state_boundaries": len(observations),
                "forward_inputs": len(forward_inputs),
                "link_gaps": len(gaps),
                "images": sum(len(page["frames"]) for page in pages),
                "images_meaning": (
                    "All available declared boundary images and archived VLA input images; thumbnails are navigation. Physical-step images are not inferred from boundary, chunk or video frame indices."
                    if include_visuals
                    else "Boundary images remain available as immutable originals; this mutation workspace does not duplicate overview sheets."
                ),
            },
            "tool_counts": dict(Counter(row["tool_name"] for row in executions)),
            "execution_refs": [row["execution_ref"] for row in executions],
            "features": {
                plane: telemetry["feature_names"][plane]
                for plane in ("public_evidence", "critic_private_evidence")
            },
            "visual_pages": pages,
            "media": [
                {
                    "kind": "video",
                    "source": episode.ref(name),
                    "alignment": "Video frame indices and nominal video seconds are not physical step or wall-clock indices; verify the recorder and observation timeline before alignment.",
                }
                for name in sorted(evidence.files)
                if Path(name).suffix.lower() in {".mp4", ".webm", ".mov"}
            ],
            "files": inventory_episode(temporary),
        }
        atomic_text(
            temporary / "index.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        previous = None
        if destination.exists():
            previous = (
                workspace
                / "context/retired_trace_indices"
                / evidence.evidence_id
                / uuid.uuid4().hex
            )
            previous.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(previous)
        try:
            temporary.rename(destination)
        except BaseException:
            if previous is not None:
                previous.rename(destination)
            raise
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {
        "path": manifest_path.relative_to(workspace).as_posix(),
        **{
            key: value
            for key, value in artifact_record(manifest_path).items()
            if key != "path"
        },
    }


def read_video(
    workspace: Path,
    evidence_roots: dict[str, Path],
    *,
    evidence_id: str,
    path: str,
    frames: list[int],
) -> dict:
    """Decode explicitly requested frames of one indexed immutable episode video."""
    import base64

    import imageio.v2 as imageio
    from PIL import Image

    if evidence_id not in evidence_roots:
        raise ValueError("video evidence is outside this author request")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("video path must be relative to its episode")
    if (
        not isinstance(frames, list)
        or any(type(frame) is not int or frame < 0 for frame in frames)
        or frames != sorted(set(frames))
    ):
        raise ValueError("video frames must be sorted unique nonnegative indices")
    index = _read(workspace / "context/trace" / evidence_id / "index.json")
    reference = next(
        (
            item["source"]
            for item in index["media"]
            if item["source"]["path"] == f".archive/episodes/{evidence_id}/{path}"
        ),
        None,
    )
    root = evidence_roots[evidence_id].resolve()
    original = root / path
    if reference is None or original.resolve() != root / path or not original.is_file():
        raise ValueError(
            "video path is absent from the episode inventory or follows a link"
        )
    destination = workspace / ".scratch/media" / evidence_id / reference["artifact_id"]
    if destination.resolve() != destination.absolute():
        raise ValueError("derived video directory must not follow links")
    destination.mkdir(parents=True, exist_ok=True)
    selected, images = [], []
    reader = imageio.get_reader(str(original))
    try:
        metadata = reader.get_meta_data()
        count = reader.count_frames()
        if frames and frames[-1] >= count:
            raise ValueError(f"video frame index outside [0, {count})")
        for frame in frames:
            output = destination / f"frame-{frame:06d}.png"
            if output.is_symlink():
                raise ValueError("derived video frame must not follow links")
            Image.fromarray(reader.get_data(frame)).save(output)
            raw = output.read_bytes()
            selected.append(
                {
                    "frame_index": frame,
                    "nominal_video_time_s": frame / metadata["fps"],
                    "path": str(output),
                    **{
                        key: value
                        for key, value in artifact_record(output).items()
                        if key != "path"
                    },
                }
            )
            images.append(
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": base64.b64encode(raw).decode(),
                }
            )
    finally:
        reader.close()
    return {
        "source": reference,
        "frame_count": count,
        "fps": metadata["fps"],
        "frames": selected,
        "physical_step_alignment": None,
        "meaning": "Requested decoded video frames, PNG at recorded resolution; originals unmodified. Nominal video time is not episode work time. Establish step alignment from recorder/timeline evidence; never infer it solely from matching frame counts.",
        "_images": images,
    }


def _trace_rows(workspace, reference, indices, evidence_roots):
    """Decode only selected original events, in one scan of their JSONL file."""
    remaining = set(indices)
    rows = {}
    if not remaining:
        return rows
    path = _trace_reference_path(workspace, reference, evidence_roots)
    if path.suffix != ".jsonl" or reference["pointer"]:
        raise ValueError("telemetry source must name a complete JSONL original")
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index not in remaining:
                continue
            row = restore_record(json.loads(line), path.parent)
            if not isinstance(row, dict) or row.get("event_index") != index:
                raise ValueError(
                    "selected original event_index differs from its record"
                )
            rows[index] = row
            remaining.remove(index)
            if not remaining:
                break
    if remaining:
        raise ValueError("selected telemetry original record is absent")
    return rows


def query_trace(
    workspace: Path,
    *,
    evidence_id: str,
    view: str,
    execution_ref: str | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    features: list[str] | None = None,
    evidence_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Select complete aligned rows, with explicit coverage and no read limit.

    Step filters are inclusive recorded physical indices (not state boundaries).
    Feature names retain their public/private plane; unknown names are rejected.
    Timeline execution filters include linked Planner calls and state boundaries.
    """
    if (
        not isinstance(evidence_id, str)
        or not evidence_id
        or Path(evidence_id).name != evidence_id
        or evidence_id in {".", ".."}
    ) or view not in {
        "timeline",
        "executions",
        "telemetry",
        "diagnostics",
        "online_workflows",
    }:
        raise ValueError("invalid evidence id or trace view")
    for value in (start_step, end_step):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("physical step bounds must be non-negative integers")
    if start_step is not None and end_step is not None and start_step > end_step:
        raise ValueError("physical step bounds are reversed")
    root = workspace / "context/trace" / evidence_id
    if view != "telemetry" and (
        start_step is not None or end_step is not None or features is not None
    ):
        raise ValueError("step and feature selection applies only to telemetry")
    if view == "online_workflows":
        if execution_ref is not None:
            raise ValueError(
                "online W revisions are episode-wide; query physical executions separately"
            )
        index = _read(root / "index.json")
        timeline = _read(root / "timeline.json")
        native = index.get("native", {})
        data = {
            "meaning": "Online W uses the ordinary WorkflowSpec and adaptation chain. Original sources and actual executions are evidence, not automatic inheritance or an independently measured final Bundle.",
            "adaptation_receipt": native.get("adaptation_receipt"),
            "revision_records": native.get("revision_records", []),
            "events": [
                row
                for row in timeline["bundle_events"]
                if row["kind"]
                in {
                    "workflow_written",
                    "workflow_write_rejected",
                    "workflow_started",
                    "workflow_completed",
                    "workflow_failed",
                    "workflow_timing",
                    "native_decision_timing",
                }
            ],
        }
        return {
            "evidence_id": evidence_id,
            "view": view,
            "filters": {
                "execution_ref": None,
                "start_step": None,
                "end_step": None,
                "features": None,
            },
            "coverage": {"selection_complete": True},
            "data": data,
        }
    value = _read(root / ("timeline.json" if view == "executions" else f"{view}.json"))
    timeline = value if view in {"timeline", "executions"} else None
    if execution_ref is not None:
        # Telemetry has its own complete execution identity column. Avoid
        # loading unrelated model/vision payloads for a physical-step query.
        identities = (
            set(_read(root / "index.json")["execution_refs"])
            if view == "telemetry"
            else None
        )
        if identities is None:
            timeline = (
                timeline if timeline is not None else _read(root / "timeline.json")
            )
            identities = {row["execution_ref"] for row in timeline["executions"]}
        if execution_ref not in identities:
            raise ValueError("unknown execution_ref")
    if view == "executions":
        rows = value["executions"]
        selected = [
            row
            for row in rows
            if execution_ref is None or row["execution_ref"] == execution_ref
        ]
        selected_calls = {
            identity for row in selected for identity in row["planner_call_ids"]
        }
        planner_calls = [
            call
            for call in value["planner_calls"]
            if execution_ref is None or call["tool_call_id"] in selected_calls
        ]
        data = {
            "executions": selected,
            # Unknown links cannot be filtered by the identity they lack.
            # Keep episode-wide gaps visible even in an execution selection.
            "gaps": value["gaps"],
            # A Workflow may return or fail before submitting a physical action.
            # Preserve its lifecycle independently of physical-execution links,
            # including nested invocations and starts without a recorded end.
            "workflow_events": [
                row
                for row in value["bundle_events"]
                if row["kind"]
                in {"workflow_started", "workflow_completed", "workflow_failed"}
            ],
            "program_feedback": [
                row
                for row in value["program_feedback"]
                if execution_ref is None or row["execution_ref"] == execution_ref
            ],
            "planner_feedback": [
                {
                    "tool_call_id": call["tool_call_id"],
                    "name": call["name"],
                    "request": call["request"],
                    "source": call["feedback"],
                }
                for call in planner_calls
            ],
        }
        coverage = {
            "total_executions": len(rows),
            "selected_executions": len(selected),
            "total_planner_calls": len(value["planner_calls"]),
            "selected_planner_calls": len(planner_calls),
            "workflow_events_scope": "episode",
            "selection_complete": True,
        }
    elif view == "telemetry":
        selected = [
            index
            for index, step in enumerate(value["control_step_indices"])
            if (
                execution_ref is None or value["execution_refs"][index] == execution_ref
            )
            and (start_step is None or step >= start_step)
            and (end_step is None or step <= end_step)
        ]
        names = set(value["feature_names"]["public_evidence"]) | set(
            value["feature_names"]["critic_private_evidence"]
        )
        if features is not None and set(features) - names:
            raise ValueError(
                f"unknown telemetry features: {sorted(set(features) - names)}"
            )
        data = {
            key: [value[key][index] for index in selected]
            for key in (
                "control_step_indices",
                "event_indices",
                "telemetry_event_indices",
                "telemetry_errors",
                "execution_refs",
                "physical_call_refs",
                "forward_refs",
                "action_indices",
                "tool_names",
                "terminated",
                "truncated",
            )
        }
        # Unacknowledged actions have no confirmed physical-step index. Keep the
        # episode-wide uncertainty visible even when selecting confirmed steps.
        data["unconfirmed_actions"] = value["unconfirmed_actions"]
        data["physical_steps_complete"] = value["physical_steps_complete"]
        original_indices = {
            index
            for position in selected
            for index in (
                value["event_indices"][position],
                value["telemetry_event_indices"][position],
            )
            if index is not None
        }
        originals = _trace_rows(
            workspace, value["source"], original_indices, evidence_roots
        )
        from reweave.robots.execution_trace import merge_control_telemetry

        controls = {
            row["event_index"]: row
            for row in merge_control_telemetry(
                [originals[index] for index in sorted(originals)]
            )
            if row["event_type"] == "control_step"
        }
        rows = []
        for position in selected:
            row = controls.get(value["event_indices"][position])
            if (
                row is None
                or row["control_step_index"] != value["control_step_indices"][position]
                or row["execution_ref"] != value["execution_refs"][position]
            ):
                raise ValueError(
                    "selected telemetry differs from its indexed control step"
                )
            rows.append(row)
        data["requested_actions"] = [row["action"] for row in rows]
        for plane in ("public_evidence", "critic_private_evidence"):
            data[plane] = {
                name: [row.get(plane, {}).get(name) for row in rows]
                for name in value["feature_names"][plane]
                if features is None or name in features
            }
        data["evidence_source"] = value["source"]
        data["evidence_event_indices"] = {
            plane: {
                name: [row["evidence_event_indices"][plane].get(name) for row in rows]
                for name in data[plane]
            }
            for plane in ("public_evidence", "critic_private_evidence")
        }
        data["evidence_pointer_rule"] = (
            "Each non-null source index refers to that exact original JSONL event. "
            "Resolve /<event_index>/<plane>/<key>, escaping ~ as ~0 and / as ~1 in the key. "
            "Null means that measurement is unrecorded at this control step."
        )
        data["alignment"] = value["alignment"]
        coverage = {
            "total_physical_steps": len(value["control_step_indices"]),
            "selected_physical_steps": len(selected),
            "selection_complete": True,
        }
    else:
        data = value
        if execution_ref is not None:
            keys = (
                ("planner_calls", "executions", "observations", "forward_inputs")
                if view == "timeline"
                else ("physical_calls",)
            )
            data = {
                **value,
                **{
                    key: [
                        row
                        for row in value[key]
                        if row.get("execution_ref") == execution_ref
                        or execution_ref in row.get("execution_refs", ())
                    ]
                    for key in keys
                },
            }
            if view == "diagnostics":
                steps = set(
                    next(
                        row
                        for row in timeline["executions"]
                        if row["execution_ref"] == execution_ref
                    )["control_step_indices"]
                )
                data["discrete_changes"] = [
                    row
                    for row in value["discrete_changes"]
                    if row["control_step_index"] in steps
                ]
        coverage = {"selection_complete": True}
    return {
        "evidence_id": evidence_id,
        "view": view,
        "filters": {
            "execution_ref": execution_ref,
            "start_step": start_step,
            "end_step": end_step,
            "features": features,
        },
        "coverage": coverage,
        "data": data,
    }
