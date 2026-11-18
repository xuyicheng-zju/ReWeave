"""Host-measured deployed C/W/L size and task-balanced retention metrics."""

from __future__ import annotations

import json
from difflib import SequenceMatcher

from reweave.evolution.bundle import AgentBundle


def harness_delta_size(bundle: AgentBundle, reference: AgentBundle) -> dict:
    """Added/replacement UTF-8 lines against a fixed H0, never net size growth.

    Compare each deployed source/text/schema at its stable component identity.
    Unchanged lines cost zero; removed bytes are audited separately and cannot
    cancel additions elsewhere. New/renamed components count in full. This is a
    syntactic delta proxy, not a proof of semantic novelty or minimal patch size.
    """

    def components(value):
        parts = {}
        programs = [
            ("context", value.context.program),
            ("loop", value.loop.program),
            *((f"workflow/{w.workflow_id}", w.program) for w in value.workflows),
        ]
        for scope, program in programs:
            for file in program.files:
                parts[("source", scope, file.path)] = file.source
        parts[("text", "context", "instructions")] = value.context.instructions
        for skill in value.context.skills:
            parts[("text", "skill", skill.skill_id)] = skill.text
        for workflow in value.workflows:
            for field in ("description", "applicability"):
                parts[("text", workflow.workflow_id, field)] = getattr(workflow, field)
            for field in ("input_schema", "output_schema"):
                parts[("interface", workflow.workflow_id, field)] = json.dumps(
                    getattr(workflow, field),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
        return parts

    before, after = components(reference), components(bundle)
    added = dict.fromkeys(("source", "text", "interface"), 0)
    removed = 0
    for key in sorted(before.keys() | after.keys()):
        old = before.get(key, "").splitlines(keepends=True)
        new = after.get(key, "").splitlines(keepends=True)
        for tag, a, b, c, d in SequenceMatcher(
            None, old, new, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                continue
            added[key[0]] += len("".join(new[c:d]).encode("utf-8"))
            removed += len("".join(old[a:b]).encode("utf-8"))
    total = sum(added.values())
    return {
        "reference_bundle_id": reference.record_id,
        "metric": "fixed_reference_added_replacement_lines_utf8_v1",
        **{f"{kind}_bytes": count for kind, count in added.items()},
        "total_bytes": total,
        "size_kb": total / 1000,
        "removed_bytes": removed,
    }


def harness_size(bundle: AgentBundle) -> dict[str, int | float]:
    """Count deployed UTF-8 source/text, excluding runtime, receipts and media.

    Every deployed copy counts, even when two workflows contain equal source.
    This measures representation size, not semantic or computational complexity.
    """
    programs = (
        bundle.context.program,
        bundle.loop.program,
        *(workflow.program for workflow in bundle.workflows),
    )
    source = [file.source for program in programs for file in program.files]
    texts = [bundle.context.instructions, *(s.text for s in bundle.context.skills)]
    for workflow in bundle.workflows:
        texts.extend((workflow.description, workflow.applicability))
    interface_bytes = sum(
        len(
            json.dumps(
                schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        for workflow in bundle.workflows
        for schema in (workflow.input_schema, workflow.output_schema)
    )
    source_bytes = sum(len(value.encode("utf-8")) for value in source)
    text_bytes = sum(len(value.encode("utf-8")) for value in texts)
    return {
        "source_bytes": source_bytes,
        "text_bytes": text_bytes,
        "interface_bytes": interface_bytes,
        "total_bytes": source_bytes + text_bytes + interface_bytes,
        "size_kb": (source_bytes + text_bytes + interface_bytes) / 1000,
        "source_lines": sum(len(value.splitlines()) for value in source),
        "workflows": len(bundle.workflows),
    }


def retention(scores: dict[str, float], anchors: dict[str, float]) -> dict:
    """Require every anchor task; report positive losses with equal task weight."""
    if not anchors.keys() <= scores.keys():
        raise ValueError("retention requires every anchor task")
    drops = {task: max(0.0, value - scores[task]) for task, value in anchors.items()}
    return {
        "per_task_drop": drops,
        "mean_drop": sum(drops.values()) / len(drops) if drops else 0.0,
        "worst_drop": max(drops.values(), default=0.0),
    }
