# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Offline experience is required, preserved, and separate from executable skill assets."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from reweave.evolution.authoring.candidate import (
    capture_bundle,
    read_candidate,
    write_bundle_source,
)
from reweave.evolution.bundle import TextSkill
from reweave.evolution.contracts import Candidate
from reweave.evolution.validation.admission import admit_mutation
from tests.unit_tests.harness.evolution.fixtures import (
    author_candidate,
    baseline,
)
from tests.unit_tests.harness.evolution.test_bundle_author import workspace


@pytest.mark.parametrize("text", [None, "", " \n\t"])
def test_final_admission_preserves_optional_experience_without_fabricating_notes(
    tmp_path, text
):
    root = workspace(tmp_path)
    note = root / "candidate/SKILL.md"
    if text is None:
        note.unlink()
    else:
        note.write_text(text)
        assert read_candidate(root, require_plan=False)
    candidate = asyncio.run(admit_mutation(root))[0]
    assert candidate.experience == (text if text is not None else "")
    assert json.loads(candidate.summary)["diagnosis"]


def test_uncertain_experience_survives_admission_without_changing_online_bundle(
    tmp_path,
):
    root = workspace(tmp_path)
    note = root / "candidate/SKILL.md"
    original = asyncio.run(admit_mutation(root))[0]
    lesson = "My hypothesis may be wrong: inspect fresh feedback before retrying."
    note.write_text(lesson)
    revised = asyncio.run(admit_mutation(root))[0]
    assert revised.experience == lesson
    assert revised.experience != original.experience
    assert revised.bundle.same_harness(original.bundle)
    assert revised.bundle.context.skills == ()
    assert lesson not in revised.bundle.context.instructions
    assert revised != original


def test_experience_alone_cannot_count_as_a_new_harness(tmp_path):
    root = workspace(tmp_path)
    plan_path = root / "candidate/plan.json"
    plan = json.loads(plan_path.read_text())
    plan["scopes"] = []
    author_candidate(baseline(), root / "candidate")
    plan_path.write_text(json.dumps(plan))
    (root / "candidate/SKILL.md").write_text(
        "New experience, unchanged executable code."
    )
    with pytest.raises(ValueError, match="differ from every reference"):
        read_candidate(root)


@pytest.mark.parametrize("text", [None, "", " \n\t"])
def test_receipt_can_omit_additional_experience(text):
    payload = {"branch_id": "repair", "bundle": baseline(), "summary": "synthetic"}
    if text is not None:
        payload["experience"] = text
    assert Candidate.model_validate(payload).experience == (text or "")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_scoped_c_assets_round_trip_once_with_exact_values(tmp_path, newline):
    root = workspace(tmp_path)
    skills = (
        TextSkill(
            skill_id="a_family",
            scope="task-family",
            text="Conditional, unverified family lesson.",
        ),
        TextSkill(
            skill_id="this-task",
            scope="task-specific",
            text="---\nid: this-task\nscope: task-specific\n---\nObserved task only.",
        ),
        TextSkill(
            skill_id="verify-held-object",
            scope="global",
            text="""---
id: verify-held-object
scope: global
kind: strategy
title: Verify possession from fresh evidence
applies_when: A grasp outcome is ambiguous
symptom: [closed gripper, occlusion]
evidence: {cells: [], attempts: 0, solved_seeds: [], failed_seeds: []}
confidence: unverified
related: []
---
A closed gripper alone may not establish possession.

**Why:** observed, cause unknown; synthetic interface fixture only.
**How to apply:** Inspect current public feedback; physical effect untested.
**Falsify:** Current evidence conclusively establishes possession without this check.
**Related:** None.
""",
        ),
    )
    skills = tuple(
        skill.model_copy(update={"text": skill.text.replace("\n", newline)})
        for skill in skills
    )
    package = baseline().context.model_copy(update={"skills": skills})
    bundle = baseline().model_copy(update={"context": package})
    write_bundle_source(bundle, root / "candidate")
    parsed = capture_bundle(root / "candidate", bundle.runtime).bundle
    assert parsed.context == package
    for skill in skills:
        path = (
            root
            / "candidate"
            / skill.scope
            / "context/skills"
            / skill.skill_id
            / "SKILL.md"
        )
        assert path.read_bytes() == skill.text.encode("utf-8")
    assert len(list((root / "candidate").rglob("SKILL.md"))) == 4
    other = tmp_path / "roundtrip"
    write_bundle_source(parsed, other)
    assert capture_bundle(other, bundle.runtime).bundle.context == package


@pytest.mark.parametrize("scope", [None, "unknown", "suite"])
def test_c_skill_requires_current_explicit_scope(scope):
    payload = {"skill_id": "lesson", "text": "Unverified lesson"}
    if scope is not None:
        payload["scope"] = scope
    with pytest.raises(ValidationError, match="scope"):
        TextSkill.model_validate(payload)


@pytest.mark.parametrize("scope", ["task-family", "task-specific"])
def test_scope_changes_actual_assets_without_allowing_duplicate_ids(scope):
    parent = baseline()
    skill = {"skill_id": "lesson", "scope": "global", "text": "Conditional claim"}
    payload = parent.context.model_dump(mode="json")
    payload["skills"] = [skill]
    original = parent.context.model_validate(payload)
    payload["skills"] = [{**skill, "scope": scope}]
    changed = parent.context.model_validate(payload)
    assert changed != original
    assert not parent.model_copy(update={"context": changed}).same_harness(
        parent.model_copy(update={"context": original})
    )
    payload["skills"].append(skill)
    with pytest.raises(ValidationError, match="sorted and unique"):
        parent.context.model_validate(payload)


@pytest.mark.parametrize("scope", ["global", "unknown"])
def test_candidate_rejects_removed_skill_layout(tmp_path, scope):
    root = workspace(tmp_path)
    directory = root / "candidate/task-family/context/skills" / scope / "nested"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("A nested removed layout")
    with pytest.raises(ValueError, match="each scoped skill"):
        read_candidate(root)
