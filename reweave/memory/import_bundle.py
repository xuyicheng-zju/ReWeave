# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Explicit offline publication of local text into a new Bundle's C assets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reweave.evolution.bundle import AgentBundle, Asset, TextSkill
from reweave.memory.experience import ExperienceId, ExperienceScope
from reweave.utils.persistence import atomic_path, sync_directory


class LocalSource(BaseModel):
    """An original local file explicitly selected by the corpus publisher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(
        min_length=1,
        description="Local file, absolute or relative to the manifest directory; never downloaded or executed.",
    )

    @field_validator("path")
    @classmethod
    def _local_path(cls, value: str) -> str:
        if not value.strip() or "://" in value or "\x00" in value:
            raise ValueError("source must name an explicit local file")
        return value


class CorpusText(Asset):
    """One published text and its original evidence, with no verification claim."""

    skill_id: ExperienceId = Field(
        description="New C skill identity; duplicates and existing skill IDs are rejected."
    )
    scope: ExperienceScope = Field(
        description="Publisher's applicability category; not a correctness or transfer certificate."
    )
    source: LocalSource = Field(
        description="Complete UTF-8 text selected for the frozen online C asset."
    )
    evidence: tuple[LocalSource, ...] = Field(
        description="Explicit original evidence files; an empty list preserves an unsupported experience as unverified. Evidence presence or count never certifies its explanation."
    )

    @model_validator(mode="after")
    def _unique_evidence(self) -> CorpusText:
        if len({item.path for item in self.evidence}) != len(self.evidence):
            raise ValueError("original evidence paths must be unique")
        return self


class CorpusManifest(Asset):
    """The complete explicit input inventory for one offline text import."""

    texts: tuple[CorpusText, ...] = Field(
        min_length=1,
        description="Text assets deliberately selected by the publisher, with exact source bytes and original evidence.",
    )

    @model_validator(mode="after")
    def _unique_ids(self) -> CorpusManifest:
        if len({item.skill_id for item in self.texts}) != len(self.texts):
            raise ValueError("corpus skill IDs must be unique")
        return self


def _read_source(source: LocalSource, root: Path) -> tuple[bytes, dict]:
    path = Path(source.path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"source is not a local file: {source.path}")
    raw = path.read_bytes()
    return raw, {"artifact_id": uuid4().hex, "path": str(path), "bytes": len(raw)}


def import_corpus(
    bundle: AgentBundle, manifest: CorpusManifest, *, source_root: Path
) -> tuple[AgentBundle, tuple[dict, ...]]:
    """Freeze selected text and provenance into C; never merge or certify claims.

    Existing C assets remain intact. A revision uses the existing offline author
    path; this import rejects collisions instead of silently replacing knowledge.
    """
    existing = {skill.skill_id for skill in bundle.context.skills}
    collisions = existing & {item.skill_id for item in manifest.texts}
    if collisions:
        raise ValueError(f"C skill IDs already exist: {sorted(collisions)}")
    skills = list(bundle.context.skills)
    inventory = []
    for item in sorted(manifest.texts, key=lambda text: text.skill_id):
        raw, source = _read_source(item.source, source_root)
        text = raw.decode("utf-8")
        if not text.strip():
            raise ValueError(f"corpus text is blank: {item.source.path}")
        evidence = [_read_source(value, source_root)[1] for value in item.evidence]
        if len({value["path"] for value in evidence}) != len(evidence):
            raise ValueError("original evidence resolves to duplicate files")
        provenance = {
            "skill_id": item.skill_id,
            "source": source,
            "evidence": evidence,
            "verification": "unverified",
        }
        guidance = (
            "Imported reference text. Its claims remain unverified; source references "
            "record provenance only. Check applicability against current observations "
            "and original evidence. Never replay historical actions blindly.\n\n"
            "Import provenance:\n```json\n"
            + json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n```\n\n"
            + text
        )
        skill = TextSkill(skill_id=item.skill_id, scope=item.scope, text=guidance)
        skills.append(skill)
        inventory.append({**provenance, "scope": item.scope})
    payload = bundle.model_dump(mode="json")
    payload["record_id"] = uuid4().hex
    payload["context"]["skills"] = [
        skill.model_dump(mode="json")
        for skill in sorted(skills, key=lambda skill: skill.skill_id)
    ]
    return AgentBundle.model_validate(payload), tuple(inventory)


def _publish_json(path: Path, value: dict) -> None:
    """Atomically expose one complete file, refusing an existing destination."""
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    # Unlike rename, a hard link cannot overwrite a concurrently created file.
    with atomic_path(path, Path(stream.name), publish=os.link):
        with stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    sync_directory(path.parent)


def publish_import(manifest_path: Path, bundle_path: Path, output: Path) -> dict:
    """Publish a new Bundle and its audit receipt without changing source assets."""
    receipt_path = output.with_name(output.name + ".receipt.json")
    if (
        output.exists()
        or output.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise FileExistsError("import output and receipt must both be new files")
    manifest_raw = manifest_path.read_bytes()
    bundle_raw = bundle_path.read_bytes()
    manifest = CorpusManifest.model_validate_json(manifest_raw)
    original = AgentBundle.model_validate_json(bundle_raw)
    imported, inventory = import_corpus(
        original, manifest, source_root=manifest_path.resolve().parent
    )
    receipt = {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "artifact_id": uuid4().hex,
            "bytes": len(manifest_raw),
        },
        "input_bundle": {
            "path": str(bundle_path.resolve()),
            "bytes": len(bundle_raw),
            "record_id": original.record_id,
        },
        "output_bundle": {
            "path": str(output.resolve()),
            "record_id": imported.record_id,
        },
        "texts": list(inventory),
        "verification": "unverified",
    }
    # Publish the receipt first; a visible Bundle always has its complete audit.
    _publish_json(receipt_path, receipt)
    try:
        _publish_json(output, imported.model_dump(mode="json"))
    except BaseException:
        receipt_path.unlink()
        raise
    return receipt
