"""Reuse a published parent's synthetic fixtures without inheriting its verdict."""

from __future__ import annotations

import json
from pathlib import Path

from reweave.evolution.contracts import EvidenceFile
from reweave.utils.persistence import atomic_json

SUITE = ".scratch/tests/interaction.json"


def seed_parent_interactions(workspace, parent_id, receipts):
    """Copy only the exact parent's published suite; keep in-progress edits intact."""
    destination = Path(workspace) / SUITE
    if destination.exists():
        return
    for receipt in receipts:
        bundles = (
            [receipt["bundle"]]
            if "bundle" in receipt
            else [item["bundle"] for item in receipt.get("candidates", ())]
        )
        if not any(bundle["record_id"] == parent_id for bundle in bundles):
            continue
        artifacts = receipt.get("artifacts", ())
        if receipt.get("interaction_suite") is not None:
            artifacts = (*artifacts, receipt["interaction_suite"])
        for raw in artifacts:
            if not raw["path"].endswith("/workspace/" + SUITE):
                continue
            original = EvidenceFile.model_validate(raw)
            original.verify()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(original.path).read_bytes())
            atomic_json(
                Path(workspace) / "context/parent_interaction.json",
                {
                    "bundle_id": parent_id,
                    "source": original.model_dump(mode="json"),
                    "fixture": SUITE,
                    "meaning": "Parent fixture scaffold only. Adapt coverage and rerun phase=interaction for the current candidate; no prior verdict is inherited.",
                },
            )
            return


def history_parent_interactions(workspace, parent_id, originals):
    """Resolve only parent author receipts already admitted by task-local history."""
    navigation = originals["navigation"]
    receipts = []
    for version in navigation.get("versions", {}).values():
        if version["bundle_id"] != parent_id or not version.get("author_attempt"):
            continue
        raw = navigation["attempts"][version["author_attempt"]]["receipt"]
        original = EvidenceFile.model_validate(
            {key: raw[key] for key in ("path", "artifact_id", "bytes")}
        )
        original.verify()
        receipts.append(json.loads(Path(original.path).read_text()))
    seed_parent_interactions(workspace, parent_id, receipts)
