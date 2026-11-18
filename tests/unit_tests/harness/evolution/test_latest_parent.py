"""Removing global parent choice preserves global final selection."""

import pytest

from reweave.evolution.contracts import EvolutionProtocol, VersionReference
from reweave.evolution.graph.selection import policy_parent
from tests.unit_tests.harness.evolution.fixtures import protocol


def test_latest_measured_parent_can_regress_but_final_selection_stays_global():
    raw = protocol().model_dump(mode="json")
    raw.update(
        parent_selection="latest_evaluated",
        width_mode="dynamic",
        max_authors_per_round=1,
    )
    declared = EvolutionProtocol.model_validate(raw)
    versions = tuple(
        VersionReference(version_id=v, bundle_id=v)
        for v in ("baseline", "child", "infra")
    )
    progress = {
        "versions": {v.version_id: v.bundle_id for v in versions},
        "cells": {},
        "episodes": {},
        "validation_cells": {},
    }
    for version, count, outcome in zip(versions, (2, 1, 0), (False, False, None)):
        key = version.version_id
        progress["cells"][key] = {"condition": key}
        progress["episodes"][key] = {
            "receipt_id": key,
            "bundle_id": key,
            "condition_id": "condition",
            "completed_nodes": list(range(count)),
            "progress": count / 3,
            "official_success": outcome,
        }
    assert policy_parent(progress, versions, declared).version_id == "child"
    assert (
        policy_parent(
            progress, versions, declared, purpose="final_selection"
        ).version_id
        == "baseline"
    )
    standard = declared.model_copy(update={"parent_selection": "global"})
    assert policy_parent(progress, versions, standard).version_id == "baseline"


@pytest.mark.parametrize(
    "changes",
    [
        {"max_authors_per_round": 2},
    ],
)
def test_latest_parent_cannot_silently_change_another_mechanism(changes):
    raw = protocol().model_dump(mode="json")
    raw.update(
        parent_selection="latest_evaluated",
        width_mode="dynamic",
        max_authors_per_round=1,
    )
    raw.update(changes)
    with pytest.raises(ValueError, match="latest-evaluated"):
        EvolutionProtocol.model_validate(raw)
