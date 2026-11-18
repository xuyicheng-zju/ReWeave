# Modified for the ReWeave namespace.
"""One native session owns one pre-materialized Harness mutation."""

import asyncio

import pytest

from reweave.evolution.authoring.candidate import candidate_root
from reweave.evolution.validation.admission import admit_mutation
from tests.unit_tests.harness.evolution.test_bundle_author import workspace


def test_session_admits_the_single_candidate_with_author_identity(tmp_path):
    root = workspace(tmp_path)
    cases = []
    (candidate,) = asyncio.run(admit_mutation(root, case_results=cases))
    assert candidate.branch_id == "author_000"
    assert cases and all("branch_id" not in case for case in cases)


def test_candidate_cannot_skip_its_interaction(tmp_path):
    root = workspace(tmp_path)
    (root / ".scratch/tests/interaction.json").unlink()
    with pytest.raises(ValueError, match=".scratch/tests/interaction.json"):
        asyncio.run(admit_mutation(root))


def test_candidate_root_rejects_a_symbolic_link(tmp_path):
    root = workspace(tmp_path)
    candidate = root / "candidate"
    moved = root / ".scratch/candidate"
    candidate.rename(moved)
    candidate.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        candidate_root(root)
