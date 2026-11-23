# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Resolve evolution attempts and explicitly imported completed receipts."""

from pathlib import Path


def attempt_directory(root, key, attempt):
    """Resolve an explicitly imported completed receipt without rewriting its bytes."""
    original = attempt.get("result_directory")
    if original is not None:
        if attempt["status"] != "completed" or not Path(original).is_absolute():
            raise ValueError(
                "only completed attempts can reference an original directory"
            )
        return Path(original)
    return root / "attempts" / key
