# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Resolve graph program slices against their actual executable Bundle."""

from reweave.evolution.bundle import AgentBundle
from reweave.evolution.evidence.trajectory import resolve_original_pointer


def program_assets(program, original):
    """Reference actual admitted source, without claiming independent executability."""
    bundle = AgentBundle.model_validate(original)
    selected = {
        pointer: resolve_original_pointer(original, pointer)
        for pointer in program.pointers
    }
    return {
        "bundle_id": bundle.record_id,
        "assets": [
            {
                "pointer": pointer,
                "scope": {"workflows": "W", "context": "C", "loop": "L"}[
                    pointer.split("/")[1]
                ],
            }
            for pointer in selected
        ],
    }
