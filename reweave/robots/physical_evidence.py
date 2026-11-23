# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Side-effect-free physical scalar measurements shared by simulator adapters.

The wire dictionary describes each feature at collection time. Evaluator calls
are deliberately excluded: physical state does not certify official success.
"""

from __future__ import annotations

import math

import numpy as np


def feature(value, unit: str, source: str, derivation: str = "direct measurement"):
    """Encode a scalar with explicit source, units and missingness."""
    if isinstance(value, (np.bool_, bool)):
        value = bool(value)
    elif isinstance(value, (int, float, np.number)):
        value = float(value)
        if not math.isfinite(value):
            value = None
    elif value is not None and not isinstance(value, str):
        raise TypeError("physical evidence requires a scalar")
    return {
        "value": value,
        "unit": unit,
        "source": source,
        "derivation": derivation,
        "availability": "unavailable" if value is None else "observed",
    }


def libero_features(values):
    """Declare only channels with known LIBERO sensor semantics; keep raw originals."""
    result = {}
    for name, value in values.items():
        if isinstance(value, bool):
            unit = "boolean"
        elif isinstance(value, str):
            unit = "text"
        elif name.startswith(("robot.eef.", "robot.gripper.")) and not name.endswith(
            "available"
        ):
            unit = "m"
        elif name.endswith(("_m", ".position.x", ".position.y", ".position.z")):
            unit = "m"
        elif (
            name.startswith("privileged.joint.")
            and ".unit" not in name
            and (name.rsplit(".", 1)[0] + ".unit") in values
        ):
            unit = values[name.rsplit(".", 1)[0] + ".unit"]
        elif name.endswith("_n"):
            unit = "N"
        elif "fraction_from_lower_limit" in name or name.endswith(
            ("count", ".predicate_fraction")
        ):
            unit = "dimensionless"
        else:
            # Unannotated historical scalars remain accessible in their originals.
            continue
        if name.endswith("max_normal_force_n") and not values.get(
            "privileged.contact.force_available"
        ):
            value = None
        result[name] = feature(value, unit, "LIBERO:" + name)
    return result
