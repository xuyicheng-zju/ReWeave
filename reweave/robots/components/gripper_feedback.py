# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PyRUA-Lean Contributors.
# Modified for ReWeave.
"""PyRUA-Lean's measured two-action stroke convergence, with explicit units.

Adapters supply measured opening and tolerance in the SAME units. A stable
opening proves neither contact nor possession. Exceptions never trigger retry.
"""

from __future__ import annotations

import numpy as np


class GripperSettling:
    def __init__(self, opening, *, tolerance: float):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("gripper settling tolerance must be positive and finite")
        self.tolerance = float(tolerance)
        self.start = self.last = self._reading(opening)
        self.steps = 0
        self.stable = False
        self.delta = None

    @staticmethod
    def _reading(value):
        if value is None:
            return None
        reading = np.asarray(value, dtype=float).reshape(-1)
        if not reading.size or not np.isfinite(reading).all():
            return None
        return reading.copy()

    def observe(self, opening) -> bool:
        self.steps += 1
        current = self._reading(opening)
        self.delta = None
        if (
            current is not None
            and self.last is not None
            and current.shape == self.last.shape
        ):
            self.delta = float(np.max(np.abs(current - self.last)))
        self.stable = (
            self.steps >= 2 and self.delta is not None and self.delta < self.tolerance
        )
        self.last = current
        return self.stable

    def feedback(self, *, unit: str) -> dict:
        return {
            "opening_before": None if self.start is None else self.start.tolist(),
            "measured_opening": None if self.last is None else self.last.tolist(),
            "opening_change": self.delta,
            "opening_unit": unit,
            "opening_stable": self.stable
            if self.last is not None and self.steps >= 2
            else None,
            "measurement_available": self.last is not None,
        }
