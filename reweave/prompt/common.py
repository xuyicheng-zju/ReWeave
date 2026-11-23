# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""Global prompt definitions shared by robots."""

from __future__ import annotations

from reweave.prompt.utils import BulletList

OUTPUT = BulletList(
    [
        "Brief reasoning before each tool call (1-2 sentences): observation -> decision.",
        "Don't re-read files already in this session. Don't view_env_state right after a primitive tool already returned the state.",
        "Numerical coords in 3 decimals are enough.",
        "Save artifacts BEFORE calling finish. Stop immediately after writing the audit; do not chat further.",
    ]
)
