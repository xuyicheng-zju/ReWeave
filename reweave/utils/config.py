# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Path resolution and environment-variable configuration."""

from __future__ import annotations

import os
from pathlib import Path



def get_repo_root() -> Path:
    """Return the ReWeave source root.

    Resolution: ``REWEAVE_REPO_ROOT`` env var, then the parent of
    the ``reweave/`` package directory.
    """
    env = os.environ.get("REWEAVE_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # config.py lives at <repo>/reweave/utils/config.py
    return Path(__file__).resolve().parents[2]




def get_memory_dir(robot_name: str) -> Path:
    """Return the persistent, cross-run memory directory for a robot."""
    return get_repo_root() / "memory" / robot_name


def get_pi05_checkpoint_path() -> str:
    return os.environ.get("PI05_CHECKPOINT_PATH", "")


def get_libero_type() -> str:
    return os.environ.get("LIBERO_TYPE", "pro")


def get_rlinf_repo_path() -> Path | None:
    """Return the configured RLinf checkout path, or *None*."""
    env = os.environ.get("REWEAVE_RLINF_ROOT") or os.environ.get("RLINF_REPO_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return None
