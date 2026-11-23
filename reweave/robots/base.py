# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Discover installed benchmark adapters and load their robot specs lazily."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Any

from reweave.robots.robot_spec import RobotSpec

if TYPE_CHECKING:
    from reweave.tools.toolkit import Toolkit


def _resolve_robot(name: str) -> Any:
    if not name:
        raise ValueError("robot name must be non-empty")
    robot_name = name.lower()
    module_name = f"reweave.benchmarks.{robot_name}"
    try:
        return importlib.import_module(f"{module_name}.robot_spec")
    except ModuleNotFoundError as e:
        if e.name in {module_name, f"{module_name}.robot_spec"}:
            raise ValueError(f"unknown robot: {robot_name!r}") from e
        raise


def enumerate_robots() -> tuple[str, ...]:
    """Return the names of importable robot packages under ``reweave.benchmarks``."""
    from reweave import benchmarks

    return tuple(
        sorted(
            module.name
            for module in pkgutil.iter_modules(benchmarks.__path__)
            if module.ispkg and not module.name.startswith("_")
        )
    )


def get_robot_spec(name: str) -> RobotSpec:
    return _resolve_robot(name).get_robot_spec()


def get_toolkit(name: str, **kwargs) -> Toolkit:
    """Build the robot toolkit (common tools + robot-specific tools)."""
    return _resolve_robot(name).get_toolkit(**kwargs)
