# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""LIBERO worker construction, seeding and read-only instrumentation."""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import numpy as np

from reweave.benchmarks.libero.cartesian_control import collect_cartesian_control_spec
from reweave.benchmarks.libero.privileged_state import attach_privileged_critic_state


def wrap_libero_env_factories(
    factories: list[Callable[[], Any]], *, seed: int
) -> list[Callable[[], Any]]:
    """Seed inside each spawned process, including subsequent backend resets.

    LIBERO placement samplers use NumPy's global stream. The robosuite-backed
    wrapper seeds its own Generator, which does not seed that separate stream.
    Fixture body poses sampled at reset are not restored by flattened qpos/qvel.
    """
    wrapped: list[Callable[[], Any]] = []
    for factory in factories:

        def worker_factory(factory: Callable[[], Any] = factory) -> Any:
            np.random.seed(seed)
            env = factory()
            backend_seed = env.seed

            def seed_worker(value: int) -> Any:
                np.random.seed(value)
                return backend_seed(value)

            env.seed = seed_worker
            attach_privileged_critic_state(env)
            env.harness_cartesian_control_spec = types.MethodType(
                collect_cartesian_control_spec, env
            )
            return env

        wrapped.append(worker_factory)
    return wrapped


def install_libero_worker(libero_env_class: type) -> None:
    """Extend the existing RLinF worker factory before it spawns subprocesses."""
    if getattr(libero_env_class, "_harness_worker_extension", False):
        return
    original = libero_env_class.get_env_fns

    def get_env_fns(self: Any) -> list[Callable[[], Any]]:
        # RLinF passes self.seed to each LIBERO factory; each episode here owns
        # one environment. A later seed RPC uses its actual supplied value.
        return wrap_libero_env_factories(original(self), seed=self.seed)

    libero_env_class.get_env_fns = get_env_fns
    libero_env_class._harness_worker_extension = True
