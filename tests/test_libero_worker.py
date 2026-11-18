# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Worker seeding checks; no simulator or physical actions are used."""

from __future__ import annotations

import json
import subprocess
import sys

import cloudpickle
import numpy as np

from reweave.benchmarks.libero.worker import (
    install_libero_worker,
    wrap_libero_env_factories,
)


class _Environment:
    def __init__(self):
        self.constructed_fixture = np.random.uniform(size=3).tolist()

    def seed(self, value):
        # The installed robosuite adapter resets only its Generator. LIBERO
        # fixture samplers independently consume np.random's global stream.
        self.rng = np.random.default_rng(value)
        return [value]


def _sample_worker(seed):
    factory = wrap_libero_env_factories([_Environment], seed=seed)[0]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """import cloudpickle, json, sys, numpy as np
env = cloudpickle.loads(sys.stdin.buffer.read())()
result = {'constructed': env.constructed_fixture}
for value in (73, 19, 73):
    returned = env.seed(value)
    result.setdefault('resets', []).append({
        'returned': returned,
        'fixture': np.random.uniform(size=3).tolist(),
        'backend': env.rng.uniform(size=3).tolist(),
    })
result['sensors'] = [callable(env.harness_privileged_critic_state),
                     callable(env.harness_cartesian_control_spec)]
print(json.dumps(result))
""",
        ],
        input=cloudpickle.dumps(factory),
        capture_output=True,
        check=True,
        timeout=20,
    )
    return json.loads(result.stdout)


def test_spawned_worker_seeds_construction_and_repeated_reset_streams():
    first = _sample_worker(41)
    assert first == _sample_worker(41)
    other = _sample_worker(42)
    assert first["constructed"] != other["constructed"]
    assert first["resets"] == other["resets"]
    assert first["resets"][0] == first["resets"][2]
    assert first["resets"][0] != first["resets"][1]
    assert first["resets"][0]["returned"] == [73]
    assert first["sensors"] == [True, True]


def test_worker_installation_uses_configured_seed_once():
    class RLinF:
        seed = 41

        def get_env_fns(self):
            return [_Environment]

    state = np.random.get_state()
    try:
        install_libero_worker(RLinF)
        installed = RLinF.get_env_fns
        install_libero_worker(RLinF)
        assert RLinF.get_env_fns is installed
        first = RLinF().get_env_fns()[0]()
        np.random.uniform(size=100)
        second = RLinF().get_env_fns()[0]()
        assert first.constructed_fixture == second.constructed_fixture
    finally:
        np.random.set_state(state)
