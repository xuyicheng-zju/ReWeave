# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Explicit CUDA placement must not silently select a different renderer."""

import os

import pytest

from reweave.utils import egl


def test_explicit_device_overrides_inherited_pin(monkeypatch):
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    monkeypatch.setattr(egl, "cuda_to_egl_map", lambda: {0: 2, 1: 3})
    egl.configure_egl_device(1)
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "3"


def test_absent_device_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(egl, "cuda_to_egl_map", lambda: {0: 2})
    with pytest.raises(RuntimeError, match="no matching EGL"):
        egl.configure_egl_device(1)


def test_probe_failure_is_an_initialization_error(monkeypatch):
    def probe():
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(egl, "cuda_to_egl_map", probe)
    with pytest.raises(RuntimeError, match="driver unavailable"):
        egl.configure_egl_device(0)
