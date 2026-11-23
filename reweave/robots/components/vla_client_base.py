# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Unified VLA client base class."""

from __future__ import annotations


class BaseVLAClient:
    """Unified VLA client base class."""

    _TIMEOUT_S: dict[str, float] = {"default": 30.0, "predict": 120.0}

    def __init__(self, client):
        self._client = client

    def predict(self, obs, options=None):
        """Request one backend-defined VLA prediction response.

        Args:
            obs: observation data.
            options: optional dict.

        Returns:
            Backend response. Robot-specific clients own and validate its
            action semantics.
        """
        return self._client.call(
            "vla.predict",
            args=(obs, options),
            timeout_s=self._TIMEOUT_S["predict"],
        )
