# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Regression coverage for LIBERO comparison-video rendering."""

from __future__ import annotations

import numpy as np

from reweave.benchmarks.libero.make_comparison import _pad_even


def test_pad_even_preserves_pixels_and_satisfies_h264_dimensions() -> None:
    frame = np.full((281, 515, 3), 127, dtype=np.uint8)

    padded = _pad_even(frame)

    assert padded.shape == (282, 516, 3)
    np.testing.assert_array_equal(padded[:281, :515], frame)
    assert not padded[-1].any()
    assert not padded[:, -1].any()


def test_pad_even_returns_aligned_frame_without_copy() -> None:
    frame = np.zeros((282, 516, 3), dtype=np.uint8)

    assert _pad_even(frame) is frame
