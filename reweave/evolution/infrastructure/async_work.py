# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Blocking job-local work that must finish before its owner releases locks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def run_blocking(
    function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Keep I/O off the event loop without abandoning writes on cancellation.

    Threads cannot be forcibly cancelled. Hold the caller (and its job lock)
    until the worker settles, even under repeated cancellation. A worker error
    remains the raised error, with cancellation as its context.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        worker.result()
        raise
