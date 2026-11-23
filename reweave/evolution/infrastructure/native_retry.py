# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Retry classification for durable native coding-agent sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path

from reweave.utils.provider_admission import is_quota_exhausted

_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "IncompleteRead",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "TimeoutException",
        "WriteTimeout",
    }
)
_RETRYABLE_TEXT = (
    "api error: 408",
    "api error: 409",
    "api error: 429",
    "api error: 500",
    "api error: 502",
    "api error: 503",
    "api error: 504",
    "bad gateway",
    "connection aborted",
    "connection reset",
    "connection failed: error sending request",
    "stream disconnected before completion:",
    "claude upstream stream ended before message_delta, reason=eof",
    "gateway timeout",
    "internal server error",
    "overloaded",
    "rate limit",
    "server disconnected",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "too many requests",
)
_PERMANENT_TEXT = (
    "authentication",
    "invalid api key",
    "invalid_api_key",
    "model not found",
    "permission denied",
    "unauthorized",
    "unknown model",
)

# Every outage has a finite same-session recovery budget across restarts.
NATIVE_RETRY_LIMIT = 3


class NativeTransportFailure(RuntimeError):
    """A retryable provider failure from one native task's own model query."""

    def __init__(self, task_directory: Path, original: Exception):
        super().__init__(str(original))
        self.task_directory = task_directory.resolve()
        self.original = original


class RetryableNativeProviderError(RuntimeError):
    """A provider-declared transient error that can resume the native session."""


def _error_chain(error: BaseException):
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def is_retryable_native_error(error: BaseException) -> bool:
    """Return whether a native CLI failure can resume the same durable session.

    Retry requires transport evidence or an explicit transient provider error.
    A generic 400 is not evidence of a transient gateway failure. Permanent
    causes anywhere in the exception chain override a retryable outer wrapper.
    """
    if isinstance(error, asyncio.CancelledError):
        return False
    chain = tuple(_error_chain(error))
    # Inspect the whole chain before accepting a generic SDK wrapper such as
    # ResultError; its outer message can omit a permanent auth/model cause.
    has_retryable_status = False
    for cause in chain:
        blob = f"{type(cause).__name__}: {cause}".lower()
        if is_quota_exhausted(blob):
            return False
        if any(marker in blob for marker in _PERMANENT_TEXT):
            return False
        terminal_reason = getattr(cause, "terminal_reason", None)
        if terminal_reason and terminal_reason != "api_error":
            return False
        status = getattr(cause, "api_error_status", None)
        if status is not None:
            try:
                status = int(status)
            except (TypeError, ValueError):
                return False
            if status not in {408, 409, 429} and not 500 <= status < 600:
                return False
            has_retryable_status = True
    if has_retryable_status or any(
        isinstance(cause, RetryableNativeProviderError) for cause in chain
    ):
        return True
    for cause in chain:
        blob = f"{type(cause).__name__}: {cause}".lower()
        if type(cause).__name__ in _RETRYABLE_EXCEPTION_NAMES:
            return True
        if any(marker in blob for marker in _RETRYABLE_TEXT):
            return True
    return False


def native_retry_wait_s(attempt: int) -> float:
    """Short exponential backoff for the Nth same-session resume."""
    return min(2.0 * (2 ** max(attempt - 1, 0)), 60.0)
