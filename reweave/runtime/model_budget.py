# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Shared controller allowance with a complete, separate inference ledger."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


class ModelServiceFailure(RuntimeError):
    """A fixed provider failed; must not be relabeled as authored program failure."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        # Request-layer classification survives wrapping and episode finalization.
        self.retryable = retryable


class ModelBudgetExceeded(RuntimeError):
    """Root and local controllers used their shared decision allowance."""


class ModelServiceBudgetExceeded(ModelServiceFailure):
    """Failed service attempts spent the cost cap before full decisions arrived."""


class ModelCallBudget:
    def __init__(self, maximum: int, evidence: Path):
        self.maximum, self.evidence = maximum, evidence
        self.attempts = 0
        self.controller_attempts = 0
        self.failed_controller_attempts = 0
        self._lock = threading.Lock()
        self.controller_kind: ContextVar[str] = ContextVar(
            "controller_kind", default="planner"
        )

    def snapshot(self) -> dict:
        """Read shared root/local allowance before the next charged native turn."""
        with self._lock:
            return {
                "unit": "native_turn",
                "limit": self.maximum,
                "used": self.controller_attempts,
                "remaining": max(0, self.maximum - self.controller_attempts),
                "failed_attempts": self.failed_controller_attempts,
            }

    @contextmanager
    def request(self, kind: str, *, request_id: str | None = None):
        """Charge one actual attempt, optionally bound to a Host logical request.

        request_attempt is this budget's charged ordinal, not a provider turn ID
        or physical step. Reconnected reads keep the same context; continuations
        create another ordinal while retaining their original Host request_id.
        """
        with self._lock:
            if kind == "controller":
                if self.controller_attempts >= self.maximum:
                    if self.failed_controller_attempts:
                        raise ModelServiceBudgetExceeded(
                            "controller attempt cost cap exhausted after service failures; full decision allowance was not delivered",
                            retryable=False,
                        )
                    raise ModelBudgetExceeded(
                        "fixed episode controller request budget exhausted"
                    )
                self.controller_attempts += 1
            self.attempts += 1
            identity = self.attempts
        started = time.monotonic()
        status, error = "failed", None
        try:
            yield
            status = "completed"
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            with self._lock:
                if kind == "controller" and status == "failed":
                    self.failed_controller_attempts += 1
                self.evidence.parent.mkdir(parents=True, exist_ok=True)
                with self.evidence.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "request_attempt": identity,
                                "request_id": request_id,
                                "kind": kind,
                                "controller_kind": self.controller_kind.get()
                                if kind == "controller"
                                else None,
                                "status": status,
                                "error": error,
                                "wall_time_s": time.monotonic() - started,
                            }
                        )
                        + "\n"
                    )
