"""Bounded, process-local HTTP receipts; lookup never dispatches task work."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _Receipt:
    signature: bytes
    created: float
    response: bytes | None = None
    dispatch_s: float = 0.0
    encode_s: float = 0.0


class RequestReceipts:
    """Keep pending calls and a bounded completed-response recovery window.

    Missing/expired means unknown, never evidence that dispatch did not occur.
    Clients must never resubmit mutations, including after receipt expiry.
    Serialized responses are immutable snapshots, not mutable facade results.
    """

    def __init__(self, *, max_entries=4096, max_bytes=256 * 1024 * 1024, ttl_s=3600.0):
        if max_entries < 1 or max_bytes < 1 or ttl_s <= 0:
            raise ValueError("Receipt limits must be positive")
        self.max_entries, self.max_bytes, self.ttl_s = max_entries, max_bytes, ttl_s
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def _prune(self):
        now = time.monotonic()
        for key, entry in list(self._entries.items()):
            if entry.response is not None and (
                now - entry.created > self.ttl_s
                or len(self._entries) > self.max_entries
                or self._bytes > self.max_bytes
            ):
                self._bytes -= len(entry.response)
                del self._entries[key]

    def begin(self, session_id, request_id, signature):
        with self._lock:
            self._prune()
            key = (session_id, request_id)
            entry = self._entries.get(key)
            if entry is not None:
                if entry.signature != signature:
                    raise ValueError("RPC request ID reused with different input")
                return False, entry.response
            if len(self._entries) >= self.max_entries:
                # Make space only from completed work; never discard pending.
                for old_key, old in self._entries.items():
                    if old.response is not None:
                        self._bytes -= len(old.response)
                        del self._entries[old_key]
                        break
                else:
                    raise RuntimeError("RPC receipt capacity occupied by pending work")
            self._entries[key] = _Receipt(signature, time.monotonic())
            return True, None

    def complete(self, session_id, request_id, response, dispatch_s, encode_s):
        with self._lock:
            entry = self._entries[(session_id, request_id)]
            entry.response = bytes(response)
            entry.created = time.monotonic()
            entry.dispatch_s, entry.encode_s = dispatch_s, encode_s
            self._bytes += len(entry.response)
            self._prune()

    def lookup(self, session_id, request_id):
        with self._lock:
            self._prune()
            entry = self._entries.get((session_id, request_id))
            if entry is None:
                return {"state": "unknown"}
            if entry.response is None:
                return {"state": "pending"}
            return {
                "state": "completed",
                "response_json": entry.response.decode("utf-8"),
                "dispatch_s": entry.dispatch_s,
                "encode_s": entry.encode_s,
            }
