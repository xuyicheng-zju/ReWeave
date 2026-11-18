"""Expose existing account locks over an authenticated loopback SSH channel."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import select
import socketserver
import time
from pathlib import Path

from reweave.utils.persistence import atomic_json
from reweave.utils.provider_admission import ProviderAdmission


class InFlightReservations:
    """Count existing requests until their original Host records completion.

    A gate handoff must not grant 32 new leases in addition to requests already
    running against the provider. Missing evidence keeps its reservation held.
    """

    def __init__(self, gate, records, status_path):
        if len(records) > gate.capacity:
            raise ValueError("In-flight migration exceeds shared request capacity")
        self.status_path = status_path
        self.pending = []
        try:
            for slot, record in enumerate(records):
                handle = gate._try_lock(gate.root / f"slot-{slot:02d}.lock")
                if handle is None:
                    raise RuntimeError("New gate pool already has an owner")
                self.pending.append((record, handle))
        except BaseException:
            self.close()
            raise
        self.released = 0
        self.poll()

    def poll(self):
        remaining = []
        for record, handle in self.pending:
            try:
                current = json.loads(Path(record["audit_path"]).read_text())
                done = current.get("started_at") != record["started_at"] or current.get(
                    "status"
                ) in {"completed", "failed"}
                done = done or any(
                    Path(path).is_file() for path in record.get("terminal_receipts", [])
                )
            except (OSError, ValueError):
                done = False
            if done:
                handle.close()
                self.released += 1
            else:
                remaining.append((record, handle))
        self.pending = remaining
        atomic_json(
            self.status_path,
            {"at": time.time(), "remaining": len(remaining), "released": self.released},
        )

    def watch(self, stopping):
        while not stopping.wait(0.2) and self.pending:
            self.poll()

    def close(self):
        for _, handle in self.pending:
            handle.close()
        self.pending = []


class AdmissionServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, gate, token):
        self.gate, self.token = gate, token
        super().__init__(address, AdmissionHandler)


class AdmissionHandler(socketserver.StreamRequestHandler):
    def send(self, value):
        self.wfile.write(json.dumps(value).encode() + b"\n")
        self.wfile.flush()

    def handle(self):
        lease = None
        succeeded = False
        try:
            self.request.settimeout(10)
            value = json.loads(self.rfile.readline(1024 * 1024))
            if not hmac.compare_digest(str(value.get("token", "")), self.server.token):
                self.send({"status": "unauthorized"})
                return
            command = value.get("command")
            gate = self.server.gate
            if command in {"rate_limit", "transport_failure", "quota_exhausted"}:
                note = {
                    "rate_limit": gate.note_rate_limit,
                    "transport_failure": gate.note_transport_failure,
                    "quota_exhausted": gate.note_quota_exhausted,
                }[command]
                self.send({"status": "ok", "recorded": note(value.get("error"))})
                return
            if command != "lease":
                self.send({"status": "invalid_command"})
                return
            self.request.settimeout(None)
            while lease is None:
                lease, wait = gate._acquire()
                if lease is None and select.select([self.request], [], [], wait)[0]:
                    return
            self.send(
                {
                    "status": "admitted",
                    "capacity": gate.effective_capacity,
                    "recovery_probe": lease[1] is not None,
                    "recovery_capacity": min(4, gate.effective_capacity)
                    if lease[1] is not None
                    else gate.effective_capacity,
                }
            )
            completed = self.rfile.readline(1024)
            succeeded = bool(
                completed and json.loads(completed).get("succeeded") is True
            )
        except (OSError, ValueError, KeyError):
            pass
        finally:
            if lease is not None:
                self.server.gate._release(lease, succeeded)
                try:
                    self.send({"status": "closed"})
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--route", default="codex-auth")
    parser.add_argument("--capacity", type=int)
    args = parser.parse_args()
    os.umask(0o077)
    gate = ProviderAdmission(
        args.route, root=args.root, capacity=args.capacity, start_interval_s=0
    )
    with AdmissionServer(
        ("127.0.0.1", args.port), gate, args.token_file.read_text().strip()
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
