"""Gate replacement retains the charge for requests that are already running."""

import json

import pytest

from reweave.utils.provider_admission import ProviderAdmission
from reweave.utils.provider_admission_server import InFlightReservations


def test_inflight_request_holds_capacity_until_original_receipt_changes(tmp_path):
    gate = ProviderAdmission(
        "handoff", root=tmp_path / "pool", capacity=32, start_interval_s=0
    )
    gate.root.mkdir(parents=True)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"started_at": 1, "status": "admitted"}))
    held = InFlightReservations(
        gate, [{"audit_path": str(audit), "started_at": 1}], tmp_path / "status.json"
    )
    assert gate._try_lock(gate.root / "slot-00.lock") is None
    audit.unlink()
    held.poll()
    assert len(held.pending) == 1
    audit.write_text(json.dumps({"started_at": 2, "status": "waiting"}))
    held.poll()
    assert not held.pending and held.released == 1
    lock = gate._try_lock(gate.root / "slot-00.lock")
    assert lock is not None
    lock.close()


def test_gate_handoff_rejects_more_than_shared_capacity(tmp_path):
    gate = ProviderAdmission("handoff", root=tmp_path / "pool", capacity=32)
    with pytest.raises(ValueError, match="capacity"):
        InFlightReservations(gate, [{}] * 33, tmp_path / "status.json")
