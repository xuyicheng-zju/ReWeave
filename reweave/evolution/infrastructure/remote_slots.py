"""Retain remote resource ownership across local process or connection loss."""

import json
import time

from reweave.evolution.infrastructure.campaign_slots import campaign_root, enabled
from reweave.utils.persistence import atomic_json


def retain_remote_slot(output, slot, identity, remote_output):
    root = campaign_root(output)
    if root is None or not enabled(root):
        return None
    path = root / "scheduler/online" / f"slot-{slot:02d}.remote.json"
    if path.exists():
        previous = json.loads(path.read_text())
        same = (
            previous["service_instance_id"] == identity
            and previous["remote_output"] == remote_output
        )
        if previous["status"] != "exited" and not same:
            raise RuntimeError("Remote slot still has an unsettled owner")
        if same:
            return path
    atomic_json(
        path,
        {
            "status": "unsettled",
            "slot": slot,
            "service_instance_id": identity,
            "remote_output": remote_output,
            "local_output": str(output),
            "at": time.time(),
        },
    )
    return path


def settle_remote_slot(path, state):
    if path is None:
        return
    value = json.loads(path.read_text())
    if (
        state.get("service_instance_id") != value["service_instance_id"]
        or state.get("status") != "exited"
        or state.get("exit", {}).get("parent_closed") is not True
    ):
        raise ValueError("Remote slot release requires its owner's closed exit")
    value.update(status="exited", exit=state["exit"], closed_at=time.time())
    atomic_json(path, value)
