# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Durable native usage journals: completion receipts, totals and reconciliation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from reweave.evolution.contracts import Usage
from reweave.utils.persistence import atomic_json


def native_completion(
    directory: Path, usage: Usage, session_id: str, checkpoint: dict[str, Any]
) -> None:
    atomic_json(
        directory / "native_result.json",
        {
            "session_id": session_id,
            "native_checkpoint": checkpoint,
            "usage": usage.model_dump(mode="json"),
        },
    )


def complete_query(directory, usage, session_id, store, native_path=None):
    """Checkpoint and publish a finished query before honoring cancellation.

    Once the native stream ends, recovery must be able to publish that output
    without another provider query. The whole completion commit is one worker.
    """
    store.bind(session_id)
    if native_path is not None:
        store.capture(native_path)
    checkpoint = store.checkpoint()
    native_completion(directory, usage, session_id, checkpoint)


def failed_initializations(audit: Path) -> list[str]:
    """Only terminal SDK initialization failures permit a fresh native session.

    Missing identity alone is not evidence: the model may already have run.
    Receipts and unknown usage remain charged across this explicit recovery.
    """
    queries = sorted((audit / "queries").glob("*.json"))
    if not queries:
        return []
    events = [
        json.loads(line) for line in (audit / "events.jsonl").read_text().splitlines()
    ]
    failures = {
        event["query"]
        for event in events
        if event.get("event") == "native_query_failed"
        and event.get("phase") == "sdk_initialization"
    }
    for path in queries:
        usage = Usage.model_validate_json(path.read_text())
        if path.name not in failures or not usage.wall_time_complete:
            raise ValueError(
                "native session evidence exists without a recoverable identity"
            )
    return [path.name for path in queries]


def cumulative_usage(audit: Path) -> Usage:
    """Charge every pre-query interval and native query across recovery."""
    preparation_usage = audit / "preparation_usage"
    values = [
        Usage.model_validate_json(path.read_text())
        for path in (
            *sorted(preparation_usage.glob("*.json")),
            *sorted((audit / "retry_usage").glob("*.json")),
            *sorted((audit / "queries").glob("*.json")),
        )
    ]
    return Usage(
        physical_steps=sum(value.physical_steps for value in values),
        model_calls=sum(value.model_calls for value in values)
        if values and all(value.model_calls is not None for value in values)
        else None,
        wall_time_s=sum(value.wall_time_s for value in values),
        wall_time_charge_s=sum(value.charged_wall_time_s for value in values)
        if values
        and all(
            value.wall_time_complete or value.wall_time_charge_s is not None
            for value in values
        )
        and any(value.wall_time_charge_s is not None for value in values)
        else None,
        wall_time_complete=bool(values)
        and all(value.wall_time_complete for value in values),
        cost_usd=sum(value.cost_usd for value in values)
        if values and all(value.cost_usd is not None for value in values)
        else None,
    )


def with_admission_usage(queries: Usage, audit: Path) -> Usage:
    """Include admission and nested model calls without repeating native elapsed time."""
    checks = [
        json.loads(path.read_text())
        for path in sorted((audit / "checks").glob("*.json"))
    ]
    charge_known = (
        queries.wall_time_complete or queries.wall_time_charge_s is not None
    ) and all(
        item["wall_time_complete"] or item.get("wall_time_charge_s") is not None
        for item in checks
    )
    has_estimate = queries.wall_time_charge_s is not None or any(
        not item["wall_time_complete"] for item in checks
    )
    model_work = [
        Usage.model_validate_json(p.read_text())
        for p in (
            *(audit / "planner_probes").glob("*/usage.json"),
            *(audit / "session_inquiry").glob("usage.json"),
        )
    ]
    return queries.model_copy(
        update={
            "model_calls": queries.model_calls + sum(p.model_calls for p in model_work)
            if queries.model_calls is not None
            and all(p.model_calls is not None for p in model_work)
            else None,
            "cost_usd": queries.cost_usd + sum(p.cost_usd for p in model_work)
            if queries.cost_usd is not None
            and all(p.cost_usd is not None for p in model_work)
            else None,
            "wall_time_s": queries.wall_time_s
            + sum(item["wall_time_s"] for item in checks),
            "wall_time_complete": queries.wall_time_complete
            and all(item["wall_time_complete"] for item in checks),
            "wall_time_charge_s": queries.charged_wall_time_s
            + sum(
                item["wall_time_charge_s"]
                if item.get("wall_time_charge_s") is not None
                else item["wall_time_s"]
                for item in checks
            )
            if charge_known and has_estimate
            else None,
        }
    )


def reconcile_interrupted_checks(audit: Path) -> None:
    """Close an interrupted check's charging interval before starting another query."""
    for path in sorted((audit / "checks").glob("*.json")):
        prior = json.loads(path.read_text())
        if prior["wall_time_complete"] or prior.get("wall_time_charge_s") is not None:
            continue
        now = time.time()
        if "started_at" not in prior or now < prior["started_at"]:
            raise ValueError(
                "interrupted admission timing requires valid original start evidence"
            )
        prior["wall_time_charge_s"] = max(
            prior["wall_time_s"], now - prior["started_at"]
        )
        prior["reconciled_at"] = now
        atomic_json(path, prior)
        with (audit / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "interrupted_admission_time_reconciled",
                        "check": path.name,
                        **prior,
                    }
                )
                + "\n"
            )


def reconcile_interrupted_timing(directory: Path) -> None:
    """Called under the job lock after proving the old native process is gone.

    Preserve measured lower bounds. Charge the whole calendar interval through
    recovery, including downtime, and retain the calculation as an audit event.
    Missing start evidence or a backwards clock cannot authorize another launch.
    """
    audit = directory / "audit"
    events = audit / "events.jsonl"
    if not events.exists():
        return
    starts = {}
    for line in events.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "native_query_started":
            if event["query"] in starts:
                raise ValueError("duplicate native query start evidence")
            starts[event["query"]] = event["started_at"]
    reconciled_at = time.time()
    for path in sorted((audit / "queries").glob("*.json")):
        value = Usage.model_validate_json(path.read_text())
        if value.wall_time_complete or value.wall_time_charge_s is not None:
            continue
        started_at = starts.get(path.name)
        if started_at is None or reconciled_at < started_at:
            raise ValueError(
                "interrupted query timing requires valid original start evidence"
            )
        charge = max(value.wall_time_s, reconciled_at - started_at)
        with events.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "interrupted_time_reconciled",
                        "query": path.name,
                        "started_at": started_at,
                        "reconciled_at": reconciled_at,
                        "measured_lower_bound_s": value.wall_time_s,
                        "wall_time_charge_s": charge,
                        "includes_downtime": True,
                    }
                )
                + "\n"
            )
        atomic_json(
            path,
            value.model_copy(update={"wall_time_charge_s": charge}).model_dump(
                mode="json"
            ),
        )
    if (audit / "queries").exists():
        atomic_json(
            audit / "usage.json",
            with_admission_usage(cumulative_usage(audit), audit).model_dump(
                mode="json"
            ),
        )
