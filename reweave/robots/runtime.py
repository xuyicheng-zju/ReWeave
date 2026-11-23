# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The RPent Authors.
# Modified for ReWeave.

"""Shared spawn/wait/stop helpers for robot runtime components.

These orchestrate the lifecycle of subprocess-backed servers (env, vla,
sam3) selected through the ``init_runtime`` contract on
:class:`~reweave.robots.robot_spec.RobotSpec`. Each helper reports state
transitions to a :class:`~reweave.dashboard.events.DashboardEventSink` so
the dashboard UI stays in sync with the actual processes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from reweave.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from reweave.utils.daemon import ProcessDaemon
from reweave.utils.logging import get_logger
from reweave.utils.rpc.client_utils import wait_for_ready

if TYPE_CHECKING:
    from reweave.utils.rpc.rpc_client import RpcClient

logger = get_logger("robot_runtime")


def stop_owned_daemons(
    daemons: dict[str, ProcessDaemon],
    dashboard_events: DashboardEventSink,
) -> None:
    """Attempt every owned stop in reverse order without masking startup failure.

    Only a completed stop marks a component pending. A failed stop remains
    explicitly failed in the dashboard and log; its process may still be alive.
    """
    for component, daemon in reversed(list(daemons.items())):
        try:
            daemon.stop()
        except Exception as exc:
            logger.error("%s cleanup failed: %s", component, exc, exc_info=True)
            dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
        else:
            dashboard_events.emit(RuntimeStatusEvent(component, "pending"))


def try_spawn_server(
    daemons: dict[str, ProcessDaemon],
    dashboard_events: DashboardEventSink,
    component: str,
    spawn_fn: Callable[[], tuple[ProcessDaemon | None, RpcClient]],
) -> tuple[ProcessDaemon | None, RpcClient]:
    dashboard_events.emit(RuntimeStatusEvent(component, "starting"))
    try:
        daemon, rpc = spawn_fn()
        if daemon is not None:
            daemons[component] = daemon
        return daemon, rpc
    except Exception as exc:
        stop_owned_daemons(daemons, dashboard_events)
        dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
        raise RuntimeError(f"[{component}] spawn failed: {exc}") from exc


def try_wait_server(
    daemons: dict[str, ProcessDaemon],
    dashboard_events: DashboardEventSink,
    component: str,
    rpc: RpcClient,
    daemon: ProcessDaemon | None,
    timeout_s: float,
    *,
    post_fn: Callable[[], Any] | None = None,
):
    try:
        wait_for_ready(rpc, daemon=daemon, timeout_s=timeout_s)
        result = None
        if post_fn is not None:
            result = post_fn()
    except Exception as exc:
        stop_owned_daemons(daemons, dashboard_events)
        dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
        raise RuntimeError(
            f"[{component}] wait / client connect failed: {exc}"
        ) from exc
    dashboard_events.emit(RuntimeStatusEvent(component, "ready"))
    return result
