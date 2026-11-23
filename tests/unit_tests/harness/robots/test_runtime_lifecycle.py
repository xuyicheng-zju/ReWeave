# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RPent Contributors
# Modified for ReWeave.
"""Startup failures retain their cause and report each owned service cleanup."""

from types import SimpleNamespace

import pytest

from reweave.robots.runtime import stop_owned_daemons, try_spawn_server, try_wait_server


def _daemons(stopped):
    def daemon(name, error=None):
        def stop():
            stopped.append(name)
            if error is not None:
                raise error

        return SimpleNamespace(stop=stop)

    return {
        "env": daemon("env"),
        "vla": daemon("vla", OSError("owned VLA did not exit")),
        "sam": daemon("sam"),
    }


def test_owned_cleanup_continues_and_never_marks_failed_stop_pending(caplog):
    stopped, events = [], []
    stop_owned_daemons(_daemons(stopped), SimpleNamespace(emit=events.append))
    assert stopped == ["sam", "vla", "env"]
    assert [(event.component, event.status) for event in events] == [
        ("sam", "pending"),
        ("vla", "failed"),
        ("env", "pending"),
    ]
    assert str(events[1].error) == "owned VLA did not exit"
    assert "owned VLA did not exit" in caplog.text


@pytest.mark.parametrize("phase", ["spawn", "wait"])
def test_startup_cleanup_preserves_primary_error(monkeypatch, phase):
    stopped, events = [], []
    daemons = _daemons(stopped)
    primary = ConnectionError("initial startup failure")

    def fail(*args, **kwargs):
        raise primary

    sink = SimpleNamespace(emit=events.append)
    with pytest.raises(RuntimeError, match="initial startup failure") as raised:
        if phase == "spawn":
            try_spawn_server(daemons, sink, "new", fail)
        else:
            monkeypatch.setattr("reweave.robots.runtime.wait_for_ready", fail)
            try_wait_server(daemons, sink, "new", object(), None, 1)
    assert raised.value.__cause__ is primary
    assert stopped == ["sam", "vla", "env"]
    assert events[-1].error is primary
    assert (
        next(event for event in events if event.component == "vla").status == "failed"
    )
