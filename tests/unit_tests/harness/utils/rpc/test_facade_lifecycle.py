# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""RPC startup is atomic and shutdown releases requests not yet dispatched."""

import queue
import threading

import pytest

from reweave.utils.rpc.rpc_facade import RpcFacade
from reweave.utils.rpc.main_thread_serve import MainThreadServeMixin


class MainThreadFacade(MainThreadServeMixin, RpcFacade):
    pass


@pytest.mark.parametrize("facade_type", [RpcFacade, MainThreadFacade])
@pytest.mark.parametrize("transport", ["http", "socket"])
def test_invalid_session_sweep_does_not_bind_a_listener(facade_type, transport):
    facade = facade_type(enable_sessions=True)
    servers = []
    bind = facade._bind_and_announce

    def capture(*args):
        server = bind(*args)
        servers.append(server)
        return server

    facade._bind_and_announce = capture
    try:
        with pytest.raises(ValueError, match="session_sweep_s"):
            facade.serve(transport=transport, host="127.0.0.1", port=0)
        assert not servers, "invalid startup left a bound RPC socket in the traceback"
    finally:
        for server in servers:
            server.server_close()


@pytest.mark.parametrize("facade_type", [RpcFacade, MainThreadFacade])
@pytest.mark.parametrize("interval", [float("nan"), float("inf")])
def test_session_sweep_requires_a_finite_interval(facade_type, interval, monkeypatch):
    facade = facade_type(enable_sessions=True)

    def no_bind(*args):
        pytest.fail("invalid session sweep interval reached transport binding")

    monkeypatch.setattr(facade, "_bind_and_announce", no_bind)
    with pytest.raises(ValueError, match="session_sweep_s"):
        facade.serve(
            transport="http", host="127.0.0.1", port=0, session_sweep_s=interval
        )


def test_main_thread_shutdown_releases_queued_request_without_executing_it(monkeypatch):
    facade = MainThreadFacade()
    queued = threading.Event()
    outcome = []
    executed = []
    facade._rpc["mutate"] = lambda: executed.append("must not execute")

    class ObservedQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            super().put(item, *args, **kwargs)
            queued.set()

    monkeypatch.setattr(
        "reweave.utils.rpc.main_thread_serve.queue.Queue", ObservedQueue
    )

    def request():
        try:
            facade._dispatch_main_thread("mutate", (), {})
        except RuntimeError as exc:
            outcome.append(str(exc))

    requester = threading.Thread(target=request, daemon=True)

    class Server:
        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    def bind(*args):
        requester.start()
        assert queued.wait(2), "transport did not enqueue its request"
        facade._shutdown_event.set()
        return Server()

    monkeypatch.setattr(facade, "_bind_and_announce", bind)
    facade.serve(transport="http", host="127.0.0.1", port=0)
    requester.join(2)
    assert not requester.is_alive(), "shutdown stranded the RPC transport thread"
    assert executed == []
    assert len(outcome) == 1 and "shut down" in outcome[0]
    with pytest.raises(RuntimeError, match="shut down"):
        facade._dispatch_main_thread("mutate", (), {})


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_dispatch_releases_its_current_request(monkeypatch, interruption):
    facade = MainThreadFacade()
    queued = threading.Event()
    outcome, closed = [], []

    def interrupt():
        raise interruption("dispatch interrupted")

    facade._rpc["mutate"] = interrupt

    class ObservedQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            super().put(item, *args, **kwargs)
            queued.set()

    monkeypatch.setattr(
        "reweave.utils.rpc.main_thread_serve.queue.Queue", ObservedQueue
    )
    monkeypatch.setattr(facade, "close", lambda: closed.append(True))

    def request():
        try:
            facade._dispatch_main_thread("mutate", (), {})
        except RuntimeError as exc:
            outcome.append(str(exc))

    requester = threading.Thread(target=request, daemon=True)

    class Server:
        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    def bind(*args):
        requester.start()
        assert queued.wait(2)
        return Server()

    monkeypatch.setattr(facade, "_bind_and_announce", bind)
    with pytest.raises(interruption, match="dispatch interrupted"):
        facade.serve(transport="http", host="127.0.0.1", port=0)
    requester.join(2)
    assert not requester.is_alive(), "interrupted dispatch stranded its transport"
    assert len(outcome) == 1 and "dispatch interrupted" in outcome[0]
    assert closed == [True]
