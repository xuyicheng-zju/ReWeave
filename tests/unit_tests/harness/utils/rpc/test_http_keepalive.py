"""Direct RPC socket reuse preserves independent dispatch and receipt recovery."""

import concurrent.futures
import socket
import threading
from contextlib import contextmanager

from reweave.utils.rpc.http_rpc import HttpRpcClient, HttpRpcServer


@contextmanager
def running_service(dispatch):
    class Server(HttpRpcServer):
        connections = 0

        def get_request(self):
            result = super().get_request()
            self.connections += 1
            return result

    server = Server(("127.0.0.1", 0), dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpRpcClient(f"http://127.0.0.1:{server.server_port}")
    try:
        yield server, client
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_sequential_mutations_reuse_socket_and_preserve_every_result():
    calls = []
    with running_service(
        lambda method, args, kwargs: calls.append(args[0]) or args[0]
    ) as (server, client):
        for value in range(20):
            assert client.call("mutate", args=(value,)) == value
        assert calls == list(range(20))
        assert server.connections == 1
        assert client.last_call_metrics.recovery_queries == 0
        client.close()
        assert client._opener._idle is None


def test_idle_peer_close_is_discarded_before_next_mutation():
    calls = []
    with running_service(
        lambda method, args, kwargs: calls.append(method) or len(calls)
    ) as (server, client):
        assert client.call("first") == 1
        # A readable idle socket (EOF) must not receive the next mutation.
        client._opener._idle.sock.shutdown(socket.SHUT_RDWR)
        assert client.call("second") == 2
        assert calls == ["first", "second"]
        assert server.connections == 2
        assert client.last_call_metrics.recovery_queries == 0


def test_concurrent_cancel_does_not_wait_for_inflight_action():
    started = threading.Event()
    cancelled = threading.Event()

    def dispatch(method, args, kwargs):
        if method == "action":
            started.set()
            assert cancelled.wait(2)
            return "settled"
        if method == "cancel":
            cancelled.set()
        return "ok"

    with (
        running_service(dispatch) as (server, client),
        concurrent.futures.ThreadPoolExecutor() as pool,
    ):
        client.call("healthz")
        future = pool.submit(client.call, "action")
        assert started.wait(1)
        assert client.call("cancel") == "ok"
        assert future.result(timeout=2) == "settled"
        assert server.connections == 2


def test_long_idle_tunnel_socket_is_retired_before_sending_mutation(monkeypatch):
    import time

    from reweave.utils.rpc import http_rpc

    calls = []
    with running_service(
        lambda method, args, kwargs: calls.append(method) or len(calls)
    ) as (server, client):
        assert client.call("first") == 1
        idle = client._opener._idle
        client._opener._idle_since = time.monotonic() - 30
        # EOF is not yet visible through the tunnel. The old transport would
        # write the second mutation onto this socket based only on readiness.
        monkeypatch.setattr(http_rpc.select, "select", lambda *args: ([], [], []))

        def stale_write(*args, **kwargs):
            raise AssertionError("mutation submitted on stale tunnel socket")

        monkeypatch.setattr(idle, "request", stale_write)
        assert client.call("second") == 2
        assert calls == ["first", "second"]
        assert server.connections == 2
        assert client.last_call_metrics.recovery_queries == 0
