"""Exact PNG bytes cross HTTP without raw pixel expansion or ambiguous tags."""

import json
import threading

import numpy as np
import pytest

from reweave.utils.rpc.http_rpc import (
    HttpRpcClient,
    HttpRpcServer,
    _from_json,
    _NumpyEncoder,
)
from reweave.utils.rpc.rpc_client import RpcError


@pytest.mark.parametrize(
    "value", [b"", b"\x00\xffPNG\r\n", {"camera": b"pixels", "state": np.arange(3)}]
)
def test_bytes_codec_round_trip(value):
    restored = _from_json(json.loads(json.dumps(value, cls=_NumpyEncoder)))
    if isinstance(value, dict):
        assert restored["camera"] == value["camera"]
        np.testing.assert_equal(restored["state"], value["state"])
    else:
        assert type(restored) is bytes and restored == value


@pytest.mark.parametrize(
    "payload",
    [
        {"__bytes__": "not valid b64"},
        {"__bytes__": 123},
        {"__bytes__": "YQ==", "extra": 1},
        {"__bytes__": "YQ==", "__ndarray__": "YQ=="},
    ],
)
def test_malformed_and_ambiguous_tags_are_rejected(payload):
    with pytest.raises(ValueError, match="RPC bytes tag"):
        _from_json(payload)


def test_http_bytes_response_and_request_and_no_ambiguous_dispatch():
    requests = []

    def dispatch(method, args, kwargs):
        requests.append((method, args, kwargs))
        return {"cam_head": args[0]}

    server = HttpRpcServer(("127.0.0.1", 0), dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HttpRpcClient(f"http://127.0.0.1:{server.server_address[1]}")
        payload = bytes(range(256)) * 1024
        assert client.call("read_rgb", args=(payload,)) == {"cam_head": payload}
        assert len(requests) == 1
        with pytest.raises(RpcError, match="ambiguous RPC bytes tag"):
            client.call("step", args=({"__bytes__": "YQ==", "extra": 1},))
        assert len(requests) == 1, (
            "invalid request must not dispatch or retry an action"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
