import sys

import pytest

from reweave.utils import provider_admission_server as module


@pytest.mark.parametrize("explicit", [False, True])
def test_gate_cli_preserves_configured_capacity(tmp_path, monkeypatch, explicit):
    token = tmp_path / "token"
    token.write_text("test-only")
    argv = [
        "gate",
        "--root",
        str(tmp_path / "pool"),
        "--token-file",
        str(token),
        "--port",
        "18732",
    ]
    if explicit:
        argv += ["--capacity", "48"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("REWEAVE_NATIVE_MODEL_CONCURRENCY", "48")
    monkeypatch.setattr(module.os, "umask", lambda value: 0o077)
    observed = []

    class Server:
        def __init__(self, address, gate, token):
            observed.append(gate.capacity)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def serve_forever(self):
            pass

    monkeypatch.setattr(module, "AdmissionServer", Server)
    module.main()
    assert observed == [48]
