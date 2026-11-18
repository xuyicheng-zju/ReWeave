import json

from reweave.evolution.infrastructure.native_retry import is_retryable_native_error
from reweave.utils.provider_admission import (
    ProviderAdmission,
    is_quota_exhausted,
    is_rate_limit,
)


def test_sharing_quota_is_not_transient_rate_limiting(tmp_path):
    message = "stream disconnected before completion: The ChatGPT user has reached their Subscription Sharing usage limit."
    error = {"message": message, "codex_error_info": "other"}
    assert is_quota_exhausted(error)
    assert not is_rate_limit(error)
    assert not is_retryable_native_error(RuntimeError("HTTP 429: " + message))
    assert not is_quota_exhausted({"used_percent": 100})
    assert is_rate_limit({"message": "429 Too Many Requests"})
    gate = ProviderAdmission("codex-api", root=tmp_path, capacity=48)
    lease, _ = gate._acquire()
    assert lease is not None
    assert gate.note_quota_exhausted(error)
    gate._release(lease, succeeded=True)
    assert gate._acquire()[0] is None
    state = json.loads((gate.root / "state.json").read_text())
    assert state["quota_blocked"] is True
    assert state["probe_required"] is True
    assert state["capacity"] == 48


def test_hard_quota_can_be_recorded_through_shared_rpc(tmp_path, monkeypatch):
    import threading

    from reweave.utils.provider_admission_server import AdmissionServer

    gate = ProviderAdmission("codex-api", root=tmp_path, capacity=48)
    token = tmp_path / "token"
    token.write_text("test-only")
    with AdmissionServer(("127.0.0.1", 0), gate, "test-only") as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            monkeypatch.setenv(
                "REWEAVE_PROVIDER_ADMISSION_ENDPOINT",
                f"127.0.0.1:{server.server_address[1]}",
            )
            monkeypatch.setenv("REWEAVE_PROVIDER_ADMISSION_TOKEN_FILE", str(token))
            client = ProviderAdmission("codex-api")
            assert client.note_quota_exhausted({"code": "usage_limit_reached"})
            assert gate._acquire()[0] is None
        finally:
            server.shutdown()
            thread.join()
