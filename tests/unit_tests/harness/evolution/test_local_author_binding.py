# Modified for the ReWeave namespace.
"""Local native sessions reject changed ownership before any provider work."""

import asyncio
import json
import sys

import pytest

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.evolution.authoring.local import LocalCodexAuthor


@pytest.mark.parametrize(
    "change",
    [
        {"model": "different-model"},
        {"codex_auth_mode": "api"},
        {"role": "author"},
    ],
)
def test_bound_local_session_rejects_configuration_changes_before_provider(
    tmp_path, change
):
    class Channel:
        async def call(self, *args, **kwargs):
            pytest.fail("Changed session binding must not contact remote tools")

    author = LocalCodexAuthor(Channel(), tmp_path, codex_bin=sys.executable)
    policy = AuthorPolicy(author="codex").model_dump(mode="json")
    original_policy = dict(policy)
    original_policy.pop("codex_auth_mode")
    directory = author._record_directory(tmp_path, "graph-session")
    directory.mkdir()
    binding = directory / "remote.json"
    original = json.dumps(
        {
            "session_key": "graph-session",
            "role": "graph_analysis",
            "policy": original_policy,
        },
        indent=2,
    ).encode()
    binding.write_bytes(original)
    request = {
        "session_key": "graph-session",
        "request_id": "graph-session",
        "role": change.get("role", "graph_analysis"),
        "policy": policy
        | {key: value for key, value in change.items() if key != "role"},
    }
    with pytest.raises(ValueError, match="session model, provider or role changed"):
        asyncio.run(author.query(request))
    assert binding.read_bytes() == original
    assert not (directory / "requests").exists()
    assert not (directory / "session").exists()
