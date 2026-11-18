# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Native Codex account authentication, installed CLI and model-policy boundaries."""

import json
import sys
from pathlib import Path

import pytest

from reweave.evolution.authoring.config import AuthorPolicy
from reweave.planner import codex_connection


@pytest.fixture
def config_capture(tmp_path, monkeypatch):
    account = tmp_path / "login"
    account.mkdir()
    (account / "auth.json").write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "test-secret"}})
    )
    monkeypatch.setenv("CODEX_HOME", str(account))
    monkeypatch.setenv("CODEX_BIN", sys.executable)
    monkeypatch.setattr(
        codex_connection.openai_codex, "AsyncCodex", lambda config: config
    )
    return account


def test_codex_uses_account_and_explicit_local_mcp_grants(
    tmp_path, monkeypatch, config_capture
):
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "CODEX_BASE_URL",
        "CODEX_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(key, "must-not-reach-author")
    config = codex_connection.create_codex_client(
        workspace=tmp_path,
        output=tmp_path / "output",
        control=tmp_path / "control",
        mcp_servers={"programs": "http://127.0.0.1:12345/mcp/"},
    )
    assert config.config_overrides[:3] == (
        'mcp_servers.programs.url="http://127.0.0.1:12345/mcp/"',
        'mcp_servers.programs.default_tools_approval_mode="approve"',
        "mcp_servers.programs.required=true",
    )
    assert config.config_overrides[3:] == (
        *codex_connection.codex_connection_overrides("auth"),
        "model_context_window=500000",
        'service_tier="default"',
        "features.fast_mode=false",
    )
    assert "must-not-reach-author" not in str(config.env)
    assert {"localhost", "127.0.0.1"} <= set(config.env["NO_PROXY"].split(","))
    auth = tmp_path / ".scratch/runtime/codex/auth.json"
    assert auth.read_bytes() == (config_capture / "auth.json").read_bytes()
    assert auth.stat().st_mode & 0o777 == 0o600
    assert config.cwd == str(tmp_path.resolve())


def test_account_login_enables_native_response_continuation():
    mode, provider = "auth", "openai"
    configuration = codex_connection.codex_connection_config(mode)
    assert configuration[f"model_providers.{provider}.supports_websockets"] is True
    overrides = codex_connection.codex_connection_overrides(mode)
    assert f"model_providers.{provider}.supports_websockets=true" in overrides
    # Both online stdio and offline authors consume these same overrides.
    assert configuration["features.unbounded_connection_retries"] is False


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_api_login_is_rejected(tmp_path, config_capture, missing):
    source = config_capture / "auth.json"
    if missing:
        source.unlink()
    else:
        source.write_text(
            json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "test-key"})
        )
    with pytest.raises((FileNotFoundError, ValueError)):
        codex_connection.create_codex_client(
            workspace=tmp_path,
            output=tmp_path / "output",
            control=tmp_path / "control",
            mcp_servers={},
        )


def test_resume_requires_original_account_state(tmp_path, config_capture):
    arguments = {
        "workspace": tmp_path,
        "output": tmp_path / "output",
        "control": tmp_path / "control",
        "mcp_servers": {},
    }
    codex_connection.create_codex_client(**arguments)
    auth = tmp_path / ".scratch/runtime/codex/auth.json"
    auth.unlink()
    with pytest.raises(ValueError, match="refusing a replacement login"):
        codex_connection.create_codex_client(**arguments, resume=True)
    assert not auth.exists()


def test_installed_cli_is_resolved_and_missing_cli_never_uses_sdk_binary(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("CODEX_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="Installed Codex executable is missing"):
        codex_connection.codex_executable()
    assert codex_connection.codex_executable(sys.executable) == str(
        Path(sys.executable).resolve()
    )
    seen = []
    monkeypatch.setattr(
        codex_connection.shutil,
        "which",
        lambda name: seen.append(name) or sys.executable,
    )
    codex_connection.codex_executable()
    assert seen == ["codex"]


def test_npm_cli_resolves_matching_native_installation_without_node(
    tmp_path, monkeypatch
):
    package = tmp_path / "node_modules/@openai/codex"
    entry = package / "bin/codex.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("#!/usr/bin/env node\n")
    entry.chmod(0o700)
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "@openai/codex",
                "optionalDependencies": {
                    "@openai/codex-darwin-arm64": "npm:@openai/codex@0.160.0-darwin-arm64"
                },
            }
        )
    )
    dependency = package / "node_modules/@openai/codex-darwin-arm64"
    binary = dependency / "vendor/aarch64-apple-darwin/bin/codex"
    binary.parent.mkdir(parents=True)
    binary.write_text("native executable fixture")
    binary.chmod(0o700)
    (dependency / "package.json").write_text(
        json.dumps({"version": "0.160.0-darwin-arm64"})
    )
    monkeypatch.setattr(codex_connection.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(codex_connection.platform, "machine", lambda: "arm64")
    assert codex_connection.codex_executable(str(entry)) == str(binary.resolve())
    (dependency / "package.json").write_text(
        json.dumps({"version": "0.147.0-darwin-arm64"})
    )
    with pytest.raises(ValueError, match="version does not match"):
        codex_connection.codex_executable(str(entry))
    binary.unlink()
    with pytest.raises(ValueError, match="native dependency is missing"):
        codex_connection.codex_executable(str(entry))


def test_selected_author_owns_model_and_reasoning_defaults(config_capture):
    claude = AuthorPolicy()
    codex = AuthorPolicy(author="codex")
    assert (claude.model, claude.reasoning_effort) == ("claude-sonnet-4-6", "xhigh")
    assert (codex.model, codex.reasoning_effort) == ("gpt-5.4", "xhigh")
    for backend in ("codex", "claude_code"):
        with pytest.raises(ValueError, match="reasoning_effort"):
            AuthorPolicy(author=backend, reasoning_effort=None)


def test_remote_mcp_server_cannot_receive_local_author_grants(tmp_path, config_capture):
    with pytest.raises(ValueError, match="local Host"):
        codex_connection.create_codex_client(
            workspace=tmp_path,
            output=tmp_path / "output",
            control=tmp_path / "control",
            mcp_servers={"programs": "https://remote.invalid/mcp"},
        )


def test_codex_native_state_and_launcher_are_branch_owned(tmp_path, config_capture):
    config = codex_connection.create_codex_client(
        workspace=tmp_path,
        output=tmp_path / "output",
        control=tmp_path / "control",
        mcp_servers={},
    )
    assert config.env["CODEX_HOME"] == str(tmp_path / ".scratch/runtime/codex")
    assert config.env["TMPDIR"] == str(tmp_path / ".scratch/runtime/tmp")
    assert config.codex_bin == str(tmp_path / "control/codex")
    boundary = json.loads((tmp_path / "control/workspace.json").read_text())
    assert str(tmp_path / "output") in boundary["writable"]
    assert str(tmp_path / "control") not in boundary["writable"]


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.4", None])
def test_isolated_author_context_policy_does_not_depend_on_global_config(
    tmp_path, config_capture, model
):
    config = codex_connection.create_codex_client(
        workspace=tmp_path,
        output=tmp_path / "output",
        control=tmp_path / "control",
        mcp_servers={},
        model=model,
    )
    assert "model_context_window=500000" in config.config_overrides
    assert not any(
        "model_context_window=256000" in value for value in config.config_overrides
    )


@pytest.mark.parametrize(
    "model,window",
    [("gpt-5.4", 500_000), ("gpt-5.4", 500_000), (None, 500_000)],
)
def test_explicit_online_astra_and_offline_sol_windows(model, window):
    value = codex_connection.codex_model_config(model)
    assert value == ({"model_context_window": window} if window is not None else {})


@pytest.mark.parametrize(
    "existing",
    [{}, {"HTTPS_PROXY": "http://chosen:80"}, {"all_proxy": "socks5://chosen:90"}],
)
def test_native_websocket_uses_enabled_system_proxy_without_overriding_explicit_route(
    monkeypatch, existing
):
    from types import SimpleNamespace

    monkeypatch.setattr(codex_connection.platform, "system", lambda: "Darwin")
    calls = []

    def system_proxy(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            stdout="""<dictionary> {
  HTTPEnable : 1
  HTTPProxy : 127.0.0.1
  HTTPPort : 1082
  HTTPSEnable : 1
  HTTPSProxy : 127.0.0.1
  HTTPSPort : 1082
}"""
        )

    monkeypatch.setattr(codex_connection.subprocess, "run", system_proxy)
    source = {
        **existing,
        "NO_PROXY": "localhost,127.0.0.1",
        "MODEL_UNRELATED": "retained",
    }
    result = codex_connection.codex_proxy_environment(source)
    assert set(source["NO_PROXY"].split(",")) <= set(result["NO_PROXY"].split(","))
    assert result["MODEL_UNRELATED"] == "retained"
    if existing:
        assert result == source and calls == []
    else:
        assert (
            result["HTTP_PROXY"]
            == result["HTTPS_PROXY"]
            == result["ALL_PROXY"]
            == "http://127.0.0.1:1082"
        )
        assert len(calls) == 1 and "HTTP_PROXY" not in source


@pytest.mark.parametrize(
    "settings",
    [
        "",
        "HTTPSProxy : remote-host\nHTTPSPort : 1082\nHTTPSEnable : 1",
        "HTTPSProxy : localhost\nHTTPSPort : 1082\nHTTPSEnable : 0",
    ],
)
def test_native_proxy_does_not_enable_disabled_or_unverified_system_routes(
    monkeypatch, settings
):
    from types import SimpleNamespace

    monkeypatch.setattr(codex_connection.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        codex_connection.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=settings),
    )
    assert codex_connection.codex_proxy_environment({"NO_PROXY": "localhost"}) == {
        "NO_PROXY": "localhost"
    }
