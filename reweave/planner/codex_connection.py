# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Native Codex connection routing; credentials stay in the local process."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import openai_codex

from reweave.evolution.authoring.workspace import native_author_launcher


def claude_executable() -> str:
    """Resolve an explicit or source-local Claude runtime before the SDK bundle."""
    source = Path(
        os.environ.get("REWEAVE_REPO_ROOT", Path(__file__).resolve().parents[2])
    )
    explicit = os.environ.get("REWEAVE_CLAUDE_BIN")
    local = source / ".native/claude"
    if explicit or local.is_file():
        executable = Path(explicit) if explicit else local
        return str(executable.resolve(strict=True))
    spec = importlib.util.find_spec("claude_agent_sdk")
    if spec is not None and spec.origin:
        bundled = Path(spec.origin).parent / "_bundled/claude"
        if bundled.is_file():
            return str(bundled.resolve())
    executable = shutil.which("claude")
    if executable is None:
        raise ValueError("Claude Code executable is unavailable")
    return str(Path(executable).resolve())


_ROUTING_ENVIRONMENT_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_API_KEY",
        "CODEX_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CURSOR_API_KEY",
        "CURSOR_AUTH_TOKEN",
    }
)
_ANTHROPIC_ENVIRONMENT_KEYS = {
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
}


def default_codex_auth_mode() -> str:
    mode = os.environ.get("REWEAVE_CODEX_AUTH_MODE", "auth")
    codex_connection_config(mode)
    return mode


def codex_connection_config(codex_auth_mode: str = "auth") -> dict:
    """Resolve nonsecret settings for the selected Codex transport."""
    if codex_auth_mode == "auth":
        return {
            "model_provider": "openai",
            "model_providers.openai.supports_websockets": True,
            "forced_login_method": "chatgpt",
            "cli_auth_credentials_store": "file",
            "features.unbounded_connection_retries": False,
        }
    if codex_auth_mode == "api":
        return {
            "model_provider": "responses_api",
            "forced_login_method": "api",
            "model_providers.responses_api.name": "Responses API",
            "model_providers.responses_api.base_url": os.environ.get(
                "REWEAVE_CODEX_API_BASE_URL", "https://api.openai.com/v1"
            ),
            "model_providers.responses_api.env_key": "OPENAI_API_KEY",
            "model_providers.responses_api.wire_api": "responses",
            "model_providers.responses_api.requires_openai_auth": False,
            "model_providers.responses_api.supports_websockets": False,
            "features.unbounded_connection_retries": False,
            "shell_environment_policy.exclude": ["OPENAI_API_KEY"],
        }
    raise ValueError("Codex auth mode must be auth or api")


def codex_connection_overrides(codex_auth_mode: str = "auth") -> tuple[str, ...]:
    """Serialize the one selected provider configuration without credentials."""
    return tuple(
        f"{key}={json.dumps(value)}"
        for key, value in codex_connection_config(codex_auth_mode).items()
    )


def codex_environment(
    native_home: Path,
    *,
    codex_auth_mode: str = "auth",
    resume: bool,
    environment=None,
):
    """Pass the selected credential to the native process."""
    codex_connection_config(codex_auth_mode)
    origin = dict(os.environ if environment is None else environment)
    if codex_auth_mode == "auth":
        return codex_proxy_environment(
            codex_account_environment(native_home, resume=resume, environment=origin)
        )
    key = origin.get("OPENAI_API_KEY", "").strip()
    if not key or any(character.isspace() for character in key):
        raise ValueError("Codex API mode requires OPENAI_API_KEY")
    return {
        **{
            name: value
            for name, value in origin.items()
            if name not in _ROUTING_ENVIRONMENT_KEYS
        },
        "OPENAI_API_KEY": key,
    }


def claude_environment() -> dict[str, str]:
    """Forward Anthropic configuration without exposing other provider credentials."""
    return {
        **dict.fromkeys(_ROUTING_ENVIRONMENT_KEYS - _ANTHROPIC_ENVIRONMENT_KEYS, ""),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_DISABLE_THINKING": "1",
        "MAX_THINKING_TOKENS": "0",
    }


def codex_proxy_environment(environment):
    """Carry an enabled macOS loopback proxy into native WebSocket connections.

    The native HTTPS fallback can use the system proxy while WebSocket setup
    repeatedly resets without explicit proxy variables. Respect any explicit
    process proxy allocation; never change provider, authentication or TLS.
    """
    result = dict(environment)
    if platform.system() != "Darwin" or any(
        key.lower() in {"http_proxy", "https_proxy", "all_proxy"} and value
        for key, value in result.items()
    ):
        return result
    try:
        proxy = subprocess.run(
            ["/usr/sbin/scutil", "--proxy"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return result
    fields = dict(re.findall(r"^\s*(HTTP\w*)\s*:\s*(\S+)\s*$", proxy, re.MULTILINE))
    for scheme in ("HTTP", "HTTPS"):
        host, port = fields.get(scheme + "Proxy"), fields.get(scheme + "Port", "")
        if (
            fields.get(scheme + "Enable") == "1"
            and host in {"127.0.0.1", "localhost", "::1"}
            and port.isdecimal()
            and 0 < int(port) <= 65535
        ):
            host = f"[{host}]" if host == "::1" else host
            result[scheme + "_PROXY"] = f"http://{host}:{port}"
    if "HTTPS_PROXY" in result:
        result["ALL_PROXY"] = result["HTTPS_PROXY"]
    if "HTTP_PROXY" in result or "HTTPS_PROXY" in result:
        bypass = dict.fromkeys(
            value.strip()
            for key in ("NO_PROXY", "no_proxy")
            for value in result.get(key, "").split(",")
            if value.strip()
        )
        bypass.update(dict.fromkeys(("localhost", "127.0.0.1", "::1")))
        result["NO_PROXY"] = result["no_proxy"] = ",".join(bypass)
    return result


def codex_model_config(model: str | None) -> dict[str, int]:
    """One 500k context cap for every isolated online and offline Codex path."""
    return {"model_context_window": 500_000}


def codex_executable(explicit: str | None = None) -> str:
    """Resolve the installed CLI, never an independently bundled SDK binary."""
    path = shutil.which(explicit or os.environ.get("CODEX_BIN") or "codex")
    if path is None:
        raise ValueError(
            "Installed Codex executable is missing; install Codex and run codex login"
        )
    executable = Path(path).resolve(strict=True)
    if executable.name != "codex.js":
        return str(executable)
    # The installed npm entrypoint only selects and spawns its native binary.
    # Resolve that same installation before entering the OS boundary, so Node
    # and unrelated package-manager directories need no additional read grants.
    package = json.loads((executable.parent.parent / "package.json").read_text())
    if package.get("name") != "@openai/codex":
        raise ValueError("Codex JavaScript entrypoint is not the installed npm package")
    target = {
        ("Darwin", "arm64"): ("darwin-arm64", "aarch64-apple-darwin"),
        ("Darwin", "x86_64"): ("darwin-x64", "x86_64-apple-darwin"),
        ("Linux", "aarch64"): ("linux-arm64", "aarch64-unknown-linux-musl"),
        ("Linux", "x86_64"): ("linux-x64", "x86_64-unknown-linux-musl"),
    }.get((platform.system(), platform.machine()))
    if target is None:
        raise ValueError(
            "Unsupported installed Codex platform; specify a native binary"
        )
    suffix, triple = target
    dependency = f"@openai/codex-{suffix}"
    for parent in executable.parent.parents:
        directory = parent / "node_modules" / dependency
        binary = directory / "vendor" / triple / "bin/codex"
        if not binary.is_file():
            continue
        installed = json.loads((directory / "package.json").read_text())
        expected = f"npm:@openai/codex@{installed['version']}"
        if package.get("optionalDependencies", {}).get(dependency) != expected:
            raise ValueError("Installed Codex native dependency version does not match")
        if not os.access(binary, os.X_OK):
            raise ValueError("Installed Codex native binary is not executable")
        return str(binary.resolve(strict=True))
    raise ValueError("Installed Codex native dependency is missing; reinstall Codex")


def codex_account_environment(native_home: Path, *, resume: bool, environment=None):
    """Keep account credentials in local native state and exclude API routing."""
    origin = dict(os.environ if environment is None else environment)
    auth = native_home / "auth.json"
    if not auth.exists():
        if resume:
            raise ValueError(
                "Native account state missing; refusing a replacement login"
            )
        source = (
            Path(origin.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
        )
        raw = source.read_bytes()
        if json.loads(raw).get("auth_mode") != "chatgpt":
            raise ValueError(
                "Local Codex requires an existing ChatGPT account login, not an API key"
            )
        descriptor = os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
    if json.loads(auth.read_bytes()).get("auth_mode") != "chatgpt":
        raise ValueError("Native author state does not contain a ChatGPT account login")
    return {
        key: value
        for key, value in origin.items()
        if key not in _ROUTING_ENVIRONMENT_KEYS
    }


def create_codex_client(
    *,
    workspace: Path,
    output: Path,
    control: Path,
    mcp_servers: dict[str, str],
    model: str | None = None,
    readable_roots: tuple[Path, ...] | None = None,
    writable_roots: tuple[Path, ...] | None = None,
    environment: dict[str, str] | None = None,
    native_runtime_root: Path | None = None,
    resume: bool = False,
    codex_auth_mode: str = "auth",
) -> openai_codex.AsyncCodex:
    """Use the selected native connection; model and effort belong to author policy."""
    env = dict(os.environ)
    if environment is not None:
        env.update(environment)
    bypass = dict.fromkeys(
        host.strip()
        for value in (
            env.get("NO_PROXY", ""),
            env.get("no_proxy", ""),
            "127.0.0.1,localhost",
        )
        for host in value.split(",")
        if host.strip()
    )
    env["NO_PROXY"] = env["no_proxy"] = ",".join(bypass)
    overrides: list[str] = []
    for name, url in mcp_servers.items():
        if urlsplit(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("author MCP tools must be served by the local Host")
        overrides.extend(
            (
                f"mcp_servers.{name}.url={json.dumps(url)}",
                f'mcp_servers.{name}.default_tools_approval_mode="approve"',
                f"mcp_servers.{name}.required=true",
            )
        )
    overrides.extend(codex_connection_overrides(codex_auth_mode))
    overrides.extend(
        f"{key}={value}" for key, value in codex_model_config(model).items()
    )
    overrides.extend(('service_tier="default"', "features.fast_mode=false"))
    executable = codex_executable(env.get("CODEX_BIN"))
    launcher, runtime_env = native_author_launcher(
        workspace,
        output,
        control,
        backend="codex",
        cli=executable,
        readable_roots=readable_roots,
        writable_roots=writable_roots,
        native_runtime_root=native_runtime_root,
    )
    env = codex_environment(
        Path(runtime_env["CODEX_HOME"]),
        codex_auth_mode=codex_auth_mode,
        resume=resume,
        environment=env,
    )
    env.update(runtime_env)
    return openai_codex.AsyncCodex(
        openai_codex.CodexConfig(
            cwd=str(workspace.resolve()),
            env=env,
            config_overrides=tuple(overrides),
            codex_bin=launcher,
        )
    )
