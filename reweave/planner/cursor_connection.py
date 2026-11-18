"""Cursor CLI routing. Subscription state stays in the private reweave directory."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

_API_KEY_ENVIRONMENT = frozenset({"CURSOR_API_KEY", "CURSOR_AUTH_TOKEN"})
_EFFORTS = ("xhigh", "high", "medium", "low")
DEFAULT_CURSOR_MODEL = "grok-4.7-xhigh"


def resolve_cursor_executable(name: str, environment: dict) -> str:
    """Find the CLI on PATH, or the private install under the account home."""
    found = shutil.which(name, path=environment.get("PATH"))
    if found is None:
        candidate = Path(name)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found = str(candidate)
    if found is None:
        candidate = Path(environment.get("HOME", "")) / ".local/bin" / Path(name).name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found = str(candidate)
    if found is None:
        raise ValueError("Installed Cursor CLI was not found")
    return str(Path(found).resolve())


def cursor_private_directory(home: Path) -> Path:
    """Owner-only directory. Other accounts on a shared machine must not enter it."""
    return Path(home) / ".config" / "reweave" / "cursor"


def cursor_auth_file(home: Path) -> Path:
    """Linux file login. Darwin keeps the same account in the login keychain."""
    return cursor_private_directory(home) / "auth.json"


def prepare_cursor_private_directory(home: Path, *, require_auth: bool) -> Path:
    """Create the private directory and reject group or world access."""
    reweave = Path(home) / ".config" / "reweave"
    private = reweave / "cursor"
    reweave.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in (reweave, private):
        if path.is_symlink():
            raise ValueError("Cursor subscription directory must not be a symlink")
        os.chmod(path, 0o700)
        info = path.stat()
        if info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ValueError(
                "Cursor subscription directory must be private to this user"
            )
    auth = private / "auth.json"
    if auth.exists() or auth.is_symlink():
        if auth.is_symlink() or not auth.is_file():
            raise ValueError("Cursor subscription credentials must be a regular file")
        os.chmod(auth, 0o600)
        info = auth.stat()
        if info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ValueError(
                "Cursor subscription credentials must have private 0600 permissions"
            )
    elif require_auth:
        raise ValueError(
            "Cursor subscription login is missing from the private reweave directory"
        )
    return private


def cursor_model_spec(model: str) -> dict:
    """Map a CLI catalog id onto ACP's model plus parameter options.

    ``grok-4.7-xhigh`` is the non-fast extra-high subscription variant.
    ACP selects base model ``grok-4.7`` and then ``reasoning_effort`` / ``fast``.
    """
    if not isinstance(model, str) or not model or model != model.strip():
        raise ValueError("Cursor model must be a nonempty model name")
    fast = model.endswith("-fast")
    name = model[: -len("-fast")] if fast else model
    effort = None
    for candidate in _EFFORTS:
        suffix = "-" + candidate
        if name.endswith(suffix):
            effort = candidate
            name = name[: -len(suffix)]
            break
    if not name or name.endswith("-"):
        raise ValueError("Cursor model must name a model")
    if name == "grok-4.7" and effort is None and model != "grok-4.7":
        raise ValueError("Unsupported Grok 4.7 effort")
    if name.startswith("grok-4.7-"):
        raise ValueError("Unsupported Grok 4.7 effort")
    return {
        "catalog_id": model,
        "model": name,
        "reasoning_effort": effort,
        "fast": fast,
        "context": "256k" if name == "grok-4.7" else None,
    }


def cursor_environment(environment=None, *, system: str | None = None) -> dict:
    """Child environment for subscription billing inside the private directory.

    API keys are removed so the CLI cannot bill a shared key. On Linux the
    login file is ``~/.config/reweave/cursor/auth.json``. On macOS the account
    remains in the login keychain, and CLI state still uses that private directory.
    """
    origin = dict(os.environ if environment is None else environment)
    home = Path(origin.get("HOME") or Path.home())
    system_name = platform.system() if system is None else system
    private = prepare_cursor_private_directory(
        home, require_auth=system_name != "Darwin"
    )
    environment = {
        key: value for key, value in origin.items() if key not in _API_KEY_ENVIRONMENT
    }
    environment["CURSOR_CONFIG_DIR"] = str(private)
    if system_name != "Darwin":
        # The CLI reads $XDG_CONFIG_HOME/cursor/auth.json. Point that at reweave/.
        environment["XDG_CONFIG_HOME"] = str(private.parent)
    return environment
