"""Source-local placement for new online runtimes; resume locators win."""

from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path

from reweave.utils.persistence import atomic_json

from .base import NativeProtocolError


def online_runtime_home(directory: Path, *, source_root: Path) -> tuple[Path, bool]:
    """Read optional .reweave-online-runtime.json only when allocating a runtime.

    The marker contains exactly {"runtime_root": "/absolute/private/directory"}.
    It is local deployment configuration, never copied into episode evidence.
    """
    locator = directory / "runtime_home.json"
    if locator.exists():
        native_home = Path(json.loads(locator.read_text())["path"])
        if not native_home.is_absolute() or not native_home.is_dir():
            raise NativeProtocolError("Original online native runtime is missing")
        return native_home, True

    marker = source_root / ".reweave-online-runtime.json"
    runtime_root = Path.home() / ".codex/harness-online"
    if marker.exists():
        try:
            config = json.loads(marker.read_text())
            if not isinstance(config, dict) or set(config) != {"runtime_root"}:
                raise ValueError("Expected runtime_root only")
            value = config["runtime_root"]
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError("Expected absolute path")
            runtime_root = Path(value)
            if not runtime_root.is_absolute() or runtime_root.resolve() != runtime_root:
                raise ValueError("Expected canonical absolute path")
            if runtime_root.is_relative_to(directory.resolve()):
                raise ValueError("Runtime cannot be inside evidence")
            runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            info = runtime_root.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ValueError(
                    "Runtime root must be owned by this user with mode 0700"
                )
        except (OSError, ValueError, TypeError) as exc:
            raise NativeProtocolError(
                "Invalid source-local online runtime configuration"
            ) from exc
    else:
        runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    session_root = runtime_root / uuid.uuid4().hex
    session_root.mkdir(mode=0o700)
    native_home = session_root / "codex"
    native_home.mkdir(mode=0o700)
    atomic_json(locator, {"path": str(native_home)})
    return native_home, False
