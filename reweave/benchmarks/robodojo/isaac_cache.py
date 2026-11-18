"""Remove one owner's disposable cache only after its process group closes."""

import os
import shutil
from pathlib import Path


def clear_closed_cache(cache_root, identity, state):
    if (
        len(identity) != 32
        or any(c not in "0123456789abcdef" for c in identity)
        or state.get("service_instance_id") != identity
        or state.get("status") != "exited"
        or state.get("exit", {}).get("parent_closed") is not True
    ):
        raise ValueError("Cache cleanup requires its exact closed owner")
    pid = state["exit"].get("pid")
    if type(pid) is not int or pid <= 1:
        return {"status": "deferred", "reason": "environment PID unavailable"}
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        pass
    except PermissionError:
        return {"status": "deferred", "reason": "process group cannot be inspected"}
    else:
        return {"status": "deferred", "reason": "process group remains alive"}
    root = Path(cache_root)
    target = root / identity
    if root.is_symlink() or target.is_symlink():
        raise ValueError("Episode cache cannot be a symbolic link")
    if target.exists():
        shutil.rmtree(target)
    return {"status": "removed", "service_instance_id": identity, "path": str(target)}
