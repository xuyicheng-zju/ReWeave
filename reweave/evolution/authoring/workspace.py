# Modified for the ReWeave namespace.
"""Filesystem boundary for the native coding CLI, not an author loop.

The CLI and every child inherit the same boundary. Inputs remain read-only;
only the current candidate and scratch are writable. Linux uses Landlock
without namespaces (also works on our development container); macOS uses
Seatbelt. There is no unsandboxed fallback.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import re
import runpy
import stat
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Literal


def _linux_restrict(
    readable: list[str],
    writable: list[str],
    metadata_calls: tuple[str, ...],
) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)  # landlock_create_ruleset: VERSION
    if abi < 1:
        raise RuntimeError("Native author requires Linux Landlock support")
    # Library discovery can invoke system tools that are outside the read grants.
    # Load it before entering Landlock; install the filter after filesystem rules.
    library = ctypes.util.find_library("seccomp")
    if library is None:
        raise RuntimeError("Native author requires libseccomp with Landlock")
    seccomp = ctypes.CDLL(library, use_errno=True)
    # ABI 1 handles read/execute, writes, removal and creation. Later ABIs
    # additionally handle cross-directory reparenting and truncation.
    access = (1 << 13) - 1
    if abi >= 2:
        access |= 1 << 13
    if abi >= 3:
        access |= 1 << 14
    read_access = (1 << 0) | (1 << 2) | (1 << 3)

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    ruleset = Ruleset(access)
    fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for names, rights in (
            (readable, read_access),
            (writable, access),
        ):
            for name in names:
                path = Path(name)
                if not path.exists():
                    continue
                descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    # File rules cannot carry directory access rights.
                    allowed = (
                        rights
                        if path.is_dir()
                        else rights & ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 14))
                    )
                    rule = PathRule(allowed, descriptor)
                    if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                        raise OSError(ctypes.get_errno(), f"landlock_add_rule: {path}")
                finally:
                    os.close(descriptor)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if libc.syscall(446, fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)
    return _linux_protect_metadata(seccomp, abi, metadata_calls)


def _linux_protect_metadata(
    lib: ctypes.CDLL, abi: int, metadata_calls: tuple[str, ...]
) -> int:
    """Landlock does not mediate metadata changes; ABI 1/2 omit truncate."""
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_notify_fd.argtypes = [ctypes.c_void_p]
    lib.seccomp_notify_fd.restype = ctypes.c_int
    context = lib.seccomp_init(0x7FFF0000)  # ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    denied = [
        "chown",
        "fchown",
        "lchown",
        "fchownat",
        "ptrace",
        "process_vm_writev",
        "process_vm_readv",
        "pidfd_getfd",
        "mount",
        "umount2",
        "open_by_handle_at",
        "io_uring_setup",
    ]
    if abi < 3:
        denied.extend(("truncate", "truncate64"))
    try:
        for name in metadata_calls:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if (
                number >= 0
                and lib.seccomp_rule_add(context, 0x7FC00000, number, 0) != 0
            ):
                raise RuntimeError(f"seccomp metadata notification failed: {name}")
        for name in denied:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if (
                number >= 0
                and lib.seccomp_rule_add(context, 0x00050000 | 13, number, 0) != 0
            ):
                raise RuntimeError(f"seccomp rule failed: {name}")
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
        listener = lib.seccomp_notify_fd(context)
        if listener < 0:
            raise RuntimeError("Native author requires seccomp metadata notifications")
        return os.dup(listener)
    finally:
        lib.seccomp_release(context)


def launch(config: dict, arguments: list[str]) -> None:
    """Enter the OS boundary before starting the unmodified native CLI."""
    writable = list(config["writable"])
    native_config = os.environ.get("CLAUDE_CONFIG_DIR")
    if "CLAUDE_CONFIG_DIR" in config["env"] and native_config:
        if Path(native_config).resolve() != Path(config["env"]["CLAUDE_CONFIG_DIR"]):
            # SessionStore resume materializes its own directory. The SDK and
            # CLI must use that same path, including for transcript mirroring.
            resumed = _claude_resume_directory(config, arguments, native_config)
            writable.append(str(resumed))
        else:
            native_config = config["env"]["CLAUDE_CONFIG_DIR"]
    os.environ.update(config["env"])
    if "CLAUDE_CONFIG_DIR" in config["env"] and native_config:
        os.environ["CLAUDE_CONFIG_DIR"] = native_config
    system = platform.system()
    if system == "Darwin" and config.get("restrict_reads", False):
        # Apple's tool shims otherwise open a GUI installer when the isolated
        # process cannot discover the already-installed developer directory.
        # An explicit directory makes unavailable tools fail normally; it does
        # not grant access to the toolchain or change the system selection.
        developer = Path("/Library/Developer/CommandLineTools")
        if developer.is_dir():
            os.environ.setdefault("DEVELOPER_DIR", str(developer))
    if system == "Linux":
        # The generated launcher is deliberately standalone. Load its trusted
        # sibling before restricting reads; cwd/PYTHONPATH must not choose a
        # different installed (or workspace-provided) privileged supervisor.
        metadata = runpy.run_path(str(Path(__file__).with_name("metadata.py")))
        metadata["write_process_receipt"](
            Path(config["process_receipt"]), descendants_drained=False
        )
    elif config.get("process_receipt"):
        Path(config["process_receipt"]).write_text(
            json.dumps({"pid": os.getpid()}) + "\n"
        )
    # Canonicalize native Codex aliases, including persisted online launchers.
    # Preserve other interpreters: resolving a venv Python loses its environment.
    cli = config["cli"]
    if "CODEX_HOME" in config["env"] and Path(cli).name == "codex":
        cli = str(Path(cli).resolve())
    command = [cli, *arguments]
    if system == "Linux":
        # Bun/JSC reads its mappings at startup and memory counters before tools. Bind
        # this rule in the exec'ing process: resolving /proc/self in the Host
        # would grant the wrong PID. Do not expose other processes or all /proc.
        metadata["exec_with_metadata_broker"](
            command,
            lambda: _linux_restrict(
                [
                    *config["readable"],
                    "/proc/self/maps",
                    "/proc/self/stat",
                    "/proc/self/statm",
                ],
                writable,
                metadata["METADATA_CALLS"],
            ),
            writable,
            Path(config["process_receipt"]).with_name("metadata.jsonl"),
            Path(config["process_receipt"]),
        )
    elif system == "Darwin":
        profile = ["(version 1)", "(allow default)", "(deny file-write*)"]
        if config.get("restrict_reads", False):
            profile.append("(deny file-read*)")
            profile.append('(allow file-read* (literal "/"))')
            # macOS /tmp is a symlink; canonical parent grants do not permit
            # lstat of the alias itself during Node's recursive mkdir.
            for alias in ("/tmp", "/etc", "/var"):
                profile.append(f'(allow file-read-metadata (literal "{alias}"))')
            for alias in config.get("readable_metadata", []):
                profile.append(
                    f"(allow file-read-metadata (literal {json.dumps(alias, ensure_ascii=False)}))"
                )
            parents = {
                str(parent)
                for name in (*config["readable"], *writable)
                for parent in Path(name).parents
            }
            profile.extend(
                f"(allow file-read-metadata (literal {json.dumps(path, ensure_ascii=False)}))"
                for path in sorted(parents)
            )
        for right, paths in (
            ("file-read*", config["readable"]),
            ("file-read* file-write*", writable),
        ):
            files: dict[Path, set[str]] = {}
            for path in paths:
                target = Path(path)
                if target.is_file():
                    files.setdefault(target.parent, set()).add(target.name)
                else:
                    kind = (
                        "literal"
                        if target.exists() and not target.is_dir()
                        else "subpath"
                    )
                    selector = f"({kind} {json.dumps(path, ensure_ascii=False)})"
                    profile.append(f"(allow {right} {selector})")
            for parent, names in files.items():
                # Share parent matchers to stay below Seatbelt's compiled data
                # limit. Fixed component count plus exact basenames preserves
                # file-only access, including after a file becomes a directory.
                prefix = "^" + "/[^/]+" * (len(parent.parts) - 1) + "/("
                batches: list[list[str]] = [[]]
                for name in sorted(names):
                    escaped = re.escape(name)
                    candidate = prefix + "|".join([*batches[-1], escaped]) + ")$"
                    # Seatbelt also bounds individual profile string literals.
                    if (
                        batches[-1]
                        and len(json.dumps(candidate, ensure_ascii=False).encode())
                        > 900
                    ):
                        batches.append([])
                    batches[-1].append(escaped)
                for batch in batches:
                    pattern = prefix + "|".join(batch) + ")$"
                    selector = (
                        "(require-all (subpath "
                        + json.dumps(str(parent), ensure_ascii=False)
                        + ") (regex "
                        + json.dumps(pattern, ensure_ascii=False)
                        + "))"
                    )
                    profile.append(f"(allow {right} {selector})")
        # A profile with thousands of historical evidence paths must not also
        # consume the exec argument/environment byte budget. The Host-owned
        # control directory is readable but outside every author write root.
        profile_path = Path(config["process_receipt"]).with_suffix(".sb")
        profile_path.write_text("\n".join(profile) + "\n")
        command = ["/usr/bin/sandbox-exec", "-f", str(profile_path), *command]
    else:
        raise RuntimeError(f"Unsupported native author filesystem boundary: {system}")
    os.execvpe(command[0], command, os.environ)


def _claude_resume_directory(config: dict, arguments: list[str], value: str) -> Path:
    """Accept only the SDK's private materialized session, never a broad root.

    The Host supplies this environment before entering the OS boundary. Native
    child processes cannot expand the inherited boundary by rerunning a launcher.
    """
    directory = Path(value)
    try:
        resume_values = [
            arg.removeprefix("--resume=")
            for arg in arguments
            if arg.startswith("--resume=")
        ]
        if len(resume_values) != 1:
            raise ValueError("expected one SDK resume argument")
        session_id = str(uuid.UUID(resume_values[0]))
        if resume_values[0] != session_id:
            raise ValueError("noncanonical session id")
        if (
            directory.is_symlink()
            or directory.resolve().parent != Path(config["sdk_temp_root"])
            or not directory.name.startswith("claude-resume-")
            or directory.stat().st_mode & 0o777 != 0o700
        ):
            raise ValueError("not a private SDK temporary directory")
        for path in (directory, *directory.rglob("*")):
            info = path.lstat()
            if (
                info.st_uid != os.getuid()
                or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))
                or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)
            ):
                raise ValueError("unsafe SDK session tree")
        transcript = (
            directory / "projects" / config["project_key"] / f"{session_id}.jsonl"
        )
        if not transcript.is_file() or not transcript.stat().st_size:
            raise ValueError("requested session transcript is absent")
    except (ValueError, OSError, KeyError) as exc:
        raise RuntimeError("Invalid SDK Claude resume directory") from exc
    return directory.resolve()


def native_output(workspace: Path) -> Path:
    """The sole deliverable directory owned by this prepared native job."""
    kind = json.loads((workspace / "context/input.json").read_text())["kind"]
    return (
        workspace
        / {
            "mutation": "candidate",
            "no_topology": "candidate",
            "success_consolidation": "candidate",
            "shared_distillation": "candidate",
            "graph_analysis": "analysis",
        }[kind]
    )


def native_author_launcher(
    workspace: Path,
    output: Path,
    control: Path,
    *,
    backend: Literal["claude", "codex", "cursor"],
    cli: str,
    readable_roots: tuple[Path, ...] | None = None,
    writable_roots: tuple[Path, ...] | None = None,
    native_runtime_root: Path | None = None,
    native_state_roots: tuple[Path, ...] = (),
) -> tuple[str, dict[str, str]]:
    """Return the Host-owned launcher and job-local native environment.

    control holds immutable launch settings, outside candidate/scratch write roots.
    Scratch contains the CLI's own state and temporary analysis, never deliverables.
    """
    workspace, output, control = (p.resolve() for p in (workspace, output, control))
    if not output.is_relative_to(workspace) or output == workspace:
        raise ValueError("Author output must be a child of its workspace")
    scratch = (workspace / ".scratch").resolve()
    if not scratch.is_relative_to(workspace):
        raise ValueError("Author scratch must remain within its workspace")
    if any(
        control.is_relative_to(path) or path.is_relative_to(control)
        for path in (output, scratch)
    ):
        raise ValueError(
            "Author control must be separate from writable output and scratch"
        )
    for path in (
        output,
        control,
        *(
            scratch / name
            for name in (
                "analysis",
                "tests",
                "media",
                "runtime/tmp",
                "runtime/pycache",
                f"runtime/{backend}",
            )
        ),
    ):
        path.mkdir(parents=True, exist_ok=True)
    # Like the native Bash sandbox, system/dependency reads remain available.
    # Authority is the write boundary, shared by all tools and descendants.
    readable = (
        ["/"] if readable_roots is None else [str(p.resolve()) for p in readable_roots]
    )
    if readable_roots is not None:
        if "/" in readable or str(Path.home()) in readable:
            raise ValueError("isolated author cannot read a filesystem or home root")
        readable.extend((str(workspace), str(control), str(Path(cli).resolve())))
        if backend == "codex":
            # New Codex versions execute tools through this separate binary.
            # Grant the installed sibling itself, never the whole bin directory.
            helper = Path(cli).resolve().with_name("codex-code-mode-host")
            if helper.is_file():
                readable.append(str(helper.resolve()))
    allowed_outputs = [output] if writable_roots is None else list(writable_roots)
    for path in allowed_outputs:
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("scope write roots must stay inside the candidate output")
    runtime = (
        (native_runtime_root / backend).resolve()
        if native_runtime_root is not None
        else scratch / f"runtime/{backend}"
    )
    runtime.mkdir(parents=True, exist_ok=True)
    readable.append(str(runtime))
    config = {
        "cli": cli,
        "process_receipt": str(control / "process.json"),
        "restrict_reads": readable_roots is not None,
        "readable": sorted(set(readable)),
        # A venv executable may target a version-alias symlink outside its
        # canonical base prefix. Permit only metadata on those exact aliases;
        # executable/library contents still require the existing read roots.
        "readable_metadata": sorted(
            {
                str(path)
                for name in (sys.executable, cli, *sys.path)
                if Path(name).is_absolute()
                for path in (Path(name), *Path(name).parents)
                if path.is_symlink()
            }
        ),
        "writable": [
            *(str(p.resolve()) for p in allowed_outputs),
            str(scratch),
            str(runtime),
            *(str(p.resolve()) for p in native_state_roots),
            "/dev/null",
            "/dev/tty",
        ],
        "env": {
            "TMPDIR": str(scratch / "runtime/tmp"),
            "PYTHONPYCACHEPREFIX": str(scratch / "runtime/pycache"),
            # Python scripts under .scratch must import the same workspace
            # client as -c/heredoc commands; cwd is not sys.path[0] for scripts.
            "PYTHONPATH": os.pathsep.join(
                value
                for value in (str(workspace), os.environ.get("PYTHONPATH"))
                if value
            ),
            **{
                {
                    "claude": "CLAUDE_CONFIG_DIR",
                    "codex": "CODEX_HOME",
                    "cursor": "CURSOR_DATA_DIR",
                }[backend]: str(runtime)
            },
        },
    }
    config_path = control / "workspace.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    launcher = control / backend
    launcher.write_text(
        f"#!{sys.executable}\nimport json, runpy, sys\n"
        f"module = runpy.run_path({str(Path(__file__).resolve())!r})\n"
        f"config = json.load(open({str(config_path)!r}))\n"
        "module['launch'](config, sys.argv[1:])\n"
    )
    launcher.chmod(0o700)
    return str(launcher), config["env"]


def claude_workspace_options(
    workspace: Path,
    output: Path,
    control: Path,
    *,
    readable_roots: tuple[Path, ...] | None = None,
    writable_roots: tuple[Path, ...] | None = None,
    native_runtime_root: Path | None = None,
) -> dict:
    """Native Claude tools run without asks inside the shared IO boundary."""
    from reweave.planner.codex_connection import claude_executable

    cli = claude_executable()
    launcher, env = native_author_launcher(
        workspace,
        output,
        control,
        backend="claude",
        cli=cli,
        readable_roots=readable_roots,
        writable_roots=writable_roots,
        native_runtime_root=native_runtime_root,
    )
    from claude_agent_sdk._internal.sessions import project_key_for_directory

    config_path = control / "workspace.json"
    config = json.loads(config_path.read_text())
    config.update(
        sdk_temp_root=str(Path(tempfile.gettempdir()).resolve()),
        project_key=project_key_for_directory(workspace),
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return {
        "cli_path": launcher,
        "env": env,
        # SDK None restores its 1 MiB default. Use the addressable container limit,
        # not an application cap: complete image/mirror messages must survive.
        "max_buffer_size": sys.maxsize,
        "permission_mode": "bypassPermissions",
        "setting_sources": [],
    }
