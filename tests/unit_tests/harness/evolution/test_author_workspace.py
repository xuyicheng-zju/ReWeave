# Modified for the ReWeave namespace.
"""Real subprocess tests of author IO boundaries, including child Python tools."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reweave.evolution.authoring.workspace import claude_workspace_options


def test_codex_grants_only_its_exact_tool_helper(tmp_path):
    from reweave.evolution.authoring.workspace import native_author_launcher

    install = tmp_path / "install/bin"
    install.mkdir(parents=True)
    cli = install / "codex"
    helper = install / "codex-code-mode-host"
    cli.write_text("native cli")
    helper.write_text("native helper")
    root, control = tmp_path / "author", tmp_path / "control"
    native_author_launcher(
        root,
        root / "candidate",
        control,
        backend="codex",
        cli=str(cli),
        readable_roots=(Path(sys.prefix),),
    )
    config = json.loads((control / "workspace.json").read_text())
    assert str(helper.resolve()) in config["readable"]
    assert str(install.resolve()) not in config["readable"]
    assert str(helper.resolve()) not in config["writable"]


@pytest.mark.parametrize("selected", [None, "/host/custom-developer"])
def test_macos_hidden_developer_tools_fail_without_install_prompt(tmp_path, selected):
    import platform

    from reweave.evolution.authoring.workspace import native_author_launcher

    installed = Path("/Library/Developer/CommandLineTools")
    if platform.system() != "Darwin" or not installed.is_dir():
        pytest.skip("macOS with installed Command Line Tools")
    root = tmp_path / "author"
    launcher, env = native_author_launcher(
        root,
        root / "candidate",
        tmp_path / "control",
        backend="codex",
        cli="/usr/bin/xcrun",
        readable_roots=(
            Path("/System"),
            Path("/usr"),
            Path("/bin"),
            Path(sys.prefix),
            Path(sys.base_prefix),
        ),
    )
    environ = {k: v for k, v in os.environ.items() if k != "DEVELOPER_DIR"}
    if selected is not None:
        environ["DEVELOPER_DIR"] = selected
    result = subprocess.run(
        [launcher, "--find", "clang"],
        env={**environ, **env},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert (
        "missing DEVELOPER_DIR path: " + (selected or str(installed)) in result.stderr
    )
    profile = (tmp_path / "control/process.sb").read_text()
    assert '(subpath "/Library/Developer' not in profile


def test_macos_large_evidence_allowlist_keeps_file_and_directory_boundaries(tmp_path):
    import platform

    from reweave.evolution.authoring.workspace import native_author_launcher

    if platform.system() != "Darwin":
        pytest.skip("macOS compiled Seatbelt profile size and exact file grants")
    evidence = tmp_path / "evidence"
    files = []
    for index in range(860):
        directory = evidence / f"snapshot-{index:06d}"
        directory.mkdir(parents=True)
        for name in ("cam_head.json", "cam_left_wrist.json", "cam_right_wrist.json"):
            path = directory / name
            path.write_text("recorded")
            files.append(path)
    secret = evidence / "snapshot-000000/not-granted.json"
    secret.write_text("private")
    replaceable = evidence / "was-a-file"
    replaceable.mkdir()
    (replaceable / "child").write_text("private")
    # A path granted as a regular file must not authorize descendants after
    # another process replaces it with a directory.
    file_before_launch = evidence / "replace-after-launch"
    file_before_launch.write_text("recorded")
    special_directory = evidence / '中文 "[parent]'
    special_directory.mkdir()
    special_file = special_directory / 'quoted"(file).json'
    special_file.write_text("recorded")
    for denied in (
        special_directory / 'quoted"(file).json\n',
        special_directory / "nested" / special_file.name,
    ):
        denied.parent.mkdir(exist_ok=True)
        denied.write_text("private")
    root = tmp_path / "author"
    launcher, env = native_author_launcher(
        root,
        root / "candidate",
        tmp_path / "control",
        backend="codex",
        cli=sys.executable,
        readable_roots=(
            Path("/System"),
            Path("/usr"),
            Path("/bin"),
            Path(sys.prefix),
            Path(sys.base_prefix),
            *files,
            file_before_launch,
            special_file,
        ),
    )
    code = """
import pathlib, sys
root, evidence = map(pathlib.Path, sys.argv[1:])
for index in range(860):
    for name in ('cam_head.json', 'cam_left_wrist.json', 'cam_right_wrist.json'):
        path = evidence/f'snapshot-{index:06d}'/name
        assert path.read_text() == 'recorded'
try: path.write_text('changed')
except PermissionError: pass
else: raise AssertionError('evidence became writable')
for path in (evidence/'snapshot-000000/not-granted.json', evidence/'was-a-file/child'):
    try: path.read_text()
    except PermissionError: pass
    else: raise AssertionError('read escaped: ' + str(path))
special = evidence/'中文 "[parent]'/ 'quoted"(file).json'
assert special.read_text() == 'recorded'
for path in (special.with_name(special.name+'\\n'), special.parent/'nested'/special.name):
    try: path.read_text()
    except PermissionError: pass
    else: raise AssertionError('file matcher escaped: ' + str(path))
(root/'candidate/new-directory').mkdir()
(root/'candidate/new-directory/result').write_text('owned')
print('large_allowlist_verified', flush=True)
# Pause after the OS profile is compiled, while the parent replaces a file.
sys.stdin.readline()
try: (evidence/'replace-after-launch/child').read_text()
except PermissionError: pass
else: raise AssertionError('file grant expanded to a directory')
"""
    process = subprocess.Popen(
        [launcher, "-c", code, str(root), str(evidence)],
        cwd=root,
        env={**os.environ, **env},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "large_allowlist_verified"
        file_before_launch.unlink()
        file_before_launch.mkdir()
        (file_before_launch / "child").write_text("private")
        stdout, stderr = process.communicate("continue\n", timeout=30)
        assert process.returncode == 0, (stdout, stderr)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_failed_descendant_drain_cannot_publish_completion(tmp_path, monkeypatch):
    from reweave.evolution.authoring import metadata as author_metadata

    receipt = tmp_path / "process.json"
    author_metadata.write_process_receipt(receipt, descendants_drained=False)
    monkeypatch.setattr(author_metadata, "_supervise_metadata", lambda *args: 0)

    def fail():
        raise OSError("owned child could not be reaped")

    monkeypatch.setattr(author_metadata, "_drain_children", fail)
    with pytest.raises(OSError, match="could not be reaped"):
        author_metadata.exec_with_metadata_broker(
            [], None, [], tmp_path / "metadata.jsonl", receipt
        )
    assert json.loads(receipt.read_text())["descendants_drained"] is False


def test_linux_failed_native_exec_records_completed_drain(tmp_path):
    import platform

    from reweave.evolution.authoring.workspace import native_author_launcher

    if platform.system() != "Linux":
        pytest.skip("Linux native author child subreaper")
    launcher, env = native_author_launcher(
        tmp_path / "author",
        tmp_path / "author/candidate",
        tmp_path / "control",
        backend="claude",
        cli=str(tmp_path / "nonexistent-cli"),
    )
    result = subprocess.run(
        [launcher],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 125
    assert "FileNotFoundError" in result.stderr
    assert (
        json.loads((tmp_path / "control/process.json").read_text())[
            "descendants_drained"
        ]
        is True
    )


@pytest.mark.parametrize("termination", ["exit", "term", "kill"])
def test_linux_author_records_whether_detached_writers_are_drained(
    tmp_path, termination
):
    import platform

    from reweave.evolution.authoring.workspace import native_author_launcher

    if platform.system() != "Linux":
        pytest.skip("Linux native author child subreaper")
    root = tmp_path / "author"
    launcher, env = native_author_launcher(
        root,
        root / "candidate",
        tmp_path / "control",
        backend="claude",
        cli=sys.executable,
    )
    code = """
import os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
writer = root/'candidate/writer.pid'
if os.fork() == 0:
    os.setsid()
    # Detached subprocesses must not keep the test driver's output pipe open.
    null = os.open('/dev/null', os.O_RDWR)
    for descriptor in (0, 1, 2): os.dup2(null, descriptor)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    writer.write_text(str(os.getpid()))
    while True:
        (root/'candidate/background.txt').write_text(str(time.time_ns()))
        time.sleep(0.01)
while not writer.exists(): time.sleep(0.01)
print('writer_ready', flush=True)
if sys.argv[2] == 'wait':
    while True: time.sleep(0.01)
"""
    process = subprocess.Popen(
        [launcher, "-c", code, str(root), "exit" if termination == "exit" else "wait"],
        cwd=root,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "writer_ready"
        writer = int((root / "candidate/writer.pid").read_text())
        if termination == "term":
            process.terminate()
        elif termination == "kill":
            process.kill()
        _, stderr = process.communicate(timeout=15)
        assert (
            process.returncode == {"exit": 0, "term": 143, "kill": -9}[termination]
        ), stderr
        receipt = json.loads((tmp_path / "control/process.json").read_text())
        assert receipt["descendants_drained"] is (termination != "kill")
        if termination == "kill":
            # SIGKILL cannot run finally. The surviving owned test writer is
            # cleaned below; the false receipt must prevent Host publication.
            os.kill(writer, 0)
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(writer, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        writer_file = root / "candidate/writer.pid"
        if writer_file.exists():
            try:
                os.kill(int(writer_file.read_text()), 9)
            except ProcessLookupError:
                pass


def test_native_launcher_reaps_fast_children(tmp_path):
    """A listener hangup must not become a blocking timestamp receive."""
    from reweave.evolution.authoring.workspace import native_author_launcher

    launcher, env = native_author_launcher(
        tmp_path / "workspace",
        tmp_path / "workspace/candidate",
        tmp_path / "control",
        backend="claude",
        cli=sys.executable,
    )
    for _ in range(12):
        result = subprocess.run(
            [launcher, "-c", "pass"],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr


def test_native_timestamp_updates_keep_the_write_scope(tmp_path):
    from reweave.evolution.authoring.workspace import native_author_launcher

    root = tmp_path / "author"
    output = root / "candidate"
    launcher, env = native_author_launcher(
        root, output, tmp_path / "control", backend="claude", cli=sys.executable
    )
    evidence = root / "evidence.txt"
    evidence.write_text("immutable")
    os.utime(evidence, ns=(123456789, 123456789))
    (output / "link").symlink_to(evidence)
    code = """
import errno, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
folder = root/'.scratch/runtime/claude/tasks/test/.lock.lock'
folder.mkdir(parents=True)
os.utime(folder, ns=(1000000000, 2000000000))
assert folder.stat().st_mtime_ns == 2000000000
target = root/'candidate/created.txt'
target.write_text('owned')
with target.open('rb') as stream:
    os.utime(stream.fileno(), ns=(3000000000, 4000000000))
assert target.stat().st_mtime_ns == 4000000000
directory = os.open(target.parent, os.O_RDONLY)
try: os.utime('created.txt', dir_fd=directory, ns=(5000000000, 6000000000))
finally: os.close(directory)
assert target.stat().st_mtime_ns == 6000000000
subprocess.run([sys.executable, '-c', 'import os,sys; os.utime(sys.argv[1],None)', str(target)], check=True)
for path in [root/'evidence.txt', root/'candidate/link']:
    try: os.utime(path, ns=(9000000000, 9000000000))
    except PermissionError: pass
    else: raise AssertionError(f'timestamp escaped: {path}')
with (root/'evidence.txt').open('rb') as stream:
    try: os.utime(stream.fileno(),None)
    except PermissionError: pass
    else: raise AssertionError('fd timestamp escaped')
try: os.utime(root/'candidate/missing',None)
except FileNotFoundError: pass
else: raise AssertionError('missing timestamp target became success')
print('timestamps_verified')
"""
    result = subprocess.run(
        [launcher, "-c", code, str(root)],
        cwd=root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "timestamps_verified"
    assert evidence.stat().st_mtime_ns == 123456789


def test_native_mode_changes_keep_the_write_scope(tmp_path):
    from reweave.evolution.authoring.workspace import native_author_launcher

    root = tmp_path / "author"
    output = root / "candidate"
    launcher, env = native_author_launcher(
        root, output, tmp_path / "control", backend="codex", cli=sys.executable
    )
    evidence = root / "evidence.txt"
    evidence.write_text("immutable")
    evidence.chmod(0o644)
    (output / "link").symlink_to(evidence)
    code = """
import os, pathlib, stat, subprocess, sys
root = pathlib.Path(sys.argv[1])
folder = root/'.scratch/runtime/codex/tmp/arg0'
folder.mkdir(parents=True)
folder.chmod(0o700)
assert stat.S_IMODE(folder.stat().st_mode) == 0o700
target = root/'candidate/created.txt'
target.write_text('owned')
target.chmod(0o640)
assert stat.S_IMODE(target.stat().st_mode) == 0o640
with target.open('rb') as stream: os.fchmod(stream.fileno(),0o600)
assert stat.S_IMODE(target.stat().st_mode) == 0o600
directory = os.open(target.parent,os.O_RDONLY)
try: os.chmod('created.txt',0o644,dir_fd=directory)
finally: os.close(directory)
assert stat.S_IMODE(target.stat().st_mode) == 0o644
subprocess.run([sys.executable,'-c','import os,sys;os.chmod(sys.argv[1],0o600)',str(target)],check=True)
for path in [root/'evidence.txt',root/'candidate/link']:
    try: path.chmod(0o600)
    except PermissionError: pass
    else: raise AssertionError(f'mode change escaped: {path}')
with (root/'evidence.txt').open('rb') as stream:
    try: os.fchmod(stream.fileno(),0o600)
    except PermissionError: pass
    else: raise AssertionError('fd mode change escaped')
assert stat.S_IMODE((root/'evidence.txt').stat().st_mode) == 0o644
print('modes_verified')
"""
    result = subprocess.run(
        [launcher, "-c", code, str(root)],
        cwd=root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "modes_verified"
    assert evidence.read_text() == "immutable"


def test_linux_metadata_broker_rejects_hardlinked_evidence(tmp_path):
    import platform

    import pytest

    from reweave.evolution.authoring.workspace import native_author_launcher

    if platform.system() != "Linux":
        pytest.skip("Linux seccomp metadata supervisor")
    root = tmp_path / "author"
    output = root / "candidate"
    launcher, env = native_author_launcher(
        root, output, tmp_path / "control", backend="claude", cli=sys.executable
    )
    evidence = root / "evidence.txt"
    evidence.write_text("immutable")
    os.utime(evidence, ns=(123456789, 123456789))
    os.link(evidence, output / "alias")
    code = """
import os,sys,stat
before=stat.S_IMODE(os.stat(sys.argv[1]).st_mode)
try: os.utime(sys.argv[1], None)
except PermissionError: pass
else: raise AssertionError('timestamp changed a hardlinked input')
try: os.chmod(sys.argv[1],0o600)
except PermissionError: pass
else: raise AssertionError('mode changed a hardlinked input')
assert stat.S_IMODE(os.stat(sys.argv[1]).st_mode)==before
"""
    result = subprocess.run(
        [launcher, "-c", code, str(output / "alias")],
        cwd=root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert evidence.stat().st_mtime_ns == 123456789


def test_standalone_linux_launcher_ignores_pythonpath_for_its_supervisor(
    tmp_path, monkeypatch
):
    import platform

    from reweave.evolution.authoring.workspace import native_author_launcher

    if platform.system() != "Linux":
        pytest.skip("Linux supervisor loading")
    root = tmp_path / "author"
    spoof = root / "harness/evolution"
    spoof.mkdir(parents=True)
    (root / "harness/__init__.py").write_text("")
    (spoof / "__init__.py").write_text("")
    (spoof / "author_metadata.py").write_text(
        "raise AssertionError('workspace replaced the trusted supervisor')\n"
    )
    monkeypatch.setenv("PYTHONPATH", ".")
    launcher, env = native_author_launcher(
        root,
        root / "candidate",
        tmp_path / "control",
        backend="claude",
        cli=sys.executable,
    )
    result = subprocess.run(
        [launcher, "-c", "print('started')"],
        cwd=root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "started"
    receipt = json.loads((tmp_path / "control/process.json").read_text())
    assert receipt["descendants_drained"] is True


def test_native_author_workspace_allows_analysis_and_protects_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "unrelated-global-config"))
    root = tmp_path / "author"
    output = root / "output"
    options = claude_workspace_options(root, output, root / "control")
    assert options["permission_mode"] == "bypassPermissions"
    assert "extra_args" not in options
    assert "allowed_tools" not in options
    (root / "evidence.json").write_text('{"values": [1, 2, 3]}')
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched")
    (output / "link").symlink_to(outside)
    config_path = root / "control/workspace.json"
    config = json.loads(config_path.read_text())
    config["cli"] = sys.executable
    config_path.write_text(json.dumps(config))
    code = """
import json, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
assert os.environ['CLAUDE_CONFIG_DIR'] == str(root/'.scratch/runtime/claude')
values = json.loads((root/'evidence.json').read_text())['values']
(root/'output/result.json').write_text(json.dumps({'sum': sum(values)}))
(root/'.scratch/tests/probe.py').write_text('print(42)')
assert subprocess.check_output([sys.executable, str(root/'.scratch/tests/probe.py')]).strip() == b'42'
for path in (root/'evidence.json', root/'control/workspace.json', root/'control/process.json', root.parent/'outside.txt', root/'output/link'):
    try:
        path.write_text('corrupted')
    except PermissionError:
        pass
    else:
        raise AssertionError(f'write escaped: {path}')
    try:
        os.truncate(path, 0)
    except PermissionError:
        pass
    else:
        raise AssertionError(f'truncate escaped: {path}')
print('boundary_verified')
"""
    result = subprocess.run(
        [options["cli_path"], "-c", code, str(root)],
        cwd=root,
        env={**os.environ, **options["env"]},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "boundary_verified"
    assert json.loads((output / "result.json").read_text()) == {"sum": 6}
    assert outside.read_text() == "untouched"
    assert json.loads((root / "evidence.json").read_text()) == {"values": [1, 2, 3]}


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_native_author_reads_complete_correction_feedback(tmp_path, backend):
    from reweave.evolution.authoring.workspace import native_author_launcher

    root = tmp_path / "workspace"
    corrections = tmp_path / "audit/corrections"
    corrections.mkdir(parents=True)
    feedback = corrections / "feedback.json"
    payload = {"error": "validation failed", "result": "完整结果" * 100000}
    feedback.write_text(json.dumps(payload, ensure_ascii=False))
    original = feedback.read_bytes()
    launcher, env = native_author_launcher(
        root,
        root / "candidate",
        tmp_path / "control",
        backend=backend,
        cli=str(Path(sys.executable).resolve()),
        readable_roots=(
            corrections,
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            *(
                Path(p)
                for p in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/System", "/dev")
                if Path(p).exists()
            ),
        ),
    )
    code = """
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
assert value == {'error': 'validation failed', 'result': '完整结果' * 100000}
try:
    path.write_text('changed')
except PermissionError:
    pass
else:
    raise AssertionError('correction feedback is writable')
print('complete_feedback_read')
"""
    result = subprocess.run(
        [launcher, "-c", code, str(feedback)],
        cwd=root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "complete_feedback_read"
    assert feedback.read_bytes() == original


def test_author_workspace_cannot_grant_entire_workspace_write(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="child"):
        claude_workspace_options(tmp_path, tmp_path, tmp_path / "control")


def test_author_control_cannot_overlap_writable_roots(tmp_path):
    import pytest

    for control in (
        tmp_path / "output/control",
        tmp_path / ".scratch/control",
        tmp_path,
    ):
        with pytest.raises(ValueError, match="control must be separate"):
            claude_workspace_options(tmp_path, tmp_path / "output", control)


def test_sdk_transport_preserves_large_fragmented_image_and_following_message(tmp_path):
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    payload = {"type": "image", "data": "完整图像" * (1024 * 1024)}
    following = {"type": "result", "result": "done"}
    wire = json.dumps(payload, ensure_ascii=False) + "\n" + json.dumps(following) + "\n"

    class Process:
        async def wait(self):
            return 0

    async def chunks():
        for offset in range(0, len(wire), 16381):
            yield wire[offset : offset + 16381]

    async def read():
        options = ClaudeAgentOptions(
            **claude_workspace_options(
                tmp_path, tmp_path / "output", tmp_path / "control"
            )
        )
        transport = SubprocessCLITransport(prompt="test", options=options)
        transport._process = Process()
        transport._stdout_stream = chunks()
        return [message async for message in transport.read_messages()]

    assert asyncio.run(read()) == [payload, following]


def test_sdk_resume_launcher_and_mirror_share_materialized_directory(tmp_path):
    """Real SDK materialization and OS launcher, without a model/network call."""
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk._internal import session_resume
    from claude_agent_sdk._internal.sessions import project_key_for_directory
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    from reweave.evolution.infrastructure.session import ClaudeSessionStore

    root = tmp_path / "author"
    options = claude_workspace_options(root, root / "output", root / "control")
    assert options["env"]["CLAUDE_CONFIG_DIR"] == str(root / ".scratch/runtime/claude")
    evidence = root / "evidence.json"
    evidence.write_text("immutable")
    config_path = root / "control/workspace.json"
    config = json.loads(config_path.read_text())
    config["cli"] = sys.executable
    config_path.write_text(json.dumps(config))
    session_id = "11111111-1111-4111-8111-111111111111"
    key = {"session_id": session_id, "project_key": project_key_for_directory(root)}
    first = {"uuid": "first", "type": "user", "message": {"content": "before"}}
    second = {"uuid": "second", "type": "assistant", "message": {"content": "after"}}
    store = ClaudeSessionStore(root / "session")

    async def exercise():
        await store.append(key, [first])
        configured = ClaudeAgentOptions(
            cwd=root, session_store=store, resume=session_id, **options
        )
        materialized = await session_resume.materialize_resume_session(configured)
        assert materialized is not None
        resumed = session_resume.apply_materialized_options(configured, materialized)
        native_command = SubprocessCLITransport(
            prompt="test", options=resumed
        )._build_command()
        resume_arguments = [
            arg for arg in native_command if arg.startswith("--resume=")
        ]
        assert resume_arguments == [f"--resume={session_id}"]
        code = """
import json, os, pathlib, sys
directory = pathlib.Path(os.environ['CLAUDE_CONFIG_DIR'])
transcript = directory/'projects'/sys.argv[1]/(sys.argv[2]+'.jsonl')
assert json.loads(transcript.read_text().splitlines()[0])['uuid'] == 'first'
entry = json.loads(sys.argv[3])
with transcript.open('a') as stream:
    stream.write(json.dumps(entry)+'\\n')
for path in (pathlib.Path(sys.argv[4]), directory.parent/'escaped.txt'):
    try:
        path.write_text('corrupt')
    except PermissionError:
        pass
    else:
        raise AssertionError(f'write escaped: {path}')
print(json.dumps({'filePath': str(transcript), 'entries': [entry]}))
"""
        try:
            result = subprocess.run(
                [
                    options["cli_path"],
                    "-c",
                    code,
                    key["project_key"],
                    session_id,
                    json.dumps(second),
                    str(evidence),
                    *resume_arguments,
                ],
                cwd=root,
                env={**os.environ, **resumed.env},
                text=True,
                capture_output=True,
            )
            assert result.returncode == 0, result.stderr
            frame = json.loads(result.stdout)
            assert frame["filePath"].startswith(str(materialized.config_dir))
            errors = []

            async def on_error(key, message):
                errors.append(message)

            mirror = session_resume.build_mirror_batcher(
                store, materialized, resumed.env, on_error
            )
            mirror.enqueue(frame["filePath"], frame["entries"])
            await mirror.close()
            assert not errors
            assert await store.load(key) == [first, second]
            assert evidence.read_text() == "immutable"
        finally:
            await materialized.cleanup()

    asyncio.run(exercise())


def test_resume_directory_rejects_unrelated_or_unsafe_environment(tmp_path):
    import tempfile

    import pytest

    from reweave.evolution.authoring.workspace import _claude_resume_directory

    options = claude_workspace_options(
        tmp_path, tmp_path / "output", tmp_path / "control"
    )
    config = json.loads((tmp_path / "control/workspace.json").read_text())
    session_id = "11111111-1111-4111-8111-111111111111"
    arguments = [f"--resume={session_id}"]
    with pytest.raises(RuntimeError, match="Invalid SDK"):
        _claude_resume_directory(config, arguments, str(tmp_path))
    with tempfile.TemporaryDirectory(prefix="claude-resume-") as value:
        from pathlib import Path

        directory = Path(value)
        with pytest.raises(RuntimeError, match="Invalid SDK"):
            _claude_resume_directory(config, arguments, value)
        transcript = (
            directory / "projects" / config["project_key"] / f"{session_id}.jsonl"
        )
        transcript.parent.mkdir(parents=True)
        transcript.write_text('{"uuid":"original"}\n')
        assert _claude_resume_directory(config, arguments, value) == directory.resolve()
        with pytest.raises(RuntimeError, match="Invalid SDK"):
            _claude_resume_directory(config, [], value)
        (directory / "link").symlink_to(tmp_path / "output")
        with pytest.raises(RuntimeError, match="Invalid SDK"):
            _claude_resume_directory(config, arguments, value)
        (directory / "link").unlink()
        os.link(transcript, directory / "hardlink")
        with pytest.raises(RuntimeError, match="Invalid SDK"):
            _claude_resume_directory(config, arguments, value)
        (directory / "hardlink").unlink()
        directory.chmod(0o755)
        with pytest.raises(RuntimeError, match="Invalid SDK"):
            _claude_resume_directory(config, arguments, value)
    assert options["env"]["CLAUDE_CONFIG_DIR"].endswith("/.scratch/runtime/claude")


def test_linux_launcher_grants_only_its_own_memory_map(tmp_path, monkeypatch):
    import platform
    from pathlib import Path

    import pytest

    from reweave.evolution.authoring.workspace import native_author_launcher

    if platform.system() != "Linux":
        pytest.skip("Linux procfs and Landlock boundary")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    root = tmp_path / "author"
    secret = tmp_path / "secret"
    secret.write_text("private")
    launcher, env = native_author_launcher(
        root,
        root / "output",
        root / "control",
        backend="claude",
        cli=sys.executable,
        readable_roots=tuple(
            Path(p).resolve()
            for p in (
                "/usr",
                "/bin",
                "/lib",
                "/lib64",
                "/etc",
                sys.prefix,
                sys.base_prefix,
            )
            if Path(p).exists()
        ),
    )
    code = """
import pathlib, sys
for name in ('maps', 'stat', 'statm'):
    assert pathlib.Path('/proc/self', name).read_text()
for name in ('/proc/self/environ', '/proc/1/maps', sys.argv[1]):
    try:
        pathlib.Path(name).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('read escaped: ' + name)
print('verified')
"""
    result = subprocess.run(
        [launcher, "-c", code, str(secret)],
        cwd=root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "verified"


def test_author_sandbox_reads_the_systemd_resolver_directory(monkeypatch):
    """/etc/resolv.conf links there on systemd hosts; without it DNS fails (EAI_AGAIN)."""
    from reweave.evolution.authoring.author import system_read_roots

    resolver = Path("/run/systemd/resolve")
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: path == resolver or exists(path))
    assert resolver in system_read_roots()
