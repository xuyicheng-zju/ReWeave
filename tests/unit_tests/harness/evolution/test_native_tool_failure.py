import pytest

from reweave.evolution.infrastructure.native_tool_failure import tool_startup_failure


@pytest.mark.parametrize(
    "message",
    [
        "Permission denied (os error 13) launching codex-code-mode-host",
        "codex-code-mode-host: No such file or directory",
        "failed to spawn codex-code-mode-host",
    ],
)
def test_helper_launch_fault_is_infrastructure(message):
    assert tool_startup_failure(
        {
            "method": "item/completed",
            "payload": {"item": {"text": message}},
        }
    )


@pytest.mark.parametrize(
    "event",
    [
        {"method": "item/completed", "payload": {"text": "result.json missing"}},
        {"method": "item/completed", "payload": {"text": "Permission denied: data"}},
        {"method": "item/completed", "payload": {"text": "codex-code-mode-host OK"}},
        {
            "method": "turn/started",
            "payload": {"input": "codex-code-mode-host: Permission denied"},
        },
    ],
)
def test_unrelated_output_does_not_trigger_helper_fault(event):
    assert not tool_startup_failure(event)


def test_tool_fault_never_enters_format_correction_loop(tmp_path):
    import asyncio

    from reweave.evolution.authoring.author import NativeSession
    from reweave.evolution.infrastructure.native_tool_failure import (
        NativeToolStartupError,
    )

    calls = []

    class BrokenTools(NativeSession):
        async def _execute(self, *args, **kwargs):
            calls.append(kwargs)
            raise NativeToolStartupError("codex-code-mode-host: Permission denied")

    native = BrokenTools(bundle_loader=None, policy=None, evidence_workspace=tmp_path)
    with pytest.raises(NativeToolStartupError):
        asyncio.run(
            native._run_native(None, tmp_path / "job", resume=False, before_launch=None)
        )
    assert len(calls) == 1
