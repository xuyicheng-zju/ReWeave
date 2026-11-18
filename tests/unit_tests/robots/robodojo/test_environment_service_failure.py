"""Read transport loss is infrastructure; submitted action loss stays unknown."""

import pytest

from examples.robodojo_demo import SimulatedEnvironmentRPC, create_simulated_toolkit
from reweave.runtime.capability import (
    EnvironmentServiceFailure,
    PhysicalOutcomeUnknown,
)
from reweave.utils.rpc.rpc_client import RpcError, RpcTransportError


@pytest.mark.parametrize("method", ["env.is_success", "env.segment_ref"])
def test_exhausted_read_transport_is_latched_without_action_replay(tmp_path, method):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    original_call = rpc.call
    calls = []

    def call(name, *args, **kwargs):
        if name == method:
            calls.append(name)
            raise RpcTransportError(name, "SIMULATED network unavailable")
        return original_call(name, *args, **kwargs)

    rpc.call = call
    try:
        with pytest.raises(EnvironmentServiceFailure) as failed:
            if method == "env.is_success":
                toolkit.solved()
            else:
                toolkit.execute_tool("segment", {"text_prompt": "ring"})
        assert failed.value.retryable is False
        assert isinstance(failed.value.__cause__, RpcTransportError)
        with pytest.raises(EnvironmentServiceFailure) as blocked:
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert blocked.value is failed.value
        assert calls == [method] and rpc.actions == 0
        assert not toolkit._primitives.env._outcome_unknown
    finally:
        toolkit.close()


def test_remote_application_error_does_not_poison_environment_transport(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    original_call = rpc.call

    def call(name, *args, **kwargs):
        if name == "env.segment_ref":
            raise RpcError(name, "SIMULATED invalid segmentation argument")
        return original_call(name, *args, **kwargs)

    rpc.call = call
    try:
        result = toolkit.execute_tool("segment", {"text_prompt": "ring"})
        assert "invalid segmentation argument" in result.result["error"]
        assert toolkit._primitives.env._service_failure is None
        assert toolkit.solved() is False
        assert rpc.actions == 0
    finally:
        toolkit.close()


def test_step_transport_loss_never_becomes_a_retryable_read_failure(tmp_path):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    original_call = rpc.call
    submissions = []

    def call(name, *args, **kwargs):
        if name == "env.step":
            submissions.append(name)
            original_call(name, *args, **kwargs)
            raise RpcTransportError(name, "SIMULATED lost physical ACK")
        return original_call(name, *args, **kwargs)

    rpc.call = call
    try:
        with pytest.raises(PhysicalOutcomeUnknown):
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        with pytest.raises(PhysicalOutcomeUnknown):
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert submissions == ["env.step"] and rpc.actions == 1
        assert toolkit._primitives.env._service_failure is None
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "method", ["env.plan_position_path", "env.plan_joint_path", "env.plan_pose_path"]
)
@pytest.mark.parametrize("rpc_exception", [False, True])
def test_planner_cuda_oom_is_latched_as_infrastructure(tmp_path, method, rpc_exception):
    rpc = SimulatedEnvironmentRPC(terminal_after=99)
    toolkit, _, _ = create_simulated_toolkit(tmp_path, rpc=rpc)
    original_call = rpc.call

    def call(name, *args, **kwargs):
        if name == method:
            if rpc_exception:
                raise RpcError(name, "CUDA out of memory. Tried to allocate 256 MiB")
            return {
                "status": "Failure",
                "error": "CUDA out of memory. Tried to allocate 256 MiB",
            }
        return original_call(name, *args, **kwargs)

    rpc.call = call
    try:
        env = toolkit._primitives.env
        with pytest.raises(EnvironmentServiceFailure) as failed:
            env._plan_call(method)
        assert failed.value.retryable is False
        with pytest.raises(EnvironmentServiceFailure) as blocked:
            toolkit.execute_tool("set_gripper", {"arm": "right", "gripper": 1})
        assert blocked.value is failed.value
        assert rpc.actions == 0 and not env._outcome_unknown
    finally:
        toolkit.close()
