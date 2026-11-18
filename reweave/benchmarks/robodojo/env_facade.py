# Attribution retained in NOTICE. Modified for the ReWeave namespace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main-thread RPC facade for the RoboDojo agent environment."""

from __future__ import annotations

from reweave.robots.components.env_facade_base import BaseEnvFacade
from reweave.utils.rpc.main_thread_serve import MainThreadServeMixin


class RoboDojoEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """Serve compact action ACKs and explicit settled-snapshot reads."""

    def __init__(self, env, *, inference_config=None) -> None:
        self.env = env
        self.observations = env.observations
        if inference_config is not None:
            self.observations.configure_inference(inference_config)
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        del self._rpc["env.render_camera"]
        self._readonly_methods.discard("env.render_camera")
        self._rpc.update(
            {
                "env.get_obs": self.get_obs,
                "env.step_trajectory": self.step_trajectory,
                "env.get_status": self.get_status,
                "env.plan_position_path": self.plan_position_path,
                "env.plan_joint_path": self.plan_joint_path,
                "env.solve_ik_pose": self.solve_ik_pose,
                "env.plan_pose_path": self.plan_pose_path,
                "env.is_success": self.is_success,
                "env.close": self.request_close,
            }
        )
        snapshot_methods = (
            "read_sensor_snapshot",
            "retain_observation",
            "read_rgb",
            "back_project_ref",
            "segment_ref",
            "locate_ref",
            "grasps_ref",
            "predict_ref",
        )
        for method in snapshot_methods:
            self._rpc[f"env.{method}"] = getattr(self.observations, method)
        self._readonly_methods.update(f"env.{method}" for method in snapshot_methods)
        self._readonly_methods.update(
            [
                "env.get_env_meta",
                "env.get_task_language",
                "env.get_camera_meta",
                "env.get_obs",
                "env.get_status",
                "env.is_success",
                "env.plan_position_path",
                "env.plan_joint_path",
                "env.solve_ik_pose",
                "env.plan_pose_path",
            ]
        )

    def get_env_meta(self) -> dict:
        return self.env.get_env_meta()

    def get_task_language(self) -> str:
        return self.env.get_task_language()

    def get_camera_meta(self, camera_name: str, height=None, width=None) -> dict:
        return self.env.get_camera_meta(camera_name, height, width)

    def reset(self) -> dict:
        return self.env.reset_episode()

    def step(self, flat_action):
        return self.env.step_native(flat_action)

    def step_trajectory(self, arm, positions, velocities, gripper=0):
        return self.env.step_trajectory(arm, positions, velocities, gripper)

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        raise NotImplementedError("RoboDojo uses step for action control")

    def get_obs(self) -> dict:
        return self.env.get_obs()

    def get_status(self) -> dict:
        return self.env.get_status()

    def plan_position_path(self, arm: str, xyz: list) -> dict:
        return self.env.plan_position_path(arm, xyz)

    def plan_joint_path(self, arm: str, joints: list) -> dict:
        return self.env.plan_joint_path(arm, joints)

    def solve_ik_pose(self, arm: str, pose: list) -> dict:
        return self.env.solve_ik_pose(arm, pose)

    def plan_pose_path(
        self, arm: str, poses: list, linear_axis: str | None = None
    ) -> dict:
        return self.env.plan_pose_path(arm, poses, linear_axis)

    def is_success(self) -> bool:
        return self.env.is_success()

    def request_close(self) -> None:
        # The parent may terminate this process as soon as it receives the ACK.
        self.env.finish_recording()
        self._shutdown_event.set()

    def close(self) -> None:
        self.env.close()
