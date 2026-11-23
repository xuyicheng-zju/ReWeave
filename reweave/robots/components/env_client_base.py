# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""Unified env client base class."""

from __future__ import annotations

import copy


class BaseEnvClient:
    """Unified env client base class."""

    _TIMEOUT_S: dict[str, float] = {
        "default": 30.0,
        "env.reset": 120.0,
        "env.step": 60.0,
        "env.chunk_step": 120.0,
        "env.render_camera": 120.0,
    }

    def __init__(self, client, *, expected_meta: dict):
        self._client = client
        # Only synchronous simulator subclasses opt into these read helpers.
        # A submitted mutation invalidates all camera/calibration reads, even
        # when its acknowledgement is lost. No server or physical state cache
        # is shared between clients.
        self._observation_cache = {}
        self._observation_settled = False
        server_meta = self._client.call(
            "env.get_env_meta",
            timeout_s=self._TIMEOUT_S["default"],
        )
        if server_meta != expected_meta:
            raise ValueError(
                f"env_meta mismatch: expected={expected_meta!r} actual={server_meta!r}. "
                "The env_server was launched with different args than this client "
                "expects — kill the stale env_server and relaunch."
            )
        self.reset()

    def reset(self):
        """Reset the env and return the initial obs. Also updates the
        ``self.last_obs`` cache."""
        self._invalidate_observation_cache()
        self.last_obs = self._client.call(
            "env.reset",
            timeout_s=self._TIMEOUT_S["env.reset"],
        )
        self._mark_observation_settled()
        return self.last_obs

    def step(self, flat_action):
        """Execute one env action. Returns the gym 5-tuple
        ``(obs, rew, term, trunc, info)``.

        Also updates the ``self.last_obs`` cache with the first element (obs)
        of the returned tuple.
        """
        self._invalidate_observation_cache()
        result = self._client.call(
            "env.step",
            args=(flat_action,),
            timeout_s=self._TIMEOUT_S["env.step"],
        )
        self.last_obs = result[0]
        self._mark_observation_settled()
        return result

    def chunk_step(self, flat_actions, *, return_all_frames: bool = False):
        """Execute N actions in one batch. Returns the 5-tuple
        ``(obs, rew, term, trunc, info)``.

        - ``obs_or_list``: ``list[Obs]`` when ``return_all_frames=True`` (one
          per step, carrying the per-step render); the final obs dict when
          ``False``.
        - Updates the ``self.last_obs`` cache: if the returned obs is a list,
          take the last element; otherwise assign directly.

        ``return_all_frames`` is the standard perf-vs-video switch. ``True``
        renders agentview per step (high-density video); ``False`` renders only
        the final obs (fast).
        """
        self._invalidate_observation_cache()
        result = self._client.call(
            "env.chunk_step",
            args=(flat_actions,),
            kwargs={"return_all_frames": return_all_frames},
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        obs_field = result[0]
        if isinstance(obs_field, list):
            self.last_obs = obs_field[-1]
        else:
            self.last_obs = obs_field
        self._mark_observation_settled()
        return result

    def _invalidate_observation_cache(self) -> None:
        """Discard reads before a mutation; failures leave the cache unusable."""
        self._observation_cache = {}
        self._observation_settled = False

    def _mark_observation_settled(self) -> None:
        """Allow snapshot reads only after the subclass accepts its action ACK."""
        self._observation_settled = True

    def _cached_observation_call(self, method, *, kwargs=None, timeout_s=None):
        """Reuse a synchronous simulator read until its next submitted mutation.

        Keys include all scalar camera/resolution arguments. Returned values
        are caller-owned copies, so image edits cannot corrupt later perception.
        This is opt-in: the generic rendering methods below still perform live
        RPCs for backends whose environment can progress independently.
        """
        if not self._observation_settled:
            raise RuntimeError("simulator observation has no settled action boundary")
        arguments = kwargs or {}
        key = (method, tuple(sorted(arguments.items())))
        if key not in self._observation_cache:
            self._observation_cache[key] = self._client.call(
                method,
                kwargs=arguments,
                timeout_s=timeout_s
                if timeout_s is not None
                else self._TIMEOUT_S["default"],
            )
        return copy.deepcopy(self._observation_cache[key])

    def _cached_render_camera(self, camera_name, **kwargs):
        """Reuse matching RGB-D for RGB reads, preserving native pixel geometry."""
        arguments = {"camera_name": camera_name, **kwargs}
        arguments.setdefault("depth", False)
        if not arguments["depth"]:
            depth_arguments = {**arguments, "depth": True}
            depth_key = ("env.render_camera", tuple(sorted(depth_arguments.items())))
            if self._observation_settled and depth_key in self._observation_cache:
                return copy.deepcopy(self._observation_cache[depth_key][0])
        return self._cached_observation_call(
            "env.render_camera",
            kwargs=arguments,
            timeout_s=self._TIMEOUT_S["env.render_camera"],
        )

    def get_camera_meta(self, camera_name, **kwargs):
        return self._client.call(
            "env.get_camera_meta",
            kwargs={"camera_name": camera_name, **kwargs},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def render_camera(self, camera_name, **kwargs):
        return self._client.call(
            "env.render_camera",
            kwargs={"camera_name": camera_name, **kwargs},
            timeout_s=self._TIMEOUT_S["env.render_camera"],
        )

    def get_task_language(self) -> str | None:
        """Return the language description of the current task."""
        return self._client.call(
            "env.get_task_language", timeout_s=self._TIMEOUT_S["default"]
        )
