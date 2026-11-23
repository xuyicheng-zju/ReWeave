# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.

"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""

from __future__ import annotations

import copy
import json
from functools import partial
from pathlib import Path
from typing import Any

from reweave.benchmarks.libero import tools as libero_tools
from reweave.dashboard.events import DashboardEventSink
from reweave.memory.corpus import ReadOnlyCorpus
from reweave.robots.active_execution import ActiveExecution
from reweave.robots.execution_trace import ExecutionTrace
from reweave.session.base import EnvState
from reweave.tools.toolkit import Toolkit, readonly
from reweave.utils.logging import get_logger, get_output_dir
from reweave.utils.persistence import atomic_text

logger = get_logger("libero_toolkit")


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO robot."""

    def native_image_contract(self) -> dict:
        return {
            "observation": 3,
            "reads": {
                **super().native_image_contract()["reads"],
                "view_env_state": 3,
                "segment": 1,
                "read_image": 1,
                "view_camera_meta": 0,
                "back_project": 0,
                "generate_grasps": 0,
                "read_text_file": 0,
                "list_dir": 0,
            },
        }

    def harness_source_files(self) -> dict[str, str]:
        """Include the control mapping behind the physical tool implementation."""
        sources = super().harness_source_files()
        repository = Path(__file__).resolve().parents[3]
        for name in ("cartesian_control.py", "env_client.py", "env_server.py"):
            path = Path(__file__).resolve().parent / name
            sources[path.relative_to(repository).as_posix()] = path.read_text()
        primitives = getattr(self, "_primitives", None)
        if getattr(primitives, "_graspgen_client", None) is not None:
            path = repository / "reweave/robots/components/graspgen_client.py"
            sources[path.relative_to(repository).as_posix()] = path.read_text()
        return sources

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: ReadOnlyCorpus,
        mode: str = "evaluation",
        attempts_per_session: int = 0,
        state_output_dir: Path | str | None = None,
    ) -> None:
        toolkit_kwargs = dict(primitives_kwargs)
        frozen_bundle = toolkit_kwargs.pop("frozen_bundle", False)
        filesystem_tools_enabled = toolkit_kwargs.pop("filesystem_tools_enabled", True)
        if frozen_bundle:
            filesystem_tools_enabled = False
        self._state_output_dir = Path(state_output_dir or get_output_dir())
        state = EnvState(self._state_output_dir)
        self._execution_trace = ExecutionTrace(
            self._state_output_dir / "execution_trace.jsonl"
        )
        self._execution_controller = ActiveExecution(self._execution_trace)
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
            active_execution=self._execution_controller,
            filesystem_tools_enabled=filesystem_tools_enabled,
        )
        if mode not in {"evaluation", "exploration"}:
            raise ValueError(f"unsupported LIBERO toolkit mode: {mode!r}")
        self._mode = mode
        self._solved: bool = False
        self._attempt: int = 1
        # Bound the resettable attempts owned by this planner session.
        self._attempts_per_session: int = max(0, int(attempts_per_session))
        self._session_attempt: int = 1
        self.init_primitives(primitives_kwargs=toolkit_kwargs)
        self._register_libero_tools()

    def _register_libero_tools(self) -> None:
        # These read-only handlers need the run's EnvState bound in. Every
        # other spec binds to a primitive-driver method and captures state by
        # default unless that method is explicitly marked @readonly.
        state_handlers = {
            "view_env_state": partial(libero_tools.view_env_state, state=self._state),
            "view_camera_meta": partial(
                libero_tools.view_camera_meta, state=self._state
            ),
            "back_project": partial(libero_tools.back_project, state=self._state),
            "segment": partial(self._primitives.segment, state=self._state),
            "release": partial(self._primitives.release, state=self._state),
        }
        graspgen_enabled = (
            getattr(self._primitives, "_graspgen_client", None) is not None
        )
        if graspgen_enabled:
            state_handlers["generate_grasps"] = partial(
                self._primitives.generate_grasps, state=self._state
            )
        # The parent planner always sees the upstream Harness tool contract.
        # Runtime-only controls belong to generated code/capabilities; exposing
        # them here would change H0 before any experience had been learned.
        specs = copy.deepcopy(libero_tools.TOOLS_SPEC)
        for spec in specs:
            name = spec["name"]
            if name == "generate_grasps" and not graspgen_enabled:
                continue
            if name == "reset" and self._mode != "exploration":
                continue
            if name in state_handlers:
                handler = state_handlers[name]
            else:
                handler = getattr(self._primitives, name, None)
                if handler is None:
                    raise ValueError(f"LIBERO tool {name!r} has no implementation")
            self.add_tool(name, spec, handler)
        if self._mode == "exploration":
            reset_spec = next(spec for spec in specs if spec["name"] == "reset")
            self.add_tool("reset", reset_spec, self._reset_episode)
            finish_spec, finish_handler = self._tools["finish"]
            self.add_tool(
                "finish", finish_spec, partial(self._guarded_finish, finish_handler)
            )

    @readonly
    def _guarded_finish(self, inner: Any, **kwargs: Any) -> dict[str, Any]:
        """Refuse to end an unsolved session while attempts remain."""
        budget = self._attempts_per_session
        if budget and not self.solved() and self._session_attempt < budget:
            remaining = budget - self._session_attempt
            return {
                "error": "finish refused",
                "reason": (
                    f"This session has {remaining} of its {budget} attempts left "
                    "and the task is not solved. Archive this attempt, call "
                    "`reset`, and try another approach."
                ),
            }
        return inner(**kwargs)

    def _reset_episode(self, reason: str) -> dict[str, Any]:
        """Restart the episode while preserving the full exploration trace."""
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            return {
                "error": "reset refused",
                "reason": (
                    f"This session's attempt budget is spent ({budget} attempts). "
                    "Archive the attempt, update the handoff notes, and call "
                    "`finish` so the next session can continue."
                ),
            }
        self._attempt += 1
        self._session_attempt += 1
        result = self._primitives.reset_episode(reason=reason)
        result["attempt"] = self._attempt
        result["notice"] = (
            f"Episode restarted; this is attempt {self._attempt}. The original "
            "layout was restored. Re-run perception before acting."
        )
        return result

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()
        record = libero_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
        )
        self._solved |= record.terminated
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
            except Exception as e:
                logger.warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )
        out = libero_tools.view_env_state(record.step_idx, state=self._state)
        out["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted"):
            out.update(result)
        return out

    def bind_model_budget(self, budget) -> None:
        """Host-only binding; every VLA/SAM3 request charges the controller's allowance."""
        self._primitives.model_budget = budget

    def init_primitives(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        self._state.reset()

        primitives = libero_tools.LiberoPrimitives(
            check_cancelled=self.raise_if_cancelled,
            execution_trace=self._execution_trace,
            active_execution=self._execution_controller,
            **primitives_kwargs,
        )
        primitives.start_recording()
        primitives.reset()
        self._action_frame_cursor = primitives.recorded_frame_count()
        record = libero_tools.dump_state(primitives, self._state, log=None)
        self._primitives = primitives
        graspgen_client = getattr(primitives, "_graspgen_client", None)
        if graspgen_client is not None:
            atomic_text(
                self._state_output_dir / "graspgen_config.json",
                json.dumps(graspgen_client.config.model_dump(mode="json"), indent=2)
                + "\n",
            )
        self._solved |= record.terminated
        self._publish_step(record)

    def discard_active_vla_suffix(self) -> int:
        """Drop an agent-side VLA suffix after failure or cancellation."""
        return self._primitives.discard_active_vla_suffix()

    def leftover_vla_suffix_steps(self) -> int:
        """Return leftover actions the host failed to dump on interrupt."""
        return self._primitives.leftover_vla_suffix_steps()

    def control_step_budget(self) -> dict[str, Any]:
        """Return the live LIBERO control-step horizon for generated code."""
        return self._primitives.control_step_budget()

    def close(self) -> None:
        """Flush the agent-side video buffer through ``EnvState``."""
        super().close()
        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            # The runner is in the cleanup path; never let a video save
            # abort it.
            logger.warning(f"failed to save episode video: {e}")

    def solved(self) -> bool:
        """Return whether this run has completed the task."""
        return self._solved

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(
            self._state, recipe_tag, output_dir=get_output_dir()
        )
