# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""RPC server wrapping the Pi0.5 VLA."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Any

import numpy as np

from reweave.robots.components.vla_facade_base import BaseVLAFacade
from reweave.utils.config import (
    get_pi05_checkpoint_path,
    get_repo_root,
    get_rlinf_repo_path,
)
from reweave.utils.logging import get_logger

logger = get_logger("vla_server")

REWEAVE_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (REWEAVE_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))



def build_model_cfg(
    model_path: str,
    *,
    embodiment: str = "libero",
    norm_stats_path: str | None = None,
    repo_id: str | None = None,
) -> Any:
    """OmegaConf for ``rlinf.models.embodiment.openpi.get_model``."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 5,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 5,
                "num_steps": 5,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": None,
                "use_dsrl": False,
            },
        }
    )
    if embodiment == "robodojo":
        cfg.pi05 = True
        cfg.num_action_chunks = 50
        cfg.action_dim = 14
        cfg.openpi.update(
            {
                "config_name": "pi05_robodojo_arx_x5",
                "task": "eval",
                "model_action_dim": 32,
                "max_token_len": 200,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "discrete_state_input": True,
                "torch_compile": False,
                "num_images_in_input": 3,
                "action_chunk": 50,
                "action_env_dim": 14,
            }
        )
    elif embodiment != "libero":
        raise ValueError(f"unsupported traced Pi0.5 embodiment: {embodiment}")
    if norm_stats_path is not None or repo_id is not None:
        cfg.openpi_data = {}
        if norm_stats_path is not None:
            cfg.openpi_data.norm_stats_path = norm_stats_path
        if repo_id is not None:
            cfg.openpi_data.repo_id = repo_id
    return cfg




class Pi05VLAFacade(BaseVLAFacade):
    """Implements :class:`reweave.robots.components.pi05_vla_client.Pi05VLAClient` over a Pi0.5 model.

    Loads the model once at construction; each ``predict`` call runs one
    inference batch and returns its NumPy action array.
    """

    def __init__(
        self,
        model_path: str,
        *,
        embodiment: str = "libero",
        norm_stats_path: str | None = None,
        repo_id: str | None = None,
    ):
        import torch
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        self._embodiment = embodiment
        if embodiment == "libero":
            os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
        cfg = build_model_cfg(
            model_path=model_path,
            embodiment=embodiment,
            norm_stats_path=norm_stats_path,
            repo_id=repo_id,
        )
        t0 = time.time()
        logger.info("loading Pi0.5 (model_path=%s) ...", cfg["model_path"])
        self._model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
        self._inference_context = torch.no_grad
        model_config = getattr(self._model, "config", None)
        prediction_horizon = getattr(model_config, "action_horizon", None)
        if (
            isinstance(prediction_horizon, bool)
            or not isinstance(prediction_horizon, int)
            or prediction_horizon <= 0
        ):
            raise RuntimeError(
                "loaded Pi0.5 model does not expose a positive config.action_horizon"
            )
        self._prediction_horizon_steps = prediction_horizon
        logger.info("model ready in %.1fs", time.time() - t0)
        super().__init__()

    def predict(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(observation, dict):
            raise TypeError(f"Pi0.5 observation must be a mapping, got {observation!r}")
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise TypeError(f"Pi0.5 options must be a mapping, got {options!r}")
        unexpected_options = set(options) - {"mode"}
        if unexpected_options:
            raise ValueError(
                f"unsupported Pi0.5 options: {sorted(unexpected_options)!r}"
            )

        env_obs = dict(observation)
        main_images = np.asarray(env_obs.get("main_images"))
        if main_images.ndim != 4 or main_images.shape[-1] != 3:
            raise ValueError(
                f"main_images must be [B,H,W,3]; got shape {main_images.shape}"
            )
        env_obs["main_images"] = main_images.astype(np.uint8, copy=False)

        batch_size = main_images.shape[0]
        states = np.asarray(env_obs.get("states"), dtype=np.float32)
        if states.ndim != 2 or states.shape[0] != batch_size:
            raise ValueError(
                "states must be [B,state_dim] with the image batch size; "
                f"got shape {states.shape}"
            )
        if not np.isfinite(states).all():
            raise ValueError("Pi0.5 state must be finite")
        if self._embodiment == "robodojo" and states.shape[1] != 14:
            raise ValueError("RoboDojo policy requires 14 native joint state values")
        env_obs["states"] = states

        task_descriptions = env_obs.get("task_descriptions")
        if (
            not isinstance(task_descriptions, list)
            or len(task_descriptions) != batch_size
        ):
            raise ValueError("task_descriptions must contain one string per batch item")
        env_obs["task_descriptions"] = [str(value) for value in task_descriptions]

        for key in ("wrist_images", "extra_view_images"):
            view = env_obs.get(key)
            if view is None:
                if self._embodiment == "robodojo":
                    raise ValueError(f"RoboDojo policy requires {key}")
                env_obs[key] = None
                continue
            array = np.asarray(view)
            if array.ndim != 4 or array.shape[0] != batch_size or array.shape[-1] != 3:
                raise ValueError(f"{key} must be [B,H,W,3]; got shape {array.shape}")
            env_obs[key] = array.astype(np.uint8, copy=False)

        inference_started = time.perf_counter()
        with self._inference_context():
            actions, _ = self._model.predict_action_batch(
                env_obs,
                mode=options.get("mode", "eval"),
            )
        inference_elapsed_s = time.perf_counter() - inference_started
        actions_np = (
            actions.detach().cpu().numpy()
            if (
                hasattr(actions, "detach")
                and hasattr(actions, "cpu")
                and hasattr(actions, "numpy")
            )
            else np.asarray(actions)
        ).astype(np.float32)
        if (
            actions_np.ndim != 3
            or actions_np.shape[0] != batch_size
            or actions_np.shape[1] <= 0
            or actions_np.shape[2] <= 0
            or not np.isfinite(actions_np).all()
        ):
            raise RuntimeError(
                "Pi0.5 predict_action_batch must return finite "
                f"[B,K,D] actions; got {actions_np.shape}"
            )
        if self._embodiment == "robodojo" and actions_np.shape[2] != 14:
            raise RuntimeError("RoboDojo policy requires 14 joint action values")
        execution_horizon_steps = int(actions_np.shape[1])
        if execution_horizon_steps > self._prediction_horizon_steps:
            raise RuntimeError(
                "Pi0.5 returned more executable actions than its configured "
                "prediction horizon: "
                f"{execution_horizon_steps} > {self._prediction_horizon_steps}"
            )
        return {
            "actions": actions_np,
            "forward_id": uuid.uuid4().hex,
            "prediction_horizon_steps": self._prediction_horizon_steps,
            "execution_horizon_steps": execution_horizon_steps,
            "inference_elapsed_s": inference_elapsed_s,
        }




def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Pi0.5 checkpoint (defaults to PI05_CHECKPOINT_PATH env)",
    )
    p.add_argument("--embodiment", choices=["libero", "robodojo"], default="libero")
    p.add_argument("--norm-stats-path", default=os.environ.get("PI05_NORM_STATS_PATH"))
    p.add_argument(
        "--repo-id", default=None, help="Explicit OpenPI normalization dataset key"
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target

    model_path = args.model_path or get_pi05_checkpoint_path()
    if not model_path:
        raise RuntimeError(
            "PI05_CHECKPOINT_PATH is not set; provide the Pi0.5 checkpoint "
            "path via --model-path or the environment."
        )

    facade = Pi05VLAFacade(
        model_path=model_path,
        embodiment=args.embodiment,
        norm_stats_path=args.norm_stats_path,
        repo_id=args.repo_id,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
