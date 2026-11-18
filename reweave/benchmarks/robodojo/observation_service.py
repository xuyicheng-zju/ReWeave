"""Episode-owned settled RGB-D snapshots and server-local inference."""

from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from reweave.benchmarks.robodojo.access import (
    CAMERAS,
    public_observation,
    robot_state_features,
)


class ObservationService:
    """Keep one current snapshot; archive only explicitly retained boundaries."""

    def __init__(self, directory, *, service_instance_id=None):
        self.directory = Path(directory) / "server_observations"
        self.service_instance_id = (
            service_instance_id
            or os.environ.get("REWEAVE_SERVICE_INSTANCE_ID")
            or uuid4().hex
        )
        self._current = None
        self._reference = None
        self._step = None
        self.initialized = False
        self._retained = {}
        self._masks = {}
        self.vla = self.sam3 = self.graspgen = None

    def configure_inference(self, configuration):
        from reweave.robots.components.graspgen_client import (
            GraspGenXClient,
            GraspGenXConfig,
            load_graspgen_config,
        )
        from reweave.robots.components.pi05_vla_client import Pi05VLAClient
        from reweave.robots.components.sam3_client import Sam3Client
        from reweave.utils.rpc import make_rpc_client

        unknown = set(configuration) - {
            "vla_endpoint",
            "sam3_endpoint",
            "graspgen_config",
        }
        if unknown:
            raise ValueError(
                f"unknown server inference configuration: {sorted(unknown)}"
            )
        if configuration.get("vla_endpoint"):
            self.vla = Pi05VLAClient(
                make_rpc_client(configuration["vla_endpoint"]), embodiment="robodojo"
            )
        if configuration.get("sam3_endpoint"):
            self.sam3 = Sam3Client(make_rpc_client(configuration["sam3_endpoint"]))
        if configuration.get("graspgen_config") is not None:
            config = configuration["graspgen_config"]
            if not isinstance(config, GraspGenXConfig):
                config = load_graspgen_config(config)
            self.graspgen = GraspGenXClient(config)

    @property
    def settled(self):
        return self._current is not None

    def invalidate(self):
        self._current = self._reference = self._step = None
        self._masks.clear()

    def publish(self, observation, step):
        if type(step) is not int or step < 0:
            raise ValueError("observation physical step must be a nonnegative integer")
        observation = copy.deepcopy(public_observation(observation))
        robot_state_features(observation)
        if set(observation["vision"]) != set(CAMERAS):
            raise ValueError("observation must contain all three RoboDojo cameras")
        for camera, value in observation["vision"].items():
            color = np.asarray(value.get("color"))
            if (
                color.ndim != 3
                or color.shape[2] != 3
                or color.dtype != np.uint8
                or min(color.shape[:2]) <= 0
            ):
                raise ValueError(f"invalid RGB for {camera}")
            for name, shape in (
                ("intrinsic_matrix", (3, 3)),
                ("extrinsic_matrix", (4, 4)),
            ):
                array = np.asarray(value.get(name))
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"invalid {camera} {name}")
            for key in ("depth", "distance_to_image_plane"):
                if (
                    value.get(key) is not None
                    and np.asarray(value[key]).shape != color.shape[:2]
                ):
                    raise ValueError(f"invalid {camera} {key} pixel grid")
        self.invalidate()
        self.initialized = True
        self._current, self._step = observation, step
        self._reference = f"{self.service_instance_id}:{step}:{uuid4().hex}"
        return self.compact()

    def _require(self, observation_ref):
        if type(observation_ref) is not str or not observation_ref:
            raise ValueError(
                "observation_ref must be a nonempty string, not an image or array"
            )
        if self._current is None or observation_ref != self._reference:
            raise ValueError("observation_ref is stale, foreign, or unsettled")
        return self._current

    def compact(self):
        obs = self._require(self._reference)
        vision = {}
        for camera, value in obs["vision"].items():
            vision[camera] = {
                "shape": list(np.asarray(value["color"]).shape),
                "color_dtype": str(np.asarray(value["color"]).dtype),
                "has_depth": any(
                    value.get(key) is not None
                    for key in ("depth", "distance_to_image_plane")
                ),
                "intrinsic_matrix": np.asarray(value["intrinsic_matrix"]).tolist(),
                "extrinsic_matrix": np.asarray(value["extrinsic_matrix"]).tolist(),
            }
        return {
            "observation_ref": self._reference,
            "service_instance_id": self.service_instance_id,
            "physical_step": self._step,
            "instruction": obs["instruction"],
            "state": robot_state_features(obs),
            "vision": vision,
        }

    def retain_observation(self, observation_ref):
        obs = self._require(observation_ref)
        if observation_ref in self._retained:
            return copy.deepcopy(self._retained[observation_ref])
        path = self.directory / observation_ref.rsplit(":", 1)[1]
        path.mkdir(parents=True, exist_ok=False)
        arrays, metadata = {}, self.compact()
        for camera, value in obs["vision"].items():
            metadata["vision"][camera]["arrays"] = {}
            for key in (
                "color",
                "depth",
                "distance_to_image_plane",
                "intrinsic_matrix",
                "extrinsic_matrix",
            ):
                if value.get(key) is not None:
                    name = f"{camera}__{key}"
                    array = np.asarray(value[key])
                    arrays[name] = array
                    metadata["vision"][camera]["arrays"][key] = {
                        "key": name,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                    }
        archive = path / "rgbd.npz"
        np.savez_compressed(archive, **arrays)
        metadata_path = path / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        receipt = {
            "observation_ref": observation_ref,
            "path": str(path),
            "archive_path": str(archive),
            "metadata_path": str(metadata_path),
            "rgb_paths": {},
        }
        with (self.directory / "manifest.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")
        self._retained[observation_ref] = receipt
        return copy.deepcopy(receipt)

    def read_sensor_snapshot(self, observation_ref):
        """Read an explicitly retained archive; never capture or change the scene."""
        if type(observation_ref) is not str or observation_ref not in self._retained:
            raise ValueError(
                "observation_ref has no explicitly retained sensor snapshot"
            )
        receipt = self._retained[observation_ref]
        return {
            "observation_ref": observation_ref,
            "archive": Path(receipt["archive_path"]).read_bytes(),
            "metadata": json.loads(Path(receipt["metadata_path"]).read_text()),
        }

    def read_rgb(self, observation_ref, cameras):
        if type(observation_ref) is not str:
            raise ValueError("observation_ref must be a string")
        if (
            not isinstance(cameras, (list, tuple))
            or not cameras
            or len(set(cameras)) != len(cameras)
            or any(
                type(camera) is not str or camera not in CAMERAS for camera in cameras
            )
        ):
            raise ValueError("cameras must name unique RoboDojo RGB cameras")
        if observation_ref == self._reference and self._current is not None:
            source = {
                camera: self._current["vision"][camera]["color"] for camera in cameras
            }
        elif observation_ref in self._retained:
            with np.load(
                self._retained[observation_ref]["archive_path"], allow_pickle=False
            ) as archive:
                source = {camera: archive[f"{camera}__color"] for camera in cameras}
        else:
            raise ValueError("observation_ref is stale, foreign, or not retained")
        result = {}
        for camera, color in source.items():
            buffer = io.BytesIO()
            Image.fromarray(color).save(buffer, format="PNG")
            result[camera] = buffer.getvalue()
            if observation_ref in self._retained:
                receipt = self._retained[observation_ref]
                path = Path(receipt["path"]) / f"{camera}.png"
                if not path.exists():
                    path.write_bytes(result[camera])
                receipt["rgb_paths"][camera] = str(path)
        return result

    def back_project_ref(self, observation_ref, row, col, camera="cam_head"):
        from reweave.benchmarks.robodojo.perception import back_project_observation

        obs = self._require(observation_ref)
        self.retain_observation(observation_ref)
        return back_project_observation(obs, row=row, col=col, camera=camera)

    def segment_ref(
        self, observation_ref, text_prompt, camera="cam_head", min_score=0.2
    ):
        from reweave.benchmarks.robodojo.perception import segment_observation

        obs = self._require(observation_ref)
        if self.sam3 is None:
            raise RuntimeError("server SAM3 is not configured")
        self.retain_observation(observation_ref)
        result, mask = segment_observation(
            obs,
            sam3_client=self.sam3,
            text_prompt=text_prompt,
            camera=camera,
            min_score=min_score,
        )
        if mask is not None:
            mask_ref = uuid4().hex
            self._masks[mask_ref] = (
                observation_ref,
                camera,
                text_prompt,
                result["score"],
                np.asarray(mask, dtype=bool).copy(),
            )
            result = {**result, "mask_ref": mask_ref}
        return result

    def locate_ref(
        self,
        observation_ref,
        text_prompt,
        camera="cam_head",
        min_score=0.2,
        support_z=None,
    ):
        from reweave.benchmarks.robodojo.perception import locate_observation

        obs = self._require(observation_ref)
        if self.sam3 is None:
            raise RuntimeError("server SAM3 is not configured")
        self.retain_observation(observation_ref)
        return {
            **locate_observation(
                obs,
                sam3_client=self.sam3,
                text_prompt=text_prompt,
                camera=camera,
                min_score=min_score,
                support_z=support_z,
            ),
            "observation_ref": observation_ref,
        }

    def grasps_ref(self, observation_ref, mask_ref, arm, max_candidates=5):
        from reweave.benchmarks.robodojo.perception import grasps_from_segment

        obs = self._require(observation_ref)
        if type(mask_ref) is not str or mask_ref not in self._masks:
            raise ValueError("mask_ref is stale or unknown")
        ref, camera, prompt, score, mask = self._masks[mask_ref]
        if ref != observation_ref:
            raise ValueError("mask_ref belongs to a different observation")
        if self.graspgen is None:
            raise RuntimeError("server GraspGen-X is not configured")
        return grasps_from_segment(
            obs,
            graspgen_client=self.graspgen,
            mask=mask,
            mask_score=score,
            text_prompt=prompt,
            arm=arm,
            camera=camera,
            max_candidates=max_candidates,
            observation_step=self._step,
        )

    def predict_ref(self, observation_ref, prompt, mode="eval"):
        from reweave.benchmarks.robodojo.tasks import validate_instruction

        obs = self._require(observation_ref)
        if self.vla is None:
            raise RuntimeError("server VLA is not configured")
        obs = {**obs, "instruction": validate_instruction(prompt)}
        self.retain_observation(observation_ref)
        prediction = self.vla.predict_action_batch(obs, mode=mode)
        return {
            "actions": prediction.actions[None],
            "forward_id": prediction.forward_id,
            "prediction_horizon_steps": prediction.prediction_horizon_steps,
            "execution_horizon_steps": prediction.execution_horizon_steps,
            "inference_elapsed_s": prediction.inference_elapsed_s,
        }
