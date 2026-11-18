# Copyright (c) 2026 RPent Contributors
# Modified for the ReWeave namespace.
"""Simulator-side semantic evidence reserved for the LIBERO critic.

The collector reads MuJoCo state and LIBERO's declared task predicates without
advancing or mutating the environment.  Its output is attached only to the
private execution trace; it is never inserted into planner or VLA images,
proprioception, prompts, or tool results.
"""

from __future__ import annotations

import re
import types
from collections.abc import Mapping
from typing import Any

import numpy as np

_FEATURE_TOKEN = re.compile(r"[^a-z0-9]+")
_LOGICAL_PREDICATES = {"and", "or", "not"}


def _feature_token(value: str) -> str:
    token = _FEATURE_TOKEN.sub("_", value.casefold()).strip("_")
    return token or "unnamed"


def _unwrap_robosuite_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current


def _id_to_name(model: Any, kind: str, identifier: int) -> str:
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        value = method(int(identifier))
        if value is not None:
            return str(value)
    native_model = getattr(model, "_model", model)
    try:
        import mujoco

        object_type = {
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "site": mujoco.mjtObj.mjOBJ_SITE,
        }[kind]
        value = mujoco.mj_id2name(
            native_model,
            object_type,
            int(identifier),
        )
        if value is not None:
            return str(value)
    except (ImportError, KeyError, TypeError):
        pass
    return f"{kind}_{int(identifier)}"


def _name_to_id(model: Any, kind: str, name: str) -> int:
    modern = getattr(model, kind, None)
    if callable(modern):
        try:
            return int(modern(name).id)
        except Exception:
            pass
    legacy = getattr(model, f"{kind}_name2id", None)
    if callable(legacy):
        try:
            return int(legacy(name))
        except Exception:
            pass
    raise KeyError(f"cannot resolve MuJoCo {kind} {name!r}")


def _flatten_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        result: set[str] = set()
        for item in value.values():
            result.update(_flatten_names(item))
        return result
    try:
        result = set()
        for item in value:
            result.update(_flatten_names(item))
        return result
    except TypeError:
        return set()


def _robot_geom_names(raw_env: Any, model: Any) -> set[str]:
    names: set[str] = set()
    for robot in getattr(raw_env, "robots", ()) or ():
        for owner in (
            robot,
            getattr(robot, "robot_model", None),
            getattr(robot, "gripper", None),
        ):
            if owner is None:
                continue
            for attribute in ("contact_geoms", "visual_geoms", "collision_geoms"):
                names.update(_flatten_names(getattr(owner, attribute, None)))
    for identifier in range(int(getattr(model, "ngeom", 0))):
        name = _id_to_name(model, "geom", identifier)
        lowered = name.casefold()
        if (
            lowered.startswith(("robot0_", "gripper0_", "panda_"))
            or "finger" in lowered
            or "hand_collision" in lowered
        ):
            names.add(name)
    return names


def _gripper_geom_names(raw_env: Any, model: Any) -> set[str]:
    names: set[str] = set()
    for robot in getattr(raw_env, "robots", ()) or ():
        gripper = getattr(robot, "gripper", None)
        for attribute in ("contact_geoms", "collision_geoms"):
            names.update(_flatten_names(getattr(gripper, attribute, None)))
    for identifier in range(int(getattr(model, "ngeom", 0))):
        name = _id_to_name(model, "geom", identifier)
        lowered = name.casefold()
        if "finger" in lowered or "gripper" in lowered:
            names.add(name)
    return names


def _contact_normal_force(simulator: Any, contact_index: int) -> float | None:
    native_model = getattr(simulator.model, "_model", simulator.model)
    native_data = getattr(simulator.data, "_data", simulator.data)
    wrench = np.zeros(6, dtype=np.float64)
    try:
        import mujoco

        mujoco.mj_contactForce(native_model, native_data, int(contact_index), wrench)
        return float(abs(wrench[0]))
    except Exception:
        pass
    try:
        from mujoco_py import functions

        functions.mj_contactForce(
            native_model,
            native_data,
            int(contact_index),
            wrench,
        )
        return float(abs(wrench[0]))
    except Exception:
        return None


def _descends_from(model: Any, body_id: int, ancestor_id: int) -> bool:
    parents = getattr(model, "body_parentid", None)
    if parents is None:
        return body_id == ancestor_id
    current = int(body_id)
    seen: set[int] = set()
    while current >= 0 and current not in seen:
        if current == ancestor_id:
            return True
        seen.add(current)
        parent = int(parents[current])
        if parent == current:
            break
        current = parent
    return False


def _position_and_quaternion(
    simulator: Any,
    *,
    kind: str,
    identifier: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    data = simulator.data
    if kind == "body":
        positions = getattr(data, "body_xpos", getattr(data, "xpos", None))
        quaternions = getattr(data, "body_xquat", getattr(data, "xquat", None))
    elif kind == "site":
        positions = getattr(data, "site_xpos", None)
        quaternions = None
    else:
        positions = getattr(data, "geom_xpos", None)
        quaternions = None
    if positions is None:
        raise KeyError(f"MuJoCo data has no {kind} positions")
    position = np.asarray(positions[int(identifier)], dtype=np.float64).reshape(-1)
    if position.size != 3 or not np.isfinite(position).all():
        raise ValueError(f"invalid {kind} position")
    quaternion = None
    if quaternions is not None:
        candidate = np.asarray(
            quaternions[int(identifier)],
            dtype=np.float64,
        ).reshape(-1)
        if candidate.size == 4 and np.isfinite(candidate).all():
            quaternion = candidate
    return position, quaternion


def _owner_name(owner: Any, fallback: str | None = None) -> str | None:
    for value in (fallback, getattr(owner, "name", None)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _entity_owners(raw_env: Any) -> dict[str, Any]:
    owners: dict[str, Any] = {}
    for attribute in ("objects_dict", "fixtures_dict"):
        value = getattr(raw_env, attribute, None)
        if isinstance(value, Mapping):
            for name, owner in value.items():
                resolved = _owner_name(owner, str(name))
                if resolved:
                    owners.setdefault(resolved, owner)
    for attribute in ("objects", "fixtures", "object_sites_dict"):
        value = getattr(raw_env, attribute, None)
        if isinstance(value, Mapping):
            items = value.items()
        else:
            try:
                items = ((None, owner) for owner in (value or ()))
            except TypeError:
                continue
        for fallback, owner in items:
            resolved = _owner_name(owner, str(fallback) if fallback else None)
            if resolved:
                owners.setdefault(resolved, owner)
    return owners


def _owner_contact_geoms(owner: Any) -> set[str]:
    names: set[str] = set()
    for attribute in ("contact_geoms", "collision_geoms"):
        names.update(_flatten_names(getattr(owner, attribute, None)))
    return names


def _owner_root_body(owner: Any) -> str | None:
    for attribute in ("root_body", "root_body_name"):
        value = getattr(owner, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _model_name_ids(model: Any, kind: str) -> dict[str, int]:
    count_attribute = "njnt" if kind == "joint" else f"n{kind}"
    return {
        name: identifier
        for identifier in range(int(getattr(model, count_attribute, 0)))
        if (name := _id_to_name(model, kind, identifier))
    }


def _resolve_entity_record(
    simulator: Any,
    *,
    name: str,
    owner: Any | None,
) -> dict[str, Any] | None:
    model = simulator.model
    candidates: list[tuple[str, str]] = []
    root_body = _owner_root_body(owner) if owner is not None else None
    if root_body:
        candidates.append(("body", root_body))
    candidates.extend((kind, name) for kind in ("body", "site", "geom"))
    for kind, candidate in candidates:
        try:
            identifier = _name_to_id(model, kind, candidate)
            position, quaternion = _position_and_quaternion(
                simulator,
                kind=kind,
                identifier=identifier,
            )
            break
        except (KeyError, ValueError, IndexError):
            continue
    else:
        normalized = _feature_token(name)
        identifier = -1
        kind = ""
        position = np.asarray([], dtype=np.float64)
        quaternion = None
        for candidate_kind in ("body", "site", "geom"):
            for candidate_name, candidate_id in _model_name_ids(
                model,
                candidate_kind,
            ).items():
                candidate_token = _feature_token(candidate_name)
                if candidate_token != normalized and not candidate_token.startswith(
                    f"{normalized}_"
                ):
                    continue
                try:
                    position, quaternion = _position_and_quaternion(
                        simulator,
                        kind=candidate_kind,
                        identifier=candidate_id,
                    )
                except (KeyError, ValueError, IndexError):
                    continue
                identifier = candidate_id
                kind = candidate_kind
                break
            if identifier >= 0:
                break
        if identifier < 0:
            return None

    geom_ids: set[int] = set()
    for geom_name in _owner_contact_geoms(owner) if owner is not None else ():
        try:
            geom_ids.add(_name_to_id(model, "geom", geom_name))
        except KeyError:
            continue
    if kind == "body":
        geom_body_ids = getattr(model, "geom_bodyid", None)
        if geom_body_ids is not None:
            for geom_id in range(int(getattr(model, "ngeom", 0))):
                if _descends_from(
                    model,
                    int(geom_body_ids[geom_id]),
                    int(identifier),
                ):
                    geom_ids.add(geom_id)
    elif kind == "geom":
        geom_ids.add(int(identifier))
    return {
        "name": name,
        "position": position,
        "quaternion": quaternion,
        "geom_ids": geom_ids,
        "geom_names": {_id_to_name(model, "geom", geom_id) for geom_id in geom_ids},
    }


def _goal_atoms(value: Any) -> list[tuple[str, ...]]:
    atoms: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        predicate = value.get("predicate") or value.get("name")
        arguments = value.get("arguments") or value.get("args")
        if isinstance(predicate, str) and isinstance(arguments, (list, tuple)):
            atoms.append((predicate, *(str(item) for item in arguments)))
        else:
            for item in value.values():
                atoms.extend(_goal_atoms(item))
        return atoms
    if not isinstance(value, (list, tuple)) or not value:
        return atoms
    first = value[0]
    if isinstance(first, str):
        predicate = first.casefold()
        if predicate in _LOGICAL_PREDICATES:
            for item in value[1:]:
                atoms.extend(_goal_atoms(item))
        elif all(not isinstance(item, (list, tuple, Mapping)) for item in value[1:]):
            atoms.append(tuple(str(item) for item in value))
        else:
            for item in value[1:]:
                atoms.extend(_goal_atoms(item))
        return atoms
    for item in value:
        atoms.extend(_goal_atoms(item))
    return atoms


def _goal_state(raw_env: Any) -> Any:
    for owner in (raw_env, getattr(raw_env, "unwrapped", None)):
        if owner is None:
            continue
        for attribute in ("parsed_problem", "_parsed_problem"):
            problem = getattr(owner, attribute, None)
            if isinstance(problem, Mapping):
                for key in ("goal_state", "goal", "goals"):
                    if key in problem:
                        return problem[key]
    return None


def _predicate_value(raw_env: Any, atom: tuple[str, ...]) -> bool | None:
    evaluator = getattr(raw_env, "_eval_predicate", None)
    if not callable(evaluator):
        return None
    for value in (list(atom), tuple(atom)):
        try:
            return bool(evaluator(value))
        except Exception:
            continue
    return None


def _eef_position(raw_env: Any, simulator: Any) -> np.ndarray | None:
    for owner in (raw_env, *(getattr(raw_env, "robots", ()) or ())):
        for attribute in ("_eef_xpos", "eef_xpos"):
            value = getattr(owner, attribute, None)
            if value is None:
                continue
            candidate = np.asarray(value, dtype=np.float64).reshape(-1)
            if candidate.size == 3 and np.isfinite(candidate).all():
                return candidate
    for name in ("gripper0_grip_site", "robot0_grip_site", "eef_site"):
        try:
            identifier = _name_to_id(simulator.model, "site", name)
            position, _ = _position_and_quaternion(
                simulator,
                kind="site",
                identifier=identifier,
            )
            return position
        except (KeyError, ValueError, IndexError):
            continue
    return None


def _contact_summary(
    raw_env: Any,
    entities: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    simulator = raw_env.sim
    model = simulator.model
    robot_names = _robot_geom_names(raw_env, model)
    gripper_names = _gripper_geom_names(raw_env, model)
    robot_ids = {
        identifier
        for identifier in range(int(getattr(model, "ngeom", 0)))
        if _id_to_name(model, "geom", identifier) in robot_names
    }
    gripper_ids = {
        identifier
        for identifier in range(int(getattr(model, "ngeom", 0)))
        if _id_to_name(model, "geom", identifier) in gripper_names
    }
    pairs: list[tuple[int, int]] = []
    robot_contacts = 0
    gripper_contacts = 0
    max_normal_force = 0.0
    force_available = False
    for contact_index in range(int(getattr(simulator.data, "ncon", 0))):
        contact = simulator.data.contact[contact_index]
        first, second = int(contact.geom1), int(contact.geom2)
        pairs.append((first, second))
        robot_contacts += int(first in robot_ids or second in robot_ids)
        gripper_contacts += int(first in gripper_ids or second in gripper_ids)
        force = _contact_normal_force(simulator, contact_index)
        if force is not None:
            force_available = True
            max_normal_force = max(max_normal_force, force)

    per_entity: dict[str, dict[str, bool]] = {}
    for name, record in entities.items():
        entity_ids = set(record["geom_ids"])
        robot_contact = False
        gripper_contact_ids: set[int] = set()
        for first, second in pairs:
            robot_contact |= (first in entity_ids and second in robot_ids) or (
                second in entity_ids and first in robot_ids
            )
            if first in entity_ids and second in gripper_ids:
                gripper_contact_ids.add(second)
            if second in entity_ids and first in gripper_ids:
                gripper_contact_ids.add(first)
        grasped: bool | None = None
        checker = getattr(raw_env, "_check_grasp", None)
        robots = tuple(getattr(raw_env, "robots", ()) or ())
        if callable(checker) and robots and record["geom_names"]:
            try:
                grasped = bool(
                    checker(
                        getattr(robots[0], "gripper", None),
                        sorted(record["geom_names"]),
                    )
                )
            except Exception:
                grasped = None
        if grasped is None:
            grasped = len(gripper_contact_ids) >= 2
        per_entity[name] = {
            "robot_contact": robot_contact,
            "gripper_contact": bool(gripper_contact_ids),
            "grasped": grasped,
        }
    return {
        "robot_count": robot_contacts,
        "gripper_count": gripper_contacts,
        "force_available": force_available,
        "max_normal_force_n": max_normal_force,
        "per_entity": per_entity,
    }


def _write_pose_features(
    result: dict[str, Any],
    prefix: str,
    record: Mapping[str, Any],
) -> None:
    for axis, value in zip(
        ("x", "y", "z"),
        np.asarray(record["position"], dtype=np.float64),
        strict=True,
    ):
        result[f"{prefix}.position.{axis}"] = float(value)
    quaternion = record.get("quaternion")
    if quaternion is not None:
        for axis, value in zip(
            ("w", "x", "y", "z"),
            np.asarray(quaternion, dtype=np.float64),
            strict=True,
        ):
            result[f"{prefix}.orientation.{axis}"] = float(value)


def _write_joint_features(result: dict[str, Any], simulator: Any) -> None:
    model = simulator.model
    data = simulator.data
    count = 0
    for identifier in range(int(getattr(model, "njnt", 0))):
        if int(model.jnt_type[identifier]) not in (2, 3):
            continue
        try:
            position = float(data.qpos[int(model.jnt_qposadr[identifier])])
            joint_range = np.asarray(
                model.jnt_range[identifier],
                dtype=np.float64,
            ).reshape(-1)
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(position) or joint_range.size != 2:
            continue
        name = _id_to_name(model, "joint", identifier)
        prefix = f"privileged.joint.{_feature_token(name)}"
        lower, upper = float(joint_range[0]), float(joint_range[1])
        result[f"{prefix}.name"] = name
        result[f"{prefix}.unit"] = (
            "m" if int(model.jnt_type[identifier]) == 2 else "rad"
        )
        result[f"{prefix}.position"] = position
        result[f"{prefix}.range.lower"] = lower
        result[f"{prefix}.range.upper"] = upper
        result[f"{prefix}.distance_to_lower"] = abs(position - lower)
        result[f"{prefix}.distance_to_upper"] = abs(upper - position)
        span = upper - lower
        if bool(model.jnt_limited[identifier]) and np.isfinite(span) and span > 1e-12:
            result[f"{prefix}.fraction_from_lower_limit"] = (position - lower) / span
        count += 1
    result["privileged.joint.count"] = count


def collect_privileged_critic_state(outer_env: Any) -> dict[str, Any]:
    """Read task semantics, object state, contacts and joints for the critic."""
    raw_env = _unwrap_robosuite_env(outer_env)
    simulator = getattr(raw_env, "sim", None)
    if simulator is None:
        raise RuntimeError("LIBERO privileged critic requires the MuJoCo simulator")

    owners = _entity_owners(raw_env)
    atoms = _goal_atoms(_goal_state(raw_env))
    goal_names = {
        argument
        for atom in atoms
        for argument in atom[1:]
        if argument and argument.casefold() not in {"true", "false"}
    }
    entities: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys((*owners.keys(), *sorted(goal_names))):
        record = _resolve_entity_record(
            simulator,
            name=name,
            owner=owners.get(name),
        )
        if record is not None:
            entities[name] = record

    eef = _eef_position(raw_env, simulator)
    contacts = _contact_summary(raw_env, entities)
    result: dict[str, Any] = {
        "privileged.available": True,
        "privileged.task.semantic_available": bool(atoms and entities),
        "privileged.task.goal.predicate_count": len(atoms),
        "privileged.contact.robot.count": int(contacts["robot_count"]),
        "privileged.contact.gripper.count": int(contacts["gripper_count"]),
        "privileged.contact.force_available": bool(contacts["force_available"]),
        "privileged.contact.max_normal_force_n": float(contacts["max_normal_force_n"]),
    }
    _write_joint_features(result, simulator)

    predicate_values = [_predicate_value(raw_env, atom) for atom in atoms]
    known_values = [value for value in predicate_values if value is not None]
    result["privileged.task.goal.evaluable_count"] = len(known_values)
    result["privileged.task.goal.satisfied_count"] = sum(known_values)
    result["privileged.task.goal.predicate_fraction_available"] = bool(known_values)
    if known_values:
        result["privileged.task.goal.predicate_fraction"] = float(
            sum(known_values) / len(known_values)
        )
    for index, (atom, value) in enumerate(zip(atoms, predicate_values, strict=True)):
        result[f"privileged.task.goal.predicate.{index}.name"] = atom[0]
        if value is not None:
            result[f"privileged.task.goal.predicate.{index}.satisfied"] = value

    entity_contact = contacts["per_entity"]
    for name, record in entities.items():
        prefix = f"privileged.entity.{_feature_token(name)}"
        result[f"{prefix}.name"] = name
        _write_pose_features(result, prefix, record)
        if eef is not None:
            result[f"{prefix}.distance_to_eef_m"] = float(
                np.linalg.norm(np.asarray(record["position"]) - eef)
            )
        for key, value in entity_contact[name].items():
            result[f"{prefix}.{key}"] = bool(value)

    resolved_atoms: list[tuple[int, tuple[str, ...]]] = []
    for index, atom in enumerate(atoms):
        resolved = [argument for argument in atom[1:] if argument in entities]
        if len(resolved) >= 2:
            resolved_atoms.append((index, (atom[0], *resolved[:2])))
    if not resolved_atoms:
        return result

    primary_index, primary_atom = next(
        (
            (index, atom)
            for index, atom in resolved_atoms
            if predicate_values[index] is False
        ),
        resolved_atoms[0],
    )
    relation, manipulated_name, target_name = primary_atom
    manipulated = entities[manipulated_name]
    target = entities[target_name]
    manipulated_position = np.asarray(manipulated["position"], dtype=np.float64)
    target_position = np.asarray(target["position"], dtype=np.float64)
    offset = target_position - manipulated_position
    result.update(
        {
            "privileged.task.primary_relation": relation,
            "privileged.task.manipulated_object.name": manipulated_name,
            "privileged.task.target.name": target_name,
            "privileged.task.target_identity_confidence": 1.0,
            "privileged.task.manipulated_object.distance_to_target_m": float(
                np.linalg.norm(offset)
            ),
            "privileged.task.manipulated_object.target_offset.x": float(offset[0]),
            "privileged.task.manipulated_object.target_offset.y": float(offset[1]),
            "privileged.task.manipulated_object.target_offset.z": float(offset[2]),
        }
    )
    _write_pose_features(result, "privileged.task.manipulated_object", manipulated)
    _write_pose_features(result, "privileged.task.target", target)
    for key, value in entity_contact[manipulated_name].items():
        result[f"privileged.task.manipulated_object.{key}"] = bool(value)
    for key, value in entity_contact[target_name].items():
        result[f"privileged.task.target.{key}"] = bool(value)
    if eef is not None:
        result["privileged.task.manipulated_object.distance_to_eef_m"] = float(
            np.linalg.norm(manipulated_position - eef)
        )
        result["privileged.task.target.distance_to_eef_m"] = float(
            np.linalg.norm(target_position - eef)
        )
    relation_value = predicate_values[primary_index]
    if relation_value is not None:
        result["privileged.task.primary_relation_satisfied"] = relation_value
        if relation.casefold() in {"in", "inside", "on", "ontop", "on_top"}:
            result["privileged.task.manipulated_object.in_target"] = relation_value

    movable_names = set(
        getattr(raw_env, "objects_dict", {}).keys()
        if isinstance(getattr(raw_env, "objects_dict", None), Mapping)
        else ()
    )
    comparison_names = movable_names | {target_name}
    peer_distances = sorted(
        (
            float(
                np.linalg.norm(np.asarray(record["position"]) - manipulated_position)
            ),
            name,
        )
        for name, record in entities.items()
        if name != manipulated_name and name in comparison_names
    )
    if peer_distances:
        result["privileged.task.nearest_entity.name"] = peer_distances[0][1]
        result["privileged.task.nearest_entity.distance_m"] = peer_distances[0][0]
        result["privileged.task.target.is_nearest_entity"] = (
            peer_distances[0][1] == target_name
        )
        result["privileged.task.target.distance_rank"] = 1 + next(
            index
            for index, (_, name) in enumerate(peer_distances)
            if name == target_name
        )
    return result


def _enrich_temporal_state(
    state: dict[str, Any],
    history: dict[str, Any],
) -> dict[str, Any]:
    grasp_key = "privileged.task.manipulated_object.grasped"
    if grasp_key not in state:
        history.clear()
        return state
    grasped = bool(state[grasp_key])
    previously_grasped = bool(history.get("grasped", False))
    ever_grasped = bool(history.get("ever_grasped", False) or grasped)
    released_now = previously_grasped and not grasped
    ever_released = bool(history.get("ever_released", False) or released_now)
    state["privileged.task.manipulated_object.ever_grasped"] = ever_grasped
    state["privileged.task.manipulated_object.retained"] = ever_grasped and grasped
    state["privileged.task.manipulated_object.released_now"] = released_now
    state["privileged.task.manipulated_object.ever_released"] = ever_released
    distance = state.get("privileged.task.manipulated_object.distance_to_target_m")
    previous_distance = history.get("target_distance_m")
    progress_available = isinstance(distance, (int, float)) and isinstance(
        previous_distance,
        (int, float),
    )
    state["privileged.task.manipulated_object.target_progress_available"] = (
        progress_available
    )
    if progress_available:
        delta = float(distance) - float(previous_distance)
        state["privileged.task.manipulated_object.target_distance_delta_m"] = delta
        state["privileged.task.manipulated_object.target_progress_m"] = -delta

    history.update(
        {
            "grasped": grasped,
            "ever_grasped": ever_grasped,
            "ever_released": ever_released,
        }
    )
    if isinstance(distance, (int, float)):
        history["target_distance_m"] = float(distance)
    else:
        history.pop("target_distance_m", None)
    return state


def _bound_critic_state(
    self: Any,
    *,
    reset_tracker: bool = False,
) -> dict[str, Any]:
    if reset_tracker or not hasattr(self, "_harness_critic_history"):
        self._harness_critic_history = {}
    try:
        state = collect_privileged_critic_state(self)
    except Exception:
        self._harness_critic_history.clear()
        raise
    return _enrich_temporal_state(state, self._harness_critic_history)


def attach_privileged_critic_state(env: Any) -> None:
    """Attach the read-only semantic sensor to an existing worker environment."""
    env.harness_privileged_critic_state = types.MethodType(_bound_critic_state, env)


__all__ = [
    "collect_privileged_critic_state",
    "attach_privileged_critic_state",
]
