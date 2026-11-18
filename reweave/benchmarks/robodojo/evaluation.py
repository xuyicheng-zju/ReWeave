"""Host-only terminal metrics matching RoboDojo's native evaluation writer."""

from __future__ import annotations

import math


def terminal_metrics(env) -> dict:
    """Read the native success gate and process score without a robot action.

    Native eval_env.run_eval gives successful episodes 1.0 and otherwise uses
    reward_manager.get_score()[env_idx] / 100.0 when the task defines get_score.
    The task's score checks must be registered immediately after scene reset.
    Invalid scenes have no benchmark verdict; callers retain unknown outcomes
    independently even when a server-side final metric can be collected.
    """
    if 0 in env.unstable_envs:
        return {"valid": False, "success": None, "score": None}
    success = bool(env.is_success(env_idx=0))
    score = (
        1.0
        if success
        else float(env.reward_manager.get_score()[0]) / 100.0
        if hasattr(env, "get_score")
        else 0.0
    )
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("native RoboDojo score must be within [0, 100]")
    return {"valid": True, "success": success, "score": score}
