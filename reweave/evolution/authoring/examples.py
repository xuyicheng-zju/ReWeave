# Modified for the ReWeave namespace.
"""Executable W/C/L illustrations: evidence -> mechanism -> falsifiable change."""

import json

from reweave.evolution.bundle import (
    ContextPackage,
    LoopPolicy,
    PluginCase,
    ProgramLimits,
    PythonFile,
    PythonProgram,
    TextSkill,
    WorkflowSpec,
)


def author_examples() -> dict:
    """Illustrative fixtures, never claimed to be evidence from the current task."""
    workflow = WorkflowSpec(
        workflow_id="release_checked",
        description="Release and read a fresh public observation; report interrupted execution as a local failure.",
        applicability="Planner has decided release is appropriate and needs a post-action observation.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        permissions=("release", "view_env_state"),
        observations=("view_env_state",),
        limits=ProgramLimits(
            runtime_calls=2, physical_calls=1, model_calls=0, wall_time_s=30
        ),
        program=PythonProgram(
            entrypoint="workflow:run",
            files=(
                PythonFile(
                    path="workflow.py",
                    source="""def run(runtime, context):
    result = runtime.act("release", {"max_steps": 2})
    if result.get("error") or result.get("truncated") or result.get("interrupted"):
        return {"status": "failed", "summary": "Release did not finish normally", "output": result}
    observation = runtime.observe("view_env_state", {"step": -1})
    output = {"action": result, "observation": observation}
    if observation.get("error") or observation.get("observation_errors"):
        return {"status": "failed", "summary": "Post-action observation failed", "output": output}
    return {"status": "completed", "summary": "Release returned; inspect the post-action evidence", "output": output}
""",
                ),
            ),
        ),
        contract_cases=(
            {
                "arguments": {},
                "observation": {},
                "responses": [
                    {
                        "operation": "release",
                        "arguments": {"max_steps": 2},
                        "result": {
                            "capability_result": {"steps_used": 2},
                            "terminated": False,
                            "truncated": False,
                            "interrupted": False,
                            "error": None,
                        },
                    },
                    {
                        "operation": "view_env_state",
                        "arguments": {"step": -1},
                        "result": {"state": {"fixture": "fresh observation"}},
                    },
                ],
                "expected_status": "completed",
            },
            {
                "arguments": {},
                "observation": {},
                "responses": [
                    {
                        "operation": "release",
                        "arguments": {"max_steps": 2},
                        "result": {
                            "capability_result": {"interrupted": True},
                            "terminated": False,
                            "truncated": False,
                            "interrupted": True,
                            "error": None,
                        },
                    }
                ],
                "expected_status": "failed",
            },
        ),
    )
    context = ContextPackage(
        program=PythonProgram(
            entrypoint="context:build",
            files=(
                PythonFile(
                    path="context.py",
                    source="""import json

def build(runtime, context):
    # Promote endpoint fields, not the image-bearing observation envelope.
    # The complete feedback and images already travel in messages.
    observation = context["observation"]
    result = observation.get("capability_result") or {}
    endpoint = {
        key: observation[key]
        for key in ("step", "terminated", "truncated") if key in observation
    }
    fields = ("target_reached", "stop_reason", "final_dist_m",
              "final_pitch_error", "final_yaw_error", "steps_used", "max_steps")
    endpoint.update({key: result[key] for key in fields if key in result})
    skills = context["assets"]["skills"]
    guidance = next(skill["text"] for skill in skills if skill["skill_id"] == "endpoint_evidence")
    return {"instructions": context["base_instructions"] + "\\n\\n" + guidance + "\\nCurrent endpoint evidence: " + json.dumps(endpoint),
            "messages": context["messages"], "tools": context["tools"], "state": context["state"]}
""",
                ),
            ),
        ),
        skills=(
            TextSkill(
                scope="global",
                skill_id="endpoint_evidence",
                text="""# Interpret local completion

A physical tool returning normally does not establish that its target was reached.
Inspect `target_reached`, `stop_reason`, and the current observation when those
fields are available. If the endpoint was not reached, identify the remaining
error before requesting more motion. A workflow's `completed` status describes
its local program boundary. Use the environment-owned verdict for task success.
""",
            ),
        ),
    )
    loop = LoopPolicy(
        program=PythonProgram(
            entrypoint="policy:run",
            files=(
                PythonFile(
                    path="policy.py",
                    source="""def run(runtime, context):
    checked_step = None
    while True:
        view = runtime.snapshot()
        if view["terminal"]:
            return {"summary": "Environment ended"}
        observation = view["observation"]
        result = observation.get("capability_result") or {}
        step = observation.get("step")
        # LIBERO view_env_state exposes the captured state identity as step;
        # observation remains the last physical result across read-only tools.
        if result.get("target_reached") is False and step is None:
            raise ValueError("Unmet endpoint feedback is missing its state step")
        if result.get("target_reached") is False and step != checked_step:
            checked_step = step
            response = runtime.plan(
                "The previous physical endpoint was not reached. Confirm the next "
                "evidence-gathering or recovery call before continuing."
            )
        else:
            response = runtime.plan()
        if not response["tool_calls"]:
            continue
        calls = response["tool_calls"]
        for index, call in enumerate(calls):
            outcome = runtime.execute(call)
            if runtime.snapshot()["terminal"]:
                return {"summary": "Environment ended"}
            if (call["name"].startswith("workflow_") and
                    isinstance(outcome, dict) and outcome.get("status") == "failed"):
                for skipped in calls[index + 1:]:
                    runtime.feedback(skipped["id"], {
                        "status": "not_executed",
                        "reason": "A preceding Workflow failed; replan from its feedback.",
                    })
                break
""",
                ),
            ),
        )
    )
    context_input = {
        "base_instructions": "Complete the stated task.",
        "assets": {
            "skills": [skill.model_dump(mode="json") for skill in context.skills]
        },
        "observation": {"capability_result": {"target_reached": False}},
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {"part_kind": "user-prompt", "content": "Inspect the endpoint."}
                ],
            }
        ],
        "tools": [{"name": "view_env_state", "input_schema": {"type": "object"}}],
        "state": {"prior": "retained"},
    }
    context = context.model_copy(
        update={
            "contract_cases": (
                PluginCase(
                    input=context_input,
                    expected={
                        "instructions": context_input["base_instructions"]
                        + "\n\n"
                        + context.skills[0].text
                        + "\nCurrent endpoint evidence: "
                        + json.dumps(context_input["observation"]["capability_result"]),
                        "messages": context_input["messages"],
                        "tools": context_input["tools"],
                        "state": context_input["state"],
                    },
                ),
            )
        }
    )
    loop = loop.model_copy(update={"contract_cases": _loop_cases()})
    return {
        "status": "synthetic teaching examples; actual proposal must cite its own development evidence",
        "W": {
            "hypothesis": "A repeated action sequence omits its final observation. Test this against cases where the observation is already sufficient.",
            "prediction": "One local callable returns current evidence and stops on interruption; it cannot claim official success.",
            "workflow": workflow.model_dump(mode="json"),
        },
        "C": {
            "hypothesis": "Feedback contains the necessary fields, but the controller interprets normal return as target completion.",
            "prediction": "The context builder presents endpoint evidence and the relevant skill at the model request, with the same physical tools and loop.",
            "context": context.model_dump(mode="json"),
        },
        "L": {
            "hypothesis": "The Planner is not explicitly asked to reassess an unmet physical endpoint at the next decision boundary.",
            "prediction": "The Loop marks each new unmet endpoint in the next plan objective while leaving the Planner responsible for the executable recovery call.",
            "loop": loop.model_dump(mode="json"),
        },
    }


def _loop_cases() -> tuple[PluginCase, ...]:
    """Ordered production RPC fixtures, with no alternate runtime implementation."""

    def response(method, name, arguments, result):
        return {
            "method": method,
            "name": name,
            "arguments": arguments,
            "result": result,
        }

    def snapshot(step, reached=False, terminal=False):
        return response(
            "control",
            "episode:snapshot",
            {},
            {
                "terminal": terminal,
                "observation": {
                    "step": step,
                    "capability_result": {"target_reached": reached},
                },
            },
        )

    final_call = {
        "id": "release_last",
        "name": "release",
        "arguments": {"max_steps": 2},
    }
    done = {
        "text": "Continue recovery until the environment ends",
        "tool_calls": [final_call],
    }
    execute_last = response(
        "control", "tool:execute", {"call": final_call}, {"truncated": True}
    )
    call = {"id": "read_1", "name": "view_env_state", "arguments": {"step": -1}}
    proceed = {"text": "Inspect", "tool_calls": [call]}
    execute = response("control", "tool:execute", {"call": call}, {})
    physical_call = {
        "id": "release_1",
        "name": "release",
        "arguments": {"max_steps": 2},
    }
    move = {"text": "Release as requested", "tool_calls": [physical_call]}
    execute_physical = response("control", "tool:execute", {"call": physical_call}, {})

    def plan(result, objective=None):
        return response("control", "model:plan", {"objective": objective}, result)

    reassess = (
        "The previous physical endpoint was not reached. Confirm the next "
        "evidence-gathering or recovery call before continuing."
    )

    sequences = [
        [
            snapshot(1, reached=True),
            plan(done),
            execute_last,
            snapshot(1, terminal=True),
        ],
        [
            snapshot(1),
            plan(proceed, reassess),
            execute,
            snapshot(1),
            snapshot(1),
            plan(done),
            execute_last,
            snapshot(1, terminal=True),
        ],
        [
            snapshot(1),
            plan(proceed, reassess),
            execute,
            snapshot(1),
            snapshot(1),
            plan(done),
            execute_last,
            snapshot(1, terminal=True),
        ],
        [
            snapshot(1),
            plan(move, reassess),
            execute_physical,
            snapshot(2),
            snapshot(2),
            plan(done, reassess),
            execute_last,
            snapshot(2, terminal=True),
        ],
        [snapshot(1, terminal=True)],
        [
            snapshot(1, reached=True),
            plan(move),
            execute_physical,
            snapshot(2, terminal=True),
        ],
    ]
    return tuple(
        PluginCase(
            input={},
            responses=tuple(calls),
            expected={"summary": "Environment ended"},
        )
        for calls in sequences
    )
