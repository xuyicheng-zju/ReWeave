# Modified for the ReWeave namespace.
"""Published execution changes preserve original assets and measurement meaning."""

import pytest
from pydantic import ValidationError

from reweave.evolution.contracts import (
    BudgetRevision,
    EvolutionProtocol,
    EvolutionState,
    ExecutionRevision,
)
from reweave.evolution.runtime_manifest import (
    RuntimeAuthoringContract,
    RuntimeManifest,
    capture_baseline,
)
from reweave.planner.bundle_runtime import BundleRuntime
from reweave.runtime.kernel import EpisodeLimits
from tests.unit_tests.harness.evolution.fixtures import baseline, protocol
from tests.unit_tests.harness.evolution.test_runtime_manifest import fixture


def revised_protocol(previous):
    return previous.model_copy(
        update={
            "record_id": previous.record_id + "-repaired",
            "execution_id": previous.execution_id + "-repaired",
            "execution_runtime": baseline().runtime.model_copy(
                update={
                    "revision_id": "repaired-runtime",
                    "dependencies": {"native_sdk": "installed version"},
                    "tool_api": ({"name": "repaired_tool"},),
                    "environment": {"adapter": "repaired"},
                }
            ),
        }
    )


def empty_state(current, **revisions):
    return EvolutionState(
        baseline=baseline().record_id,
        protocol=current.record_id,
        versions={},
        order=(),
        proposals={},
        attempts={},
        completed=False,
        stop_reason=None,
        **revisions,
    )


def test_saved_protocol_roundtrip_retains_ordinary_record_references():
    original = protocol()
    published = original.model_dump(mode="json")
    restored = EvolutionProtocol.model_validate(published)
    assert restored == original
    assert restored.record_id == original.record_id
    assert restored.execution_id == original.execution_id
    current = revised_protocol(restored)
    assert current.execution == restored.execution
    assert current.execution_id != restored.execution_id
    assert current.model_dump(mode="json")["execution_runtime"]


@pytest.mark.parametrize("field", ["graph", "author"])
def test_explicit_offline_cap_removal_retains_execution_and_other_budget(field):
    previous = protocol()
    value = previous.model_dump(mode="json")
    value["record_id"] = "uncapped-offline-" + field
    if field == "graph":
        value["graph"]["analysis_wall_time_s"] = None
    else:
        value["author_wall_time_s"] = None
    revision = BudgetRevision(previous=previous, current=value)
    assert revision.current.execution_id == previous.execution_id
    assert revision.current.execution == previous.execution
    assert BudgetRevision.model_validate_json(revision.model_dump_json()) == revision
    with pytest.raises(ValidationError, match="budget revision may only"):
        BudgetRevision(previous=revision.current, current=previous)


def test_offline_cap_removal_cannot_change_both_limits_at_once():
    previous = protocol()
    value = previous.model_dump(mode="json")
    value.update(record_id="both-uncapped", author_wall_time_s=None)
    value["graph"]["analysis_wall_time_s"] = None
    with pytest.raises(ValidationError, match="budget revision may only"):
        BudgetRevision(previous=previous, current=value)


def test_frozen_graph_hands_off_to_uncapped_new_authors_only_with_revision_chain():
    from reweave.evolution.contracts import execution_handoff_allowed

    previous = protocol()
    graph_uncapped = previous.model_copy(
        update={
            "record_id": "uncapped-graph",
            "graph": previous.graph.model_copy(update={"analysis_wall_time_s": None}),
        }
    )
    authors_uncapped = graph_uncapped.model_copy(
        update={
            "record_id": "uncapped-authors",
            "author_wall_time_s": None,
        }
    )
    graph_revision = BudgetRevision(previous=previous, current=graph_uncapped)
    author_revision = BudgetRevision(previous=graph_uncapped, current=authors_uncapped)
    assert not execution_handoff_allowed(previous, authors_uncapped, ())
    assert not execution_handoff_allowed(previous, authors_uncapped, (author_revision,))
    assert execution_handoff_allowed(
        previous, authors_uncapped, (graph_revision, author_revision)
    )
    assert previous.author_wall_time_s is not None
    assert previous.graph.analysis_wall_time_s is not None


def test_environment_change_cannot_be_published_as_a_budget_revision():
    previous = protocol()
    current = revised_protocol(previous)
    with pytest.raises(ValidationError, match="budget revision may only"):
        BudgetRevision(previous=previous, current=current)
    revision = ExecutionRevision(
        previous=previous, current=current, reason="Correct public action schema."
    )
    assert revision.current.execution_id != revision.previous.execution_id


def test_explicit_sampling_revision_preserves_historical_effort_and_budgets():
    from reweave.evolution.contracts import execution_handoff_allowed

    previous = protocol().model_copy(
        update={
            "execution": protocol().execution.model_copy(
                update={"model_sampling": {"reasoning_effort": "high"}}
            )
        }
    )
    current = revised_protocol(previous)
    current = current.model_copy(
        update={
            "execution": current.execution.model_copy(
                update={"model_sampling": {"reasoning_effort": "medium"}}
            )
        }
    )
    revision = ExecutionRevision(
        previous=previous,
        current=current,
        reason="User selected online medium; retain original high receipt.",
    )
    assert execution_handoff_allowed(previous, current, (revision,))
    assert previous.execution.model_sampling == {"reasoning_effort": "high"}
    assert current.execution.model_requests == previous.execution.model_requests


@pytest.mark.parametrize(
    "change",
    [
        "budget",
        "condition",
        "request_unit",
        "adaptation",
        "missing",
        "noop",
        "record_only",
        "reason",
    ],
)
def test_execution_revision_rejects_unrelated_or_undeclared_changes(change):
    previous = protocol()
    current = revised_protocol(previous)
    reason = "Correct actual runtime."
    if change == "budget":
        current = current.model_copy(
            update={
                "execution": current.execution.model_copy(update={"wall_time_s": None})
            }
        )
    elif change == "condition":
        current = current.model_copy(update={"target_success_rate": 0.5})
    elif change in {"request_unit", "adaptation"}:
        update = (
            {"controller_request_unit": "api_attempt"}
            if change == "request_unit"
            else {"adaptation_mode": "adaptive"}
        )
        current = current.model_copy(
            update={"execution": current.execution.model_copy(update=update)}
        )
    elif change == "missing":
        current = previous
    elif change == "noop":
        previous = current
    elif change == "record_only":
        previous = previous.model_copy(
            update={"execution_runtime": current.execution_runtime}
        )
        current = current.model_copy(
            update={
                "execution_runtime": current.execution_runtime.model_copy(
                    update={"revision_id": "another-record"}
                )
            }
        )
    else:
        reason = "  "
    with pytest.raises(ValidationError, match="change only the published execution"):
        ExecutionRevision(previous=previous, current=current, reason=reason)


def test_budget_and_environment_revisions_form_one_continuous_history():
    original = protocol()
    uncapped = original.model_copy(
        update={
            "record_id": "uncapped-protocol",
            "execution_id": "uncapped-execution",
            "execution": original.execution.model_copy(update={"wall_time_s": None}),
        }
    )
    current = revised_protocol(uncapped)
    budget = BudgetRevision(previous=original, current=uncapped)
    environment = ExecutionRevision(
        previous=uncapped, current=current, reason="Correct active environment."
    )
    state = empty_state(
        current, budget_revisions=(budget,), execution_revisions=(environment,)
    )
    assert state.protocol == current.record_id
    with pytest.raises(ValidationError, match="revision chain"):
        empty_state(current, budget_revisions=(budget,))
    with pytest.raises(ValidationError, match="must not branch"):
        empty_state(current, execution_revisions=(environment, environment))


def test_search_round_extension_preserves_execution_and_protocol_history():
    original = protocol().model_copy(update={"max_rounds": 5})
    repaired = revised_protocol(original)
    environment = ExecutionRevision(
        previous=original, current=repaired, reason="Synthetic active runtime repair."
    )
    extended = repaired.model_copy(
        update={"record_id": "extended-protocol", "max_rounds": 10}
    )
    search = BudgetRevision(previous=repaired, current=extended)
    assert search.previous.execution_id == search.current.execution_id
    assert search.previous.model_dump(
        exclude={"record_id", "max_rounds"}
    ) == search.current.model_dump(exclude={"record_id", "max_rounds"})
    assert (
        empty_state(
            extended, budget_revisions=(search,), execution_revisions=(environment,)
        ).protocol
        == extended.record_id
    )
    with pytest.raises(ValidationError, match="revision chain"):
        empty_state(extended, execution_revisions=(environment,))


@pytest.mark.parametrize("rounds", [5, 4, 0, -1, True, 10.5, float("inf"), None])
@pytest.mark.parametrize("typed", [False, True])
def test_search_round_extension_requires_a_larger_finite_integer(rounds, typed):
    previous = protocol().model_copy(update={"max_rounds": 5})
    current = (
        previous.model_copy(update={"max_rounds": rounds})
        if typed
        else {**previous.model_dump(mode="json"), "max_rounds": rounds}
    )
    with pytest.raises(ValidationError):
        BudgetRevision(previous=previous, current=current)


@pytest.mark.parametrize(
    "change", ["graph", "online", "author", "runtime", "conditions"]
)
def test_search_round_extension_cannot_change_another_protocol_field(change):
    previous = protocol().model_copy(update={"max_rounds": 5})
    value = previous.model_dump(mode="json")
    value["max_rounds"] = 10
    if change == "graph":
        value["graph"]["analysis_wall_time_s"] *= 2
    elif change == "online":
        value["execution"]["wall_time_s"] = None
    elif change == "author":
        value["author_wall_time_s"] += 1
    elif change == "runtime":
        value["execution_runtime"] = revised_protocol(
            previous
        ).execution_runtime.model_dump(mode="json")
    else:
        value["target_success_rate"] = 0.5
    with pytest.raises(ValidationError, match="budget revision may only"):
        BudgetRevision(previous=previous, current=value)


def test_actual_gateway_export_makes_no_model_or_tool_calls(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    robot = values["toolkit"]
    monkeypatch.setattr(
        values["planner"],
        "run",
        lambda *args, **kwargs: pytest.fail("export must not query a model"),
        raising=False,
    )
    runtime = BundleRuntime(
        baseline(),
        robot,
        "Synthetic CPU contract export",
        episode_limits=EpisodeLimits(**values["limits"]["episode"]),
    )
    contract = RuntimeAuthoringContract.from_runtime(runtime)
    assert robot.calls == 0 and not robot.terminal
    assert contract.operations["model:plan"]["effect"] == "model"
    assert contract.operations["inspect_scene"]["effect"] == "read"
    assert {spec["name"] for spec in contract.tools} <= set(contract.operations)
    assert {item["name"] for item in contract.capability_catalog} == {
        "inspect_scene",
        "read_terminal_state",
    }


def test_published_active_manifest_reuses_origin_without_rebinding_bundle(tmp_path):
    values = fixture(tmp_path)
    destination = tmp_path / "baseline"
    bundle = capture_baseline(destination, **values)
    original_bytes = (destination / "bundle.json").read_bytes()
    manifest = RuntimeManifest.model_validate_json(
        (destination / "runtime.json").read_text()
    )
    runtime = BundleRuntime(
        bundle,
        values["toolkit"],
        "Synthetic CPU execution revision",
        episode_limits=EpisodeLimits(**values["limits"]["episode"]),
    )
    active = manifest.model_copy(
        update={
            "revision_id": "repaired-runtime",
            "execution_id": "repaired-execution",
            "dependencies": {
                **manifest.dependencies,
                "native_sdk": "installed version",
            },
            "authoring_contract": RuntimeAuthoringContract.from_runtime(runtime),
        }
    )
    previous = protocol().model_copy(
        update={"execution_id": manifest.execution_id, "execution": manifest.conditions}
    )
    current = previous.model_copy(
        update={
            "record_id": "repaired-protocol",
            "execution_id": active.execution_id,
            "execution_runtime": active.runtime_configuration(),
        }
    )
    revision = ExecutionRevision(
        previous=previous,
        current=current,
        reason="Correct native and public interfaces.",
    )
    active = active.model_copy(update={"execution_revision": revision})
    assert active.execution_id == current.execution_id
    assert active.execution_id != manifest.execution_id
    assert active.runtime_configuration() != bundle.runtime
    assert (destination / "bundle.json").read_bytes() == original_bytes
    assert values["toolkit"].calls == 0
    assert (
        active.model_copy(update={"tool_api": ()}).runtime_configuration().tool_api
        == ()
    )


def test_captured_gateway_translation_preserves_constraints():
    operations = {
        "read": {
            "effect": "read",
            "input_schema": {
                "type": "object",
                "properties": {"step": {"type": "integer", "description": "旧说明"}},
            },
        }
    }
    tools = [
        {
            "name": "read",
            "description": "Read state",
            "input_schema": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer", "description": "Latest settled state"}
                },
            },
        }
    ]
    contract = RuntimeAuthoringContract.from_capture(
        operations=operations, tools=tools, capability_catalog=[]
    )
    assert contract.operations["read"]["input_schema"] == tools[0]["input_schema"]
    assert (
        operations["read"]["input_schema"]["properties"]["step"]["description"]
        == "旧说明"
    )
    tools[0]["input_schema"]["properties"]["step"]["type"] = "string"
    with pytest.raises(ValueError, match="changes captured gateway constraints"):
        RuntimeAuthoringContract.from_capture(
            operations=operations, tools=tools, capability_catalog=[]
        )


def test_captured_gateway_preserves_description_named_data():
    schema = {
        "type": "object",
        "properties": {"description": {"const": {"description": "required value"}}},
    }
    operations = {"read": {"effect": "read", "input_schema": schema}}
    tools = [
        {
            "name": "read",
            "description": "Read",
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {"const": {"description": "changed value"}}
                },
            },
        }
    ]
    with pytest.raises(ValueError, match="changes captured gateway constraints"):
        RuntimeAuthoringContract.from_capture(
            operations=operations, tools=tools, capability_catalog=[]
        )


def test_captured_gateway_rejects_boolean_number_constraint_change():
    operations = {"read": {"effect": "read", "input_schema": {"const": False}}}
    tools = [{"name": "read", "description": "Read", "input_schema": {"const": 0}}]
    with pytest.raises(ValueError, match="changes captured gateway constraints"):
        RuntimeAuthoringContract.from_capture(
            operations=operations, tools=tools, capability_catalog=[]
        )


def test_author_cap_reduction_keeps_execution_and_every_other_budget():
    from reweave.evolution.contracts import execution_handoff_allowed

    original = protocol().model_copy(update={"max_authors_per_round": 4})
    repaired = revised_protocol(original)
    execution = ExecutionRevision(
        previous=original, current=repaired, reason="Actual runtime repair"
    )
    reduced = repaired.model_copy(
        update={"record_id": "one-author-protocol", "max_authors_per_round": 1}
    )
    revision = BudgetRevision(previous=repaired, current=reduced)
    assert revision.previous.execution_id == revision.current.execution_id
    assert repaired.model_dump(
        exclude={"record_id", "max_authors_per_round"}
    ) == reduced.model_dump(exclude={"record_id", "max_authors_per_round"})
    assert execution_handoff_allowed(original, reduced, (revision, execution))
    assert not execution_handoff_allowed(original, reduced, (execution,))
    assert not execution_handoff_allowed(original, reduced, (revision,))
    assert not execution_handoff_allowed(reduced, original, (execution, revision))
    assert (
        empty_state(
            reduced, execution_revisions=(execution,), budget_revisions=(revision,)
        ).protocol
        == reduced.record_id
    )
    assert BudgetRevision.model_validate_json(revision.model_dump_json()) == revision


@pytest.mark.parametrize("count", [4, 0, -1, True, 1.5, None])
@pytest.mark.parametrize("typed", [False, True])
def test_author_cap_revision_requires_a_different_positive_integer(count, typed):
    original = protocol().model_copy(update={"max_authors_per_round": 4})
    update = {"record_id": "invalid-author-cap", "max_authors_per_round": count}
    current = (
        original.model_copy(update=update)
        if typed
        else {**original.model_dump(mode="json"), **update}
    )
    with pytest.raises(ValidationError):
        BudgetRevision(previous=original, current=current)


def test_author_cap_increase_preserves_execution_and_requires_published_chain():
    from reweave.evolution.contracts import execution_handoff_allowed

    original = protocol().model_copy(update={"max_authors_per_round": 4})
    reduced = original.model_copy(
        update={"record_id": "one-author", "max_authors_per_round": 1}
    )
    increased = reduced.model_copy(
        update={"record_id": "two-authors", "max_authors_per_round": 2}
    )
    reduction = BudgetRevision(previous=original, current=reduced)
    increase = BudgetRevision(previous=reduced, current=increased)
    assert increased.model_dump(exclude={"record_id", "max_authors_per_round"}) == (
        reduced.model_dump(exclude={"record_id", "max_authors_per_round"})
    )
    assert execution_handoff_allowed(reduced, increased, (increase,))
    assert execution_handoff_allowed(original, increased, (increase, reduction))
    assert not execution_handoff_allowed(reduced, increased, ())
    assert not execution_handoff_allowed(increased, reduced, (increase,))
    assert (
        empty_state(increased, budget_revisions=(reduction, increase)).protocol
        == increased.record_id
    )
    assert BudgetRevision.model_validate_json(increase.model_dump_json()) == increase


@pytest.mark.parametrize(
    "change",
    [
        "graph",
        "online",
        "rounds",
        "author_time",
        "physical",
        "sampling",
        "execution_id",
        "runtime",
        "search",
        "record_id",
    ],
)
def test_author_cap_revision_rejects_every_other_change(change):
    previous = protocol().model_copy(update={"max_authors_per_round": 4})
    current = previous.model_dump(mode="json")
    current.update(record_id="reduced-author-cap", max_authors_per_round=1)
    if change == "graph":
        current["graph"]["analysis_wall_time_s"] += 1
    elif change == "online":
        current["execution"]["wall_time_s"] = None
    elif change == "rounds":
        current["max_rounds"] += 1
    elif change == "author_time":
        current["author_wall_time_s"] += 1
    elif change == "physical":
        current["execution"]["physical_steps"] += 1
    elif change == "sampling":
        current["execution"]["model_sampling"]["reasoning_effort"] = "high"
    elif change == "execution_id":
        current["execution_id"] = "different-execution"
    elif change == "runtime":
        current["execution_runtime"] = revised_protocol(
            previous
        ).execution_runtime.model_dump(mode="json")
    elif change == "search":
        current["search"]["success_value"] += 1
    else:
        current["record_id"] = previous.record_id
    with pytest.raises(ValidationError, match="budget revision may only"):
        BudgetRevision(previous=previous, current=current)
