"""Phase-specific output contracts retain dynamic allocation and distillation inputs."""

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from reweave.evolution.graph.analysis import _graph_instructions
from reweave.evolution.graph.refinement import (
    analysis_result_schema,
    validate_refinement,
)
from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request


@pytest.mark.parametrize(
    "purpose,mode,width",
    [
        ("search", "fixed", 1),
        ("search", "fixed", 3),
        ("search", "dynamic", 1),
        ("final_selection", "dynamic", 3),
    ],
)
def test_unused_forecast_is_absent_from_output_schema_and_prompt(
    tmp_path, purpose, mode, width
):
    request, result = analysis_request(tmp_path)
    request = request.model_copy(
        update={
            "purpose": purpose,
            "protocol": request.protocol.model_copy(
                update={"width_mode": mode, "max_authors_per_round": width}
            ),
        }
    )
    schema = analysis_result_schema(request)
    Draft202012Validator.check_schema(schema)
    assert "ProgressForecast" not in schema["$defs"]
    assert "forecast" not in schema["required"]
    assert "forecast.confidence" not in _graph_instructions(request, tmp_path)
    data = result.model_dump(mode="json", exclude={"forecast"})
    Draft202012Validator(schema).validate(data)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(result.model_dump(mode="json"))


def test_dynamic_width_still_requires_the_forecast(tmp_path):
    request, result = analysis_request(tmp_path)
    request = request.model_copy(
        update={
            "protocol": request.protocol.model_copy(
                update={"width_mode": "dynamic", "max_authors_per_round": 3}
            )
        }
    )
    assert request.needs_forecast
    schema = analysis_result_schema(request)
    Draft202012Validator(schema).validate(result.model_dump(mode="json"))
    with pytest.raises(ValueError, match="requires a candidate forecast"):
        validate_refinement(request, result.model_copy(update={"forecast": None}))


def test_matching_deferred_until_distillation_graph_without_retrieval(tmp_path):
    request, _ = analysis_request(tmp_path)
    search = analysis_result_schema(request)
    assert "GraphSemantics" not in search["$defs"]
    assert "graph.matching" in _graph_instructions(request, tmp_path)
    final = request.model_copy(update={"purpose": "final_selection"})
    assert "GraphSemantics" in analysis_result_schema(final)["$defs"]
    instructions = _graph_instructions(final, tmp_path)
    assert "downstream shared-harness distillation" in instructions
    assert "read_reference_graph" not in instructions
    # The model must be given a self-contained schema, including all transitive refs.
    Draft202012Validator.check_schema(analysis_result_schema(final))
    assert "ProgressForecast" not in json.dumps(analysis_result_schema(final))


def test_frozen_topology_encodes_matching_at_the_same_phase_as_full(tmp_path):
    from reweave.evolution.bundle import new_record_id
    from reweave.evolution.contracts import GraphAnalysisReceipt
    from tests.unit_tests.harness.evolution.fixtures import artifact, usage

    request, result = analysis_request(tmp_path)
    fixed = request.model_copy(
        update={
            "protocol": request.protocol.model_copy(
                update={"topology_mode": "fixed_initial"}
            )
        }
    )
    assert "GraphSemantics" not in analysis_result_schema(fixed)["$defs"]
    prior = GraphAnalysisReceipt(
        record_id=new_record_id(),
        request_id=request.record_id,
        analysis=result,
        session_id="fixture",
        usage=usage(),
        artifacts=(request.graph,),
    )
    request = request.model_copy(
        update={
            "purpose": "final_selection",
            "protocol": request.protocol.model_copy(
                update={"topology_mode": "fixed_initial"}
            ),
            "previous_analysis": artifact(
                tmp_path, "prior.json", prior.model_dump(mode="json")
            ),
        }
    )
    schema = analysis_result_schema(request)
    assert "GraphSemantics" in schema["$defs"]
    assert "downstream shared-harness distillation" in _graph_instructions(
        request, tmp_path
    )


def test_retired_inquiry_checkpoint_cannot_fork_or_start_a_thread(tmp_path):
    import asyncio

    from reweave.evolution.infrastructure.native_codex import CodexSession

    target = tmp_path / "audit/inputs/inquiry_target.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    with pytest.raises(ValueError, match="session inquiry is retired"):
        asyncio.run(
            CodexSession()._open_codex_thread(object(), None, {}, tmp_path, tmp_path)
        )
