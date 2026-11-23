# SPDX-License-Identifier: Apache-2.0
# Modified for ReWeave.
"""Shared immutable harness and execution fixtures."""

import json
from pathlib import Path

from reweave.evolution.bundle import (
    AgentBundle,
    ContextPackage,
    ProgramLimits,
    PythonFile,
    PythonProgram,
    RuntimeConfiguration,
    WorkflowSpec,
    baseline_loop,
)
from reweave.evolution.contracts import (
    EvidenceFile,
    EvolutionHistory,
    EvolutionProtocol,
    Instance,
    Usage,
)


def baseline():
    return AgentBundle(
        record_id="fixture-baseline",
        runtime=RuntimeConfiguration(
            revision_id="fixture-runtime",
            dependencies={},
            tool_api=(),
            models={},
            memory=(),
            evaluator={},
            environment={},
            configuration={},
            wait_steps=10,
            physical_steps=100,
        ),
        loop=baseline_loop(),
    )


def globalized(bundle):
    """Synthetic ownership factoring; never a transfer or physical success test."""
    value = bundle.model_dump(mode="json")
    value["record_id"] = bundle.record_id + "-organized"
    for component in ("context", "loop"):
        for source in value[component]["program"]["files"]:
            source["layer"] = "global"
    for item in value["workflows"]:
        item["layer"] = "global"
    return AgentBundle.model_validate(value)


class ConsolidatingAuthor:
    """Lifecycle fixtures use real requests with deterministic source factoring."""

    def __init__(self, workspace, author):
        self.workspace = workspace
        self.author = author

    def __getattr__(self, name):
        return getattr(self.author, name)

    async def __call__(self, proposal, directory, *, before_launch):
        if proposal.purpose == "mutation":
            return await self.author(proposal, directory, before_launch=before_launch)
        from reweave.evolution.contracts import AuthoredMutation, Candidate

        before_launch()
        return AuthoredMutation(
            record_id=proposal.record_id + "-receipt",
            proposal_id=proposal.record_id,
            candidates=(
                Candidate(
                    branch_id="organized",
                    bundle=globalized(self.workspace.bundle(proposal.parent.bundle_id)),
                    summary="Synthetic factoring",
                    experience="Synthetic lifecycle only.",
                ),
            ),
            summary="Synthetic success consolidation",
            session_id="fixture-consolidation",
            usage=Usage(physical_steps=0, model_calls=0, wall_time_s=0, cost_usd=0),
            artifacts=(artifact(directory, "synthetic.json", {}),),
        )

    def verify_result(self, proposal, directory):
        if proposal.purpose == "mutation":
            return self.author.verify_result(proposal, directory)
        from reweave.evolution.contracts import AuthoredMutation

        result = AuthoredMutation.model_validate_json(
            (directory / "result.json").read_text()
        )
        assert result.proposal_id == proposal.record_id
        return result


def workflow():
    return WorkflowSpec(
        workflow_id="probe",
        description="Observe progress",
        applicability="uncertain pose",
        input_schema={"type": "object", "properties": {"value": {"type": "number"}}},
        output_schema={"type": "object"},
        observations=(),
        permissions=(),
        contract_cases=(
            {
                "arguments": {},
                "observation": {},
                "responses": [],
                "expected_status": "completed",
            },
        ),
        limits=ProgramLimits(
            runtime_calls=4, physical_calls=2, model_calls=0, wall_time_s=10
        ),
        program=PythonProgram(
            entrypoint="probe:run",
            files=(
                PythonFile(
                    path="probe.py",
                    source='def run(runtime, context):\n    return {"status": "completed", "summary": "observed", "output": {}}\n',
                ),
            ),
        ),
    )


def mutated(parent, scope, *, bad=False):
    payload = parent.model_dump(mode="json")
    payload["record_id"] = f"{parent.record_id}-{scope}-{'bad' if bad else 'changed'}"
    if scope == "W":
        payload["workflows"] = [workflow().model_dump(mode="json")]
    elif scope == "C":
        payload["context"] = ContextPackage(
            instructions="bad" if bad else "verify progress"
        ).model_dump(mode="json")
    else:
        policy = baseline_loop().model_dump(mode="json")
        policy["program"]["files"][0]["source"] += "\n# local supervision\n"
        payload["loop"] = policy
    return AgentBundle.model_validate(payload)


def protocol():
    return EvolutionProtocol(
        record_id="fixture-protocol",
        execution_id="fixture-execution",
        manifest_id="fixture-manifest",
        graph={"analysis_wall_time_s": 600.0},
        max_authors_per_round=3,
        author_wall_time_s=2700.0,
        target_success_rate=1.0,
        dev=(
            Instance(
                record_id="fixture-condition-1",
                task="drawer",
                initialization={"seed": 1},
            ),
        ),
        max_rounds=1,
        infrastructure_retries=0,
        execution={
            "repetition": 0,
            "physical_steps": 100,
            "wait_steps": 10,
            "model_requests": 20,
            "controller_request_unit": "native_turn",
            "adaptation_mode": "static",
            "wall_time_s": 30,
            "program_calls": 20,
            "gateway_calls": 20,
            "model_sampling": {"reasoning_effort": "medium"},
        },
    )


def artifact(path: Path, name: str, value):
    target = path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))
    from reweave.evolution.evidence.records import artifact_record

    return EvidenceFile(**artifact_record(target))


def file_metadata(path: Path):
    """Published file metadata without duplicating its inventory path key."""
    from reweave.evolution.evidence.records import artifact_record

    record = artifact_record(path)
    return {key: record[key] for key in ("artifact_id", "bytes")}


def evolution_history(path: Path, *, originals=None, navigation=None):
    """Frozen shared-memory fixture with separate summary and raw index."""
    return EvolutionHistory(
        summary=artifact(path, "history.md", "# Shared Evolution History\n"),
        originals=artifact(
            path,
            "history_originals.json",
            {
                "files": {} if originals is None else originals,
                "navigation": {} if navigation is None else navigation,
            },
        ),
    )


def usage():
    return Usage(physical_steps=10, model_calls=None, wall_time_s=1, cost_usd=None)


def graph_proposal_fields(path):
    """Explicit graph inputs for native author unit fixtures (no physical claim)."""
    from reweave.evolution.contracts import GraphPolicy, GraphSnapshot, SemanticGraph

    raw = GraphSnapshot(
        record_id="fixture-physical-graph",
        policy=GraphPolicy(),
        episodes=(),
        nodes={},
        paths={},
        transitions=(),
    )
    graph = SemanticGraph(
        nodes={
            "goal": {
                "meaning": "Unobserved evaluator goal",
                "role": "goal",
                "predecessors": [],
                "references": [],
                "uncertainty": "Synthetic fixture, no observed completion",
            }
        },
        edges={},
        programs={},
        episodes={},
        unclassified=(),
    )
    fields = {
        "graph": artifact(path, "graph.json", graph.model_dump(mode="json")),
        "execution_graph": artifact(
            path, "execution_graph.json", raw.model_dump(mode="json")
        ),
        "progress_map": artifact(
            path,
            "progress_map.json",
            {
                "graph_id": "fixture-analysis/analysis/graph",
                "cells": {},
                "milestone_nodes": [],
                "transitions": {},
                "programs": {},
                "node_evidence": {"goal": []},
                "edge_statistics": {},
                "pending_episode_ids": [],
            },
        ),
        "task_protocol": protocol(),
    }
    from reweave.evolution.contracts import (
        GraphAnalysisReceipt,
        RoundDecision,
        VersionReference,
    )

    receipt = GraphAnalysisReceipt(
        record_id="fixture-analysis",
        request_id="a" * 64,
        analysis={
            "graph": graph,
            "changes": "Synthetic graph",
            "forecast": {
                "outcomes": [{"progress": 50, "probability": 1.0}],
                "confidence": 0.5,
                "rationale": "Synthetic forecast",
            },
        },
        session_id="fixture-graph-session",
        usage=usage(),
        artifacts=(fields["graph"],),
    )
    parent = VersionReference(version_id="b" * 64, bundle_id=baseline().record_id)
    decision = RoundDecision(
        record_id="fixture-decision",
        request_id=receipt.request_id,
        analysis=artifact(
            path / "graph-analysis", "result.json", receipt.model_dump(mode="json")
        ),
        graph=fields["graph"],
        progress_map=fields["progress_map"],
        references=(parent,),
        parent=parent,
        width=1,
        candidate_cost={"rollout_usd": None},
        alternatives=({"width": 1.0},),
        statistics={},
        reason="Synthetic decision",
    )
    fields["decision"] = artifact(
        path, "decision.json", decision.model_dump(mode="json")
    )
    return fields


def author_candidate(bundle, directory):
    """Simulate an author supplying a complete program, with a fresh blank plan."""
    from reweave.evolution.authoring.candidate import initialize_candidate

    initialize_candidate(directory, bundle)
    (directory / "SKILL.md").write_text(
        "## Global\nSynthetic lesson: inspect fresh evidence before trusting an old hypothesis.\n"
        "\n## Task-family\nNo task-family lesson in this synthetic check.\n"
    )
