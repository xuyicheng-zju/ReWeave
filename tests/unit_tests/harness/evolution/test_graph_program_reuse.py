"""Task-independent program evidence, measured ties and immutable originals."""

import copy

import pytest

from reweave.evolution.contracts import GraphPolicy, SemanticGraph, VersionReference
from reweave.evolution.graph.selection import select_parent
from reweave.evolution.graph.views import query_map
from tests.unit_tests.harness.evolution.test_semantic_graph import width_map


def test_no_compatibility_reference_policy_is_accepted():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="extra_forbidden"):
        GraphPolicy.model_validate({"reference_policy": "program_graph"})


def test_measured_zero_is_preferred_to_unknown_zero():
    value = width_map(samples=2, completed=0)
    versions = (
        VersionReference(version_id="0", bundle_id="a-unknown"),
        VersionReference(version_id="1", bundle_id="z-measured"),
    )
    for receipt in value["cells"]["0"].values():
        value["episodes"][receipt].update(
            progress=None, completed_nodes=None, official_success=None
        )
    assert select_parent(value, versions) == versions[1]
    # Genuine retained progress still takes priority over measurement coverage.
    first = next(iter(value["cells"]["0"].values()))
    value["episodes"][first].update(progress=0.25, completed_nodes=["earned"])
    assert select_parent(value, versions) == versions[0]


def evidence_graph(prefix=""):
    def name(label):
        return prefix + label

    nodes = {
        name("a"): {"role": "milestone", "predecessors": []},
        name("b"): {"role": "milestone", "predecessors": [name("a")]},
        name("retained"): {"role": "branch", "predecessors": [name("a")]},
        name("goal"): {"role": "goal", "predecessors": [name("b")]},
    }
    for key, node in nodes.items():
        node.update(meaning=key, references=[], uncertainty="Synthetic condition")
    programs, mapped_programs, episodes, transitions, spans = {}, {}, {}, {}, {}
    witnesses = []
    for index, outcome in enumerate(
        ("achieved", "not_achieved", "unknown", "achieved")
    ):
        program, episode = name(f"program{index}"), name(f"episode{index}")
        edge = name("alternative" if index == 3 else "interface")
        key = episode + "/interval/1-2"
        programs[program] = {
            "bundle": {
                "path": f"/archived/{index}.json",
                "artifact_id": f"source{index}",
                "bytes": 1,
                "pointer": "",
            },
            "pointers": ["/context", "/loop"],
            "mechanism": "Synthetic evidence-dependent implementation",
            "uncertainty": "Dependence and causal isolation remain unverified",
        }
        mapped_programs[program] = {
            "bundle_id": name(f"bundle{index}"),
            "assets": [
                {"pointer": "/context", "scope": "C"},
                {"pointer": "/loop", "scope": "L"},
            ],
            "transition_ids": [key],
        }
        path = [
            {
                "first_frame": 0,
                "last_frame": 1,
                "status": "node",
                "node_id": name("a"),
                "uncertainty": "Synthetic earlier acquisition",
            },
            {
                "first_frame": 2,
                "last_frame": 2,
                "status": "unknown" if index == 2 else "node",
                "node_id": None
                if index == 2
                else name("retained" if index == 1 else "b"),
                "uncertainty": "Observed exit unavailable" if index == 2 else "",
            },
        ]
        spans[episode] = path
        episodes[episode] = {
            "bundle_id": name(f"bundle{index}"),
            "condition_id": name("condition"),
            "official_success": None if index == 2 else False,
            "exit_type": "infrastructure" if index == 2 else "agent",
            "state_spans": path,
        }
        witness = {
            "episode_id": episode,
            "first_frame": 1,
            "last_frame": 2,
            "program_id": program,
            "outcome": outcome,
            "bindings": {"subject": name("entity")},
            "uncertainty": "Synthetic observed association, not causal proof",
        }
        witnesses.append((edge, witness))
        transitions[key] = {
            **witness,
            "edge_id": edge,
            "successor_id": None,
            "executions": [],
            "cost": {
                "physical_steps": None if index == 2 else 8 + index,
                "action_wall_time_s": None,
            },
        }
    edges = {}
    for edge, witness in witnesses:
        if edge not in edges:
            edges[edge] = {
                "source": name("a"),
                "target": name("b"),
                "roles": {"subject": "entity whose relation is changed"},
                "entry_conditions": "Different prerequisite"
                if edge == name("alternative")
                else "Shared declared prerequisite",
                "exit_conditions": "Earn b while retaining a",
                "explanation": "Synthetic declared interface",
                "references": [
                    {
                        "path": "/archived/trace.jsonl",
                        "artifact_id": "trace",
                        "bytes": 1,
                        "pointer": "",
                    }
                ],
                "witnesses": [],
            }
        edges[edge]["witnesses"].append(witness)
    graph = SemanticGraph(
        nodes=nodes, edges=edges, programs=programs, episodes=spans, unclassified=[]
    )
    value = {
        "programs": mapped_programs,
        "episodes": episodes,
        "transitions": transitions,
        "edge_interfaces": {
            name("interface"): "shared",
            name("alternative"): "different",
        },
    }
    return graph, value


@pytest.mark.parametrize("prefix", ["", "renamed/"])
def test_program_evidence_keeps_failures_unknowns_and_condition_differences(prefix):
    graph, value = evidence_graph(prefix)
    original = copy.deepcopy(value)
    result = query_map(value, view="program", identity=prefix + "program1", graph=graph)
    alternatives = result["conditional_interfaces"]["shared"]["programs"]
    assert set(alternatives) == {prefix + f"program{i}" for i in range(3)}
    assert alternatives[prefix + "program0"]["outcomes"]["achieved"] == 1
    assert alternatives[prefix + "program1"]["outcomes"]["not_achieved"] == 1
    assert alternatives[prefix + "program2"]["outcomes"]["unknown"] == 1
    assert (
        alternatives[prefix + "program2"]["costs"][0]["cost"]["physical_steps"] is None
    )
    assert len(alternatives[prefix + "program0"]["episode_ids"]) == 1
    assert "isolated causality" in result["meaning"]
    assert value == original
    value["transitions"] = dict(reversed(list(value["transitions"].items())))
    assert (
        query_map(value, view="program", identity=prefix + "program1", graph=graph)
        == result
    )


def test_program_view_uses_the_same_admitted_interface_and_witness_handles(tmp_path):
    from reweave.evolution.authoring.references import AuthorReferences

    graph, value = evidence_graph()
    references = AuthorReferences.create(
        tmp_path / "workspace",
        registry=tmp_path / "registry.json",
        evidence_roots={},
    )
    # Prior rounds may have allocated different public numbers for these IDs.
    references.identity("interface", "earlier_round_interface")
    mapped = references.encode({"edge_interfaces": value["edge_interfaces"]})
    result = query_map(value, view="program", identity="program1", graph=graph)
    compact = references.encode(result)
    interface = mapped["edge_interfaces"]["interface"]
    assert interface == "interface:002"
    assert set(compact["conditional_interfaces"]) == {interface}
    evidence = compact["conditional_interfaces"][interface]["programs"]["program0"]
    witness = evidence["transition_ids"][0]
    assert witness.startswith("transition:")
    assert evidence["costs"][0]["transition_id"] == witness
    assert references.decode(compact) == result


def test_actual_state_change_and_same_episode_continuation_are_visible():
    graph, value = evidence_graph()
    known = query_map(
        value, view="witness", identity="episode0/interval/1-2", graph=graph
    )["transitions"]["episode0/interval/1-2"]
    assert known["state_change"] == {"gained_nodes": ["b"], "lost_nodes": []}
    assert known["actual_source"]["retained_nodes"] == ["a"]
    unknown = query_map(
        value, view="witness", identity="episode2/interval/1-2", graph=graph
    )["transitions"]["episode2/interval/1-2"]
    assert unknown["actual_target"]["retained_nodes"] is None
    assert unknown["state_change"] == {"gained_nodes": None, "lost_nodes": None}
    assert known["continuation"] is None
    assert known["attribution_uncertainty"] == graph.programs["program0"].uncertainty
    previous = value["transitions"]["episode0/interval/1-2"]
    next_id = "episode0/interval/2-3"
    value["transitions"][next_id] = {
        **previous,
        "first_frame": 2,
        "last_frame": 3,
        "successor_id": None,
        "cost": {"physical_steps": 1, "action_wall_time_s": None},
    }
    previous["successor_id"] = next_id
    selected = query_map(
        value, view="witness", identity="episode0/interval/1-2", graph=graph
    )["transitions"]["episode0/interval/1-2"]
    assert selected["continuation"]["transition_id"] == next_id
    assert selected["continuation"]["cost"]["physical_steps"] == 1
    value["transitions"][next_id]["episode_id"] = "episode1"
    with pytest.raises(ValueError, match="same episode"):
        query_map(value, view="witness", identity="episode0/interval/1-2", graph=graph)
