# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Structural counterexamples, evidence continuity, and author retrieval boundaries."""

import asyncio
import json
from itertools import permutations

import numpy as np
import pytest

from reweave.evolution.contracts import (
    OriginalReference,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
    StateSpan,
    TransitionWitness,
)
from reweave.evolution.graph.conditions import StateCondition
from reweave.evolution.graph.matching import GraphQuery, match_graphs, semantic_query
from reweave.evolution.graph.retrieval import (
    GraphRetrieval,
    GraphRetrievalConfig,
    GraphSearch,
    TextEmbeddings,
)

CATALOG = {
    "phase": {
        "definition": "Observed physical phase of an object in this fixture.",
        "arguments": ["object"],
        "values": {
            v: v
            for v in ("held", "aligned", "released", "localized", "lifted", "other")
        },
    },
    "support": {
        "definition": "Whether a surface supports the object.",
        "arguments": ["object"],
        "values": {"yes": "supported", "no": "unsupported"},
    },
    "grasp": {
        "definition": "Grasp security.",
        "arguments": ["object"],
        "values": {
            "secure": "securely held",
            "loose": "loosely held",
            "none": "not held",
        },
    },
}


def fact(fluent, values, *arguments):
    return {
        "fluent": fluent,
        "arguments": arguments or ("object",),
        "values": (values,) if isinstance(values, str) else values,
    }


def query_graph(nodes, edges=(), *, dependencies=(), entities=None):
    return GraphQuery(
        catalog=CATALOG,
        entities=entities or {"object": {"description": "a task object"}},
        nodes={
            key: [fact("phase", value)] if isinstance(value, str) else value
            for key, value in nodes.items()
        },
        interfaces={
            f"e{i}": {"source": a, "target": b} for i, (a, b) in enumerate(edges)
        },
        dependencies=dependencies,
    )


def with_matching(graph):
    from reweave.evolution.graph.conditions import GraphSemantics

    values = {"a": "held", "b": "aligned", "c": "released"}
    query = query_graph({key: values.get(key, "other") for key in graph.nodes})
    semantics = GraphSemantics(
        **query.model_dump(exclude={"interfaces", "dependencies"}),
        interfaces={
            key: {"entry": [fact("phase", "held")] if edge.source is None else []}
            for key, edge in graph.edges.items()
        },
    )
    return graph.model_copy(update={"matching": semantics})


def test_opposite_states_do_not_use_text_similarity_and_direction_is_exact():
    query = query_graph({"s": "held", "t": "released"}, [("s", "t")])
    good = query_graph({"cup": "held", "bowl": "released"}, [("cup", "bowl")])
    reverse = query_graph({"s": "held", "t": "released"}, [("t", "s")])
    assert match_graphs(query, good)["distance"] == pytest.approx(0)
    result = match_graphs(query, reverse)
    assert result["distance"] == pytest.approx(2 / 3)
    assert result["optimal"]
    assert result["distance_lower_bound"] == pytest.approx(result["distance"])


def test_rename_and_order_do_not_change_alignment():
    query = query_graph(
        {"a": "held", "b": "aligned", "c": "released"}, [("a", "b"), ("b", "c")]
    )
    for order in permutations("xyz"):
        states = dict(zip(order, ("held", "aligned", "released"), strict=True))
        candidate = query_graph(states, list(zip(order, order[1:], strict=False)))
        assert match_graphs(query, candidate)["distance"] == pytest.approx(0)


def test_shared_branch_cannot_match_two_different_sources():
    query = query_graph(
        {"a": "held", "b": "aligned", "c": "released"}, [("a", "b"), ("a", "c")]
    )
    split = query_graph(
        {"x": "held", "y": "held", "b": "aligned", "c": "released"},
        [("x", "b"), ("y", "c")],
    )
    assert match_graphs(query, split)["distance"] == pytest.approx(0.4)


def test_candidate_extras_are_free_but_missing_query_nodes_are_charged():
    query = query_graph({"a": "held", "b": "released"}, [("a", "b")])
    large = query_graph(
        {"a": "held", "b": "released", "x": "other"}, [("a", "b"), ("b", "x")]
    )
    assert match_graphs(query, large)["distance"] == pytest.approx(0)
    small = query_graph({"a": "held"})
    assert match_graphs(query, small)["distance"] == pytest.approx(2 / 3)


def test_dependency_closure_and_relation_types():
    nodes = {"a": "held", "b": "aligned", "c": "released"}
    query = query_graph(nodes, dependencies=(("a", "c"),))
    candidate = query_graph(nodes, dependencies=(("a", "b"), ("b", "c")))
    assert match_graphs(query, candidate)["distance"] == pytest.approx(0)
    wrong_kind = query_graph(nodes, [("a", "c")])
    assert match_graphs(query, wrong_kind)["distance"] > 0
    with pytest.raises(ValueError, match="acyclic"):
        query_graph(nodes, dependencies=(("a", "b"), ("b", "a")))


def test_unknown_and_contradiction_have_different_numeric_meanings():
    query = query_graph({"s": [fact("phase", "held"), fact("support", "yes")]})
    unknown = query_graph({"s": "held"})
    contradiction = query_graph({"s": [fact("phase", "held"), fact("support", "no")]})
    assert match_graphs(query, unknown)["distance"] == pytest.approx(0.5)
    assert match_graphs(query, contradiction)["distance"] == pytest.approx(1)


def test_allowed_sets_intersect_and_do_not_create_duplicate_reward():
    query = query_graph(
        {"s": [fact("grasp", ("secure", "loose")), fact("grasp", ("loose", "none"))]}
    )
    good = query_graph({"s": [fact("grasp", "loose")]})
    assert len(query.nodes["s"]) == 1
    assert match_graphs(query, good)["distance"] == pytest.approx(0)
    with pytest.raises(ValueError, match="contradictory"):
        query_graph({"s": [fact("grasp", "secure"), fact("grasp", "none")]})
    with pytest.raises(ValueError, match="proper domain subset"):
        query_graph({"s": [fact("support", ("yes", "no"))]})


def test_one_entity_substitution_is_shared_across_states():
    query = query_graph({"a": "held", "b": "released"}, [("a", "b")])
    candidate = query_graph(
        {"a": [fact("phase", "held", "x")], "b": [fact("phase", "released", "y")]},
        [("a", "b")],
        entities={key: {"description": key} for key in ("x", "y")},
    )
    assert match_graphs(query, candidate)["distance"] == pytest.approx(2 / 3)


def test_interface_conflict_cannot_be_hidden_by_matching_only_the_nodes():
    query = query_graph({"s": "held", "t": "released"}, [("s", "t")])
    query = query.model_copy(
        update={
            "interfaces": {
                "e0": query.interfaces["e0"].model_copy(
                    update={"entry": (StateCondition(**fact("support", "yes")),)}
                )
            }
        }
    )
    bad = query.model_copy(
        update={
            "interfaces": {
                "e0": query.interfaces["e0"].model_copy(
                    update={"entry": (StateCondition(**fact("support", "no")),)}
                )
            }
        }
    )
    assert match_graphs(query, bad)["distance"] >= 2 / 3 - 1e-8
    alternatives = bad.model_copy(
        update={"interfaces": {**bad.interfaces, "good": query.interfaces["e0"]}}
    )
    assert match_graphs(query, alternatives)["distance"] == pytest.approx(0)


def test_background_contexts_can_share_reference_source_without_becoming_one_node():
    query = query_graph({"b": "aligned", "c": "released"})
    query = GraphQuery(
        **query.model_dump(exclude={"interfaces"}),
        interfaces={
            "p": {"source": None, "target": "b", "entry": [fact("phase", "held")]},
            "q": {"source": None, "target": "c", "entry": [fact("phase", "held")]},
        },
    )
    shared = query_graph(
        {"a": "held", "b": "aligned", "c": "released"}, [("a", "b"), ("a", "c")]
    )
    split = query_graph(
        {"x": "held", "y": "held", "b": "aligned", "c": "released"},
        [("x", "b"), ("y", "c")],
    )
    assert match_graphs(query, shared)["distance"] == pytest.approx(0)
    assert match_graphs(query, split)["distance"] == pytest.approx(0)
    assert match_graphs(query, shared)["query_mass"] == 4


def test_effective_interface_context_uses_endpoints_without_diluting_requirements():
    query = query_graph({"t": "released"})
    query = GraphQuery(
        **query.model_dump(exclude={"interfaces"}),
        interfaces={
            "e": {"source": None, "target": "t", "entry": [fact("support", "yes")]}
        },
    )
    candidate = query_graph(
        {"s": [fact("phase", "held"), fact("support", "yes")], "t": "released"},
        [("s", "t")],
    )
    assert match_graphs(query, candidate)["distance"] == pytest.approx(0)
    # An internally contradictory endpoint/entrance is invalid, not selectable.
    broken = candidate.model_copy(
        update={
            "interfaces": {
                "e0": candidate.interfaces["e0"].model_copy(
                    update={"entry": (StateCondition(**fact("support", "no")),)}
                )
            }
        }
    )
    with pytest.raises(ValueError, match="contradictory"):
        match_graphs(query, broken)
    unsupported = query_graph({"s": "held", "t": "released"}, [("s", "t")])
    result = match_graphs(query, unsupported)
    assert result["distance"] == pytest.approx(0.5)
    assert all(
        relation["coverage"] == pytest.approx(0) for relation in result["relations"]
    )
    explicit = query_graph({"s": "held", "t": "released"}, [("s", "t")])
    explicit = explicit.model_copy(
        update={
            "interfaces": {
                "e0": explicit.interfaces["e0"].model_copy(
                    update={"entry": (StateCondition(**fact("support", "yes")),)}
                )
            }
        }
    )
    result = match_graphs(explicit, unsupported)
    assert result["distance"] == pytest.approx(1 / 3)
    assert result["relations"][0]["coverage"] == pytest.approx(0)


def test_user_ids_cannot_collide_with_internal_relations_or_background_ports():
    query = query_graph(
        {"@0": "held", "background/e": "released"}, [("@0", "background/e")]
    )
    query = query.model_copy(update={"interfaces": {"@0": query.interfaces["e0"]}})
    assert match_graphs(query, query)["distance"] == pytest.approx(0)


def test_shared_catalog_definitions_must_be_identical():
    query = query_graph({"s": "held"})
    changed = query.model_dump()
    changed["catalog"]["phase"]["definition"] = "A different property."
    with pytest.raises(ValueError, match="conflicting definition"):
        match_graphs(query, GraphQuery(**changed))


def test_graph_matching_enforces_resource_limits(tmp_path):
    query = query_graph({"s": "held"})
    with pytest.raises(ValueError, match="candidate budget"):
        match_graphs(query, query, max_candidates=1)
    with pytest.raises(ValueError, match="positive and finite"):
        match_graphs(query, query, seconds=float("nan"))
    config = retrieval_config(tmp_path).model_dump()
    config["max_variables"] = 1000
    with pytest.raises(ValueError, match="Extra inputs"):
        GraphRetrievalConfig.model_validate(config)


def corpus_graph(tmp_path, *, failed=False, different_episode=False, mismatch=False):
    original = tmp_path / "evidence.json"
    original.write_text(json.dumps({"fixture": "original evidence"}))
    from reweave.evolution.evidence.records import artifact_record

    reference = OriginalReference(**artifact_record(original), pointer="")
    nodes = {
        "a": SemanticNode(
            meaning="held",
            role="milestone",
            predecessors=(),
            references=(reference,),
            uncertainty="fixture",
        ),
        "b": SemanticNode(
            meaning="aligned",
            role="milestone",
            predecessors=("a",),
            references=(reference,),
            uncertainty="fixture",
        ),
        "c": SemanticNode(
            meaning="released",
            role="goal",
            predecessors=("b",),
            references=(reference,),
            uncertainty="fixture",
        ),
    }
    edges = {}
    for index, (source, target) in enumerate((("a", "b"), ("b", "c"))):
        edges[f"edge{index}"] = SemanticEdge(
            source=source,
            target=target,
            roles={},
            entry_conditions="fixture entry",
            exit_conditions="fixture exit",
            explanation="fixture",
            references=(reference,),
            witnesses=(
                TransitionWitness(
                    episode_id=("2" if index and different_episode else "1") * 64,
                    first_frame=index,
                    last_frame=index + 1,
                    program_id=None,
                    bindings={},
                    outcome="not_achieved" if failed and index == 0 else "achieved",
                    uncertainty="fixture",
                ),
            ),
        )
    states = tuple(
        StateSpan(
            first_frame=i,
            last_frame=i,
            status="node",
            node_id=node,
            uncertainty="fixture",
        )
        for i, node in enumerate(("a", "a" if mismatch else "b", "c"))
    )
    return with_matching(
        SemanticGraph(
            nodes=nodes,
            edges=edges,
            programs={},
            episodes={
                "1" * 64: states,
                **({"2" * 64: states} if different_episode else {}),
            },
            unclassified=(),
        )
    )


@pytest.mark.parametrize(
    "options,connected",
    [
        ({}, True),
        ({"failed": True}, False),
        ({"different_episode": True}, False),
        ({"mismatch": True}, False),
    ],
)
def test_multi_step_paths_require_continuous_achieved_intermediates(
    tmp_path, options, connected
):
    query, paths = semantic_query(corpus_graph(tmp_path, **options), witness_paths=True)
    pairs = {(edge.source, edge.target) for edge in query.interfaces.values()}
    assert (("a", "c") in pairs) is connected
    assert ("a", "b") in pairs
    if connected:
        assert (("edge0", 0), ("edge1", 0)) in paths.values()


def test_background_entries_are_separate_evidence_backed_query_coordinates(tmp_path):
    graph = corpus_graph(tmp_path)
    graph = graph.model_copy(
        update={
            "edges": {
                key: edge.model_copy(update={"source": None})
                for key, edge in graph.edges.items()
            }
        }
    )
    query, _ = semantic_query(with_matching(graph))
    assert set(graph.nodes) == set(query.nodes) == {"a", "b", "c"}
    assert query.interfaces["edge0"].source is None
    assert query.interfaces["edge1"].source is None


@pytest.fixture
def fake_encoder(monkeypatch):
    calls = []

    class Encoder:
        max_seq_length = 8192

        def tokenizer(self, texts, **kwargs):
            return {"input_ids": [[1] * len(text.split()) for text in texts]}

        def encode(self, texts, **kwargs):
            calls.append(texts)
            return np.ones((len(texts), 8))

    monkeypatch.setattr(
        "reweave.evolution.graph.retrieval._embedding_model", lambda *args: Encoder()
    )
    return calls


def retrieval_config(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(corpus_graph(tmp_path).model_dump_json())
    return GraphRetrievalConfig(
        graphs={"teacher": path}, cache=tmp_path / "embeddings.sqlite"
    )


def test_cache_reuses_exact_text_and_rejects_different_encoder(tmp_path, fake_encoder):
    config = retrieval_config(tmp_path)
    encoder = TextEmbeddings(config)
    document = "Physical support of an object by a surface"
    vectors, first = encoder.encode([document] * 2)
    assert vectors.shape == (2, 8)
    assert first["new_texts"] == 1
    _, second = encoder.encode([document])
    assert second["new_texts"] == 0
    assert len(fake_encoder) == 1
    changed = config.model_copy(update={"embedding_revision": "a" * 40})
    with pytest.raises(ValueError, match="different model"):
        TextEmbeddings(changed).encode([document])


def test_catalog_encoding_rejects_over_budget_without_truncation(
    tmp_path, fake_encoder
):
    with pytest.raises(ValueError, match="catalog text exceeds"):
        TextEmbeddings(retrieval_config(tmp_path)).encode(["state " * 8193])
    assert fake_encoder == []


def test_search_is_compact_and_evidence_requires_explicit_read(tmp_path, fake_encoder):
    retrieval = GraphRetrieval(retrieval_config(tmp_path))
    request = GraphSearch(
        query=query_graph({"a": "held", "c": "released"}, [("a", "c")])
    )
    result = retrieval.search(request)
    assert result["matches"][0]["task"] == "teacher"
    assert "original evidence" not in json.dumps(result)
    assert "programs" not in result["matches"][0]
    edge = retrieval.read("teacher", view="transitions", identity="edge0")
    assert edge["edges"]["edge0"]["witnesses"]
    assert fake_encoder == []
    assert (
        retrieval.search(request.model_copy(update={"exclude_tasks": ("teacher",)}))[
            "matches"
        ]
        == []
    )
    with pytest.raises(ValueError, match="outside"):
        retrieval.search(request.model_copy(update={"exclude_tasks": ("unknown",)}))


def test_author_tools_retrieve_then_read_issued_original(tmp_path, fake_encoder):
    from reweave.evolution.authoring.tools import AuthorDevelopmentTools
    from tests.unit_tests.harness.evolution.test_bundle_author import workspace

    config = retrieval_config(tmp_path)
    path = tmp_path / "retrieval.json"
    path.write_text(config.model_dump_json())
    root = workspace(tmp_path / "author")
    tools = AuthorDevelopmentTools(
        root, tmp_path / "audit/development.jsonl", retrieval_config=path
    )
    names = {tool.name for tool in asyncio.run(tools.catalog())}
    assert {"retrieve_graphs", "read_reference_graph"} <= names

    def call(name, **arguments):
        return asyncio.run(tools.model_call(name, arguments, case_results=[]))

    result = call(
        "retrieve_graphs",
        query=query_graph(
            {"start": "held", "end": "released"}, [("start", "end")]
        ).model_dump(mode="json"),
    )
    assert result["matches"][0]["task"] == "teacher"
    edge = call(
        "read_reference_graph", task="teacher", view="transitions", identity="edge0"
    )
    reference = edge["edges"]["edge0"]["references"][0]
    assert reference.startswith("file:")
    assert call("read_original", reference=reference)["value"] == {
        "fixture": "original evidence"
    }
    with pytest.raises(ValueError, match="unknown corpus"):
        call("read_reference_graph", task="not-authorized")


def test_cli_indexes_and_searches_same_algorithm(tmp_path, fake_encoder, capsys):
    from reweave.cli.evolve import _evolution_command, _parser

    config = tmp_path / "retrieval.json"
    config.write_text(retrieval_config(tmp_path).model_dump_json())
    query = tmp_path / "query.json"
    query.write_text(
        GraphSearch(
            query=query_graph({"a": "held", "b": "aligned"}, [("a", "b")])
        ).model_dump_json()
    )
    parser = _parser()
    assert (
        _evolution_command(parser.parse_args(["index-graphs", "--config", str(config)]))
        == 0
    )
    assert json.loads(capsys.readouterr().out)["tasks"] == 1
    assert (
        _evolution_command(
            parser.parse_args(
                [
                    "retrieve-graphs",
                    "--config",
                    str(config),
                    "--query",
                    str(query),
                ]
            )
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["matches"]
    query.write_text("{}")
    assert (
        _evolution_command(
            parser.parse_args(
                [
                    "retrieve-graphs",
                    "--config",
                    str(config),
                    "--query",
                    str(query),
                    "--graph",
                    str(
                        GraphRetrievalConfig.model_validate_json(
                            config.read_text()
                        ).graphs["teacher"]
                    ),
                ]
            )
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["query"] == {
        "source": "current_graph",
        "nodes": 3,
    }


def test_default_query_refreshes_whole_graph_and_never_retains_local_override(
    tmp_path, fake_encoder
):
    from reweave.evolution.authoring.tools import AuthorDevelopmentTools
    from tests.unit_tests.harness.evolution.test_bundle_author import workspace

    config = retrieval_config(tmp_path)
    config_path = tmp_path / "retrieval.json"
    config_path.write_text(config.model_dump_json())
    root = workspace(tmp_path / "author")
    tools = AuthorDevelopmentTools(
        root, tmp_path / "audit/development.jsonl", retrieval_config=config_path
    )
    graph_path = root / "context/graph.json"
    graph_path.write_text("null")

    def call(**arguments):
        return asyncio.run(
            tools.model_call("retrieve_graphs", arguments, case_results=[])
        )

    with pytest.raises(ValueError, match="no current semantic graph"):
        call()
    graph = corpus_graph(tmp_path)
    graph_path.write_text(
        json.dumps(tools.references.encode(graph.model_dump(mode="json")))
    )
    first = call()
    assert first["query"] == {"source": "current_graph", "nodes": 3}
    local = call(
        query=query_graph({"s": "held", "t": "released"}).model_dump(mode="json")
    )
    assert local["query"] == {"source": "explicit", "nodes": 2}
    assert call()["query"] == first["query"]
    assert fake_encoder == []

    revised = graph.model_dump(mode="json")
    revised["nodes"]["a"]["meaning"] = "new round: another state"
    revised["matching"]["nodes"]["a"] = [fact("phase", "other")]
    graph_path.write_text(json.dumps(tools.references.encode(revised)))
    refreshed = call()
    assert refreshed["query"] == first["query"]
    assert refreshed["matches"][0]["distance"] > first["matches"][0]["distance"]
    assert fake_encoder == []


def test_graph_analyst_decides_search_and_shares_results_with_independent_authors(
    tmp_path, fake_encoder
):
    from pathlib import Path

    from reweave.evolution.authoring.author import NativeBundleAuthor, _artifact
    from reweave.evolution.authoring.config import AuthorPolicy
    from reweave.evolution.authoring.references import AuthorReferences
    from reweave.evolution.authoring.tools import AuthorDevelopmentTools
    from reweave.evolution.contracts import (
        GraphAnalysisReceipt,
        Proposal,
        RoundDecision,
    )
    from reweave.evolution.graph.analysis import NativeGraphAnalysis
    from reweave.utils.persistence import atomic_json
    from tests.unit_tests.harness.evolution.fixtures import (
        baseline,
        graph_proposal_fields,
    )
    from tests.unit_tests.harness.evolution.test_graph_analysis import analysis_request

    config_path = tmp_path / "retrieval.json"
    config_path.write_text(retrieval_config(tmp_path).model_dump_json())
    policy = AuthorPolicy(graph_retrieval=config_path)
    (tmp_path / "evidence").mkdir()
    request, draft = analysis_request(tmp_path / "evidence")
    draft = draft.model_copy(update={"graph": with_matching(draft.graph)})
    root = tmp_path / "analyst/workspace"
    NativeGraphAnalysis(
        bundle_loader=lambda _: baseline(),
        policy=policy,
        evidence_workspace=tmp_path / "navigation",
    )._prepare(request, root)
    tools = AuthorDevelopmentTools(
        root, root.parent / "audit/development.jsonl", retrieval_config=config_path
    )
    shared = root.parent / "audit/graph_retrieval.json"
    assert not shared.exists()
    assert fake_encoder == []
    catalog = AuthorReferences.load(root)
    atomic_json(
        root / "analysis/result.json", catalog.encode(draft.model_dump(mode="json"))
    )
    result = asyncio.run(tools.model_call("retrieve_graphs", {}, case_results=[]))
    assert result["query"]["source"] == "current_graph"
    assert shared.exists()
    assert json.loads(shared.read_text())[0]["request"]["query"] == semantic_query(
        draft.graph
    )[0].model_dump(mode="json")

    fields = graph_proposal_fields(tmp_path / "round")
    decision = RoundDecision.model_validate_json(
        Path(fields["decision"].path).read_text()
    )

    analysis_path = Path(decision.analysis.path)
    published = analysis_path.parent / "audit/graph_retrieval.json"
    published.parent.mkdir(parents=True)
    published.write_bytes(shared.read_bytes())
    receipt = GraphAnalysisReceipt.model_validate_json(analysis_path.read_text())
    atomic_json(
        analysis_path,
        receipt.model_copy(
            update={"artifacts": (*receipt.artifacts, _artifact(published))}
        ).model_dump(mode="json"),
    )
    decision = decision.model_copy(update={"analysis": _artifact(analysis_path)})
    atomic_json(Path(fields["decision"].path), decision.model_dump(mode="json"))
    fields["decision"] = _artifact(Path(fields["decision"].path))
    author = NativeBundleAuthor(
        bundle_loader=lambda _: baseline(),
        policy=policy,
        evidence_workspace=tmp_path / "navigation",
    )
    outputs = []
    for slot in range(2):
        proposal = Proposal(
            **fields,
            record_id=f"branch{slot}",
            references=decision.references,
            parent=decision.parent,
            round_index=0,
            author_index=slot,
            scopes=("W", "C", "L"),
            history=request.history,
            evidence=request.evidence,
        )
        destination = tmp_path / f"author{slot}/workspace"
        author._prepare(proposal, destination)
        outputs.append((destination / "context/reference_matches.json").read_bytes())
    assert outputs[0] == outputs[1] == published.read_bytes()


def test_catalog_lookup_encodes_only_definitions_and_returns_exact_domains(
    tmp_path, fake_encoder
):
    retrieval = GraphRetrieval(retrieval_config(tmp_path))
    full = retrieval.read_catalog()
    assert len(full["catalog"]) == len(CATALOG)
    assert fake_encoder == []
    selected = retrieval.read_catalog("a secure grasp", top_k=1)
    assert len(selected["catalog"]) == 1
    assert selected["embedding"]["new_texts"] == len(CATALOG) + 1
    assert (
        retrieval.read_catalog("a secure grasp", top_k=1)["embedding"]["new_texts"] == 0
    )
    assert (
        retrieval.read("teacher", view="conditions")["matching"]["catalog"]
        == full["catalog"]
    )


def test_text_only_graph_cannot_silently_fall_back_to_cosine(tmp_path):
    graph = corpus_graph(tmp_path).model_copy(update={"matching": None})
    with pytest.raises(ValueError, match="requires matching conditions"):
        semantic_query(graph)
    with pytest.raises(ValueError, match="exactly the canonical"):
        data = corpus_graph(tmp_path).model_dump()
        del data["matching"]["nodes"]["a"]
        SemanticGraph.model_validate(data)


def test_catalog_and_context_conflicts_are_rejected_at_publication(tmp_path):
    data = corpus_graph(tmp_path).model_dump()
    data["matching"]["entities"]["object"]["conditions"] = [fact("phase", "other")]
    with pytest.raises(ValueError, match="contradictory"):
        SemanticGraph.model_validate(data)


def test_bounded_solve_reports_gap_instead_of_claiming_exact_rank(monkeypatch):
    from reweave.evolution.graph import alignment as graph_alignment

    # Expire before expansion: the empty feasible incumbent has a valid bound.
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(graph_alignment.time, "perf_counter", lambda: next(ticks, 2.0))
    from reweave.evolution.graph.matching import ConditionGraph

    graph = ConditionGraph.from_query(query_graph({"s": "held"}))
    result = graph_alignment.GraphAlignment(graph, graph, 1000).solve(1)
    assert not result["optimal"]
    assert result["distance"] == 1
    assert result["distance_lower_bound"] == 0
    assert result["search_nodes"] == 0


def test_witness_path_cannot_hide_inconsistent_same_frame_interface_conditions(
    tmp_path,
):
    graph = corpus_graph(tmp_path)
    interfaces = dict(graph.matching.interfaces)
    interfaces["edge0"] = interfaces["edge0"].model_copy(
        update={"exit": (StateCondition(**fact("grasp", "secure")),)}
    )
    interfaces["edge1"] = interfaces["edge1"].model_copy(
        update={"entry": (StateCondition(**fact("grasp", "none")),)}
    )
    graph = graph.model_copy(
        update={
            "matching": graph.matching.model_copy(update={"interfaces": interfaces})
        }
    )
    query, paths = semantic_query(graph, witness_paths=True)
    assert set(paths) == {"edge0", "edge1"}
    assert len(query.uncertainties) == 1
    assert "contradictory" in query.uncertainties[0]
    assert graph.edges["edge0"].witnesses and graph.edges["edge1"].witnesses


def test_explicit_task_phases_distinguish_requirements_at_one_physical_frame(tmp_path):
    data = corpus_graph(tmp_path).model_dump()
    matching = data["matching"]
    matching["catalog"]["inspection_required"] = {
        "definition": "Whether this object requires inspection in the given task phase.",
        "arguments": ["object", "task phase"],
        "values": {"yes": "Required", "no": "Explicitly not required"},
    }
    matching["entities"].update(
        first_phase={"description": "Initial task phase"},
        next_phase={"description": "Subsequent task phase"},
    )
    matching["interfaces"]["edge0"]["exit"] = [
        fact("inspection_required", "no", "object", "first_phase")
    ]
    matching["interfaces"]["edge1"]["entry"] = [
        fact("inspection_required", "yes", "object", "next_phase")
    ]
    graph = SemanticGraph.model_validate(data)
    query, paths = semantic_query(graph, witness_paths=True)
    assert (("edge0", 0), ("edge1", 0)) in paths.values()
    assert not query.uncertainties

    # Opposing requirements in the SAME declared phase still contradict.
    matching["interfaces"]["edge1"]["entry"] = [
        fact("inspection_required", "yes", "object", "first_phase")
    ]
    graph = SemanticGraph.model_validate(data)
    query, paths = semantic_query(graph, witness_paths=True)
    assert set(paths) == {"edge0", "edge1"}
    assert len(query.uncertainties) == 1


def test_unmatched_phase_is_reported_as_unknown_without_losing_physical_support():
    data = query_graph({"a": "held", "b": "released"}, [("a", "b")]).model_dump()
    data["catalog"]["inspection_required"] = {
        "definition": "Whether this object requires inspection in the given task phase.",
        "arguments": ["object", "task phase"],
        "values": {"yes": "Required", "no": "Explicitly not required"},
    }
    data["entities"]["phase"] = {"description": "Declared task phase"}
    data["interfaces"]["e0"]["entry"] = [
        fact("inspection_required", "yes", "object", "phase")
    ]
    query = GraphQuery.model_validate(data)
    data["interfaces"]["e0"]["entry"] = [
        fact("inspection_required", "no", "object", "phase")
    ]
    result = match_graphs(query, GraphQuery.model_validate(data))
    assert result["distance"] == pytest.approx(1 / 3)
    assert result["unmatched_entities"] == ["phase"]
    assert result["entity_mapping"] == {"object": "object"}
    assert result["unsupported_conditions"][0]["condition"]["fluent"] == (
        "inspection_required"
    )
