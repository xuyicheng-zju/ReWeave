"""Current student evidence drives queries without pruning the fixed teachers."""

import asyncio
import json

import pytest

from reweave.evolution.distillation.adaptation import unpack
from reweave.evolution.distillation.engine import (
    SharedEvolution,
    SharedInitialObservation,
)
from reweave.evolution.distillation.progress import TopologyObservation, progress_scores
from reweave.evolution.distillation.student_gaps import (
    build_gap_report,
    retrieve,
    teacher_unit_bank,
)
from reweave.evolution.distillation.topology import occurrence_key
from reweave.utils.persistence import atomic_json
from tests.unit_tests.harness.evolution.test_graph_retrieval import corpus_graph
from tests.unit_tests.harness.evolution.test_shared import Author, Evaluator
from tests.unit_tests.harness.evolution.test_shared_topology_integration import (
    shared_plan,
)


def fixture(tmp_path, *, extra_seed=False):
    graphs = {}
    for name in ("alpha", "beta"):
        directory = tmp_path / name
        directory.mkdir()
        graphs[name] = corpus_graph(directory)
    p = shared_plan(graphs).model_copy(
        update={
            "source_graphs": graphs,
            "fixed_topology_scoring": True,
            "teacher_retention": False,
            "retention_weight": 0,
            "sequential_updates": True,
            "checkpoint_selection": "min_loss",
        }
    )
    if extra_seed:
        p = p.model_copy(
            update={
                "sources": tuple(
                    source.model_copy(
                        update={
                            "dev": (
                                *source.dev,
                                source.dev[0].model_copy(
                                    update={
                                        "record_id": source.name + "-dev2",
                                        "initialization": {"seed": 2},
                                    }
                                ),
                            )
                        }
                    )
                    for source in p.sources
                )
            }
        )
    rows = asyncio.run(
        SharedEvolution(
            tmp_path / "data", None, Evaluator(lambda r, b: False)
        ).evaluate(p, p.initial, p.sources, tmp_path / "data/dev")
    )
    return p, rows


def project(p, rows, node):
    return progress_scores(
        p,
        rows,
        [
            TopologyObservation(
                record_id=r.record_id,
                node_id=node,
                citations=(r.evidence[0].path,),
                explanation="observed retained state",
            )
            for r in rows
            if not r.official_success
        ],
    )


def report(p, rows, node, bank=None):
    return build_gap_report(
        p,
        bank if bank is not None else teacher_unit_bank(p.source_graphs),
        p.initial.record_id,
        rows,
        project(p, rows, node),
    )


def f(name, args, value="yes"):
    return {"fluent": name, "arguments": args, "values": [value]}


def donor(effects, entry=(), *, achieved=True):
    return {"has_achieved_witness": achieved, "effects": effects, "entry": list(entry)}


def test_fixed_bank_current_projection_changes_queries_and_candidate_units(tmp_path):
    p, rows = fixture(tmp_path)
    bank = teacher_unit_bank(p.source_graphs)
    original = json.dumps(bank, sort_keys=True)
    first = report(p, rows, "a", bank)
    second = report(p, rows, "b", bank)
    assert len(first["mechanisms"]) == len(second["mechanisms"]) == 2
    assert {g["frontier_node"] for g in first["gaps"]} == {"b"}
    assert {g["frontier_node"] for g in second["gaps"]} == {"c"}
    assert {u for m in first["mechanisms"] for u in m["candidate_units"]} == {
        occurrence_key(t, "edge0") for t in ("alpha", "beta")
    }
    assert {u for m in second["mechanisms"] for u in m["candidate_units"]} == {
        occurrence_key(t, "edge1") for t in ("alpha", "beta")
    }
    assert first["summary"]["case_queries"] == 2
    assert first["summary"]["candidate_units"] == 2
    assert json.dumps(bank, sort_keys=True) == original
    assert all(g["retained_node"] == "b" for g in second["gaps"])
    assert all(
        g["known"] == [f("phase", ["object"], "aligned")] for g in second["gaps"]
    )
    assert "frontier_plan" not in first and "selected_units" not in first


def test_same_mechanism_groups_seeds_but_preserves_each_record_and_query(tmp_path):
    p, rows = fixture(tmp_path, extra_seed=True)
    result = report(p, rows, "a")
    assert len(result["mechanisms"]) == 2
    assert len(result["gaps"]) == 4
    for mechanism in result["mechanisms"]:
        gaps = [g for g in result["gaps"] if g["mechanism"] == mechanism["id"]]
        assert len(gaps) == 2
        assert set(mechanism["record_ids"]) == {g["record_id"] for g in gaps}
        assert set(mechanism["query_ids"]) == {g["id"] for g in gaps}
        assert {g["task"] for g in gaps} == {mechanism["task"]}
        assert {m["query"] for m in result["matches"][mechanism["id"]]} == {
            g["id"] for g in gaps
        }


def test_different_retained_states_of_one_task_are_separate_mechanisms(tmp_path):
    p, rows = fixture(tmp_path, extra_seed=True)
    projection = project(p, rows, "a")
    changed = next(r for r in rows if r.request.instance.task == "alpha")
    projection["topology_records"][changed.record_id]["node_id"] = "b"
    result = build_gap_report(
        p, teacher_unit_bank(p.source_graphs), p.initial.record_id, rows, projection
    )
    assert len(result["mechanisms"]) == 3
    mixed = [m for m in result["mechanisms"] if m["task"] == "alpha"]
    assert {m["retained_node"] for m in mixed} == {"a", "b"}
    assert all(len(m["record_ids"]) == 1 for m in mixed)
    for gap in result["gaps"]:
        expected = "held" if gap["retained_node"] == "a" else "aligned"
        assert gap["known"] == [f("phase", ["object"], expected)]


def test_success_is_protected_without_claiming_every_teacher_node(tmp_path):
    p, rows = fixture(tmp_path)
    rows[1] = rows[1].model_copy(
        update={"official_success": True, "failure_origin": "none"}
    )
    result = report(p, rows, "a")
    assert result["summary"]["failed_conditions"] == 1
    assert len(result["gaps"]) == 1
    successful = next(
        row for row in result["protected_outcomes"] if row["official_success"]
    )
    assert successful["record_id"] == rows[1].record_id
    assert not successful["retained_nodes"]
    assert all(g["record_id"] != rows[1].record_id for g in result["gaps"])


def test_unknown_projection_stays_unknown_and_unresolved(tmp_path):
    p, rows = fixture(tmp_path)
    result = report(p, rows, None)
    assert not result["mechanisms"] and len(result["unresolved"]) == 2
    assert not result["gaps"]
    assert all(not row["retained_nodes"] for row in result["protected_outcomes"])


@pytest.mark.parametrize(
    "mutation",
    ["stale", "test", "infrastructure", "unknown", "missing_record", "wrong_task"],
)
def test_only_current_complete_settled_source_dev_records_are_admitted(
    tmp_path, mutation
):
    p, rows = fixture(tmp_path)
    projection = project(p, rows, "a")
    if mutation in {"stale", "test"}:
        change = (
            {"bundle_id": "different-student"}
            if mutation == "stale"
            else {"split": "test"}
        )
        rows[0] = rows[0].model_copy(
            update={"request": rows[0].request.model_copy(update=change)}
        )
    elif mutation in {"infrastructure", "unknown"}:
        rows[0] = rows[0].model_copy(update={"failure_origin": mutation})
    elif mutation == "missing_record":
        rows.pop()
    else:
        projection["topology_records"][rows[0].record_id]["task"] = "ood"
    with pytest.raises(ValueError):
        build_gap_report(
            p, teacher_unit_bank(p.source_graphs), p.initial.record_id, rows, projection
        )


def test_projection_must_cover_exactly_current_records(tmp_path):
    p, rows = fixture(tmp_path)
    projection = project(p, rows, "a")
    projection["topology_records"].pop(rows[0].record_id)
    with pytest.raises(ValueError):
        build_gap_report(
            p, teacher_unit_bank(p.source_graphs), p.initial.record_id, rows, projection
        )


def test_cached_h0_alias_requires_exact_preserved_initial_observation(tmp_path):
    p, rows = fixture(tmp_path)
    cached = [
        row.model_copy(
            update={
                "request": row.request.model_copy(
                    update={"bundle_id": "cached-h0-alias"}
                )
            }
        )
        for row in rows
    ]
    p = p.model_copy(
        update={
            "initial_observations": tuple(
                SharedInitialObservation.model_construct(record=row) for row in cached
            )
        }
    )
    result = report(p, cached, "a")
    assert result["student_bundle_id"] == p.initial.record_id
    assert result["record_ids"] == [row.record_id for row in cached]
    changed = list(cached)
    changed[0] = changed[0].model_copy(
        update={
            "request": changed[0].request.model_copy(update={"manifest_id": "changed"})
        }
    )
    with pytest.raises(ValueError):
        report(p, changed, "a")


def test_overall_success_never_matches_an_operation(tmp_path):
    from reweave.evolution.contracts import SemanticGraph

    p, rows = fixture(tmp_path)
    graphs = {}
    for task, graph in p.source_graphs.items():
        value = graph.model_dump(mode="json")
        value["matching"]["catalog"]["official_success"] = {
            "definition": "Overall evaluator verdict",
            "arguments": [],
            "values": {"yes": "success", "no": "failure"},
        }
        value["matching"]["nodes"]["c"] = [f("official_success", [])]
        graphs[task] = SemanticGraph.model_validate(value)
    p = p.model_copy(update={"source_graphs": graphs})
    bank = teacher_unit_bank(graphs)
    result = report(p, rows, "b", bank)
    assert all(not gap["desired_results"] for gap in result["gaps"])
    assert not any(result["matches"].values())
    assert all(
        fact["fluent"] != "official_success"
        for unit in bank["units"].values()
        for fact in unit["effects"]
    )


def test_partial_suppliers_remain_available_alongside_full_supplier():
    a, b = f("pickup", ["source_object"]), f("held", ["source_object"])
    units = {"A": donor([a]), "B": donor([b]), "AB": donor([a, b])}
    goals = unpack([f("pickup", ["object"]), f("held", ["object"])])
    hits = retrieve(goals, (), units)
    assert {row["unit"] for row in hits} == set(units)
    assert all(
        row["match_kind"] == "partial" and len(row["uncovered_results"]) == 1
        for row in hits
        if row["unit"] in {"A", "B"}
    )
    full = [row for row in hits if row["match_kind"] == "full"]
    assert full and all(row["unit"] == "AB" for row in full)
    assert all(not row["uncovered_results"] for row in full)


def test_incompatible_binding_variants_never_union_into_fake_full_support():
    units = {
        "donor": donor([f("pickup", ["source_object"]), f("held", ["source_object"])])
    }
    goals = unpack([f("pickup", ["object_a"]), f("held", ["object_b"])])
    hits = retrieve(goals, (), units)
    assert hits
    assert {row["entity_mapping"]["source_object"] for row in hits} == {
        "object_a",
        "object_b",
    }
    assert all(row["match_kind"] == "partial" for row in hits)
    assert all(
        len(row["matched_results"]) == len(row["uncovered_results"]) == 1
        for row in hits
    )


def test_consistent_join_can_cover_results_with_different_local_roles():
    units = {
        "donor": donor(
            [
                f("pickup", ["source_task", "source_object"]),
                f("held", ["source_hand", "source_object"]),
            ]
        )
    }
    goals = unpack(
        [
            f("pickup", ["query_task", "query_object"]),
            f("held", ["query_hand", "query_object"]),
        ]
    )
    full = [row for row in retrieve(goals, (), units) if row["match_kind"] == "full"]
    assert len(full) == 1
    assert full[0]["entity_mapping"] == {
        "source_task": "query_task",
        "source_object": "query_object",
        "source_hand": "query_hand",
    }


def test_candidate_rename_and_bank_order_do_not_change_semantic_recall():
    units = {
        "A": donor([f("pickup", ["source_object"])]),
        "B": donor([f("held", ["source_object"])]),
        "C": donor([f("unrelated", ["source_object"])]),
    }
    goals = unpack([f("pickup", ["object"]), f("held", ["object"])])
    original = retrieve(goals, (), units)
    renamed = {"renamed-" + key: value for key, value in reversed(list(units.items()))}
    changed = retrieve(goals, (), renamed)

    def signatures(rows, *, prefix=""):
        return {
            json.dumps(
                {**row, "unit": row["unit"].removeprefix(prefix)}, sort_keys=True
            )
            for row in rows
        }

    assert signatures(original) == signatures(changed, prefix="renamed-")


def test_entry_conflicts_unknowns_and_unbound_roles_are_distinct():
    units = {
        "donor": donor(
            [f("phase", ["source_object"], "released")],
            [
                f("phase", ["source_object"], "aligned"),
                f("support", ["source_object"]),
                f("access", ["source_container"], "open"),
            ],
        )
    }
    matches = retrieve(
        unpack([f("phase", ["query_object"], "released")]),
        unpack([f("phase", ["query_object"], "held")]),
        units,
    )
    assert len(matches) == 1
    assert matches[0]["entry_conflicts"] == [f("phase", ["query_object"], "aligned")]
    assert matches[0]["missing_entry"] == [f("support", ["query_object"])]
    assert matches[0]["unbound_entry"] == [f("access", ["source_container"], "open")]
    assert matches[0]["entity_mapping"] == {"source_object": "query_object"}


def test_empty_query_and_unachieved_units_never_create_suppliers():
    units = {"failed": donor([f("held", ["source_object"])], achieved=False)}
    assert not retrieve(unpack([f("held", ["object"])]), (), units)
    assert not retrieve((), (), {"succeeded": donor([f("held", ["source_object"])])})


def test_completed_author_recovery_preserves_request_and_does_not_replay_results(
    tmp_path,
):
    p, _ = fixture(tmp_path)

    class Projector:
        async def score(self, plan, bundle, records, directory):
            return project(plan, records, "a" if bundle.record_id == "initial" else "b")

    class CachedAuthor(Author):
        def __init__(self, interrupt=False):
            super().__init__()
            self.interrupt = interrupt

        async def __call__(self, request, directory):
            file = directory / "result.json"
            if file.exists():
                return json.loads(file.read_text())
            if request["round"] == 2 and self.interrupt:
                raise RuntimeError("stop at next author")
            result = await super().__call__(request, directory)
            atomic_json(file, result)
            return result

    root = tmp_path / "run"
    evaluator = Evaluator(lambda r, b: False)
    with pytest.raises(RuntimeError, match="next author"):
        asyncio.run(
            SharedEvolution(
                root, CachedAuthor(True), evaluator, topology_assessor=Projector()
            ).run(p)
        )
    request = root / "rounds/001/request.json"
    original = request.read_bytes()
    assert json.loads(original)["student_gaps"]["student_bundle_id"] == "initial"
    before = len(evaluator.requests)
    author = CachedAuthor()
    asyncio.run(
        SharedEvolution(root, author, evaluator, topology_assessor=Projector()).run(p)
    )
    assert len(evaluator.requests) == before + len(p.sources)
    assert len(author.requests) == 1 and author.requests[0]["round"] == 2
    assert author.requests[0]["student_gaps"]["student_bundle_id"] == "compact"
    assert request.read_bytes() == original


def test_durable_json_identity_uses_json_types_without_rewriting_evidence(tmp_path):
    from reweave.evolution.distillation.engine import bind_json

    path = tmp_path / "projection.json"
    value = {"citations": (), "rate": 1.0}
    bind_json(path, value)
    original = path.read_bytes()
    bind_json(path, value)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="identity changed"):
        bind_json(path, {"citations": ("new evidence",), "rate": 1.0})
