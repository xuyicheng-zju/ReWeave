import itertools

from reweave.evolution.distillation.fragment_logic import (
    Bridge,
    Fact,
    Fragment,
    Observation,
    apply,
    certify_replay,
    compare,
    entails,
    fact,
    normalize,
    refine_guards,
    regress,
    saturate,
    solve,
)


def test_finite_symbolic_oracles():
    checks = {}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] = True

    x = fact("x")
    nx = fact("x", value="no")
    y = fact("y")
    ny = fact("y", value="no")
    check("unknown_before_conflict", entails((nx,), (y, x)) == "contradicted")
    check(
        "permutation_invariance",
        all(
            entails((nx,), p) == "contradicted" for p in itertools.permutations((x, y))
        ),
    )
    check(
        "conjunction_intersection",
        entails(
            (
                Fact("x", (), frozenset(["a", "b"])),
                Fact("x", (), frozenset(["b", "c"])),
            ),
            (fact("x", value="b"),),
        )
        == "supported",
    )
    # Exhaustive strict simulation transitivity on 9 binary pre/post contracts.
    conditions = [(), (x,), (nx,)]
    cs = [
        Fragment(str(i), p, q)
        for i, (p, q) in enumerate(itertools.product(conditions, repeat=2))
    ]

    def exact(a, b):
        return any(v["mode"] == "exact" for v in compare(a, b))

    check("strict_reflexivity", all(exact(c, c) for c in cs))
    check(
        "strict_transitivity_729",
        all(
            not (exact(a, b) and exact(b, c)) or exact(a, c)
            for a, b, c in itertools.product(cs, repeat=3)
        ),
    )
    a, b, c = [Fragment(n, p, (y,)) for n, p in [("a", (x,)), ("b", ()), ("c", (nx,))]]
    check(
        "discharge_not_transitive",
        bool(compare(a, b)) and bool(compare(b, c)) and not compare(a, c),
    )
    # First permutation is discharge; later permutation exact.
    q = Fragment("q", (fact("ready", "b"),), (fact("done", "a"), fact("done", "b")))
    r = Fragment("r", (fact("ready", "x"),), (fact("done", "x"), fact("done", "y")))
    check(
        "later_exact_mapping_retained", any(v["mode"] == "exact" for v in compare(q, r))
    )
    check(
        "opaque_obligation_preserved",
        not compare(Fragment("q", (), (x, y)), Fragment("r", (), (x,))),
    )
    bri = Bridge("rename", (x,), (y,), "hypothesis", ("manual",))
    check(
        "unvalidated_bridge_not_admitted",
        entails(saturate((x,), (bri,))[0], (y,)) == "unknown",
    )
    check("bridge_keeps_original", len(saturate((x,), (bri,), True)[0]) == 2)
    open_ = fact("open")
    closed = fact("open", value="no")
    reach = fact("reachable")
    held = fact("held")
    ms = [
        Fragment("open", (closed,), (open_,), frozenset([open_.key]), source="B"),
        Fragment("access", (open_,), (reach,), frozenset([reach.key]), source="C"),
        Fragment("grasp", (reach,), (held,), frozenset([held.key]), source="A"),
    ]
    sol = solve((closed,), (held,), ms)
    check(
        "three_teacher_complement",
        sol["plan"] == ("open", "access", "grasp")
        and certify_replay((closed,), (held,), sol["plan"], ms),
    )
    check(
        "all_single_teachers_fail",
        all(solve((closed,), (held,), [m])["plan"] is None for m in ms),
    )
    check(
        "unknown_frame_blocks_preservation",
        regress((x, y), Fragment("unknown", (), (x,))) is None,
    )
    check(
        "known_frame_allows_preservation",
        regress((x, y), Fragment("known", (), (x,), frozenset([x.key]))) == (y,),
    )
    check(
        "destructive_effect_rejected",
        regress((x, y), Fragment("bad", (), (x, ny), frozenset([x.key, y.key])))
        is None,
    )
    # Exhaustively compare regression against explicit concrete execution, and BFS vs forward reachability.
    worlds = [
        tuple(fact(n, value=v) for n, v in zip(("x", "y"), vs))
        for vs in itertools.product(("yes", "no"), repeat=2)
    ]
    conjs = [
        normalize(fs)
        for fs in itertools.product((None, x, nx), (None, y, ny))
        if True
        for fs in [tuple(f for f in fs if f is not None)]
    ]
    ops = []
    for i, (p, e) in enumerate(itertools.product(conjs, repeat=2)):
        ops.append(Fragment(str(i), p, e, frozenset(f.key for f in e)))
    cases = 0
    for m, g, s in itertools.product(ops, conjs, worlds):
        req = regress(g, m)
        if req is not None and entails(s, req) == "supported":
            after = apply(s, m)
            checkval = after is not None and entails(after, g) == "supported"
            if not checkval:
                raise AssertionError((m, g, s, req))
        cases += 1
    check("regression_soundness_2916", cases == 2916)

    def forward(s, g, ops):
        todo = [normalize(s)]
        seen = set(todo)
        while todo:
            t = todo.pop()
            if entails(t, g) == "supported":
                return True
            for op in ops:
                nxt = apply(t, op)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return False

    planning_cases = 0
    # All pairs from 20 operators with empty/single literal pre and one positive/negative effect.
    small = [m for m in ops if len(m.pre) <= 1 and len(m.post) == 1]
    for subset in itertools.combinations(small, 2):
        for s, g in itertools.product(worlds, conjs):
            result = solve(s, g, subset)
            found = result["plan"] is not None
            if found != forward(s, g, subset):
                raise AssertionError("planner mismatch")
            if found and not certify_replay(s, g, result["plan"], subset):
                raise AssertionError("invalid certificate")
            planning_cases += 1
    check("planner_matches_forward_oracle", True)
    obs = [
        Observation("p", (x, y), "success", "A"),
        Observation("n", (nx, y), "failure", "B"),
        Observation("infra", (x, y), "infrastructure", "C"),
    ]
    learned = refine_guards((), (x, y), obs)
    check("counterexample_adds_guard", learned["guards"] == [(x,)])
    check(
        "inseparable_outcomes_abstain",
        refine_guards(
            (),
            (x,),
            [
                Observation("p", (x,), "success", "A"),
                Observation("n", (x,), "failure", "B"),
            ],
        )["status"]
        == "unresolved_counterexample",
    )
    check(
        "missing_feature_not_negative",
        refine_guards(
            (),
            (x,),
            [
                Observation("p", (x,), "success", "A"),
                Observation("n", (), "failure", "B"),
            ],
        )["status"]
        == "unresolved_counterexample",
    )


def test_result_join_matches_exhaustive_entity_enumeration():
    from reweave.evolution.distillation.fragment_logic import (
        compatible_mappings,
        debt,
        mappings,
    )

    targets = ("u", "v", "w")
    query_results = (
        (),
        (fact("held", "u"),),
        (fact("held", "v"), fact("over", "v", "u")),
        (fact("over", "u", "u"),),
        (fact("ready"),),
        (Fact("held", ("u",), frozenset({"yes", "no"})),),
    )
    reference_results = (
        (fact("held", "a"),),
        (fact("held", "a"), fact("held", "b"), fact("over", "a", "b")),
        (fact("over", "a", "a"), fact("ready")),
    )
    for values in itertools.product((None, "yes", "no"), repeat=3):
        known = tuple(fact("clear", t, value=v) for t, v in zip(targets, values) if v)
        for required, delivered in itertools.product(query_results, reference_results):
            for entry in (
                (),
                (fact("clear", "a"),),
                (fact("clear", "a"), fact("clear", "b", value="no")),
            ):
                q = Fragment("q", known, required)
                r = Fragment("r", entry, delivered)
                expected = {
                    tuple(sorted(phi.items()))
                    for phi in mappings(r, targets)
                    if entails(r.rename(phi).post, q.post) == "supported"
                    and debt(q.pre, r.rename(phi).pre) is not None
                }
                actual = list(compatible_mappings(q, r, targets))
                keys = [tuple(sorted(phi.items())) for phi in actual]
                assert len(keys) == len(set(keys))
                assert set(keys) == expected


def test_fourteen_entity_results_bind_without_factorial_enumeration(monkeypatch):
    from reweave.evolution.distillation import fragment_logic

    def forbidden(*args, **kwargs):
        raise AssertionError("unconstrained factorial enumeration")

    monkeypatch.setattr(fragment_logic, "permutations", forbidden)
    q = Fragment("q", (), tuple(fact(f"role_{i}", f"q{i}") for i in range(14)))
    r = Fragment("r", (), tuple(fact(f"role_{i}", f"r{i}") for i in range(14)))
    matches = compare(q, r)
    assert len(matches) == 1
    assert matches[0]["phi"] == {f"r{i}": f"q{i}" for i in range(14)}
