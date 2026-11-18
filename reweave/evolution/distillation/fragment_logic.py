"""Evidence-carrying fragment transfer. Standalone; Python standard library only.
All formal guarantees are relative to admitted fragments, never inferred from counts.
"""

from collections import deque
from dataclasses import dataclass
from itertools import combinations, permutations


@dataclass(frozen=True, order=True)
class Fact:
    name: str
    args: tuple
    values: frozenset

    @property
    def key(self):
        return self.name, self.args

    def rename(self, phi):
        return Fact(self.name, tuple(phi.get(x, x) for x in self.args), self.values)


def fact(name, *args, value="yes"):
    return Fact(name, tuple(args), frozenset([value]))


def normalize(fs):
    d = {}
    for f in fs:
        d[f.key] = d.get(f.key, f.values) & f.values
        if not d[f.key]:
            raise ValueError("inconsistent facts")
    return tuple(Fact(n, a, v) for (n, a), v in sorted(d.items()))


def entails(known, required):
    """known |= required, with conjunction and order-independent contradiction."""
    k = {f.key: f.values for f in normalize(known)}
    r = normalize(required)
    if any(f.key in k and not k[f.key] & f.values for f in r):
        return "contradicted"
    if all(f.key in k and k[f.key] <= f.values for f in r):
        return "supported"
    return "unknown"


def debt(known, required):
    if entails(known, required) == "contradicted":
        return None
    return normalize(f for f in required if entails(known, (f,)) != "supported")


@dataclass(frozen=True)
class Fragment:
    id: str
    pre: tuple
    post: tuple
    # None means unrestricted writes; explicit set is a declared sound upper bound.
    writes: frozenset | None = None
    source: str = ""
    evidence: tuple = ()
    assurance: str = "candidate"  # candidate / observed / validated / formal
    executor: str | None = None

    def __post_init__(self):
        normalize(self.pre)
        normalize(self.post)
        if self.writes is not None:
            p = {f.key: f.values for f in self.pre}
            for f in self.post:
                if f.key not in self.writes and (
                    f.key not in p or not p[f.key] <= f.values
                ):
                    raise ValueError(
                        "postcondition changes undeclared frame coordinate"
                    )

    def rename(self, phi):
        writes = (
            None
            if self.writes is None
            else frozenset(
                (n, tuple(phi.get(a, a) for a in args)) for n, args in self.writes
            )
        )
        return Fragment(
            self.id,
            tuple(f.rename(phi) for f in self.pre),
            tuple(f.rename(phi) for f in self.post),
            writes,
            self.source,
            self.evidence,
            self.assurance,
            self.executor,
        )


def mappings(source, target):
    # Total on all source fragment parameters. Extra environmental objects must
    # be explicitly supplied by caller, not silently invented or existentialized.
    sa = sorted({a for f in (*source.pre, *source.post) for a in f.args})
    for chosen in permutations(sorted(set(target)), len(sa)):
        yield dict(zip(sa, chosen))


def compatible_mappings(query, reference, universe):
    """Exact relational join, then complete unbound variables with early rejection.

    Every supported query result must unify with a reference result atom. Joining
    these atoms first avoids materializing permutations already ruled out by the
    result. Unknown entry conditions remain admissible; only conflicts prune.
    """
    universe = set(universe)
    variables = sorted({a for f in (*reference.pre, *reference.post) for a in f.args})
    if len(variables) > len(universe):
        return
    pre = normalize(reference.pre)
    known = {f.key: f.values for f in normalize(query.pre)}
    post = normalize(reference.post)
    joins = []
    for goal in normalize(query.post):
        options = [
            f
            for f in post
            if f.name == goal.name
            and len(f.args) == len(goal.args)
            and f.values <= goal.values
        ]
        if not options:
            return
        joins.append((goal, options))
    joins.sort(key=lambda item: len(item[1]))

    def compatible(phi):
        for f in pre:
            if all(a in phi for a in f.args):
                values = known.get((f.name, tuple(phi[a] for a in f.args)))
                if values is not None and not values & f.values:
                    return False
        return True

    def unify(phi, source_args, target_args):
        result = dict(phi)
        inverse = {v: k for k, v in phi.items()}
        for source, target in zip(source_args, target_args):
            if target not in universe:
                return None
            if source in result and result[source] != target:
                return None
            if target in inverse and inverse[target] != source:
                return None
            result[source] = target
            inverse[target] = source
        return result if compatible(result) else None

    def complete(phi):
        remaining = [a for a in variables if a not in phi]
        if not remaining:
            yield phi
            return
        domains = {
            a: [
                v
                for v in sorted(universe - set(phi.values()))
                if compatible({**phi, a: v})
            ]
            for a in remaining
        }
        source = min(remaining, key=lambda a: (len(domains[a]), a))
        for target in domains[source]:
            yield from complete({**phi, source: target})

    def join(index, phi):
        if index == len(joins):
            yield from complete(phi)
            return
        goal, options = joins[index]
        for option in options:
            bound = unify(phi, option.args, goal.args)
            if bound is not None:
                yield from join(index + 1, bound)

    if compatible({}):
        yield from join(0, {})


def compare(query, reference, objects=None):
    universe = (
        objects
        if objects is not None
        else {a for f in (*query.pre, *query.post) for a in f.args}
    )
    candidates = []
    for phi in compatible_mappings(query, reference, universe):
        r = reference.rename(phi)
        if entails(r.post, query.post) != "supported":
            continue
        delta = debt(query.pre, r.pre)
        if delta is not None:
            candidates.append(
                {
                    "mode": "exact" if not delta else "discharge",
                    "phi": phi,
                    "debt": delta,
                    "reference": reference.id,
                    "assurance": reference.assurance,
                }
            )
    # Preserve the historical deterministic permutation order, without enumerating
    # permutations rejected by result unification or an entry contradiction.
    candidates.sort(key=lambda x: tuple(v for _, v in sorted(x["phi"].items())))
    # No first-discharge early return. Preserve all minimal obligation alternatives.
    return [
        x
        for x in candidates
        if not any(
            y is not x
            and entails(x["debt"], y["debt"]) == "supported"
            and entails(y["debt"], x["debt"]) != "supported"
            for y in candidates
        )
    ]


@dataclass(frozen=True)
class Bridge:
    """Directional Horn clause; never deletes the antecedent or opaque facts."""

    id: str
    lhs: tuple
    rhs: tuple
    status: str
    evidence: tuple


def saturate(fs, bridges, allow_hypotheses=False):
    out = normalize(fs)
    used = []
    while True:
        change = False
        for b in bridges:
            if b.status not in ("validated", "formal") and not allow_hypotheses:
                continue
            if (
                entails(out, b.lhs) == "supported"
                and entails(out, b.rhs) != "supported"
            ):
                out = normalize((*out, *b.rhs))
                used.append(b.id)
                change = True
        if not change:
            return out, tuple(used)


def regress(goal, mechanism):
    """Conservative sufficient precondition, NOT a claimed exact weakest precondition."""
    residual = []
    for f in normalize(goal):
        if entails(mechanism.post, (f,)) == "supported":
            continue
        if mechanism.writes is None or f.key in mechanism.writes:
            return None
        residual.append(f)
    try:
        return normalize((*mechanism.pre, *residual))
    except ValueError:
        return None


def apply(state, mechanism):
    if entails(state, mechanism.pre) != "supported":
        return None
    kept = (
        []
        if mechanism.writes is None
        else [f for f in state if f.key not in mechanism.writes]
    )
    return normalize((*kept, *mechanism.post))


def solve(initial, goal, mechanisms):
    """Finite grounded-fragment BFS; shortest certificate under unit call count.
    Exhaustive over reachable regressed conjunctions; no wall-clock truncation.
    Full finite grounding is caller's responsibility. Not physical completeness.
    """
    initial = normalize(initial)
    goal = normalize(goal)
    queue = deque([(goal, ())])
    seen = {goal}
    expanded = 0
    while queue:
        subgoal, plan = queue.popleft()
        expanded += 1
        if entails(initial, subgoal) == "supported":
            return {
                "status": "proved_relative_to_fragments",
                "plan": plan,
                "required": subgoal,
                "expanded": expanded,
                "complete": True,
            }
        for m in mechanisms:
            required = regress(subgoal, m)
            if required is None or required in seen:
                continue
            seen.add(required)
            queue.append((required, (m.id, *plan)))
    return {
        "status": "no_certificate_in_model",
        "plan": None,
        "expanded": expanded,
        "complete": True,
    }


@dataclass(frozen=True)
class Observation:
    id: str
    before: tuple
    outcome: str  # success / failure / unknown / infrastructure
    source: str


def refine_guards(base, atoms, observations):
    """Enumerate subset-minimal empirical separators in a finite predicate basis.
    A failure must contradict the guard; UNKNOWN is not negative evidence.
    No separator => need predicate/observation/variant, never invent semantics.
    """
    atoms = tuple(atoms)
    answers = []
    pos = [o for o in observations if o.outcome == "success"]
    neg = [o for o in observations if o.outcome == "failure"]
    if not pos:
        return {"status": "insufficient_evidence", "guards": []}
    for n in range(len(atoms) + 1):
        for subset in combinations(atoms, n):
            try:
                g = normalize((*base, *subset))
            except ValueError:
                continue
            if any(entails(o.before, g) != "supported" for o in pos):
                continue
            if any(entails(o.before, g) != "contradicted" for o in neg):
                continue
            if any(entails(g, a) == "supported" for a in answers):
                continue
            answers.append(g)
    return {
        "status": "empirical_only" if answers else "unresolved_counterexample",
        "guards": answers,
        "positive_ids": [o.id for o in pos],
        "negative_ids": [o.id for o in neg],
    }


def certify_replay(initial, goal, plan, mechanisms):
    by_id = {m.id: m for m in mechanisms}
    s = initial
    for key in plan:
        s = apply(s, by_id[key])
        if s is None:
            return False
    return entails(s, goal) == "supported"
