# Attribution retained in NOTICE. Modified for the ReWeave namespace.
"""Official development results for stopping, reporting and SR-only selection."""

from collections.abc import Iterable, Mapping, Sequence

from reweave.evolution.contracts import (
    BundleVersion,
    EvaluationRecord,
    EvolutionProtocol,
    same_execution_settings,
)


def evaluation_cells(
    protocol: EvolutionProtocol,
    bundles: Iterable[str],
    records: Iterable[EvaluationRecord],
    *,
    execution_ids: frozenset[str] | None = None,
) -> dict[tuple[str, str], EvaluationRecord]:
    """Select current-execution observations while preserving historical receipts.

    Keys are Bundle and condition identities. Only the declared current execution
    supplies derived cells. The latest explicit repetition supplies the current
    frontier; an infrastructure-missing latest trial remains unknown. Every older
    repetition remains evidence, and duplicate settled trials remain invalid.
    Records follow archive attempt order; an infra-only cell uses its last attempt.
    """
    pools = {
        "dev": protocol.dev,
        "validation": protocol.validation.conditions if protocol.validation else (),
    }
    manifest = protocol.manifest_id
    executions = (
        execution_ids
        if execution_ids is not None
        else frozenset({protocol.execution_id})
    )
    bundles = frozenset(bundles)
    candidates = {}
    seen = set()
    for record in records:
        request = record.request
        if request.execution not in executions:
            raise ValueError("matrix receipt has different execution conditions")
        if request.execution == protocol.execution_id and not same_execution_settings(
            request.conditions, protocol.execution
        ):
            raise ValueError("matrix receipt has different actual execution conditions")
        if request.manifest_id != manifest:
            raise ValueError("matrix receipt has different condition manifest")
        if (
            request.bundle_id not in bundles
            or request.instance not in pools[request.split]
        ):
            raise ValueError("matrix receipt is outside the frozen archive")
        if record.failure_origin != "infrastructure":
            settled_key = (
                request.bundle_id,
                request.split,
                request.instance.record_id,
                request.manifest_id,
                request.execution,
                request.conditions.repetition,
            )
            if settled_key in seen:
                raise ValueError("duplicate settled matrix cell")
            seen.add(settled_key)
        key = (request.bundle_id, request.instance.record_id)
        candidates.setdefault(key, []).append(record)
    cells = {}
    for key, observations in candidates.items():
        current = [
            record
            for record in observations
            if record.request.execution == protocol.execution_id
        ]
        if not current:
            continue
        latest = max(record.request.conditions.repetition for record in current)
        current = [
            record
            for record in current
            if record.request.conditions.repetition == latest
        ]
        measured = [
            record for record in current if record.failure_origin != "infrastructure"
        ]
        cells[key] = measured[0] if measured else current[-1]
    return cells


def result_matrix(
    protocol: EvolutionProtocol,
    bundles: Iterable[str],
    records: Iterable[EvaluationRecord],
    *,
    split: str = "dev",
    execution_ids: frozenset[str] | None = None,
) -> dict[str, tuple[bool | None, ...]]:
    """Derive official rows; infrastructure and unsettled outcomes stay unknown."""
    bundles = tuple(bundles)
    pool = (
        protocol.dev
        if split == "dev"
        else protocol.validation.conditions
        if protocol.validation
        else ()
    )
    conditions = {instance.record_id: i for i, instance in enumerate(pool)}
    rows = {bundle: [None] * len(conditions) for bundle in bundles}
    cells = evaluation_cells(protocol, bundles, records, execution_ids=execution_ids)
    for (bundle, condition), record in cells.items():
        if record.request.split == split and record.failure_origin in {"none", "agent"}:
            rows[bundle][conditions[condition]] = record.official_success
    return {bundle: tuple(row) for bundle, row in rows.items()}


def passes(row: Sequence[bool | None], threshold: float) -> bool:
    """Only a complete nonempty row can meet an official success threshold."""
    return bool(row) and None not in row and sum(row) / len(row) >= threshold


def target_passed(protocol, development, validation) -> bool:
    """Both gates must pass for the same Bundle; validation is never inferred."""
    return passes(development, protocol.target_success_rate) and (
        protocol.validation is None
        or passes(validation, protocol.validation.target_success_rate)
    )


def success_set(row: Sequence[bool | None]) -> frozenset[int]:
    if any(value is None for value in row):
        raise ValueError("incomplete official result row has no success set")
    return frozenset(i for i, success in enumerate(row) if success)


def official_rank(row: Sequence[bool | None]) -> tuple[int, int]:
    """Confirmed successes, then measured coverage; unknowns remain unscored.

    All candidates share the fixed development denominator. The first value
    orders their success-rate lower bounds; it is not a complete SR when any
    outcome is missing. Coverage breaks equal-count ties before archive recency.
    """
    return sum(value is True for value in row), sum(value is not None for value in row)


def best_version(
    order: Sequence[str],
    versions: Mapping[str, BundleVersion],
    rows: Mapping[str, Sequence[bool | None]],
) -> str | None:
    """Rank official evidence globally, preferring the newest exact tie."""
    return max(
        reversed(order),
        key=lambda v: official_rank(rows[versions[v].bundle_id]),
        default=None,
    )


def changes(
    before: Sequence[bool | None], after: Sequence[bool | None]
) -> dict[str, list[int]] | None:
    if len(before) != len(after):
        raise ValueError("delta requires the same conditions")
    if None in before or None in after:
        return None
    left, right = success_set(before), success_set(after)
    return {"gained": sorted(right - left), "lost": sorted(left - right)}
