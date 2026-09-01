# SPDX-License-Identifier: Apache-2.0
"""Correctness assertions run alongside every measurement.

A benchmark that reports a good number for a wrong result is worse than no
benchmark, so each scenario states what must be true and the run fails if it is
not -- regardless of how fast it was.

Verification never needs the source table: every generated column is a pure
function of its ``id`` (see :mod:`benchmarks.datagen`), so a row that comes back
is checked against a value recomputed from its id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pyarrow as pa

from . import datagen

__all__ = [
    "CheckResult",
    "compare_tables",
    "read_rows",
    "sample_ids",
    "verify_content",
    "verify_ids",
    "verify_schema",
]


def _short(value: Any, limit: int = 90) -> Any:
    """Keep a failure line readable when the offending value is a 1536-vector."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = repr(value)
        return value if len(text) <= limit else text[: limit - 3] + "..."
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class CheckResult:
    """One assertion and how it came out."""

    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    detail: str = ""

    def __str__(self) -> str:
        mark = "ok" if self.passed else "FAIL"
        body = f"[{mark}] {self.name}"
        if not self.passed:
            body += f": expected {self.expected!r}, got {self.actual!r}"
            if self.detail:
                body += f" ({self.detail})"
        return body


@dataclass
class CheckList:
    """Collects check results for one case."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    def that(
        self,
        name: str,
        passed: bool,
        *,
        expected: Any = None,
        actual: Any = None,
        detail: str = "",
    ) -> CheckResult:
        return self.add(CheckResult(name, bool(passed), expected, actual, detail))

    def equals(
        self, name: str, actual: Any, expected: Any, detail: str = ""
    ) -> CheckResult:
        return self.that(
            name, actual == expected, expected=expected, actual=actual, detail=detail
        )

    def at_least(
        self, name: str, actual: float, minimum: float, detail: str = ""
    ) -> CheckResult:
        return self.that(
            name,
            actual >= minimum,
            expected=f">= {minimum}",
            actual=actual,
            detail=detail,
        )

    def at_most(
        self, name: str, actual: float, maximum: float, detail: str = ""
    ) -> CheckResult:
        return self.that(
            name,
            actual <= maximum,
            expected=f"<= {maximum}",
            actual=actual,
            detail=detail,
        )

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def ok(self) -> bool:
        return not self.failures


def sample_ids(num_rows: int, *, count: int = 256, start: int = 0) -> list[int]:
    """Ids spread across the table, always including the edges.

    Deterministic: the same table always samples the same rows, so a failure is
    reproducible rather than a coin flip.
    """
    if num_rows <= 0:
        return []
    count = min(count, num_rows)
    if count == 1:
        return [start]
    step = (num_rows - 1) / (count - 1)
    return sorted({start + int(round(i * step)) for i in range(count)})


def read_rows(
    uri: str,
    table: str,
    ids: list[int],
    *,
    connect_kwargs: Optional[dict[str, Any]] = None,
    columns: Optional[list[str]] = None,
) -> pa.Table:
    """Fetch specific rows by id, using an API both backends support."""
    import lancedb

    if not ids:
        return pa.table({})
    handle = lancedb.connect(uri, **(connect_kwargs or {})).open_table(table)
    predicate = "id IN (" + ",".join(str(int(i)) for i in ids) + ")"
    query = handle.search(None).where(predicate).limit(None)
    if columns:
        query = query.select(columns)
    return query.to_arrow()


def _float_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Equality that treats NaN as equal to NaN but -0.0 as distinct from 0.0.

    A round trip that preserves a NaN is correct; one that silently flips the
    sign of a zero has lost information, so the two are judged differently.
    """
    both_nan = np.isnan(left) & np.isnan(right)
    same = left == right
    signed_zero_mismatch = (
        (left == 0) & (right == 0) & (np.signbit(left) != np.signbit(right))
    )
    result: np.ndarray = (same & ~signed_zero_mismatch) | both_nan
    return result


def compare_tables(
    actual: pa.Table, expected: pa.Table, *, sort_on: str = "id"
) -> CheckResult:
    """Exact value comparison of two tables, normalised for order and chunking."""
    missing = set(expected.column_names) - set(actual.column_names)
    if missing:
        return CheckResult(
            "content matches",
            False,
            expected=sorted(expected.column_names),
            actual=sorted(actual.column_names),
            detail=f"missing columns {sorted(missing)}",
        )

    columns = list(expected.column_names)
    left = actual.select(columns).sort_by([(sort_on, "ascending")]).combine_chunks()
    right = expected.select(columns).sort_by([(sort_on, "ascending")]).combine_chunks()

    if left.num_rows != right.num_rows:
        return CheckResult(
            "content matches",
            False,
            expected=right.num_rows,
            actual=left.num_rows,
            detail="row count differs",
        )

    for name in columns:
        lcol = left.column(name)
        rcol = right.column(name)
        if lcol.type != rcol.type:
            return CheckResult(
                "content matches",
                False,
                expected=str(rcol.type),
                actual=str(lcol.type),
                detail=f"column {name!r} changed type on the round trip",
            )
        if pa.types.is_floating(lcol.type):
            lv = lcol.to_numpy(zero_copy_only=False)
            rv = rcol.to_numpy(zero_copy_only=False)
            if not bool(_float_equal(lv, rv).all()):
                bad = int(np.argmin(_float_equal(lv, rv)))
                return CheckResult(
                    "content matches",
                    False,
                    expected=_short(rv[bad]),
                    actual=_short(lv[bad]),
                    detail=f"column {name!r} row {bad}",
                )
            continue
        if not lcol.equals(rcol):
            lp, rp = lcol.to_pylist(), rcol.to_pylist()
            bad = next(
                (i for i, (a, b) in enumerate(zip(lp, rp, strict=True)) if a != b), -1
            )
            return CheckResult(
                "content matches",
                False,
                expected=_short(rp[bad]) if bad >= 0 else "-",
                actual=_short(lp[bad]) if bad >= 0 else "-",
                detail=f"column {name!r} row {bad}",
            )

    return CheckResult(
        "content matches", True, expected=right.num_rows, actual=left.num_rows
    )


def verify_content(
    checks: CheckList,
    uri: str,
    table: str,
    dataset: str,
    *,
    num_rows: int,
    connect_kwargs: Optional[dict[str, Any]] = None,
    columns: Optional[list[str]] = None,
    sample: int = 256,
    id_start: int = 0,
) -> None:
    """Check a sample of rows against what their ids say they should contain."""
    ids = sample_ids(num_rows, count=sample, start=id_start)
    if not ids:
        checks.that("content matches", True, expected=0, actual=0, detail="empty table")
        return
    got = read_rows(uri, table, ids, connect_kwargs=connect_kwargs, columns=columns)
    want = datagen.expected_rows(dataset, ids)
    if columns:
        want = want.select(columns)
    checks.add(compare_tables(got, want))


def verify_schema(
    checks: CheckList,
    actual: pa.Schema,
    expected: pa.Schema,
    *,
    name: str = "schema preserved",
) -> None:
    """Field-by-field type check -- a silent cast is a correctness failure."""
    actual_fields = {f.name: f.type for f in actual}
    for field_ in expected:
        got = actual_fields.get(field_.name)
        if got is None:
            checks.that(
                f"{name}: {field_.name}",
                False,
                expected=str(field_.type),
                actual="missing",
            )
            continue
        checks.that(
            f"{name}: {field_.name}",
            got == field_.type,
            expected=str(field_.type),
            actual=str(got),
        )


def verify_ids(
    checks: CheckList,
    ids: Any,
    *,
    num_rows: int,
    id_start: int = 0,
    name: str = "id set intact",
) -> None:
    """Check the whole id space is present exactly once, without holding rows.

    Count, sum and extremes together pin the set: a missing row, a duplicated
    row or a shifted row all break at least one of them.
    """
    values = np.asarray(list(ids), dtype=np.int64)
    expected_sum = int(
        num_rows * id_start + (num_rows - 1) * num_rows // 2 if num_rows else 0
    )
    checks.equals(f"{name}: count", int(values.size), num_rows)
    if values.size:
        checks.equals(f"{name}: sum", int(values.sum()), expected_sum)
        checks.equals(f"{name}: min", int(values.min()), id_start)
        checks.equals(f"{name}: max", int(values.max()), id_start + num_rows - 1)
        checks.equals(f"{name}: distinct", int(np.unique(values).size), num_rows)


def relative_error(actual: float, expected: float) -> float:
    if expected == 0:
        return 0.0 if actual == 0 else math.inf
    return abs(actual - expected) / abs(expected)


@dataclass
class IdStats:
    """Aggregate description of an id column, computed without materialising it."""

    count: int
    total: int
    minimum: int
    maximum: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id_count": self.count,
            "id_sum": self.total,
            "id_min": self.minimum,
            "id_max": self.maximum,
        }


def id_stats(dataset: Any) -> IdStats:
    """Aggregate the id column of a Ray Dataset in one distributed pass."""
    from ray.data.aggregate import Max, Min, Sum

    count = int(dataset.count())
    if count == 0:
        return IdStats(0, 0, 0, 0)
    agg = dataset.aggregate(Sum("id"), Min("id"), Max("id"))
    return IdStats(
        count=count,
        total=int(agg["sum(id)"]),
        minimum=int(agg["min(id)"]),
        maximum=int(agg["max(id)"]),
    )


def verify_id_space(
    checks: CheckList,
    stats: IdStats,
    *,
    num_rows: int,
    id_start: int = 0,
    name: str = "id space",
) -> None:
    """Check the ids are exactly ``[id_start, id_start + num_rows)``.

    Count, sum and both extremes together pin the set closely enough that a lost
    row, a duplicated row or a shifted range breaks at least one of them, and
    none of the four requires holding the column.
    """
    expected_sum = (
        num_rows * id_start + (num_rows - 1) * num_rows // 2 if num_rows else 0
    )
    checks.equals(f"{name}: rows", stats.count, num_rows)
    if num_rows:
        checks.equals(f"{name}: sum", stats.total, expected_sum)
        checks.equals(f"{name}: min", stats.minimum, id_start)
        checks.equals(f"{name}: max", stats.maximum, id_start + num_rows - 1)
