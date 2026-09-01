# SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding for scenarios.

Keeps the per-scenario code down to what is actually distinctive about it: the
round-trip check bundle, table seeding, and counter collection are identical
almost everywhere and are worth writing once.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import counters as counters_mod
from .. import datagen
from ..checks import id_stats, verify_content, verify_id_space, verify_schema
from ..harness import Case

__all__ = [
    "TABLE",
    "collect_counters",
    "expected_bytes",
    "read_back",
    "seed",
    "verify_roundtrip",
]

#: Every scenario writes to a table of this name; each case has its own
#: directory, so there is nothing to collide with and one name keeps the
#: counter-collection helpers simple.
TABLE = "bench"


def expected_bytes(dataset: str, rows: int) -> int:
    return datagen.get_spec(dataset).bytes_per_row * rows


def seed(
    case: Case,
    db_dir: str,
    dataset: str,
    rows: int,
    *,
    blocks: int,
    table: str = TABLE,
    **write_kwargs: Any,
) -> str:
    """Write a table to measure *against*. Never inside a timed region."""
    from lancedb_ray import write_lancedb

    uri = case.uri(db_dir)
    source = datagen.build_dataset(dataset, rows, blocks=blocks)
    write_lancedb(
        source, table, uri=uri, mode="create", **case.connect_kwargs, **write_kwargs
    )
    return uri


def read_back(
    case: Case,
    uri: str,
    *,
    table: str = TABLE,
    columns: Optional[list[str]] = None,
    **read_kwargs: Any,
) -> Any:
    from lancedb_ray import read_lancedb

    return read_lancedb(
        table, uri=uri, columns=columns, **case.connect_kwargs, **read_kwargs
    )


def collect_counters(case: Case, uri: str, table: str = TABLE, prefix: str = "") -> Any:
    """Record the Lance-level counters for a table, if it has any.

    A real ``db://`` endpoint has no inspectable dataset -- that is the premise
    of the remote code path -- so this is a no-op there rather than a failure.
    """
    found = counters_mod.dataset_counters(
        uri, table, storage_options=case.storage_options
    )
    if found is None:
        case.note("no Lance dataset behind this table; counters unavailable")
        return None
    case.add_counters(found.as_dict(), prefix=prefix)
    return found


def verify_roundtrip(
    case: Case,
    uri: str,
    dataset: str,
    rows: int,
    *,
    table: str = TABLE,
    id_start: int = 0,
    sample: int = 256,
    check_schema: bool = True,
) -> None:
    """The standard bundle: the id space is intact and sampled rows are exact."""
    ds = read_back(case, uri, table=table)
    stats = id_stats(ds.select_columns(["id"]))
    case.add_counters(stats.as_dict())
    verify_id_space(case.checks, stats, num_rows=rows, id_start=id_start)

    if check_schema:
        verify_schema(
            case.checks, ds.schema().base_schema, datagen.get_spec(dataset).schema
        )

    verify_content(
        case.checks,
        uri,
        table,
        dataset,
        num_rows=rows,
        connect_kwargs=case.connect_kwargs,
        sample=sample,
        id_start=id_start,
    )
