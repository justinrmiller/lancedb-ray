# SPDX-License-Identifier: Apache-2.0
"""Opt-in targets: a real Cloud/Enterprise endpoint and object storage.

Neither runs by default. The Enterprise target costs money and quota and needs
credentials; the object-store target needs an endpoint (the repo's
``examples/object_storage`` compose file provides one). Both create uniquely
named tables and drop them afterwards, including tables left behind by an
earlier run that was killed.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from .. import datagen
from ..checks import id_stats, verify_id_space
from ..harness import BenchRun
from . import ALL_TIERS, register
from ._common import expected_bytes


def _drop_stale(
    uri: str, connect_kwargs: dict[str, Any], *, max_age_s: int = 3600
) -> list[str]:
    """Drop ``bench_*`` tables from runs that never got to clean up.

    The suffix carries the run's start time, so age is readable from the name
    without asking the service for metadata it may not expose.
    """
    import lancedb

    dropped: list[str] = []
    try:
        db = lancedb.connect(uri, **connect_kwargs)
        names = list(db.table_names())
    except Exception:
        return dropped

    cutoff = time.time() - max_age_s
    for name in names:
        if not name.startswith("bench_"):
            continue
        parts = name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        if int(parts[1]) > cutoff:
            continue
        try:
            db.drop_table(name)
            dropped.append(name)
        except Exception:
            continue
    return dropped


def _drop(uri: str, table: str, connect_kwargs: dict[str, Any]) -> None:
    import lancedb

    with contextlib.suppress(Exception):
        lancedb.connect(uri, **connect_kwargs).drop_table(table)


def _missing_config(backend: str) -> str:
    """Why this target cannot run, or an empty string if it can.

    Absent configuration is a skip, not a failure -- the same convention the
    live Enterprise tests follow, so a developer without credentials still gets
    a green run.
    """
    if backend == "enterprise":
        if not os.environ.get("LANCEDB_URI"):
            return "LANCEDB_URI is not set"
        if not os.environ.get("LANCEDB_API_KEY"):
            return "LANCEDB_API_KEY is not set"
    if backend == "s3" and not os.environ.get("BENCH_S3_URI"):
        return "BENCH_S3_URI is not set"
    return ""


def _roundtrip(run: BenchRun, backend: str, label: str) -> None:
    """Write, read back and verify against a shared endpoint."""
    from lancedb_ray import read_lancedb, write_lancedb

    dataset = "vector"
    rows = run.rows(dataset)

    with run.case(f"{label}_roundtrip", backend=backend, dataset=dataset) as case:
        missing = _missing_config(backend)
        if missing:
            case.skip(missing)
            return

        case.set_volume(rows, expected_bytes(dataset, rows))
        uri = case.uri("")
        table = case.table
        connect_kwargs = case.connect_kwargs

        stale = _drop_stale(uri, connect_kwargs)
        if stale:
            case.note(f"dropped {len(stale)} stale table(s) from earlier runs")

        source = datagen.build_dataset(dataset, rows, blocks=run.blocks)

        def run_write(_db_dir: str) -> str:
            # Each timed iteration starts from nothing, so the measurement is a
            # create rather than a create-then-append.
            _drop(uri, table, connect_kwargs)
            write_lancedb(source, table, uri=uri, mode="create", **connect_kwargs)
            return uri

        try:
            case.measure(run_write, fresh=True)

            ds = read_lancedb(table, uri=uri, columns=["id"], **connect_kwargs)
            stats = id_stats(ds)
            case.add_counters(stats.as_dict())
            verify_id_space(case.checks, stats, num_rows=rows)

            from ..checks import compare_tables, read_rows, sample_ids

            ids = sample_ids(rows, count=128)
            got = read_rows(uri, table, ids, connect_kwargs=connect_kwargs)
            case.checks.add(compare_tables(got, datagen.expected_rows(dataset, ids)))
        finally:
            _drop(uri, table, connect_kwargs)


@register(
    "enterprise_roundtrip",
    group="targets",
    description="Write and read a real LanceDB Cloud/Enterprise table",
    backends=("enterprise",),
    tiers=ALL_TIERS,
)
def enterprise_roundtrip(run: BenchRun, backend: str) -> None:
    _roundtrip(run, backend, "enterprise")


@register(
    "objectstore_roundtrip",
    group="targets",
    description="Write and read against S3-compatible object storage",
    backends=("s3",),
    tiers=ALL_TIERS,
)
def objectstore_roundtrip(run: BenchRun, backend: str) -> None:
    """Where the Lance IO knobs finally matter.

    Against a local SSD ``LANCE_IO_THREADS`` and ``LANCE_UPLOAD_CONCURRENCY``
    change almost nothing; against object storage they are the difference
    between a saturated link and an idle one.
    """
    _roundtrip(run, backend, "objectstore")
