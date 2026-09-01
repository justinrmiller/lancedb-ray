# SPDX-License-Identifier: Apache-2.0
"""Counts the LanceDB API calls a benchmark actually issues.

Some of what we want to assert is not visible in the finished table. Whether
``max_bytes_per_request`` really bounded a transaction, how many requests a
remote read strategy issued, whether an upsert was one merge or many -- those
are properties of the calls made along the way, and the calls happen in Ray
worker processes.

Rather than plumb an actor through, each process appends JSON lines to a file in
a shared directory (benchmarks are single-node by construction) and the driver
reads them back once the job is done. Cheap, no lifecycle to manage, and it
survives a worker being killed.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from typing import Any, Optional

import pyarrow as pa

__all__ = [
    "PROBE_DIR_ENV",
    "install_probe",
    "probe_enabled",
    "record",
]

PROBE_DIR_ENV = "LANCEDB_RAY_BENCH_PROBE_DIR"

_lock = threading.Lock()
_installed = False


def probe_enabled() -> bool:
    return bool(os.environ.get(PROBE_DIR_ENV))


def record(event: str, **fields: Any) -> None:
    """Append one event. Never raises -- a broken probe must not fail a run."""
    directory = os.environ.get(PROBE_DIR_ENV)
    if not directory:
        return
    try:
        path = os.path.join(directory, f"probe-{os.getpid()}.jsonl")
        line = json.dumps({"event": event, "pid": os.getpid(), **fields})
        with _lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:  # pragma: no cover - diagnostics must never break a run
        pass


def _nbytes(data: Any) -> int:
    for attr in ("nbytes", "get_total_buffer_size"):
        value = getattr(data, attr, None)
        if value is None:
            continue
        return int(value() if callable(value) else value)
    return 0


def _num_rows(data: Any) -> int:
    value = getattr(data, "num_rows", None)
    if value is not None:
        return int(value)
    if isinstance(data, list):
        return sum(_num_rows(item) for item in data)
    return 0


class _CountingReader:
    """Wraps a ``RecordBatchReader`` so a streamed write can still be measured.

    The library hands LanceDB a reader, not a table, precisely so one task's
    rows go up as a single streamed transaction. Consuming it here to measure it
    would consume it for the writer too, so the tally is accumulated as the
    writer pulls, and recorded when the stream is exhausted.
    """

    def __init__(self, inner: Any, op: str) -> None:
        self._inner = inner
        self._op = op
        self.rows = 0
        self.nbytes = 0
        self.batches = 0

    def _wrap(self) -> pa.RecordBatchReader:
        def gen() -> Any:
            for batch in self._inner:
                self.rows += batch.num_rows
                self.nbytes += batch.nbytes
                self.batches += 1
                yield batch
            record(
                self._op,
                rows=self.rows,
                nbytes=self.nbytes,
                batches=self.batches,
                streamed=True,
            )

        return pa.RecordBatchReader.from_batches(self._inner.schema, gen())


def _measured(data: Any, op: str) -> Any:
    """Return ``data`` unchanged, recording its size (streaming if need be)."""
    if isinstance(data, pa.RecordBatchReader):
        return _CountingReader(data, op)._wrap()
    record(op, rows=_num_rows(data), nbytes=_nbytes(data), streamed=False)
    return data


class _ProbedMergeBuilder:
    """Delegating proxy over LanceDB's merge-insert builder.

    The builder is fluent, so every chained call has to keep returning a proxy;
    only ``execute`` is actually intercepted.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def execute(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        return self._inner.execute(_measured(data, "merge_insert"), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            return _ProbedMergeBuilder(result) if result is self._inner else result

        return call


class _ProbedTable:
    """Delegating proxy over a LanceDB table that tallies the calls that cost."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def add(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        return self._inner.add(_measured(data, "add"), *args, **kwargs)

    def merge_insert(self, on: Any) -> Any:
        return _ProbedMergeBuilder(self._inner.merge_insert(on))

    def take_offsets(self, offsets: list[int]) -> Any:
        record("take_offsets", count=len(offsets))
        return self._inner.take_offsets(offsets)

    def search(self, *args: Any, **kwargs: Any) -> Any:
        record("search")
        return self._inner.search(*args, **kwargs)

    def count_rows(self, *args: Any, **kwargs: Any) -> int:
        record("count_rows")
        return int(self._inner.count_rows(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ProbedConnection:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def open_table(self, *args: Any, **kwargs: Any) -> Any:
        return _ProbedTable(self._inner.open_table(*args, **kwargs))

    def create_table(self, *args: Any, **kwargs: Any) -> Any:
        record("create_table")
        return _ProbedTable(self._inner.create_table(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def install_probe() -> None:
    """Wrap ``lancedb.connect`` so tables it hands out are counted.

    Installed *after* the fake remote backend, so it wraps whichever connection
    implementation is in play -- local or fake ``db://``.
    """
    global _installed
    if _installed or not probe_enabled():
        return

    import lancedb

    inner_connect = lancedb.connect

    def probed_connect(*args: Any, **kwargs: Any) -> Any:
        return _ProbedConnection(inner_connect(*args, **kwargs))

    lancedb.connect = probed_connect
    _installed = True


def read_events(directory: str) -> list[dict[str, Any]]:
    """Read every event recorded under ``directory``."""
    events: list[dict[str, Any]] = []
    if not os.path.isdir(directory):
        return events
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def clear_events(directory: str) -> None:
    """Drop everything recorded so far, so the next case starts clean."""
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.endswith(".jsonl"):
            # A worker may still be flushing; the next case clears it again.
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(directory, name))


def summarize(
    events: list[dict[str, Any]], event: Optional[str] = None
) -> dict[str, Any]:
    """Collapse events into the counters the checks assert on."""
    rows = [e for e in events if event is None or e.get("event") == event]
    sizes = [int(e.get("nbytes", 0)) for e in rows if "nbytes" in e]
    row_counts = [int(e.get("rows", 0)) for e in rows if "rows" in e]
    return {
        "count": len(rows),
        "total_rows": sum(row_counts),
        "total_bytes": sum(sizes),
        "max_bytes": max(sizes, default=0),
        "max_rows": max(row_counts, default=0),
    }
