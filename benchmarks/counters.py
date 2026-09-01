# SPDX-License-Identifier: Apache-2.0
"""Deterministic counters extracted from a Lance dataset.

Timings drift with the machine; these do not. A write that stops fanning out, a
projection that stops being pushed down, or an append that degenerates into one
transaction per batch all change a counter here by an exact amount, which is why
the counters are the hard gate and the timings are only a trend.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "DatasetCounters",
    "ScanCounters",
    "analyze_scan",
    "dataset_counters",
    "lance_path",
    "open_lance",
]

#: ``db://fake/tmp/x`` is served by the test fake from the local directory
#: ``/tmp/x``, so the underlying Lance dataset is inspectable even for the
#: "remote" backend. Kept in sync with ``tests/_fakes.FAKE_REMOTE_PREFIX``.
_FAKE_PREFIX = "db://fake"

_SUFFIX = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

_METRIC_RE = re.compile(
    r"(?P<key>[a-z_]+)=(?P<value>-?\d+(?:\.\d+)?)\s*(?P<suffix>[KMGT])?\s*(?P<unit>B\b)?"
)

#: The metrics worth asserting on. Everything else in the plan output is timing.
_COUNTER_KEYS = frozenset(
    {
        "output_rows",
        "output_bytes",
        "output_batches",
        "fragments_scanned",
        "ranges_scanned",
        "rows_scanned",
        "bytes_read",
        "iops",
        "requests",
        "indices_loaded",
        "parts_loaded",
    }
)


def lance_path(uri: str, table: str) -> Optional[str]:
    """Location of the Lance dataset backing ``table``, if there is one.

    Returns ``None`` for a real ``db://`` endpoint, which has no such thing --
    that is the whole reason the remote read path exists. An object-store URI is
    returned as a URI, because Lance can open one directly and the atomicity
    guarantee is most worth checking precisely where the commit crosses a
    network.
    """
    if uri.startswith(_FAKE_PREFIX):
        uri = uri[len(_FAKE_PREFIX) :]
    elif uri.startswith("db://"):
        return None
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    if "://" in uri:
        return f"{uri.rstrip('/')}/{table}.lance"
    return os.path.join(uri, f"{table}.lance")


def open_lance(
    uri: str,
    table: str,
    version: Optional[int] = None,
    storage_options: Optional[dict[str, str]] = None,
) -> Any:
    """Open the Lance dataset behind a table, or return ``None``."""
    path = lance_path(uri, table)
    if path is None:
        return None
    if "://" not in path and not os.path.isdir(path):
        return None
    import lance

    try:
        return lance.dataset(path, version=version, storage_options=storage_options)
    except Exception:
        # A table that was never created, or an endpoint that does not expose
        # one. Absence of counters is not a failure; asserting on them is.
        return None


@dataclass
class DatasetCounters:
    """What a write actually produced, as opposed to how fast it produced it."""

    rows: int = 0
    fragments: int = 0
    versions: int = 0
    latest_version: int = 0
    data_files: int = 0
    small_files: int = 0
    deleted_rows: int = 0
    disk_bytes: int = 0
    #: Rows in each fragment, so a lopsided fan-out is visible rather than
    #: hidden behind a fragment count that happens to look right.
    fragment_rows: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "rows": self.rows,
            "fragments": self.fragments,
            "versions": self.versions,
            "latest_version": self.latest_version,
            "data_files": self.data_files,
            "small_files": self.small_files,
            "deleted_rows": self.deleted_rows,
            "disk_bytes": self.disk_bytes,
        }
        if self.fragment_rows:
            data["min_fragment_rows"] = min(self.fragment_rows)
            data["max_fragment_rows"] = max(self.fragment_rows)
        return data


def dataset_counters(
    uri: str, table: str, storage_options: Optional[dict[str, str]] = None
) -> Optional[DatasetCounters]:
    """Read the counters off a local, fake-remote or object-store table."""
    dataset = open_lance(uri, table, storage_options=storage_options)
    if dataset is None:
        return None

    fragments = dataset.get_fragments()
    counters = DatasetCounters(
        rows=int(dataset.count_rows()),
        fragments=len(fragments),
        versions=len(dataset.versions()),
        latest_version=int(dataset.version),
        fragment_rows=[int(f.count_rows()) for f in fragments],
    )

    try:
        stats = dataset.stats.dataset_stats()
        counters.small_files = int(stats.get("num_small_files", 0))
        counters.deleted_rows = int(stats.get("num_deleted_rows", 0))
    except Exception:
        pass

    files = 0
    for fragment in fragments:
        for data_file in fragment.data_files():
            files += 1
            size = getattr(data_file, "file_size_bytes", None)
            if size:
                counters.disk_bytes += int(size)
    counters.data_files = files
    if not counters.disk_bytes:
        path = lance_path(uri, table) or ""
        # Walking only makes sense on a filesystem; on object storage the
        # per-file sizes above are the only source.
        counters.disk_bytes = _dir_size(path) if "://" not in path else 0
    return counters


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total


@dataclass
class ScanCounters:
    """Metrics Lance reports for an actually-executed scan.

    ``rows_scanned`` and ``bytes_read`` are what prove a filter or a projection
    reached the scan instead of being applied after the fact.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    plan: str = ""

    def get(self, key: str, default: float = 0.0) -> float:
        return self.metrics.get(key, default)

    @property
    def bytes_read(self) -> float:
        return self.get("bytes_read")

    @property
    def rows_scanned(self) -> float:
        return self.get("rows_scanned")

    @property
    def output_rows(self) -> float:
        return self.get("output_rows")

    def as_dict(self) -> dict[str, Any]:
        return {f"scan_{k}": v for k, v in sorted(self.metrics.items())}


def parse_plan_metrics(plan: str) -> dict[str, float]:
    """Pull the counter metrics out of Lance's ``analyze_plan`` output."""
    found: dict[str, float] = {}
    for match in _METRIC_RE.finditer(plan):
        key = match.group("key")
        if key not in _COUNTER_KEYS:
            continue
        value = float(match.group("value")) * _SUFFIX[match.group("suffix") or ""]
        # A plan has several nodes; the scan node reports the largest values, and
        # the parents mostly pass them through. Taking the max avoids depending
        # on which node comes first.
        found[key] = max(found.get(key, 0.0), value)
    return found


def analyze_scan(
    uri: str,
    table: str,
    *,
    columns: Optional[list[str]] = None,
    filter: Optional[str] = None,
    scanner_options: Optional[dict[str, Any]] = None,
    storage_options: Optional[dict[str, str]] = None,
) -> Optional[ScanCounters]:
    """Execute a scan through Lance and report what it had to read.

    Runs on the driver rather than through Ray on purpose: the question is what
    the *scan* does with a projection or predicate, and Ray's fan-out only adds
    noise to that.
    """
    dataset = open_lance(uri, table, storage_options=storage_options)
    if dataset is None:
        return None

    options: dict[str, Any] = dict(scanner_options or {})
    if columns is not None:
        options["columns"] = columns
    if filter is not None:
        options["filter"] = filter

    try:
        scanner = dataset.scanner(**options)
        plan = scanner.analyze_plan()
    except Exception:
        return None
    return ScanCounters(metrics=parse_plan_metrics(plan), plan=plan)
