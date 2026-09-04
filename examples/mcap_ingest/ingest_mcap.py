# SPDX-License-Identifier: Apache-2.0
"""Index a directory of MCAP recordings into a LanceDB table with Ray.

The shape of this pipeline is dictated by the source format. An MCAP file is
compressed, chunked, and indexed from a summary section at the end, so there is
no way to split one across readers -- but a recording session produces hundreds
or thousands of them, so the parallelism is there, one file per task::

    from_items(paths)  →  map_batches(decode, generator)  →  write_lancedb
    one row per file      one task per file, streamed        one atomic commit

Two things follow from that, and they are the point of the example:

**Files are read on workers, and never held whole.** The decode step is a
*generator*: it yields Arrow batches as it walks the file, so a 40 GB recording
flows through a task that never holds more than one batch. Reading the files
with ``read_binary_files`` instead would pull each one into memory as a single
value, which is the mistake this pattern exists to avoid.

**One schema covers every topic.** MCAP channels each have their own message
schema, so the table stores what is common (topic, timestamps, encoding) as
columns and keeps the message itself as bytes. Lance reads only the columns a
query projects, so counting messages per topic never touches the payloads --
which is what makes one flat table over every topic the right layout rather
than a compromise.

Run with::

    python examples/mcap_ingest/make_sample_mcap.py --out ./sample_logs
    python examples/mcap_ingest/ingest_mcap.py --logs ./sample_logs --uri ./mcap_db
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import lance
import lancedb
import pyarrow as pa
import ray
from lancedb_ray import read_lancedb, write_lancedb

#: Put this directory on the path for the imports below, and ship it to Ray
#: workers as the job's working_dir (see main): the decode task runs on a
#: worker, so ``mcap_source`` has to be importable there too.
EXAMPLE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXAMPLE_DIR))

from mcap_source import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    build_schema,
    discover_files,
    empty_batch,
    iter_record_batches,
    summarize_file,
)

#: Modes this example writes in. ``upsert`` is missing on purpose: a log
#: message has no natural key, so there is nothing to merge on.
IngestMode = Literal["create", "append", "overwrite"]


@dataclass(frozen=True)
class IngestResult:
    """What one ingest run did, in the terms worth asserting on."""

    files: int
    #: Messages the files' own summary sections say are there.
    expected_rows: int
    #: Rows this run added -- not the table's size, so an append onto an
    #: existing table is still checkable against what the files hold.
    rows_written: int
    rows_in_table: int
    fragments: int
    versions: int
    elapsed_s: float

    @property
    def complete(self) -> bool:
        """Every message in the files reached the table."""
        return self.rows_written == self.expected_rows


def decode_files(
    batch: pa.Table,
    *,
    topics: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_payload: bool = True,
) -> Iterator[pa.Table]:
    """Ray task body: turn a batch of file paths into batches of messages.

    A generator, not a function returning a table. Ray consumes the yielded
    batches one at a time, so peak memory in the task is one batch rather than
    one file -- the difference that decides whether the job survives a
    recording with an hour of camera frames in it.
    """
    for path in batch.column("path").to_pylist():
        produced = False
        for record_batch in iter_record_batches(
            path,
            topics=topics,
            batch_size=batch_size,
            include_payload=include_payload,
        ):
            produced = True
            yield pa.Table.from_batches([record_batch])
        if not produced:
            # A file with no matching messages still has to announce the
            # schema, or a run whose first file is empty has nothing to
            # create the table from.
            yield pa.Table.from_batches([empty_batch(include_payload=include_payload)])


def expected_row_count(
    files: Sequence[Path], topics: Optional[Sequence[str]] = None
) -> int:
    """How many rows the write should produce, from the files' own statistics.

    This is the ingest's correctness check, and it is nearly free: an indexed
    MCAP records its message counts in the summary section, so this is one seek
    per file rather than a read of any of them.
    """
    return sum(summarize_file(path).matching_count(topics) for path in files)


def _row_count(uri: str, table: str) -> int:
    """Rows in ``table``, or zero when it does not exist yet.

    Asking by exception rather than by listing: ``table_names()`` is deprecated
    and ``list_tables()`` is paginated, so a missing table is cheapest to
    detect by trying to open it.
    """
    try:
        return int(lancedb.connect(uri).open_table(table).count_rows())
    except ValueError:
        return 0


def ingest(
    files: Sequence[Path],
    *,
    uri: str,
    table: str = "messages",
    topics: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_payload: bool = True,
    mode: IngestMode = "create",
) -> IngestResult:
    """Read every file into ``table``, and report what landed.

    Assumes Ray is already initialised, so a caller that has its own cluster
    or its own runtime env keeps control of it.
    """
    if not files:
        raise ValueError("no MCAP files to ingest")

    started = time.perf_counter()
    expected = expected_row_count(files, topics)
    rows_before = _row_count(uri, table)

    # One block per file, so one task per file: the block is a single path and
    # the task's work is the file behind it.
    paths = ray.data.from_items(
        [{"path": str(path)} for path in files],
        override_num_blocks=len(files),
    )
    decoded = paths.map_batches(
        decode_files,
        batch_size=1,
        batch_format="pyarrow",
        fn_kwargs={
            "topics": list(topics) if topics is not None else None,
            "batch_size": batch_size,
            "include_payload": include_payload,
        },
    )

    write_lancedb(
        decoded,
        table,
        uri=uri,
        mode=mode,
        # Explicit: the decoded batches carry it already, but the table is
        # created from the first task to arrive and every other task has to
        # match it -- including one whose file was empty.
        schema=build_schema(include_payload=include_payload),
    )

    opened = lancedb.connect(uri).open_table(table)
    dataset = lance.dataset(opened.uri)  # type: ignore[attr-defined]
    rows_after = int(opened.count_rows())
    return IngestResult(
        files=len(files),
        expected_rows=expected,
        # An overwrite replaces what was there, so nothing is subtracted.
        rows_written=rows_after - (rows_before if mode == "append" else 0),
        rows_in_table=rows_after,
        fragments=len(dataset.get_fragments()),
        versions=len(dataset.versions()),  # type: ignore[no-untyped-call]
        elapsed_s=time.perf_counter() - started,
    )


def topic_counts(uri: str, table: str = "messages") -> dict[str, int]:
    """Messages per topic, computed over a projection of one column.

    The payload column is never opened: Lance reads the columns the scan asks
    for, so this walks a few MB of strings over a table that is mostly blobs.
    """
    ds = read_lancedb(table, uri=uri, columns=["topic"])
    counts = ds.groupby("topic").count().take_all()
    # Ray names the aggregate column "count()"; find it rather than hard-code.
    key = next(k for k in counts[0] if k != "topic") if counts else "count()"
    return {str(row["topic"]): int(row[key]) for row in counts}


def time_slice(
    uri: str,
    *,
    table: str = "messages",
    start_ns: int,
    end_ns: int,
    topics: Optional[Sequence[str]] = None,
    columns: Optional[list[str]] = None,
) -> ray.data.Dataset:
    """Messages logged in ``[start_ns, end_ns)``, filtered inside the scan.

    The filter is a SQL string evaluated by Lance, not a Ray stage, so rows
    outside the window are never decoded. ``log_time_ns`` rather than the
    timestamp column because an integer bound needs no literal syntax and no
    agreement about what the epoch meant.
    """
    predicate = f"log_time_ns >= {start_ns} AND log_time_ns < {end_ns}"
    if topics:
        # Doubling is how SQL escapes a quote inside a literal. Topic names
        # come from whoever recorded the log, so they are not ours to trust.
        quoted = ", ".join("'" + topic.replace("'", "''") + "'" for topic in topics)
        predicate += f" AND topic IN ({quoted})"
    return read_lancedb(table, uri=uri, columns=columns, filter=predicate)


def _table_bytes(uri: str, table: str) -> int:
    """Bytes the table occupies on disk, payloads included."""
    root = Path(lancedb.connect(uri).open_table(table).uri)  # type: ignore[attr-defined]
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _fmt_bytes(count: float) -> str:
    """Human-readable size, so a small demo run still prints a real number."""
    for unit in ("B", "KB", "MB"):
        if count < 1024:
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs", required=True, help="Directory of .mcap files, or one file"
    )
    parser.add_argument("--uri", default="./mcap_db", help="LanceDB directory")
    parser.add_argument("--table", default="messages", help="Table name")
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Only ingest these topics (default: every topic)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Rows per Arrow batch streamed out of each file",
    )
    parser.add_argument(
        "--no-payload",
        action="store_true",
        help="Index the messages without copying their bytes",
    )
    parser.add_argument(
        "--mode",
        default="create",
        choices=["create", "append", "overwrite"],
        help="How to write the table",
    )
    args = parser.parse_args()

    files = discover_files(Path(args.logs).expanduser())
    if not files:
        raise SystemExit(f"No .mcap files found under {args.logs}")

    # Resolve before Ray starts: workers run with the job's working_dir as
    # their cwd, so a relative --uri would mean a different directory there.
    uri = str(Path(args.uri).expanduser().resolve())
    print(f"Found {len(files)} MCAP file(s) under {args.logs}")

    # working_dir ships this directory to the workers and puts it on their
    # Python path, so the decode task can import ``mcap_source`` on a cluster
    # that has never seen this file.
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"working_dir": str(EXAMPLE_DIR)},
    )
    try:
        result = ingest(
            files,
            uri=uri,
            table=args.table,
            topics=args.topics,
            batch_size=args.batch_size,
            include_payload=not args.no_payload,
            mode=args.mode,
        )
        print(f"Ingested into {uri}")
        print(f"  rows written : {result.rows_written:,}")
        print(f"  expected     : {result.expected_rows:,}  (from MCAP statistics)")
        print(f"  rows in table: {result.rows_in_table:,}")
        # Fragments track tasks, not files: Ray bundles rows up to
        # rows_per_transaction per write task, so a small run lands in one
        # fragment on purpose. Versions is the number that must be 1.
        print(f"  fragments    : {result.fragments}")
        print(f"  versions     : {result.versions}  (atomic commit)")
        print(f"  elapsed      : {result.elapsed_s:.1f}s")

        assert result.complete, (
            f"wrote {result.rows_written} rows but the files hold "
            f"{result.expected_rows}"
        )
        if args.mode == "create":
            assert result.versions == 1, "expected every fragment in one commit"

        print("\nMessages per topic (reads the topic column only):")
        counts = topic_counts(uri, args.table)
        for topic, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {topic:<32} {count:>10,}")

        summaries = [summarize_file(path) for path in files]
        start_ns = min(s.start_time_ns for s in summaries)
        end_ns = max(s.end_time_ns for s in summaries) + 1
        midpoint = start_ns + (end_ns - start_ns) // 2
        window = time_slice(
            uri,
            table=args.table,
            start_ns=start_ns,
            end_ns=midpoint,
            columns=["topic", "log_time_ns", "payload_size"],
        ).materialize()
        print("\nFirst half of the recording, metadata columns only:")
        print(f"  rows         : {window.count():,}")
        print(f"  read tasks   : {window.num_blocks()}")
        print(f"  scanned      : {_fmt_bytes(window.size_bytes())} in memory")
        print(f"  table on disk: {_fmt_bytes(_table_bytes(uri, args.table))}")

        print("\nAll checks passed.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
