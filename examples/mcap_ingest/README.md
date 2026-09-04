# MCAP → LanceDB

Index a directory of [MCAP](https://mcap.dev) recordings — the container format
robotics stacks log into — as one queryable LanceDB table, reading the files in
parallel across Ray and committing them in a single atomic transaction.

The result is a table you can ask "every message on `/odom` in this ten-second
window, across four thousand recordings" and have answered by a scan that never
opens a camera frame.

## Why this example exists

The other examples read sources Ray already knows how to split. MCAP is the
opposite case, and it is the common one for real log data:

**A file is not splittable, so the file is the unit of parallelism.** MCAP
chunks are compressed and the message index lives in a summary section at the
end, so there is no byte offset a second reader could start from. The pipeline
therefore fans out over *files* — `from_items(paths)` and one task each —
rather than asking Ray to split anything.

**A file does not fit in memory, so the task streams.** Recordings are commonly
tens of GB, mostly camera payloads. The decode step is a **generator**: it
yields Arrow batches as it walks the file, so a task holds one batch rather
than one file. The obvious alternative, `ray.data.read_binary_files`, pulls
each file into memory as a single value — which works on the sample data here
and fails on the real thing.

**Channels have different schemas, so the table stores what they share.** Each
MCAP channel carries its own message schema. Rather than a table per topic,
this writes one flat table of message metadata with the message itself kept as
bytes. That is not a compromise: Lance reads only the columns a query projects,
so counting messages per topic touches the `topic` column and nothing else. On
the sample data, the metadata scan below reads **62 KB** out of a **1.5 MB**
table — and the ratio only widens as payloads grow.

```
from_items(paths)  →  map_batches(decode, generator)  →  write_lancedb
one row per file      one task per file, streamed        one atomic commit
```

## Setup

```bash
uv pip install -r examples/mcap_ingest/requirements.txt
```

`mcap` is pure Python and small. It is also part of this repository's dev
dependencies, so `make build` already installs it.

## 1. Get some recordings

Point the example at your own `.mcap` files, or generate a synthetic robot log:

```bash
python examples/mcap_ingest/make_sample_mcap.py --out ./sample_logs
```

Four recordings, four channels at different rates, interleaved by log time and
zstd-compressed — including one binary channel with no schema at all, because
that is legal MCAP and a reader has to survive it.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out` | `./sample_logs` | Output directory |
| `--files` | `4` | Recordings to write |
| `--duration` | `10.0` | Seconds of data per recording |
| `--seed` | `0` | Deterministic: the same seed writes byte-identical files |

## 2. Ingest

```bash
python examples/mcap_ingest/ingest_mcap.py --logs ./sample_logs --uri ./mcap_db
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--logs` | *required* | Directory of `.mcap` files (scanned recursively), or one file |
| `--uri` | `./mcap_db` | LanceDB directory to write into |
| `--table` | `messages` | Table name |
| `--topics` | all | Only ingest these topics — pushed into the MCAP index, not filtered afterwards |
| `--batch-size` | `8192` | Rows per Arrow batch streamed out of each file |
| `--no-payload` | off | Index the messages without copying their bytes |
| `--mode` | `create` | `create`, `append`, or `overwrite` |

## What it checks

The script asserts rather than prints, so a change that breaks it fails loudly.

**Nothing was dropped.** An indexed MCAP records its message counts in its
summary section, so the expected row count is known before the read starts —
one seek per file, no decoding. The run compares it against what actually
landed in the table.

**The write was atomic.** However many tasks wrote, the table advances by
exactly one version. A reader either sees the whole ingest or none of it.

## Expected output

```
Found 4 MCAP file(s) under ./sample_logs
Ingested into /path/to/mcap_db
  rows written : 5,040
  expected     : 5,040  (from MCAP statistics)
  rows in table: 5,040
  fragments    : 1
  versions     : 1  (atomic commit)
  elapsed      : 3.0s

Messages per topic (reads the topic column only):
  /imu                                  4,000
  /odom                                   800
  /camera/front/compressed                200
  /diagnostics                             40

First half of the recording, metadata columns only:
  rows         : 2,520
  read tasks   : 8
  scanned      : 61.6 KB in memory
  table on disk: 1.5 MB

All checks passed.
```

`fragments: 1` is not a failure to parallelise. Ray bundles rows up to
`rows_per_transaction` per write task, so a small run is deliberately one
transaction and one fragment — the alternative, a fragment per file, is how
tables end up with thousands of tiny fragments. `versions` is the number that
must be `1`.

## The table

| Column | Type | Notes |
| --- | --- | --- |
| `source_file` | `string` | Which recording the message came from |
| `topic` | `string` | The channel's topic |
| `schema_name` | `string` | Empty when the channel declares no schema |
| `message_encoding` | `string` | `json`, `cdr`, `protobuf`, or anything a producer chose |
| `sequence` | `int64` | Publisher's counter, widened from uint32 |
| `log_time` | `timestamp[ns]` | Naive — see below |
| `log_time_ns` | `int64` | The raw value, exact, and what filters are written against |
| `publish_time_ns` | `int64` | |
| `payload_size` | `int64` | Kept even when the payload is not |
| `json_payload` | `string` | Message text for JSON channels, null otherwise |
| `payload` | `large_binary` | The message bytes, dropped entirely by `--no-payload` |

## Things worth knowing

- **MCAP timestamps are uint64; Arrow's are int64.** The top half of the range
  is representable in a log and not in a table, so a timestamp above
  `2**63 - 1` is refused with an error naming the file and topic rather than
  wrapped into a negative one. In practice this only happens to logs whose
  clock is not a Unix epoch.
- **`log_time` is a naive timestamp on purpose.** MCAP defines it as
  nanoseconds since "a user-understood epoch". Usually that is the Unix epoch,
  but a robot that boots without a clock logs since boot instead. Stamping UTC
  on the second case would be a lie, so the column stays naive and
  `log_time_ns` keeps the raw value.
- **Filters are pushed down twice.** `--topics` goes into the MCAP reader,
  which uses the chunk index to skip chunks that cannot match, so a narrow
  ingest decompresses a fraction of each file. The query side pushes `columns`
  and `filter` into the Lance scan.
- **A recording with no summary section still reads.** A writer that was killed
  leaves a file with no index and no statistics; the reader falls back to a
  linear scan and the run reports that it had to.
- **Real robot logs are CDR or protobuf, not JSON.** This example stores every
  payload as bytes and decodes only JSON, so ROS 2 messages land intact but
  opaque. To decode them, pass a decoder factory from `mcap-ros2-support` or
  `mcap-protobuf-support` to `make_reader` in `mcap_source.py` and flatten the
  fields you care about into columns of their own — a per-topic table with real
  types is the natural next step once you know which topics you query.
- **Semantic search over log text** — "find where the planner complained about
  the lidar" — is the same shape as [`clip_image_search/`](../clip_image_search/):
  embed `json_payload` in a `map_batches` stage and add a vector column.

## Tests

Unlike the other examples, this one is covered by the repository's test suite
(`tests/test_mcap_example.py`) and runs in CI, because it is the kind of code
people copy. The reader is tested against deliberately awkward files — no
schema, no index, no statistics, undecodable payloads, timestamps Arrow cannot
hold — and the pipeline end to end through real Ray tasks.
