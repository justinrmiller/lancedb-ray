# lancedb-ray

Ray Data integration for [LanceDB](https://lancedb.com) and LanceDB Enterprise.

Read and write LanceDB tables as Ray Datasets, using the most parallel strategy each
backend actually supports.

```python
import lancedb_ray as ldbr

# Local / OSS
ds = ldbr.read_lancedb("embeddings", uri="/data/lancedb")
ldbr.write_lancedb(ds, "embeddings_copy", uri="/data/lancedb", mode="create")

# LanceDB Enterprise / Cloud (API key from LANCEDB_API_KEY)
ds = ldbr.read_lancedb("embeddings", uri="db://my-database", region="us-east-1")
ldbr.write_lancedb(ds, "embeddings", uri="db://my-database", mode="upsert", on="id")
```

## Why two strategies

LanceDB presents one Python API over two very different systems, and a Ray integration
that ignores the difference leaves most of the available parallelism on the table:

|                        | Local / OSS (`/path`, `s3://…`)   | Cloud / Enterprise (`db://…`)      |
| ---------------------- | --------------------------------- | ---------------------------------- |
| Underlying storage     | A real Lance dataset              | Opaque remote service              |
| `to_arrow()` / `to_lance()` | Works                        | Raises `NotImplementedError`       |
| `optimize()`           | Works                             | Silent no-op (server-managed)      |
| Fragment access        | Yes                               | No                                 |
| Bulk read primitive    | Fragment scan                     | `take_offsets` / paged queries     |

So `lancedb-ray` does this:

**Reads.** Local tables are read one task per fragment group, with `columns` and `filter`
pushed down into the scan (delegated to [`lance-ray`](https://github.com/lance-format/lance-ray)).
Remote tables have no fragments, so the row space is sharded across read tasks —
positionally via `take_offsets` when there is no filter, and by server-side pagination
when there is.

**Every read pins a table version** before planning shards. A writer landing mid-read
therefore cannot tear the result: shards planned against 100 rows never observe a
200-row table.

**Writes.** Local append-style writes take the fragment path — workers write Lance
fragments in parallel and the driver commits them as a **single atomic transaction**, so
the table advances by exactly one version no matter how many tasks participated. Upserts
and all Cloud/Enterprise writes go through LanceDB's table API with batching and retry.

```
8 Ray blocks  →  8 Lance fragments  →  1 committed version
```

## Install

Not published to PyPI. Install from the repository:

```bash
uv pip install git+ssh://git@github.com/justinrmiller/lancedb-ray.git
```

Or for local development:

```bash
git clone https://github.com/justinrmiller/lancedb-ray.git
cd lancedb-ray
uv venv --python 3.12
make build
```

Requires Python 3.12 or newer. CI tests against 3.12.

## API

### `read_lancedb(table, *, uri, ...) -> ray.data.Dataset`

| Argument | Meaning |
| --- | --- |
| `columns`, `filter` | Projection and SQL predicate, evaluated server-side |
| `version` | Table version to pin (defaults to current) |
| `remote_read_strategy` | `auto` (default), `offsets`, `pagination`, `single` |
| `batch_size` | Rows per read request — sizes each task's requests remotely, and the Lance scanner's batches locally |
| `scanner_options` | Extra Lance scanner options for a local read — `use_scalar_index`, `late_materialization`, `with_row_id` |
| `api_key`, `region`, `host_override` | Cloud/Enterprise connection |
| `storage_options`, `client_config` | Object-store and HTTP client options |
| `namespace_client_impl`, `namespace_client_properties` | Lance Namespace resolution |

### `write_lancedb(ds, table, *, uri, mode="append", ...) -> None`

| Argument | Meaning |
| --- | --- |
| `mode` | `create`, `append`, `overwrite`, `upsert` |
| `on` | Key column(s) to match on — required for `upsert` |
| `partition_on_keys` | Hash-partition on `on` before an upsert (default `True`) |
| `transform_fn` | Per-batch transform applied before writing (e.g. computing embeddings) |
| `on_batch_error` | `raise` (default) or `skip` |
| `local_write_strategy` | `auto` (default), `fragment`, `api` |
| `rows_per_transaction` | Rows Ray bundles per write task = transaction size |
| `max_rows_per_request` | Optional row ceiling; splits a task into several transactions |
| `max_bytes_per_request` | Optional ceiling on a request's payload in bytes — the one that bounds a wide schema |
| `write_parallelism` | Parts the client uploads concurrently within one transaction |
| `data_storage_version`, `enable_stable_row_ids` | Lance file format and row-ID stability for the fragment path |
| `when_matched_update_all`, `when_not_matched_insert_all`, `when_not_matched_by_source_delete` | Merge-insert semantics |

## Examples

Each example lives in its own directory with a README of its own — see
[`examples/`](examples/).

| Example | What it shows |
| --- | --- |
| [`quickstart/`](examples/quickstart/) | The core guarantees on synthetic data: a write fans out across Ray tasks yet lands as one atomic commit, and reads come back fragment-parallel. No extra dependencies. |
| [`clip_image_search/`](examples/clip_image_search/) | A realistic pipeline — scan a directory of JPGs, embed them with CLIP across Ray, write to LanceDB, build a vector index, then search them in plain English from a Streamlit app. |
| [`vllm_generate_embed/`](examples/vllm_generate_embed/) | An LLM pipeline — answer prompts with vLLM (or a small local model), embed each answer, write them to LanceDB, then search what the model said by meaning from a Streamlit app. |
| [`object_storage/`](examples/object_storage/) | Verify writes to S3-compatible object storage — a Floci emulator in Docker Compose plus a large locally generated dataset, asserting the round trip and the single atomic commit. |

## Two traps this library avoids

These are easy to get wrong when writing to LanceDB from a distributed engine, and both
were worth designing around explicitly.

### One transaction per task, not one per batch

Every LanceDB write is a transaction: it produces a new table version and at least one
fragment. The obvious way to write a Ray Dataset — issue a write per incoming batch —
therefore multiplies both. Measured on a 20,480-row append:

| Approach | Versions | Fragments |
| --- | ---: | ---: |
| One write per 1,024-row batch | 21 | 20 |
| One write for the whole task | 2 | 1 |

Thousands of tiny fragments degrade read performance until compaction catches up, and
against Cloud/Enterprise the extra requests burn quota and invite rate limiting.

So a write task hands **all** of its rows to LanceDB as a single `RecordBatchReader`. The
client streams that to the service as multiple parts under one upload and commits once, so
one task is one transaction no matter how many blocks Ray delivered. `rows_per_transaction`
controls how many rows Ray bundles per task, and therefore how large each transaction is.

Local append-style writes skip this path entirely — they write Lance fragments in parallel
and commit them in a single transaction, so the table advances by exactly one version.

### Parallel merge-inserts can silently duplicate keys

Two write tasks that each hold one row for the same key will each find the key absent and
each insert it. Neither task's input is internally ambiguous, so LanceDB accepts both and
the key ends up in the table twice — with no error:

```python
# key 7 in two concurrent merge_inserts, once each
# → table now contains id=7 twice
```

`write_lancedb` therefore hash-partitions the input on the `on` columns before an upsert,
so every row for a given key lands in exactly one task. Where a key is genuinely repeated
in the source, LanceDB then rejects it as an ambiguous merge rather than duplicating it —
a loud error instead of corrupt data.

This costs a shuffle. It changes nothing when the source's keys are already unique (a
unique key can only occupy one block), so `partition_on_keys=False` skips it when you know
that to be true.

## Other notes and trade-offs

- **Appends are at-least-once; upserts are exactly-once.** If LanceDB commits a
  write and the response is lost, no client can distinguish that from a write
  that never landed, so a retry duplicates the batch. The default retry policy
  narrows the window — an append retries only on failures that prove nothing was
  applied, such as a refused connection or a rate-limit rejection, while an
  upsert also retries ambiguous ones like a read timeout because merge-insert is
  idempotent. If you need exactly-once, write with `mode="upsert"` and an `on`
  key; replaying it converges on the same table.
- **`on_batch_error` defaults to `raise`.** Logging a failed write and continuing lets a
  job report success while having silently lost data. `skip` is available when partial
  completion genuinely is preferable, and dropped rows are counted and warned about.
- **`rows_per_transaction` sets task memory; the per-request ceilings set payload
  size.** A task holds all of its rows in memory so a failed write can be retried, and it
  materialises them before either ceiling is consulted — so peak worker memory is set by
  `rows_per_transaction` alone. That is the knob to turn down when a task is being
  OOM-killed, and it is worth sizing in bytes rather than rows: 262,144 rows is a few MB
  of scalars but 1.5GB of 1536-dimension embeddings, and unbounded if a row carries an
  image.

  `max_rows_per_request` and `max_bytes_per_request` bound something narrower — how much
  of that already-materialised data is handed to LanceDB per transaction, and so how much
  the client encodes and uploads at once. Both may be set and the tighter one closes the
  request. Either costs extra transactions.

- **`enable_stable_row_ids` is decided at creation.** Stable IDs survive compaction,
  which is what lets a later job address a row it saw earlier. They are off by default
  because they cost an index — but a table written without them cannot be switched over
  without a rewrite, so it is worth deciding up front rather than discovering later.
- **Local upserts are deliberately not highly parallel.** Concurrent merge-insert against
  a single local dataset contends on the commit lock, so local upsert defaults to
  `concurrency=4` with conflict retry. Remote upserts have no such limit — the service
  serialises for us.
- **Remote reads default to positional `take_offsets`.** The alternative,
  `remote_read_strategy="pagination"`, measured ~2x faster locally and sends a
  constant-size request (two integers) rather than an explicit list of offsets, and
  neither degraded with offset depth over a 1M-row table. That is local measurement
  though, against storage rather than the service — the positional primitive stays the
  default because it is the one with guaranteed positional semantics. If you are pushing
  volume through Enterprise, `pagination` is worth measuring against your endpoint.
- **`batch_size` sets round trips, not total payload.** The same offsets are sent either
  way, just in fewer requests, so raising it mostly buys fewer round trips. Left unset it
  uses each backend's own sizing — 50,000 rows per request remotely, and the Lance
  scanner's default locally. Setting it takes precedence over a `batch_size` named in
  `scanner_options`, the same way `columns` and `filter` do.
- **For a highly selective filter over a large table**, `remote_read_strategy="single"`
  streams the result through one task and often beats sharding it.
- **API keys should come from `LANCEDB_API_KEY`** rather than the `api_key` argument, so
  the secret stays out of Ray task definitions and logs. The connection spec's `repr`
  redacts it either way.

## Tuning

Most of the cost of a large job is inside Lance's native core, not this library, and that
core is configured by environment variables rather than by arguments. They have to be set
on the Ray **workers**, so put them in the runtime environment rather than only in the
driver's shell:

```python
ray.init(runtime_env={"env_vars": {"LANCE_IO_THREADS": "128"}})
```

| Variable | What it controls |
| --- | --- |
| `LANCE_IO_THREADS` | Concurrent read requests to storage. Raising it toward the core count helps a large scan against S3, where the default leaves the link idle. |
| `LANCE_UPLOAD_CONCURRENCY` | Concurrent multipart upload streams. The write-side counterpart, and the first thing to raise if the fragment path is not saturating your bandwidth. |
| `LANCE_INITIAL_UPLOAD_SIZE` | Starting multipart part size, in bytes. Larger parts mean fewer API calls on a big write and more memory per stream. Lance grows the part size as an upload progresses, so this only sets where it starts. |

Two things worth knowing beyond the knobs:

- **Lance is tuned for random access by default.** Point lookups, vector search and
  selective column reads are what the defaults optimise for. A scan-heavy ETL job is the
  case that benefits from raising the two thread counts above.
- **Fixed-size lists for vectors, blob encoding for large binaries.** A vector column
  stored as a fixed-size list compresses better and takes the SIMD path; large binary
  payloads stored with Lance's blob encoding are not read unless a query touches them.
  Both are properties of the schema you write, so they are decided before this library
  sees the data.

These are Lance-core knobs, so the same settings apply to
[`lance-spark`](https://github.com/lance-format/lance-spark), whose
[performance guide](https://lance.org/integrations/spark) covers them in more depth
along with the JVM-only options.

## Development

```bash
make build   # install with dev dependencies
make test    # pytest with branch coverage (gate: 95%)
make lint    # ruff check + format check + mypy
make fix     # auto-fix and format
```

Tests cover both backends. The Cloud/Enterprise paths run against an in-repo fake that
mirrors the real remote API surface — including its restrictions — so `db://` code is
exercised deterministically with no network. To additionally validate against the real
service:

```bash
LANCEDB_URI=db://your-db LANCEDB_API_KEY=... pytest -m enterprise
```

## License

Apache-2.0
