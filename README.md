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

Requires Python 3.10–3.13.

## API

### `read_lancedb(table, *, uri, ...) -> ray.data.Dataset`

| Argument | Meaning |
| --- | --- |
| `columns`, `filter` | Projection and SQL predicate, evaluated server-side |
| `version` | Table version to pin (defaults to current) |
| `remote_read_strategy` | `auto` (default), `offsets`, `pagination`, `single` |
| `batch_size` | Rows per request issued by each remote read task |
| `api_key`, `region`, `host_override` | Cloud/Enterprise connection |
| `storage_options`, `client_config` | Object-store and HTTP client options |
| `namespace_client_impl`, `namespace_client_properties` | Lance Namespace resolution |

### `write_lancedb(ds, table, *, uri, mode="append", ...) -> None`

| Argument | Meaning |
| --- | --- |
| `mode` | `create`, `append`, `overwrite`, `upsert` |
| `on` | Key column(s) to match on — required for `upsert` |
| `transform_fn` | Per-batch transform applied before writing (e.g. computing embeddings) |
| `on_batch_error` | `raise` (default) or `skip` |
| `local_write_strategy` | `auto` (default), `fragment`, `api` |
| `min_rows_per_write`, `max_rows_per_request` | Request sizing on the API path |
| `when_matched_update_all`, `when_not_matched_insert_all`, `when_not_matched_by_source_delete` | Merge-insert semantics |

## Notes and trade-offs

- **`on_batch_error` defaults to `raise`.** Logging a failed batch and continuing lets a
  write job report success while having silently lost data. `skip` is available when
  partial completion genuinely is preferable, and dropped rows are counted and warned about.
- **Local upserts are deliberately not highly parallel.** Concurrent merge-insert against
  a single local dataset contends on the commit lock, so local upsert defaults to
  `concurrency=4` with conflict retry. Remote upserts have no such limit — the service
  serialises for us.
- **Filtered remote reads use pagination**, which is O(offset) on the server for deep
  pages. For a highly selective filter over a large table, `remote_read_strategy="single"`
  is often faster.
- **API keys should come from `LANCEDB_API_KEY`** rather than the `api_key` argument, so
  the secret stays out of Ray task definitions and logs. The connection spec's `repr`
  redacts it either way.

## Development

```bash
make build   # install with dev dependencies
make test    # pytest with coverage (gate: 90%)
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
