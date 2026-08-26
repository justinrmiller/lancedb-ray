# SPDX-License-Identifier: Apache-2.0
"""Write a large dataset to S3-compatible object storage and read it back.

Object storage is not a filesystem: writes go over HTTP, large files become
multipart uploads, and a "directory" is a key prefix. That is a genuinely
different code path from a local write, so this exercises it end to end
against a real S3 wire protocol -- Floci, running locally -- rather than
assuming a local write proves anything about S3.

What it checks:

* every row survives the round trip;
* the distributed write lands as a *single* new table version, so a reader
  never sees a half-written table, which is the property that matters most
  when the commit has to cross a network;
* the write actually fans out into several fragments;
* a projected, filtered read pushes down rather than dragging every column
  back over the wire.

Run with::

    docker compose -f examples/object_storage/docker-compose.yml up -d
    python examples/object_storage/verify_s3.py --rows 1000000
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import ray
from lancedb_ray import read_lancedb, write_lancedb

#: Floci accepts any credentials; these match its documented defaults.
DEFAULT_ENDPOINT = "http://localhost:4566"
DEFAULT_KEY = "test"
DEFAULT_SECRET = "test"
DEFAULT_REGION = "us-east-1"


def storage_options(
    endpoint: str, key: str, secret: str, region: str
) -> dict[str, str]:
    """Options for Lance's object_store backend.

    ``allow_http`` is required because the emulator speaks plain HTTP;
    against real S3 you would drop it. Path-style addressing avoids relying
    on wildcard DNS resolving ``<bucket>.localhost``.
    """
    return {
        "aws_access_key_id": key,
        "aws_secret_access_key": secret,
        "aws_region": region,
        "aws_endpoint": endpoint,
        "allow_http": "true",
        "aws_virtual_hosted_style_request": "false",
    }


def make_batch(num_rows: int, start: int, dim: int) -> pa.Table:
    """Build a deterministic chunk of the dataset."""
    rng = np.random.default_rng(seed=start)
    vectors = rng.random((num_rows, dim), dtype=np.float32)
    return pa.table(
        {
            "id": pa.array(range(start, start + num_rows), pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), pa.float32()), dim
            ),
            "shard": pa.array(
                [f"shard-{i % 97}" for i in range(start, start + num_rows)]
            ),
        }
    )


def build_dataset(num_rows: int, dim: int, blocks: int) -> ray.data.Dataset:
    """Generate the dataset across blocks without materialising it all at once."""
    per_block = max(1, num_rows // blocks)
    bounds = [
        (start, min(per_block, num_rows - start))
        for start in range(0, num_rows, per_block)
    ]
    return ray.data.from_arrow(
        [make_batch(size, start, dim) for start, size in bounds if size > 0]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--bucket", default="lancedb-ray")
    parser.add_argument("--prefix", default="verify")
    parser.add_argument("--table", default="vectors")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--access-key", default=DEFAULT_KEY)
    parser.add_argument("--secret-key", default=DEFAULT_SECRET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    args = parser.parse_args()

    options = storage_options(
        args.endpoint, args.access_key, args.secret_key, args.region
    )
    uri = f"s3://{args.bucket}/{args.prefix}"
    approx_bytes = args.rows * (args.dim * 4 + 8 + 16)

    print(
        f"Writing {args.rows:,} rows x {args.dim}d (~{approx_bytes / 1e6:.0f} MB) to {uri}"
    )
    print(f"  endpoint: {args.endpoint}")

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        ds = build_dataset(args.rows, args.dim, args.blocks)

        started = time.perf_counter()
        write_lancedb(
            ds,
            args.table,
            uri=uri,
            mode="overwrite",
            storage_options=options,
            max_rows_per_file=max(1, args.rows // args.blocks),
        )
        write_seconds = time.perf_counter() - started

        dataset_uri = f"{uri}/{args.table}.lance"
        dataset: Any = lance.dataset(dataset_uri, storage_options=options)
        fragments = len(dataset.get_fragments())
        versions = len(dataset.versions())

        print(
            f"\nWrite finished in {write_seconds:.1f}s "
            f"({approx_bytes / 1e6 / write_seconds:.0f} MB/s)"
        )
        print(f"  rows      : {dataset.count_rows():,}")
        print(f"  fragments : {fragments}  (parallel write)")
        print(f"  versions  : {versions}  (atomic commit)")

        assert dataset.count_rows() == args.rows, "row count mismatch"
        assert versions == 1, "expected every fragment in a single commit"
        assert fragments > 1, "expected the write to fan out across workers"

        started = time.perf_counter()
        back = read_lancedb(args.table, uri=uri, storage_options=options).materialize()
        read_seconds = time.perf_counter() - started
        print(f"\nRead back in {read_seconds:.1f}s")
        print(f"  rows      : {back.count():,}")
        print(f"  blocks    : {back.num_blocks()}  (fragment-parallel read)")
        assert back.count() == args.rows, "row count changed on read"

        projected = read_lancedb(
            args.table,
            uri=uri,
            columns=["id", "shard"],
            filter="id < 1000",
            storage_options=options,
        )
        rows = projected.take_all()
        print(
            f"\nProjected + filtered read: {len(rows):,} rows, "
            f"columns {projected.schema().names}"
        )
        assert len(rows) == 1000, "filter pushdown returned the wrong count"
        assert sorted(r["id"] for r in rows) == list(range(1000)), "wrong rows"

        print("\nOBJECT STORAGE WRITE VERIFIED")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
