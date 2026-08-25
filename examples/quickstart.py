# SPDX-License-Identifier: Apache-2.0
"""End-to-end demonstration of lancedb-ray against a local LanceDB database.

Generates a vector dataset, writes it with Ray, and verifies the two properties
that make the local write path worth using:

1. The write fans out -- many Ray tasks produce many Lance fragments.
2. The write is atomic -- all of those fragments land in a *single* new table
   version, so a reader never sees a half-written table.

Run with::

    python examples/quickstart.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile

import lance
import lancedb
import numpy as np
import pyarrow as pa
import ray
from lancedb_ray import read_lancedb, write_lancedb


def build_dataset(num_rows: int, dim: int) -> pa.Table:
    """Create a table of ids, embeddings and labels."""
    rng = np.random.default_rng(seed=42)
    vectors = rng.random((num_rows, dim), dtype=np.float32)
    return pa.table(
        {
            "id": pa.array(range(num_rows), pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), pa.float32()), dim
            ),
            "label": pa.array([f"item-{i % 100}" for i in range(num_rows)]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--uri", default=None, help="LanceDB directory (default: temp)")
    args = parser.parse_args()

    uri = args.uri or tempfile.mkdtemp(prefix="lancedb_ray_demo_")
    cleanup = args.uri is None

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        source = build_dataset(args.rows, args.dim)
        ds = ray.data.from_arrow(source).repartition(args.blocks)
        print(f"Writing {args.rows:,} rows across {args.blocks} blocks to {uri}")

        write_lancedb(
            ds,
            "embeddings",
            uri=uri,
            mode="create",
            max_rows_per_file=args.rows // args.blocks,
        )

        table = lancedb.connect(uri).open_table("embeddings")
        dataset = lance.dataset(table.uri)  # type: ignore[attr-defined]

        fragments = len(dataset.get_fragments())
        versions = len(dataset.versions())  # type: ignore[no-untyped-call]
        print(f"  rows written : {table.count_rows():,}")
        print(f"  fragments    : {fragments}  (parallel write)")
        print(f"  versions     : {versions}  (atomic commit)")

        assert table.count_rows() == args.rows
        assert fragments > 1, "expected the write to fan out across workers"
        assert versions == 1, "expected all fragments in a single commit"

        print("\nReading back...")
        result = read_lancedb("embeddings", uri=uri).materialize()
        print(f"  rows read    : {result.count():,}")
        print(f"  blocks       : {result.num_blocks()}  (fragment-parallel read)")
        assert result.count() == args.rows

        projected = read_lancedb(
            "embeddings", uri=uri, columns=["id", "label"], filter="id < 1000"
        )
        print(
            f"  filtered rows: {projected.count():,} with columns "
            f"{projected.schema().names}"
        )
        assert projected.count() == 1000

        print("\nAll checks passed.")
    finally:
        ray.shutdown()
        if cleanup:
            shutil.rmtree(uri, ignore_errors=True)


if __name__ == "__main__":
    main()
