# SPDX-License-Identifier: Apache-2.0
"""Embed a directory of JPGs with CLIP across Ray, then index them in LanceDB.

The pipeline is four stages:

1. ``ray.data.read_binary_files`` scans the directory and reads image bytes in
   parallel, so file IO is distributed rather than done on the driver.
2. ``map_batches`` runs CLIP over each batch. The embedder is a *class*, not a
   function, so Ray keeps it alive as an actor and the model is loaded once per
   worker instead of once per batch -- the difference between a demo and
   something you would actually run.
3. ``write_lancedb`` writes the vectors. Each Ray task commits one transaction,
   and a local write lands every fragment in a single atomic version.
4. LanceDB builds a vector index over the result.

Run with::

    python examples/clip_image_search/ingest.py --images ~/Pictures --uri ./demo_db

Point ``--images`` at any directory of JPGs; it is scanned recursively.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import ray
from lancedb_ray import write_lancedb

#: This directory is put on the driver's path for the imports below, and is
#: also shipped to Ray workers as the job's working_dir (see main) -- the
#: embedder class is constructed on a worker, so ``embedding`` has to be
#: importable there too, not just here.
EXAMPLE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EXAMPLE_DIR))

from embedding import (  # noqa: E402
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    decode_image,
    embed_images,
    load_clip,
)

#: Extensions treated as JPEGs. Anything Pillow can decode would work; this
#: keeps the example's scope obvious.
JPEG_EXTENSIONS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")


def build_schema() -> pa.Schema:
    """Schema for the image table.

    Declared explicitly rather than inferred: the vector column must be a
    fixed-size list for LanceDB to index it, and the table is created before
    any row has been embedded.
    """
    return pa.schema(
        [
            pa.field("path", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("num_bytes", pa.int64()),
            # The JPEG itself lives in the table. Lance is a multimodal store,
            # so the pixels travel with the embedding rather than being a path
            # into a directory that can move, be renamed, or disappear.
            pa.field("image", pa.large_binary()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ]
    )


class ClipEmbedder:
    """Stateful Ray actor that turns image bytes into CLIP vectors.

    Ray constructs one of these per worker and reuses it across batches, so the
    model download and load happen once per worker rather than once per batch.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.device = device
        self.model, self.processor = load_clip(model_name, device)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        images = []
        paths = []
        sizes = []
        payloads = []

        for payload, path in zip(batch["bytes"], batch["path"], strict=True):
            try:
                images.append(decode_image(payload))
            except Exception as error:  # noqa: BLE001 - one bad file must not kill the job
                print(f"  skipping unreadable image {path}: {error}", file=sys.stderr)
                continue
            paths.append(str(path))
            sizes.append(len(payload))
            # Keep the original bytes: they are written alongside the vector.
            payloads.append(bytes(payload))

        if not images:
            # Ray requires the output schema to be stable even for an empty batch.
            return {
                "path": np.array([], dtype=object),
                "filename": np.array([], dtype=object),
                "num_bytes": np.array([], dtype=np.int64),
                "image": np.array([], dtype=object),
                "vector": np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
            }

        vectors = embed_images(self.model, self.processor, images, self.device)
        return {
            "path": np.array(paths, dtype=object),
            "filename": np.array([Path(p).name for p in paths], dtype=object),
            "num_bytes": np.array(sizes, dtype=np.int64),
            "image": np.array(payloads, dtype=object),
            "vector": vectors,
        }


def count_images(directory: Path) -> int:
    """Count JPGs so the run can fail early on an empty or wrong directory."""
    return sum(len(list(directory.rglob(pattern))) for pattern in JPEG_EXTENSIONS)


#: Below this, an IVF_PQ index is not worth building. Lance itself warns that
#: a dataset under 65,536 rows is too small for a meaningful index, and PQ is
#: lossy -- so on a small collection an index is both slower to build and *less*
#: accurate than the exhaustive scan it replaces. A brute-force search over a
#: few thousand 512-dim vectors takes single-digit milliseconds.
INDEX_MIN_ROWS = 65_536


def create_vector_index(table: Any, num_rows: int, force: bool = False) -> str:
    """Build a vector index if the collection is large enough to benefit.

    IVF_PQ trains IVF centroids and PQ codebooks by k-means. That training needs
    substantially more vectors than it has clusters; starved of them it produces
    empty clusters and a quantiser that loses more accuracy than the index saves
    in time.
    """
    if num_rows < INDEX_MIN_ROWS and not force:
        return (
            f"skipped -- {num_rows:,} rows is under {INDEX_MIN_ROWS:,}, where an "
            "exhaustive scan is both fast and exact. Pass --force-index to build "
            "one anyway."
        )

    # Rule of thumb: partitions ~ sqrt(rows). PQ needs the dimension to divide
    # evenly by the sub-vector count.
    num_partitions = max(1, min(4096, int(num_rows**0.5)))
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=num_partitions,
        num_sub_vectors=EMBEDDING_DIM // 16,
        replace=True,
    )
    table.wait_for_index(["vector_idx"])

    note = f"IVF_PQ built with {num_partitions} partitions"
    if num_rows < INDEX_MIN_ROWS:
        note += (
            f" -- note {num_rows:,} rows is under {INDEX_MIN_ROWS:,}, so expect "
            "k-means warnings and recall worse than an exhaustive scan"
        )
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", required=True, help="Directory of JPGs (scanned recursively)"
    )
    parser.add_argument("--uri", default="./demo_db", help="LanceDB directory")
    parser.add_argument("--table", default="images", help="Table name")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device", default="cpu", help="Torch device for inference (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Images per CLIP forward pass"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Embedder actors, i.e. how many model copies run in parallel",
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="Build the index even on a collection too small to benefit",
    )
    args = parser.parse_args()

    directory = Path(args.images).expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    # Resolve before Ray starts. Workers run with the job's working_dir as
    # their cwd, so a relative --uri would otherwise be interpreted against a
    # different directory on the driver and on each worker.
    uri = str(Path(args.uri).expanduser().resolve())

    total = count_images(directory)
    if total == 0:
        raise SystemExit(f"No JPGs found under {directory}")
    print(f"Found {total:,} JPGs under {directory}")

    # working_dir ships this directory to the workers and puts it on their
    # Python path, so the ClipEmbedder actor can import ``embedding`` wherever
    # it is constructed -- including on a cluster that never saw this file.
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"working_dir": str(EXAMPLE_DIR)},
    )
    started = time.perf_counter()
    try:
        # Distributed read: workers pull the bytes, not the driver.
        ds = ray.data.read_binary_files(str(directory), include_paths=True)

        embedded = ds.map_batches(
            ClipEmbedder,
            batch_size=args.batch_size,
            # A fixed-size actor pool: passing a class (not a function) is what
            # makes Ray hold the loaded model between batches.
            compute=ray.data.ActorPoolStrategy(size=args.concurrency),
            fn_constructor_kwargs={"model_name": args.model, "device": args.device},
        )

        print(f"Embedding with {args.model} on {args.device}...")
        write_lancedb(
            embedded,
            args.table,
            uri=uri,
            mode="overwrite",
            schema=build_schema(),
        )

        table = lancedb.connect(uri).open_table(args.table)
        num_rows = table.count_rows()
        elapsed = time.perf_counter() - started
        print(f"  embedded and wrote {num_rows:,} images in {elapsed:.1f}s")

        print("Building vector index...")
        print(f"  {create_vector_index(table, num_rows, args.force_index)}")

        print(
            f"\nDone. Explore it with:\n"
            f"  streamlit run examples/clip_image_search/app.py "
            f"--server.fileWatcherType none -- "
            f"--uri {uri} --table {args.table}"
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
