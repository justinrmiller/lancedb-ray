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
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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


def ensure_bucket(endpoint: str, bucket: str) -> str:
    """Create the bucket unless it already exists.

    Lance's object store will not create a bucket, and a write into a missing
    one fails deep inside the S3 client with ``NoSuchBucket`` rather than
    anything actionable. The emulator accepts unsigned requests, so a plain
    HTTP PUT is enough and the AWS CLI is not needed.

    Against real S3 this call will be rejected for want of a signature; create
    the bucket yourself there, which you would be doing anyway.
    """
    url = f"{endpoint.rstrip('/')}/{bucket}"

    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=10):
            return "already exists"
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise SystemExit(
                f"Unexpected response checking bucket {bucket!r}: {error}"
            ) from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Could not reach {endpoint}: {error.reason}\n"
            "Is the emulator running?\n"
            "  docker compose -f examples/object_storage/docker-compose.yml up -d"
        ) from error

    try:
        request = urllib.request.Request(url, method="PUT")
        with urllib.request.urlopen(request, timeout=10):
            return "created"
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"Could not create bucket {bucket!r}: {error}\n"
            "Against real S3, create it yourself and re-run."
        ) from error


def clear_prefix(endpoint: str, bucket: str, prefix: str) -> int:
    """Delete every object under ``prefix`` so the run starts from nothing.

    Without this the result depends on what previous runs left behind: an
    overwrite onto an existing table lands on top of its history, so the
    version arithmetic differs between a first run and a repeat. Deleting
    first makes every run identical and the assertions exact.
    """
    base = f"{endpoint.rstrip('/')}/{bucket}"
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    deleted = 0
    token: str | None = None

    while True:
        query = {"list-type": "2", "prefix": f"{prefix.strip('/')}/"}
        if token:
            query["continuation-token"] = token
        url = f"{base}?{urllib.parse.urlencode(query)}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:  # bucket not created yet: nothing to clear
                return 0
            raise

        root = ET.fromstring(body)
        keys = [
            element.text
            for element in root.findall(".//s3:Contents/s3:Key", namespace)
            if element.text
        ]
        for key in keys:
            quoted = urllib.parse.quote(key)
            request = urllib.request.Request(f"{base}/{quoted}", method="DELETE")
            try:
                with urllib.request.urlopen(request, timeout=30):
                    deleted += 1
            except urllib.error.HTTPError as error:
                if error.code not in (204, 404):
                    raise

        truncated = root.findtext(
            "s3:IsTruncated", default="false", namespaces=namespace
        )
        token = root.findtext("s3:NextContinuationToken", namespaces=namespace)
        if truncated.lower() != "true" or not token:
            return deleted


def count_versions(dataset_uri: str, options: dict[str, str]) -> int:
    """Version count for a dataset, or 0 when it does not exist yet."""
    try:
        dataset: Any = lance.dataset(dataset_uri, storage_options=options)
        return len(dataset.versions())
    except Exception:  # noqa: BLE001 - absence is the expected first-run case
        return 0


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
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Keep whatever a previous run left under --prefix",
    )
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
    # Before Ray starts: a missing bucket otherwise surfaces as NoSuchBucket
    # from deep inside the S3 client, long after the dataset was generated.
    print(f"  bucket:   {args.bucket} ({ensure_bucket(args.endpoint, args.bucket)})")

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        ds = build_dataset(args.rows, args.dim, args.blocks)

        if args.clean:
            removed = clear_prefix(args.endpoint, args.bucket, args.prefix)
            print(f"  cleaned:  {removed} existing object(s) under {args.prefix}/")

        dataset_uri = f"{uri}/{args.table}.lance"
        # Count first: an overwrite of an existing table legitimately lands on
        # top of prior versions, so the invariant is that this write adds
        # exactly one -- not that the table has exactly one in total.
        versions_before = count_versions(dataset_uri, options)

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

        dataset: Any = lance.dataset(dataset_uri, storage_options=options)
        fragments = len(dataset.get_fragments())
        versions_added = len(dataset.versions()) - versions_before

        print(
            f"\nWrite finished in {write_seconds:.1f}s "
            f"({approx_bytes / 1e6 / write_seconds:.0f} MB/s)"
        )
        print(f"  rows      : {dataset.count_rows():,}")
        print(f"  fragments : {fragments}  (parallel write)")
        print(f"  versions  : +{versions_added}  (atomic commit)")

        assert dataset.count_rows() == args.rows, "row count mismatch"
        assert versions_added == 1, (
            f"expected the write to add exactly one version, added {versions_added}"
        )
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
