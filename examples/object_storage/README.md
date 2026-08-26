# Object storage writes

Verify that `lancedb-ray` writes correctly to S3-compatible object storage, using
a large locally generated dataset and a local AWS emulator — no cloud account,
no credentials, no egress.

## Why this is its own example

Object storage is not a filesystem. Writes go over HTTP, large files become
multipart uploads, listing a "directory" is really a prefix scan, and the commit
that makes a write visible has to cross a network. A local write passing tells
you very little about any of that, so this exercises the real S3 wire protocol
rather than assuming.

[Floci](https://floci.io/) is an AWS emulator whose S3 service speaks the genuine
protocol on `:4566` and accepts any credentials. Lance's `object_store` backend
talks to it exactly as it would to AWS.

## Run it

```bash
docker compose -f examples/object_storage/docker-compose.yml up -d
python examples/object_storage/verify_s3.py --rows 2000000
```

The script creates its bucket if needed and clears anything a previous run left
under `--prefix`, so repeated runs are identical and the AWS CLI is not required.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--rows` | `1000000` | Rows to generate |
| `--dim` | `64` | Vector dimensionality |
| `--blocks` | `8` | Ray blocks, i.e. how wide the write fans out |
| `--bucket` | `lancedb-ray` | Bucket name |
| `--prefix` | `verify` | Key prefix holding the database |
| `--endpoint` | `http://localhost:4566` | S3 endpoint |
| `--no-clean` | *(off)* | Keep whatever a previous run left under `--prefix` |

Tear down with:

```bash
docker compose -f examples/object_storage/docker-compose.yml down
```

Object data lives in `examples/object_storage/data/`; delete it to start clean.

## What it asserts

It fails loudly rather than printing numbers nobody reads.

**Every row survives the round trip.** Written count, read count, and the
contents of a filtered slice all have to agree.

**The write is atomic.** The write adds exactly *one* new table version, no
matter how many workers took part. This matters more over a network than
locally: workers upload independently and the driver commits once, so a reader
either sees none of the data or all of it, never a partially-uploaded table.

The run clears its prefix first so this is exact. An overwrite onto an existing
table legitimately stacks on that table's history, so without a clean start the
version arithmetic differs between a first run and a repeat — and an assertion
loose enough to accept both would no longer be testing atomicity.

**The write fans out.** More than one fragment, so the work really was spread
across workers instead of funnelling through the driver.

**Projection and filtering push down.** A `columns` + `filter` read returns the
right rows without dragging every column back over the wire.

## The storage options that matter

```python
{
    "aws_access_key_id": "test",
    "aws_secret_access_key": "test",
    "aws_region": "us-east-1",
    "aws_endpoint": "http://localhost:4566",
    "allow_http": "true",
    "aws_virtual_hosted_style_request": "false",
}
```

`allow_http` is required because the emulator speaks plain HTTP; drop it against
real S3. Path-style addressing (`aws_virtual_hosted_style_request: false`) avoids
depending on wildcard DNS resolving `<bucket>.localhost`.

Against real S3, delete the endpoint and credentials entirely and let the usual
AWS credential chain apply — the rest of the code is unchanged, which is the
point.

## Notes

**Throughput here is not a benchmark.** The emulator, the workers, and the disk
are all the same laptop. The numbers show the pipeline works and scales with
blocks, not what S3 will do.

**The bucket must exist before writing.** Lance's object store will not create
one, and writing into a missing bucket fails deep in the S3 client with
`NoSuchBucket` rather than anything actionable — so the script checks with a
`HEAD` and creates it with a plain `PUT`. Against real S3 that unsigned `PUT`
will be rejected; create the bucket yourself there, as you would anyway.

**Verified with podman as well as Docker.** The compose file is standard; any
runtime that can run `floci/floci:latest` and publish `:4566` works.
