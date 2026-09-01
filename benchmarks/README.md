# Benchmarks

Performance *and* correctness in one pass. Every scenario measures something and
asserts what must be true of the result; a fast wrong answer fails the run.

```bash
make benchmark                    # local tier, ~5 min
make benchmark TIER=smoke         # ~80s, proves the suite itself works
make benchmark TIER=ci            # what CI runs, ~2 min here
make benchmark TIER=full          # opt-in, multi-GB
make benchmark ARGS="--scenario read --repeat 5"
make benchmark-compare            # diff against the committed baseline
make benchmark-clean              # remove stray run directories
```

`python -m benchmarks --list` prints every scenario. `--scenario` takes a
scenario name, a name prefix, or a group (`write`, `read`, `upsert`, `remote`,
`knobs`, `targets`) and is repeatable.

## Why counters, not timings, are the gate

Wall-clock on a shared runner is too noisy for a tight threshold, but most of
what is worth catching is not a timing at all:

| Regression | Counter that catches it |
| --- | --- |
| A write degenerates into one transaction per batch | `versions`, `fragments` |
| Fan-out is lost and the write serialises | `fragments`, `min/max_fragment_rows` |
| Column projection stops reaching the scan | `scan_bytes_ratio` |
| A filter stops reaching the scan | `scan_rows_scanned` vs `scan_output_rows` |
| A request ceiling stops bounding anything | `add_max_bytes`, `add_max_rows` |
| Remote sharding collapses to a single task | `read_blocks`, `take_offsets_count` |

Those are exact and they gate the build. Timings are compared against a baseline
with a wide (2.5x) tolerance and reported as a trend.

## How correctness is checked without the source table

Every generated column is a pure function of its row's `id`. A row that comes
back is verified by recomputing what that id should contain, so verification is
exact at any scale and never needs the source in memory. The id space itself is
pinned by count, sum and both extremes, which a distributed aggregation answers
without materialising anything.

## Layout

| File | What it does |
| --- | --- |
| `harness.py` | Tiers, the timed repeat loop, resource sampling, Ray setup, the cleanup contract |
| `datagen.py` | The five dataset shapes, generated on workers |
| `checks.py` | Assertions and the exact-value comparison |
| `counters.py` | Lance fragment/version counters and `analyze_plan` scan metrics |
| `probe.py` | Tallies the LanceDB calls a run actually issues, across worker processes |
| `report.py` | Terminal table, JSON, baseline diff, JUnit, GitHub job summary |
| `scenarios/` | The scenarios themselves, one module per group |

## Backends

- **local** — a temp directory. The only backend whose timings are meaningful as
  absolute numbers.
- **fake** — the test suite's `db://` stand-in (`tests/_fakes.py`), reused rather
  than copied so the benchmark measures the same fake the live Enterprise tests
  validate. It wraps a real local database narrowed to the remote API surface,
  so request *shape* is real and wall-clock is a local disk's, not a service's.
- **enterprise** — a real endpoint, opt-in via `make benchmark-enterprise` with
  `LANCEDB_URI` and `LANCEDB_API_KEY`. Never runs in CI.
- **s3** — real object storage, opt-in via `make benchmark-s3`. Not a stand-in:
  writes go over HTTP as multipart uploads through Lance's `object_store`, which
  is a genuinely different code path from a local write. Twelve scenarios run
  against it, isolated by key prefix the way local cases are isolated by temp
  directory. This is also where `LANCE_IO_THREADS` and
  `LANCE_UPLOAD_CONCURRENCY` actually move the numbers.

  Against the emulator the repo already ships:

  ```bash
  docker compose -f examples/object_storage/docker-compose.yml up -d
  curl -X PUT http://localhost:4566/lancedb-ray-bench
  BENCH_S3_URI=s3://lancedb-ray-bench/db \
  BENCH_S3_ENDPOINT=http://localhost:4566 \
  AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_REGION=us-east-1 \
    make benchmark-s3 TIER=smoke
  ```

  Missing configuration is a **skip**, not a failure, so a developer without an
  endpoint or credentials still gets a green run.

## Tiers

Tiers change sizes only. **Every correctness check runs in every tier** — the CI
tier is smaller, not weaker.

| Tier | Wall (this machine) | Largest dataset |
| --- | --- | --- |
| `smoke` | ~80s | 20K rows |
| `ci` | ~2 min | 250K narrow / 150K vector / 12K × 1536-dim |
| `local` | ~5 min | 2M narrow / 1M vector |
| `full` | minutes per scenario | 20M narrow / 8M vector |

The harness reads the machine's real CPU, RAM and free disk at startup and
refuses a tier that will not fit, rather than trusting a spec sheet.

## Verified

Last full run on this branch, Apple silicon, all backends:

| Run | Cases | Checks | Result |
| --- | ---: | ---: | --- |
| `local` tier, local + fake | 71 | 433 | green, 5 min 18 s |
| `ci` tier, local + fake | 61 | 376 | green, 1 min 54 s |
| `smoke` tier, s3 only | 14 | 88 | green, 75 s against a real object store |

An earlier run at the `ci` tier with every backend enabled (local, fake, s3 and
enterprise) was also green at 76 cases / 464 checks, with the Enterprise target
skipping for want of credentials.

The object-store run is the one worth pointing at: a fan-out write produced
`fragments=8, versions=1`, so the single-atomic-commit guarantee holds where the
commit actually crosses a network.

## Cleanup

A run leaves nothing behind, including when it is killed:

- every directory lives under one run root, removed in a `finally`, and also
  registered with `atexit` and `SIGINT`/`SIGTERM`;
- Ray's own temp directory is inside that root, and Ray is shut down alongside;
- results go to `benchmarks/results/` (gitignored) or `BENCH_OUT_DIR`, never into
  the tracked tree;
- Enterprise and S3 runs create uniquely suffixed tables, drop them in a
  `finally`, and sweep `bench_*` tables left by an earlier killed run;
- CI asserts `git status --porcelain` is empty afterwards, so a leak fails the
  build rather than being discovered later.

## In CI

The `benchmark` job runs the `ci` tier on every push and pull request, sized for
a 4-vCPU hosted runner. It gates on correctness and counters, uploads the result
JSON as an artifact, and writes a job summary. Timings are compared against
`baselines/ci-ubuntu-latest.json` when it exists — see the note there on why that
file is regenerated by hand.
