# Benchmarking system — plan

Status: **implemented**. Branch: `benchmarking-system`.

This began as a proposal and is kept as the design record. Where the build
disagreed with the plan, the plan has been corrected rather than the other way
round; see [What the first runs found](#what-the-first-runs-found).

## Goal

A `make benchmark` that answers two questions in one run:

1. **Performance** — how fast is a read/write, and what do the tuning knobs actually buy?
2. **Correctness** — did the fast path return the *right* rows, in the right shape, with
   the atomicity and idempotency this library claims?

A benchmark that reports a good number for a wrong result is worse than no benchmark, so
every scenario asserts its invariants and a failed assertion fails the run regardless of
timing.

## Non-goals

- Not a replacement for `make test`. The unit suite covers branches; this covers
  end-to-end behaviour at size.
- Not a LanceDB or Lance benchmark. We measure what *this* library adds or costs on top:
  fan-out, sharding, transaction batching, pushdown.
- Not a claim about Cloud/Enterprise throughput. The fake backend measures code-path
  shape, not service speed, and every fake-backed row is labelled as such in the output.

## The core idea: gate on counters, trend on timings

Wall-clock on a shared CI runner is noisy enough that a tight time-based gate is mostly a
flake generator. But most of what we want to catch is not a timing at all — it is a
deterministic counter:

| Regression | Counter that catches it |
| --- | --- |
| Write degenerates to one transaction per batch | `versions` delta, `fragments` |
| Fan-out lost, write serialises | `fragments`, `num_write_tasks` |
| Column projection stops being pushed down | `bytes_scanned` from the Lance scan stats |
| Filter stops being pushed down | `rows_scanned` vs `rows_returned` |
| Byte ceiling stops bounding a request | observed max request bytes |
| Remote sharding collapses to one task | `num_read_tasks`, request count |
| Upsert shuffle silently skipped | duplicate-key count in the result |

So: **counters and correctness assertions are hard gates everywhere, including CI.
Timings are recorded, compared to a baseline, and only fail on a gross (≥2.5×) regression.**

## What `make benchmark` does

```
make benchmark              # default tier, local backend + fake remote, ~3 min
make benchmark TIER=ci      # the reduced tier CI runs, ~5 min
make benchmark TIER=full    # opt-in, multi-GB, minutes
make benchmark ARGS="--scenario write_local_fragment --repeat 5"
make benchmark-compare      # run, then diff against benchmarks/baselines/<host>.json
make benchmark-enterprise   # opt-in, needs LANCEDB_URI + LANCEDB_API_KEY
make benchmark-clean        # remove any stray run directories
```

Output: a human table on stdout, plus a JSON record per run. Exit non-zero on any
correctness failure, any counter outside its expected value, or (in compare mode) a
timing regression past the threshold.

Implemented as a standalone `python -m benchmarks`, not pytest-benchmark: the pytest
config here carries `--cov`, `--cov-fail-under=95` and a 300s timeout that we would have
to fight, and pytest-benchmark's statistics are tuned for microbenchmarks, not
multi-second distributed jobs.

## Layout

```
benchmarks/
  __init__.py
  __main__.py          # CLI: tiers, scenario selection, repeat, compare, output paths
  harness.py           # run loop, warmup, timing, env capture, resource guards, cleanup
  datagen.py           # deterministic seeded dataset builders
  checks.py            # correctness assertions -> structured pass/fail records
  counters.py          # fragment/version/scan-stat extraction from a Lance dataset
  report.py            # stdout table, JSON writer, baseline diff, GH step summary
  scenarios/
    __init__.py        # registry: name -> scenario, tiers it belongs to, backends
    write_local.py
    read_local.py
    upsert.py
    remote.py
    knobs.py
  baselines/
    ci-ubuntu-latest.json    # committed; regenerated deliberately
  PLAN.md
  README.md
```

`benchmarks` gets added to `[tool.mypy] files` and stays under the repo's strict settings
and ruff rules, like `tests` and `examples` already are.

**No new dependencies.** `numpy` and `psutil` are already pulled in transitively by
`ray[data]`, `pandas` is already in `[dev]`. Nothing to add to `pyproject.toml` beyond the
mypy path.

## Datasets (deterministic, seeded)

| Name | Schema | Purpose |
| --- | --- | --- |
| `narrow` | `id int64`, `label string`, `ts timestamp[us]` | ~32 B/row. Per-row and per-task overhead, where the library's own cost is visible. |
| `vector` | `id`, `vector fixed_size_list<float32, 128>`, `label` | ~536 B/row. The realistic shape; fixed-size list takes the SIMD path. |
| `wide_vector` | `id`, `vector fixed_size_list<float32, 1536>` | ~6 KB/row. The case `max_bytes_per_request` exists for. |
| `wide_scalar` | `id` + 40 scalar columns | Projection pushdown — reading 2 of 41 columns should not read 41. |
| `fidelity` | nullable ints, unicode strings, negative/NaN floats, tz-aware timestamps, nested struct, large_binary | Correctness only, small. Round-trip type and value fidelity. |

All generated from a fixed seed so the expected digest is computable without holding the
source in memory: content checks compare a sorted-by-`id` hash of the read-back table
against a hash of the regenerated source.

## Scenario matrix

### Writes

| Scenario | Measures | Asserts |
| --- | --- | --- |
| `write_local_fragment` | rows/s, MB/s, wall | `versions` delta == 1; `fragments` == expected fan-out; row count; content digest |
| `write_local_api` | same, via `local_write_strategy="api"` | the README's 21-versions-vs-2 comparison stays true; both paths produce identical content |
| `write_create` / `write_overwrite` | wall | overwrite replaces rather than appends; **empty overwrite empties the table** (regression guard for `3ff1eca`) and keeps stable row IDs (`03448b6`) |
| `write_rows_per_transaction` | wall, peak worker RSS, transaction count at 64K / 256K / 1M | peak RSS tracks the knob (this is the documented OOM lever); content identical across all three |
| `write_max_bytes_per_request` | wall, request count on `wide_vector` at 32/128/512 MB | no request exceeds the ceiling; row count and digest unchanged |
| `write_max_rows_per_request` | request count | ceiling honoured; tighter of the two ceilings wins when both set |
| `write_concurrency` | wall vs `concurrency` 1/2/4/8 | fan-out actually scales; single commit preserved |
| `write_file_layout` | wall, fragment count, on-disk bytes for `max_rows_per_file` / `min_rows_per_file` | fragment count matches the arithmetic |
| `write_storage_version` | wall, on-disk bytes, subsequent read wall for `stable` vs `2.1` | round trip identical either way |
| `write_stable_row_ids` | write cost and size delta on/off | IDs survive a compaction when on |

### Upserts

| Scenario | Measures | Asserts |
| --- | --- | --- |
| `upsert_partitioned` | wall, shuffle cost vs `partition_on_keys=False` | final row count == distinct keys |
| `upsert_duplicate_keys` | — | a repeated key raises the ambiguous-merge error rather than landing twice — the trap the README documents |
| `upsert_idempotency` | wall of the replay | running the same upsert twice yields a byte-identical table digest (the exactly-once claim) |
| `upsert_local_concurrency` | wall at the default `concurrency=4` | commit-lock contention doesn't produce lost updates |

### Reads

| Scenario | Measures | Asserts |
| --- | --- | --- |
| `read_local_full` | rows/s, MB/s, block count | full content digest; blocks > 1 (fragment-parallel) |
| `read_local_projection` | wall and `bytes_scanned` for 2-of-41 columns vs all | bytes scanned drops roughly in proportion — this is what proves pushdown |
| `read_local_filter` | wall at 0.1% / 10% / 90% selectivity | returned row set is exactly the predicate's row set, compared against a locally evaluated expectation |
| `read_scanner_options` | wall across `batch_size`, `late_materialization`, `use_scalar_index` | content invariant under every option |
| `read_version_pinning` | — | a writer appending *during* a read does not change the read's result: shards planned against N rows never see 2N. Run with a real concurrent append. |
| `read_remote_strategies` | wall for `offsets` vs `pagination` vs `single`, at two table sizes and two selectivities | all three return identical content; keeps the README's "pagination measured ~2x faster" claim honest, or corrects it |
| `read_batch_size` | request count and wall at 10K / 50K / 200K | round trips fall, content unchanged |
| `read_fidelity` | — | every type in the `fidelity` dataset round-trips: fixed-size list stays fixed-size, timestamp keeps unit and tz, nulls preserved, no silent cast |

### Backends

| Backend | Default | Notes |
| --- | --- | --- |
| Local filesystem (temp dir) | on | The only backend whose timings are meaningful as absolute numbers. |
| Fake remote (`db://fake…`) | on | Reuses `tests/_fakes.py` via the same `sys.path` + `runtime_env` trick `conftest.py` uses, so Ray workers resolve it. Every result row is tagged `backend=fake` and the report prints a one-line caveat. |
| Real Enterprise | opt-in | `make benchmark-enterprise`, gated on `LANCEDB_URI` + `LANCEDB_API_KEY` exactly like `tests/test_enterprise_live.py`. Uniquely-named tables, dropped in `finally`. Never runs in CI. |
| S3 / object storage | opt-in, **shipped** | Reuse `examples/object_storage/docker-compose.yml`. Twelve scenarios run against it, isolated by key prefix. This is where `LANCE_IO_THREADS` and `LANCE_UPLOAD_CONCURRENCY` become measurable — they do almost nothing against a local SSD. |

## Metrics recorded per scenario

- **Timing**: wall per repetition; report median, min, and IQR. Never the mean.
- **Throughput**: rows/s and MB/s over uncompressed Arrow `nbytes`.
- **Ray**: task count, block count, and peak worker RSS sampled by `psutil` from the
  driver across the job.
- **Lance**: fragments, versions, on-disk bytes, file count, and scan stats
  (`rows_scanned`, `bytes_scanned`) where the API exposes them.
- **Library**: request/transaction count, retries, `WriteStats` totals.
- **Correctness**: one record per assertion — name, passed, and on failure the expected
  and actual.

## Noise control

- Fixed seed; data regenerated per scenario, not shared across them.
- One discarded warmup iteration, then `--repeat` (default 3, CI 2) timed iterations.
- Ray initialised once per run with an explicit `num_cpus` and `object_store_memory` so
  results don't shift with whatever else is on the machine; the Ray startup cost is timed
  once and reported separately rather than smeared into scenario timings.
- A fresh dataset directory per iteration, so the first read isn't the only cold one.
  We do **not** pretend to drop the page cache — that needs root and doesn't exist on
  macOS — so every read scenario is explicitly labelled warm or cold.
- Environment captured into the result: git SHA and dirty flag, Python, ray, lancedb,
  lance, pyarrow versions, platform, CPU count, total RAM, and whether `CI` is set.
- A calibration scenario runs first and repeats a trivial job to estimate this machine's
  run-to-run spread; the report prints it, so a 15% difference can be read against the
  noise floor rather than guessed at.

## Tiers and resource budgets

| Tier | Measured wall (Apple silicon) | Free disk required | Largest dataset |
| --- | --- | --- | --- |
| `smoke` | ~80 s | 1 GB | 20K rows |
| `ci` | ~2 min | 3 GB | 250K `narrow`, 150K `vector`, 12K `wide_vector` |
| `local` (default) | ~5 min | 8 GB | 2M `narrow`, 1M `vector` |
| `full` | minutes per scenario, opt-in | 60 GB | 20M `narrow`, 8M `vector` |

The `ci` tier was sized down from the first draft after measuring: the original
sizes took 2 min 32 s here, which left too little headroom once a 4-vCPU hosted
runner is 2-3x slower. It now measures 1 min 58 s locally.

Tier only scales row counts and sweep breadth. **Every correctness assertion runs in
every tier** — the `ci` tier is smaller, not weaker.

## GitHub Actions

A third job in `.github/workflows/ci.yml`, sized for a standard GitHub-hosted
`ubuntu-latest` runner (4 vCPU / 16 GB RAM / ~14 GB free SSD — the harness reads the
actual values and refuses to start a tier that doesn't fit, rather than trusting the spec):

```yaml
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 15          # the tier budgets 8; this is the backstop
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { enable-cache: true }
      - run: uv venv --python 3.12 && echo "$PWD/.venv/bin" >> "$GITHUB_PATH"
      - run: uv pip install -e ".[dev]"
      - name: Benchmark
        env:
          RAY_ENABLE_UV_RUN_RUNTIME_ENV: "0"
          BENCH_RUN_ROOT: ${{ runner.temp }}/bench   # never the workspace
        run: make benchmark TIER=ci ARGS="--compare benchmarks/baselines/ci-ubuntu-latest.json --junit $RUNNER_TEMP/bench.xml"
      - name: Job summary
        if: always()
        run: cat "$RUNNER_TEMP/bench/summary.md" >> "$GITHUB_STEP_SUMMARY"
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: benchmark-results
          path: ${{ runner.temp }}/bench/*.json
          retention-days: 30
      - name: Verify no leftovers
        if: always()
        run: |
          rm -rf "$RUNNER_TEMP/bench"
          test -z "$(git status --porcelain)" || { git status --porcelain; exit 1; }
```

Runner-specific adjustments the harness makes when `CI` is set:

- `ray.init(num_cpus=4, object_store_memory=2 GB, _temp_dir=$RUNNER_TEMP/ray)` — the
  default object-store sizing reads `/dev/shm`, which is small and inconsistent on hosted
  runners, and Ray's spill directory otherwise lands on the workspace disk.
- `--repeat 2` instead of 3, warmup still discarded.
- Free-disk check before each scenario; skip-with-warning rather than fail if a tier
  can't fit, so a runner spec change degrades instead of breaking the build.
- Timing comparison uses the committed `ci-ubuntu-latest.json` baseline with a **2.5×**
  threshold. Counters and correctness use exact expectations.

**On PRs** the job runs the `ci` tier and gates on correctness + counters only. **On
pushes to `main`** it also uploads the timing JSON, which is what the baseline is
regenerated from — deliberately, by a maintainer running `make benchmark-baseline`, never
automatically.

## Cleanup contract

The requirement is that a run leaves nothing behind, including when it is killed.

- Every scenario allocates its directories under a single run root
  (`$BENCH_RUN_ROOT`, else `tempfile.mkdtemp`), and the harness removes that root in a
  `finally`.
- The root is additionally registered with `atexit` and a `SIGINT`/`SIGTERM` handler, so
  Ctrl-C and a CI cancellation still clean up.
- `ray.shutdown()` in the same `finally`; Ray's own `_temp_dir` is inside the run root.
- Results and JUnit XML are the only outputs, and they go to `$BENCH_RUN_ROOT/..` or an
  explicit `--out`, never into the working tree. `benchmarks/results/` is gitignored as
  a belt-and-braces default for local runs.
- Enterprise scenarios create uniquely-suffixed tables and drop them in a `finally`, plus
  a startup sweep that drops any `bench_*` table older than an hour left by a killed run.
- `make benchmark-clean` removes stray run roots by prefix.
- CI verifies `git status --porcelain` is empty after the job, so a leak fails the build
  rather than being discovered later.

## Phases

**Phase 1 — harness and the local core.** `harness.py`, `datagen.py`, `checks.py`,
`counters.py`, `report.py`, the `write_local_fragment` / `write_local_api` /
`read_local_full` / `read_local_projection` / `read_local_filter` / `read_fidelity`
scenarios, tiers, cleanup contract, `make benchmark`. Shippable on its own.

**Phase 2 — CI.** The `ci` tier sizing, the workflow job, the step summary, the committed
baseline, `--compare`, the leftover check.

**Phase 3 — remote and upsert.** Fake-backend wiring, `read_remote_strategies`,
`read_batch_size`, the four upsert scenarios, `read_version_pinning`.

**Phase 4 — knobs and opt-in targets.** The sweep scenarios, `make benchmark-enterprise`,
and the object-storage target where the `LANCE_IO_THREADS` /
`LANCE_UPLOAD_CONCURRENCY` numbers actually mean something. Object storage ended up
a first-class backend for twelve scenarios rather than a single round trip, because
per-iteration isolation turned out to be free: the key prefix a case would have used
as a temp directory works unchanged as an object-store prefix.

## Risks and open questions

- **Ray startup dominates small scenarios.** Mitigated by initialising once and timing
  only the job, but it caps how small a scenario can usefully be.
- **Fake-backend timings are not service timings.** Guardrail: labelled in every row and
  never written to a baseline that anything gates on.
- **The README makes measured claims** (pagination ~2x, 21 versions vs 2) that were
  measured once, by hand. Phase 1 and 3 turn them into scenarios; if a number turns out
  not to reproduce, the README gets corrected rather than the benchmark tuned.
- **Hosted-runner specs change.** Hence reading actual CPU/RAM/disk at startup and
  degrading rather than asserting the spec.
- **Baseline drift.** Regenerating the baseline is a deliberate, reviewed commit — an
  auto-updating baseline would ratchet away exactly the regression it should catch.

### Decisions taken

1. **Bare `make benchmark` runs the `local` tier** (~4 min). `TIER=` selects another.
2. **The CI job gates PRs** on correctness and counters; timings only trend, compared
   against a committed baseline at a 2.5x tolerance.
3. **`benchmarks/` is outside the 95% coverage gate** but inside ruff and strict mypy. It
   is tooling, exercised by being run, not by the unit suite.

## What the first runs found

Three things the suite established on its first green run, which is the point of
building it:

- **`rows_per_transaction` is a target, not a ceiling.** Ray bundles whole blocks, so a
  transaction is the smallest number of blocks that reaches the target and overshoots by
  up to one block: asking for 65,536 rows over 37,500-row blocks produced transactions of
  75,000. The check asserts the real invariant (overshoot ≤ one block) rather than a
  ceiling the knob never promised. Worth knowing when sizing to fit worker memory, since
  the true peak is the target plus one block.
- **`max_rows_per_request` and `max_bytes_per_request` are hard ceilings.** Measured
  exactly: a 50,000-row ceiling produced a largest request of exactly 50,000 rows, and an
  8 MB ceiling produced a largest request of exactly 8.0 MB across 16 requests.
- **Projection pushdown is real and large.** Reading 2 of 41 columns reads 3.2% of the
  bytes. That ratio is now a gated counter, so a future change that quietly stops pushing
  the projection down fails the build instead of just getting slower.

A fourth finding came out of running the suite against real object storage
rather than only a local directory:

- **The single-commit guarantee holds over the network.** A fan-out write to S3
  produced `fragments=8, versions=1`, and an append advanced the table by exactly
  one version. This is the property most worth checking there and least checkable
  locally, since a local commit never has a network to fail across.

One README claim is still open: `remote_read_strategy="pagination"` is documented as
having measured ~2x faster than `offsets`. At the `ci` tier against the fake backend the
two are within noise of each other. That is a local disk rather than a service, so it
does not refute the claim -- but the scenario now exists to settle it against a real
endpoint via `make benchmark-enterprise`.
