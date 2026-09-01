# SPDX-License-Identifier: Apache-2.0
"""Command line for the benchmark suite.

    python -m benchmarks --tier ci --compare benchmarks/baselines/ci-ubuntu-latest.json

Exit status is 0 only when every correctness check passed, every counter matched
the baseline, and no timing regressed past the threshold.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
import warnings
from collections.abc import Sequence
from typing import Optional

from .harness import TIERS, BenchRun, RunConfig
from .report import (
    Comparison,
    compare_to_baseline,
    render_comparison,
    render_counters,
    render_failures,
    render_summary,
    render_table,
    write_junit,
    write_step_summary,
)
from .scenarios import all_scenarios, select

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REGRESSED = 2


def _quiet_third_party() -> None:
    """Silence the per-task chatter that would bury the results table.

    Lance's Rust logger and Ray Data's progress output are both useful when
    debugging a scenario and pure noise in a benchmark report.
    """
    # Ray forks workers and both lance and lancedb warn about it on every
    # spawn; the suite's own Ray settings are what make that safe here.
    warnings.filterwarnings("ignore", message=".*fork.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="lance.*")
    os.environ.setdefault("RUST_LOG", "error")
    os.environ.setdefault(
        "PYTHONWARNINGS", "ignore::UserWarning,ignore::RuntimeWarning"
    )
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ.setdefault("RAY_DATA_DISABLE_PROGRESS_BARS", "1")
    for name in ("ray", "ray.data", "lancedb_ray", "lance"):
        logging.getLogger(name).setLevel(logging.ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__)
    parser.add_argument(
        "--tier",
        default=os.environ.get("BENCH_TIER", "local"),
        choices=sorted(TIERS),
        help="how big the run is; every check runs in every tier",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="scenario or group name; repeatable, prefix match allowed",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=["local", "fake", "enterprise", "s3"],
        help="backends to run against (default: local and fake)",
    )
    parser.add_argument("--repeat", type=int, help="timed iterations per case")
    parser.add_argument("--warmup", type=int, help="discarded iterations per case")
    parser.add_argument("--num-cpus", type=int, help="CPUs to give Ray")
    parser.add_argument(
        "--object-store-bytes", type=int, help="Ray object store size in bytes"
    )
    parser.add_argument(
        "--run-root",
        default=os.environ.get("BENCH_RUN_ROOT"),
        help="where scratch directories go; removed when the run ends",
    )
    parser.add_argument("--out", help="write the result JSON here")
    parser.add_argument("--junit", help="write a JUnit XML of the checks here")
    parser.add_argument("--step-summary", help="write a GitHub job summary here")
    parser.add_argument("--compare", help="baseline JSON to diff against")
    parser.add_argument(
        "--regression-factor",
        type=float,
        default=2.5,
        help="fail when a median is this many times the baseline",
    )
    parser.add_argument(
        "--write-baseline", help="write this run's results as a baseline file"
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="leave case directories in place for debugging",
    )
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _list_scenarios() -> int:
    for scenario in all_scenarios():
        print(f"{scenario.name:32s} {scenario.group:8s} {','.join(scenario.backends)}")
        print(f"{'':32s} {scenario.description}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        return _list_scenarios()

    _quiet_third_party()

    backends = tuple(args.backends or ("local", "fake"))
    config = RunConfig(
        tier=args.tier,
        scenarios=args.scenarios,
        backends=backends,
        repeat=args.repeat,
        warmup=args.warmup,
        num_cpus=args.num_cpus,
        object_store_bytes=args.object_store_bytes,
        run_root=args.run_root,
        baseline=args.compare,
        regression_factor=args.regression_factor,
        fail_fast=args.fail_fast,
        keep_artifacts=args.keep_artifacts,
        quiet=args.quiet,
    )

    run = BenchRun(config)
    planned = select(config.scenarios, config.tier, backends)
    if not planned:
        print("no scenarios selected", file=sys.stderr)
        return EXIT_FAILED

    utilization = run.disk_utilization()
    if utilization >= 0.95 and not args.quiet:
        print(
            f"note: {run.run_root} is {utilization:.1%} utilised; raising Ray's spill "
            f"guard so it uses the {run.free_bytes() / 1024**3:.0f}GB actually free",
        )

    if not run.has_room(run.tier.min_free_bytes):
        free = run.free_bytes() / 1024**3
        need = run.tier.min_free_bytes / 1024**3
        print(
            f"tier {run.tier.name!r} wants {need:.0f}GB free under {run.run_root}, "
            f"found {free:.1f}GB -- pick a smaller tier or set BENCH_RUN_ROOT",
            file=sys.stderr,
        )
        run.cleanup()
        return EXIT_FAILED

    exit_code = EXIT_OK
    try:
        run.start_ray(fake_remote="fake" in backends)
        _configure_ray_data()

        if not args.quiet:
            print(
                f"tier={run.tier.name} ({run.tier.description}); "
                f"{len(planned)} scenario runs; repeat={run.repeat} warmup={run.warmup}",
                flush=True,
            )

        for scenario, backend in planned:
            started = time.perf_counter()
            if not args.quiet:
                print(f"-> {scenario.name} [{backend}]", flush=True)
            with run.scenario(scenario.name):
                try:
                    scenario.fn(run, backend)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    _record_scenario_error(run, scenario.name, backend, exc)
                    if config.fail_fast:
                        break
            if not args.quiet:
                print(f"   {time.perf_counter() - started:.1f}s", flush=True)

        exit_code = _finish(run, args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        exit_code = EXIT_FAILED
    finally:
        run.stop_ray()
        if not config.keep_artifacts:
            run.cleanup()
        elif not args.quiet:
            print(f"artifacts kept in {run.run_root}")

    return exit_code


def _configure_ray_data() -> None:
    try:
        from ray.data import DataContext

        DataContext.get_current().enable_progress_bars = False
    except Exception:
        pass


def _record_scenario_error(
    run: BenchRun, name: str, backend: str, exc: BaseException
) -> None:
    """A scenario that blew up is a result, not a crashed run."""
    from .harness import CaseResult

    run.record(
        CaseResult(
            scenario=name,
            name=name,
            backend=backend,
            error=f"{type(exc).__name__}: {exc}",
            notes=[traceback.format_exc(limit=6)],
        )
    )


def _finish(run: BenchRun, args: argparse.Namespace) -> int:
    # Default under the package rather than the run root, which is deleted on
    # exit. CI overrides BENCH_OUT_DIR so nothing is written into the workspace
    # at all; locally benchmarks/results/ is gitignored.
    default_dir = os.environ.get(
        "BENCH_OUT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    out = args.out or os.path.join(
        default_dir, f"bench-{run.tier.name}-{int(run.started)}.json"
    )
    run.write_json(out)

    if not args.quiet:
        print()
        print(render_table(run))
        counters = render_counters(run)
        if counters:
            print()
            print(counters)

    comparisons: list[Comparison] = []
    counter_regressions: list[str] = []
    if args.compare:
        if os.path.exists(args.compare):
            comparisons, counter_regressions = compare_to_baseline(
                run, args.compare, factor=args.regression_factor
            )
            print()
            print(render_comparison(comparisons, args.regression_factor))
        else:
            print(f"\nno baseline at {args.compare}; skipping comparison")

    if args.junit:
        write_junit(run, args.junit)
    if args.step_summary:
        write_step_summary(
            run,
            args.step_summary,
            comparisons=comparisons,
            factor=args.regression_factor,
        )
    if args.write_baseline:
        run.write_json(args.write_baseline)
        print(f"baseline written to {args.write_baseline}")

    print()
    print(render_summary(run))
    print(f"results: {out}")

    failures = render_failures(run)
    if failures:
        print("\nfailures:")
        print(failures)

    if counter_regressions:
        print("\ncounter regressions (these gate the build):")
        for line in counter_regressions:
            print(f"  {line}")

    slower = [c for c in comparisons if c.regressed]
    if slower:
        print(f"\ntiming regressions past {args.regression_factor:g}x:")
        for comparison in slower:
            print(f"  {comparison.key}: {comparison.factor:.2f}x")

    if failures or counter_regressions:
        return EXIT_FAILED
    if slower:
        return EXIT_REGRESSED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
