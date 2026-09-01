# SPDX-License-Identifier: Apache-2.0
"""Rendering results: a terminal table, JSON, a baseline diff, and CI output."""

from __future__ import annotations

import json
import os
import statistics
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .harness import BenchRun, CaseResult

__all__ = [
    "Comparison",
    "compare_to_baseline",
    "render_summary",
    "render_table",
    "write_junit",
    "write_step_summary",
]


def _fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _fmt_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def _fmt_rate(value: Optional[float]) -> str:
    if not value:
        return "-"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M/s"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K/s"
    return f"{value:.0f}/s"


def _spread(result: CaseResult) -> str:
    """Run-to-run spread, so a difference can be read against the noise."""
    if len(result.timings) < 2:
        return "-"
    median = statistics.median(result.timings)
    if not median:
        return "-"
    return f"±{(max(result.timings) - min(result.timings)) / median * 100:.0f}%"


def _status(result: CaseResult) -> str:
    if result.skipped:
        return "skip"
    if result.error:
        return "ERROR"
    failed = len(result.failed_checks)
    if failed:
        return f"FAIL {failed}"
    return f"ok {len(result.checks)}"


def _table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    ]
    return "\n".join([line, sep, *body])


def render_table(run: BenchRun) -> str:
    """The main terminal output: one line per case, grouped by scenario."""
    sections: list[str] = []
    by_group: dict[str, list[CaseResult]] = {}
    for result in run.results:
        by_group.setdefault(result.scenario or result.name, []).append(result)

    headers = [
        "case",
        "backend",
        "median",
        "spread",
        "rows/s",
        "MB/s",
        "peak RSS",
        "checks",
    ]
    for group in sorted(by_group):
        rows = []
        for result in by_group[group]:
            data = result.as_dict()
            rows.append(
                [
                    result.name,
                    result.backend,
                    _fmt_time(result.median),
                    _spread(result),
                    _fmt_rate(data.get("rows_per_s")),
                    f"{data['mb_per_s']:.0f}" if data.get("mb_per_s") else "-",
                    _fmt_bytes(result.peak_rss_bytes) if result.peak_rss_bytes else "-",
                    _status(result),
                ]
            )
        sections.append(_table(rows, headers))
    return "\n\n".join(sections)


def render_counters(run: BenchRun) -> str:
    """The counters that are the actual regression gate."""
    interesting = (
        "fragments",
        "versions",
        "read_blocks",
        "scan_bytes_ratio",
        "add_count",
        "add_max_bytes",
        "take_offsets_count",
    )
    rows = []
    for result in run.results:
        present = {k: result.counters[k] for k in interesting if k in result.counters}
        if not present:
            continue
        rows.append(
            [
                result.name,
                result.backend,
                ", ".join(f"{k}={_counter_str(v)}" for k, v in present.items()),
            ]
        )
    if not rows:
        return ""
    return _table(rows, ["case", "backend", "counters"])


def _counter_str(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int) and value > 100_000:
        return _fmt_bytes(value) if value > 1024**2 else f"{value:,}"
    return str(value)


def render_failures(run: BenchRun) -> str:
    lines: list[str] = []
    for result in run.results:
        if result.skipped:
            continue
        if result.error:
            lines.append(f"  {result.key}: ERROR {result.error}")
        for check in result.failed_checks:
            detail = f" ({check['detail']})" if check["detail"] else ""
            lines.append(
                f"  {result.key}: {check['name']} -- expected {check['expected']!r}, "
                f"got {check['actual']!r}{detail}"
            )
    return "\n".join(lines)


# -- baseline comparison -----------------------------------------------------


@dataclass
class Comparison:
    """One case measured against the same case in a baseline run."""

    key: str
    current: float
    baseline: float
    factor: float
    regressed: bool
    counter_diffs: list[str]


def compare_to_baseline(
    run: BenchRun, baseline_path: str, *, factor: float
) -> tuple[list[Comparison], list[str]]:
    """Diff timings and counters against a stored run.

    Timings are compared with a deliberately wide tolerance: on shared hardware
    a tight threshold is a flake generator. Counters are compared exactly,
    because they do not move unless behaviour changed.
    """
    with open(baseline_path, encoding="utf-8") as handle:
        baseline = json.load(handle)

    by_key = {
        r["name"] + "[" + r["backend"] + "]": r for r in baseline.get("results", [])
    }
    comparisons: list[Comparison] = []
    counter_regressions: list[str] = []

    for result in run.results:
        previous = by_key.get(result.key)
        if previous is None or result.median is None:
            continue
        before = previous.get("median_s")
        if not before:
            continue

        diffs: list[str] = []
        for name, value in previous.get("counters", {}).items():
            if name in ("disk_bytes", "id_sum") or name.startswith("scan_"):
                continue  # size-like, legitimately drifts with the format
            if name not in result.counters:
                continue
            if result.counters[name] != value:
                diffs.append(f"{name}: {value} -> {result.counters[name]}")

        ratio = result.median / before
        comparison = Comparison(
            key=result.key,
            current=result.median,
            baseline=before,
            factor=ratio,
            regressed=ratio > factor,
            counter_diffs=diffs,
        )
        comparisons.append(comparison)
        if diffs:
            counter_regressions.append(f"{result.key}: " + "; ".join(diffs))

    return comparisons, counter_regressions


def render_comparison(comparisons: Sequence[Comparison], factor: float) -> str:
    if not comparisons:
        return "no comparable cases in the baseline"
    rows = []
    for comparison in sorted(comparisons, key=lambda c: -c.factor):
        marker = (
            "SLOWER"
            if comparison.regressed
            else ("faster" if comparison.factor < 0.8 else "")
        )
        rows.append(
            [
                comparison.key,
                _fmt_time(comparison.baseline),
                _fmt_time(comparison.current),
                f"{comparison.factor:.2f}x",
                marker,
            ]
        )
    return _table(rows, ["case", "baseline", "now", "ratio", f"gate {factor:g}x"])


# -- machine-readable output -------------------------------------------------


def write_junit(run: BenchRun, path: str) -> None:
    """One test case per check, so CI renders the failures natively."""
    suite = ET.Element(
        "testsuite",
        name="benchmarks",
        tests="0",
        failures="0",
        errors="0",
        time=f"{sum(sum(r.timings) for r in run.results):.3f}",
    )
    tests = failures = errors = 0

    for result in run.results:
        if result.error:
            errors += 1
            tests += 1
            case = ET.SubElement(
                suite,
                "testcase",
                classname=result.scenario or "benchmarks",
                name=result.key,
            )
            ET.SubElement(case, "error", message=result.error).text = result.error
        for check in result.checks:
            tests += 1
            case = ET.SubElement(
                suite,
                "testcase",
                classname=f"{result.scenario or 'benchmarks'}.{result.key}",
                name=check["name"],
                time=f"{result.median or 0:.3f}",
            )
            if not check["passed"]:
                failures += 1
                message = (
                    f"expected {check['expected']!r}, got {check['actual']!r} "
                    f"{check['detail']}"
                ).strip()
                ET.SubElement(case, "failure", message=message).text = message
        if result.skipped:
            tests += 1
            case = ET.SubElement(
                suite,
                "testcase",
                classname=result.scenario or "benchmarks",
                name=result.key,
            )
            ET.SubElement(case, "skipped", message=result.skipped)

    suite.set("tests", str(tests))
    suite.set("failures", str(failures))
    suite.set("errors", str(errors))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def render_summary(run: BenchRun) -> str:
    total = len(run.results)
    failed = [r for r in run.results if not r.ok and not r.skipped]
    skipped = [r for r in run.results if r.skipped]
    checks = sum(len(r.checks) for r in run.results)
    failed_checks = sum(len(r.failed_checks) for r in run.results)
    env = run.environment
    return (
        f"tier={run.tier.name} cases={total} checks={checks} "
        f"failed_cases={len(failed)} failed_checks={failed_checks} "
        f"skipped={len(skipped)} "
        f"duration={run.as_dict()['duration_s']:.1f}s "
        f"ray_init={run.ray_init_seconds:.1f}s "
        f"git={env.get('git_sha') or '?'}{'-dirty' if env.get('git_dirty') else ''}"
    )


def write_step_summary(
    run: BenchRun,
    path: str,
    *,
    comparisons: Optional[Sequence[Comparison]] = None,
    factor: float = 2.5,
) -> None:
    """A GitHub job summary: the table, the failures, and the baseline diff."""
    env = run.environment
    lines = [
        "## lancedb-ray benchmarks",
        "",
        f"`{render_summary(run)}`",
        "",
        f"- runner: {env.get('platform')} / {env.get('cpu_count')} CPU / "
        f"{_fmt_bytes(env.get('total_ram_bytes', 0))} RAM",
        f"- ray {env['versions'].get('ray')}, lancedb {env['versions'].get('lancedb')}, "
        f"lance {env['versions'].get('lance')}",
        "",
        "<details><summary>Timings</summary>",
        "",
        "```",
        render_table(run),
        "```",
        "",
        "</details>",
        "",
        "<details><summary>Counters (the hard gate)</summary>",
        "",
        "```",
        render_counters(run) or "none",
        "```",
        "",
        "</details>",
    ]

    failures = render_failures(run)
    if failures:
        lines += ["", "### Failures", "", "```", failures, "```"]

    if comparisons:
        lines += [
            "",
            "### Against baseline",
            "",
            "```",
            render_comparison(comparisons, factor),
            "```",
        ]

    # Timings on a shared runner are not evidence on their own; say so where the
    # numbers are most likely to be quoted.
    lines += [
        "",
        "> Hosted-runner timings carry runner noise. Correctness and counters "
        "gate the build; timings are a trend.",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
