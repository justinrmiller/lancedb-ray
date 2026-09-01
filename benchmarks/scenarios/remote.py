# SPDX-License-Identifier: Apache-2.0
"""Remote read scenarios: the sharding strategies and their request cost.

These run against the in-repo fake Cloud/Enterprise backend, which wraps a real
local database but narrows it to the remote API surface. That makes the *shape*
of a strategy measurable -- how many requests, how large, and whether the
results agree -- but the wall-clock is a local disk's, not a service's. Results
are labelled ``fake`` for exactly that reason.
"""

from __future__ import annotations

from typing import Any

from .. import datagen
from ..checks import compare_tables, id_stats, sample_ids, verify_id_space
from ..harness import BenchRun
from . import register
from ._common import TABLE, expected_bytes, read_back, seed

_STRATEGIES = ("offsets", "pagination", "single")


@register(
    "remote_read_strategies",
    group="remote",
    description="offsets vs pagination vs single for a remote read",
    backends=("fake",),
    tiers=("smoke", "ci", "local", "full"),
)
def remote_read_strategies(run: BenchRun, backend: str) -> None:
    dataset = "narrow"
    rows = run.rows(dataset)
    fingerprints: dict[str, Any] = {}

    for strategy in _STRATEGIES:
        with run.case(
            f"remote_read.{strategy}",
            backend=backend,
            dataset=dataset,
            params={"remote_read_strategy": strategy},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))

            def build(db_dir: str) -> str:
                return seed(case, db_dir, dataset, rows, blocks=run.blocks)

            def run_read(db_dir: str, strategy: str = strategy) -> Any:
                return read_back(
                    case, case.uri(db_dir), remote_read_strategy=strategy
                ).materialize()

            outcome = case.measure(run_read, fresh=False, setup=build)
            materialized = outcome.value
            uri = case.uri(outcome.work)

            case.counter("read_blocks", int(materialized.num_blocks()))
            case.add_counters(case.probe("take_offsets"), prefix="take_offsets_")
            case.add_counters(case.probe("search"), prefix="search_")

            case.checks.equals("rows read", int(materialized.count()), rows)
            stats = id_stats(materialized.select_columns(["id"]))
            verify_id_space(case.checks, stats, num_rows=rows)
            if strategy != "single":
                case.checks.at_least(
                    "sharded across tasks", int(materialized.num_blocks()), 2
                )

            # Every strategy must return the same rows; a fast strategy that
            # returns a different set is the failure this suite exists to catch.
            ids = sample_ids(rows, count=128)
            from ..checks import read_rows

            got = read_rows(uri, TABLE, ids, connect_kwargs=case.connect_kwargs)
            case.checks.add(compare_tables(got, datagen.expected_rows(dataset, ids)))
            fingerprints[strategy] = (
                stats.count,
                stats.total,
                stats.minimum,
                stats.maximum,
            )

    if len(fingerprints) > 1:
        with run.case(
            "remote_read.agreement", backend=backend, dataset=dataset
        ) as case:
            reference = fingerprints[_STRATEGIES[0]]
            for strategy, value in fingerprints.items():
                case.checks.equals(
                    f"{strategy} agrees with {_STRATEGIES[0]}", value, reference
                )
            case.note("compares the id-space fingerprint of every strategy")


@register(
    "remote_batch_size",
    group="remote",
    description="batch_size trades round trips against request size",
    backends=("fake",),
    tiers=("smoke", "ci", "local", "full"),
)
def remote_batch_size(run: BenchRun, backend: str) -> None:
    dataset = "narrow"
    rows = run.rows(dataset)
    sizes = (10_000, 50_000, 200_000) if run.tier.full_sweeps else (10_000, 50_000)

    for batch_size in sizes:
        with run.case(
            f"remote_batch_size.{batch_size}",
            backend=backend,
            dataset=dataset,
            params={"batch_size": batch_size},
        ) as case:
            case.set_volume(rows, expected_bytes(dataset, rows))

            def build(db_dir: str) -> str:
                return seed(case, db_dir, dataset, rows, blocks=run.blocks)

            def run_read(db_dir: str, batch_size: int = batch_size) -> Any:
                return read_back(
                    case,
                    case.uri(db_dir),
                    remote_read_strategy="offsets",
                    batch_size=batch_size,
                ).materialize()

            outcome = case.measure(run_read, fresh=False, setup=build)
            materialized = outcome.value

            probe = case.probe("take_offsets")
            case.add_counters(probe, prefix="take_offsets_")
            case.checks.equals("rows read", int(materialized.count()), rows)
            if probe["count"]:
                # The knob sets how many offsets go in one request; exceeding it
                # would mean the chunking is not being applied.
                case.checks.at_most(
                    "no request exceeded batch_size",
                    max(
                        int(e.get("count", 0))
                        for e in run.probe_events()
                        if e.get("event") == "take_offsets"
                    ),
                    batch_size,
                )
