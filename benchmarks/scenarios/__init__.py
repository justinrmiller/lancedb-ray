# SPDX-License-Identifier: Apache-2.0
"""Scenario registry.

A scenario is a named function that builds one or more measured, checked cases
against a backend. Registering rather than hard-coding a list keeps the CLI's
``--scenario`` filter, the tier membership and the docs in one place.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from ..harness import BenchRun

__all__ = ["ALL_TIERS", "REGISTRY", "Scenario", "all_scenarios", "register", "select"]

ScenarioFn = Callable[[BenchRun, str], None]

#: Every tier a scenario runs in, which is currently all of them -- tiers change
#: sizes, not coverage. Named once rather than spelled out per scenario so that
#: adding a tier to ``harness.TIERS`` does not silently select nothing.
ALL_TIERS: tuple[str, ...] = ("smoke", "ci", "local", "full", "xl")


@dataclass(frozen=True)
class Scenario:
    name: str
    fn: ScenarioFn
    group: str
    description: str
    #: Backends this scenario is meaningful against. ``local`` is a directory,
    #: ``fake`` is the in-repo Cloud/Enterprise stand-in, ``enterprise`` is a
    #: real service and never runs by default.
    backends: tuple[str, ...] = ("local", "fake")
    #: Tiers the scenario participates in.
    tiers: tuple[str, ...] = ALL_TIERS


REGISTRY: dict[str, Scenario] = {}


def register(
    name: str,
    *,
    group: str,
    description: str,
    backends: tuple[str, ...] = ("local", "fake"),
    tiers: tuple[str, ...] = ALL_TIERS,
) -> Callable[[ScenarioFn], ScenarioFn]:
    def decorate(fn: ScenarioFn) -> ScenarioFn:
        if name in REGISTRY:
            raise ValueError(f"duplicate scenario {name!r}")
        REGISTRY[name] = Scenario(name, fn, group, description, backends, tiers)
        return fn

    return decorate


def _load_all() -> None:
    from . import (  # noqa: F401
        knobs,
        read_local,
        remote,
        targets,  # noqa: F401
        upsert,
        write_local,
    )


def all_scenarios() -> list[Scenario]:
    _load_all()
    return [REGISTRY[name] for name in sorted(REGISTRY)]


def select(
    names: Optional[list[str]], tier: str, backends: tuple[str, ...]
) -> list[tuple[Scenario, str]]:
    """Expand the registry into the (scenario, backend) pairs a run will do."""
    chosen: list[tuple[Scenario, str]] = []
    for scenario in all_scenarios():
        if names and not any(
            scenario.name == n or scenario.name.startswith(n) or scenario.group == n
            for n in names
        ):
            continue
        if tier not in scenario.tiers:
            continue
        for backend in scenario.backends:
            if backend in backends:
                chosen.append((scenario, backend))
    return chosen
