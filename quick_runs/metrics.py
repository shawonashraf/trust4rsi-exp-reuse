"""Consistency metrics for the quick runs (quick_runs.md §2). Same code for every probe.

A "run record" is a dict:
  {"decisions": {decision_type: canonical value str},   # executed decisions, canonical form
   "order": [decision types in the order the run first stated them],
   "answer": hashable final answer key or None,
   "score": float or None,
   "checks": {check_name: True (pass) / False (fail) / None (unknown)}}
"""
from __future__ import annotations

import itertools
import statistics


def _pairs(runs):
    return list(itertools.combinations(range(len(runs)), 2))


def decision_agreement(runs: list[dict], dtype: str) -> float | None:
    """Share of run pairs that chose the same value at this decision type.
    Runs with no value for the type are left out; None if fewer than 2 runs have one."""
    vals = [r["decisions"].get(dtype) for r in runs]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    pairs = list(itertools.combinations(vals, 2))
    return sum(a == b for a, b in pairs) / len(pairs)


def first_divergence(runs: list[dict], dtypes: list[str]) -> str | None:
    """Earliest decision type at which any two runs differ.
    'Earliest' = smallest mean position in the runs' stated order; types a run never stated
    are placed after its stated ones in the fixed `dtypes` order. None if runs never differ."""
    def pos(r, t):
        if t in r["order"]:
            return r["order"].index(t)
        return len(r["order"]) + dtypes.index(t)
    differing = [t for t in dtypes
                 if len({r["decisions"].get(t) for r in runs if r["decisions"].get(t) is not None}) > 1]
    if not differing:
        return None
    return min(differing, key=lambda t: (statistics.mean(pos(r, t) for r in runs), dtypes.index(t)))


def answer_consistency(runs: list[dict]) -> float | None:
    """Share of run pairs with the same final answer. A missing answer never matches."""
    if len(runs) < 2:
        return None
    pairs = _pairs(runs)
    return sum(runs[i]["answer"] is not None and runs[i]["answer"] == runs[j]["answer"] for i, j in pairs) / len(pairs)


def consistent_failures(runs: list[dict]) -> list[str]:
    """Checks that every run failed (all False). A check with any unknown (None) is not counted."""
    names = set().union(*(r["checks"].keys() for r in runs)) if runs else set()
    return sorted(n for n in names if all(r["checks"].get(n) is False for r in runs))


def score_spread(runs: list[dict]) -> float | None:
    """Sample SD of the task score across runs (runs without a score left out)."""
    s = [r["score"] for r in runs if r["score"] is not None]
    return statistics.stdev(s) if len(s) >= 2 else None


def task_score(runs: list[dict]) -> float | None:
    s = [r["score"] for r in runs if r["score"] is not None]
    return statistics.mean(s) if s else None
