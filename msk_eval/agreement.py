"""How far can we trust the AI grader?

Percent agreement alone is misleading: if 90% of items are "met", a grader that
always says "met" agrees 90% of the time. Cohen's kappa corrects for that.
Rough guide: above 0.8 strong, 0.6-0.8 substantial, below 0.4 weak.
"""
from __future__ import annotations

from collections import Counter


def percent_agreement(a: list[bool], b: list[bool]) -> float:
    _check(a, b)
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    _check(a, b)
    n = len(a)
    observed = percent_agreement(a, b)
    pa, pb = Counter(a), Counter(b)
    expected = sum((pa[v] / n) * (pb[v] / n) for v in (True, False))
    if expected == 1:  # both raters gave one identical label to everything
        return 1.0
    return (observed - expected) / (1 - expected)


def consistency(runs: list[list[bool]]) -> float:
    """Share of items that got the same grade in every repeated run."""
    if not runs:
        raise ValueError("need at least one run")
    columns = list(zip(*runs))
    return sum(len(set(col)) == 1 for col in columns) / len(columns)


def _check(a: list[bool], b: list[bool]) -> None:
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be the same, non-zero length")
