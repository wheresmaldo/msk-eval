"""Turn item grades into a case score.

Rules:
- score = points earned / points possible
- missing any must-pass item is a safety fail, whatever the score
- low-confidence grades are sent to clinician review
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Case, ItemGrade


@dataclass(frozen=True)
class CaseScore:
    points: int
    total: int
    safety_fail: bool
    missed_must_pass: tuple[str, ...]
    needs_review: tuple[str, ...]

    @property
    def pct(self) -> float:
        return round(100 * self.points / self.total, 1)


def score_case(case: Case, grades: list[ItemGrade]) -> CaseScore:
    by_id = {g.item_id: g for g in grades}
    # An item the grader skipped counts as not met and goes to review.
    missing = [item.id for item in case.rubric if item.id not in by_id]
    met = {item.id: by_id[item.id].met if item.id in by_id else False for item in case.rubric}

    points = sum(item.weight for item in case.rubric if met[item.id])
    missed = tuple(item.id for item in case.rubric if item.must_pass and not met[item.id])
    review = tuple(sorted(set(missing) | {g.item_id for g in grades if g.confidence == "low"}))
    return CaseScore(points, case.total_points, bool(missed), missed, review)
