"""Data shapes used across the harness.

A Case is one patient message plus the clinician rubric used to grade answers to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RubricItem:
    id: str
    text: str
    weight: int
    must_pass: bool = False


@dataclass(frozen=True)
class ReferenceAnswer:
    """An answer with grades a human already assigned. Used to check the AI grader."""
    label: str
    text: str
    grades: dict[str, bool]  # rubric item id -> met?


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    subspecialty: str
    risk: str  # "high" or "routine"
    prompt: str
    rubric: tuple[RubricItem, ...]
    variants: dict[str, str] = field(default_factory=dict)  # name -> reworded prompt
    reference_answers: tuple[ReferenceAnswer, ...] = ()

    @property
    def total_points(self) -> int:
        return sum(item.weight for item in self.rubric)


@dataclass(frozen=True)
class ItemGrade:
    item_id: str
    met: bool
    confidence: str  # "high" or "low"
    rationale: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
