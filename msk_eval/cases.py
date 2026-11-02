"""Load cases from YAML files and check they are well formed."""
from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Case, ReferenceAnswer, RubricItem


class CaseError(ValueError):
    pass


def load_case(path: Path) -> Case:
    raw = yaml.safe_load(path.read_text())
    rubric = tuple(
        RubricItem(id=r["id"], text=r["text"], weight=int(r["weight"]), must_pass=bool(r.get("must_pass", False)))
        for r in raw["rubric"]
    )
    refs = tuple(
        ReferenceAnswer(label=a["label"], text=a["text"].strip(), grades={k: bool(v) for k, v in a["grades"].items()})
        for a in raw.get("reference_answers", [])
    )
    case = Case(
        id=raw["id"],
        title=raw["title"],
        subspecialty=raw["subspecialty"],
        risk=raw["risk"],
        prompt=raw["prompt"].strip(),
        rubric=rubric,
        variants={k: v.strip() for k, v in (raw.get("variants") or {}).items()},
        reference_answers=refs,
    )
    validate(case)
    return case


def validate(case: Case) -> None:
    ids = [item.id for item in case.rubric]
    if len(ids) != len(set(ids)):
        raise CaseError(f"{case.id}: duplicate rubric ids")
    if not any(item.must_pass for item in case.rubric):
        raise CaseError(f"{case.id}: needs at least one must-pass item")
    if any(item.weight <= 0 for item in case.rubric):
        raise CaseError(f"{case.id}: weights must be positive")
    if case.risk not in {"high", "routine"}:
        raise CaseError(f"{case.id}: risk must be 'high' or 'routine'")
    for ref in case.reference_answers:
        if set(ref.grades) != set(ids):
            raise CaseError(f"{case.id}/{ref.label}: reference grades must cover every rubric item")


def load_cases(folder: Path | str) -> list[Case]:
    paths = sorted(Path(folder).glob("*.yaml"))
    if not paths:
        raise CaseError(f"no case files in {folder}")
    return [load_case(p) for p in paths]
