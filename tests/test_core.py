import json
from pathlib import Path

import pytest

from msk_eval.agreement import cohens_kappa, consistency, percent_agreement
from msk_eval.cases import CaseError, load_case, load_cases, validate
from msk_eval.grader import parse_grades
from msk_eval.providers import FakeProvider
from msk_eval.runner import ModelSpec, run_eval, summarize, validate_grader
from msk_eval.schema import Case, ItemGrade, RubricItem
from msk_eval.scoring import score_case

CASES = Path(__file__).parent.parent / "cases"


def tiny_case() -> Case:
    return Case(
        id="t", title="t", subspecialty="x", risk="high", prompt="p",
        rubric=(RubricItem("r1", "send to ER", 5, True), RubricItem("r2", "explain", 3), RubricItem("r3", "kind", 2)),
    )


# --- cases -------------------------------------------------------------

def test_all_case_files_load():
    cases = load_cases(CASES)
    assert len(cases) == 5
    assert all(any(i.must_pass for i in c.rubric) for c in cases)


def test_case_without_must_pass_is_rejected():
    c = tiny_case()
    bad = Case(**{**c.__dict__, "rubric": (RubricItem("r1", "x", 1),)})
    with pytest.raises(CaseError):
        validate(bad)


# --- scoring -----------------------------------------------------------

def test_perfect_answer_scores_100():
    s = score_case(tiny_case(), [ItemGrade(i, True, "high", "") for i in ("r1", "r2", "r3")])
    assert s.pct == 100 and not s.safety_fail


def test_missing_must_pass_fails_even_with_other_points():
    s = score_case(tiny_case(), [ItemGrade("r1", False, "high", ""), ItemGrade("r2", True, "high", ""),
                                 ItemGrade("r3", True, "high", "")])
    assert s.safety_fail and s.missed_must_pass == ("r1",)
    assert s.pct == 50.0


def test_low_confidence_and_missing_items_go_to_review():
    s = score_case(tiny_case(), [ItemGrade("r1", True, "low", ""), ItemGrade("r2", True, "high", "")])
    assert s.needs_review == ("r1", "r3")


# --- grader parsing ----------------------------------------------------

def test_parses_json_inside_code_fence():
    raw = 'Here you go:\n```json\n[{"id":"r1","met":true,"confidence":"high","rationale":"ok"}]\n```'
    grades = parse_grades(tiny_case(), raw)
    assert grades[0].met and grades[0].confidence == "high"


def test_garbled_reply_becomes_not_met_low_confidence():
    grades = parse_grades(tiny_case(), "I cannot grade this.")
    assert all(not g.met and g.confidence == "low" for g in grades)


def test_unknown_ids_are_ignored_and_strings_are_not_true():
    raw = json.dumps([{"id": "r9", "met": True}, {"id": "r1", "met": "true"}])
    grades = parse_grades(tiny_case(), raw)
    assert [g.item_id for g in grades] == ["r1", "r2", "r3"]
    assert grades[0].met is False


# --- agreement ---------------------------------------------------------

def test_kappa_perfect_and_chance():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # A grader that always says "met" agrees often but has no skill.
    human = [True] * 9 + [False]
    lazy = [True] * 10
    assert percent_agreement(human, lazy) == 0.9
    assert cohens_kappa(human, lazy) == 0.0


def test_consistency():
    assert consistency([[True, False], [True, True]]) == 0.5


# --- end to end with fake models ---------------------------------------

def test_run_eval_end_to_end_without_network():
    case = load_case(CASES / "01-tka-dvt.yaml")

    def reply(model, prompt):
        if "Rubric items" in prompt:
            answer = prompt.split("AI answer:\n<<<\n")[1].split("\n>>>")[0]
            met = "emergency" in answer.lower()
            return json.dumps([{"id": i.id, "met": met, "confidence": "high"} for i in case.rubric])
        return "Go to the emergency department now." if model == "good" else "Rest and ice."

    fake = FakeProvider(reply)
    judge = ModelSpec("judge", "judge", fake)
    rows = run_eval([case], [ModelSpec("Good", "good", fake), ModelSpec("Bad", "bad", fake)], judge, log=lambda *_: None)
    assert len(rows) == 2 * (1 + len(case.variants))
    summary = {s["model"]: s for s in summarize(rows)}
    assert summary["Good"]["safety_fails"] == 0
    assert summary["Bad"]["safety_fails"] == 1


def test_validate_grader_with_a_perfect_fake_grader():
    cases = load_cases(CASES)
    lookup = {(c.id, r.text): r.grades for c in cases for r in c.reference_answers}

    def reply(model, prompt):
        answer = prompt.split("AI answer:\n<<<\n")[1].split("\n>>>")[0]
        for (cid, text), grades in lookup.items():
            if text == answer:
                return json.dumps([{"id": k, "met": v, "confidence": "high"} for k, v in grades.items()])
        return "[]"

    result = validate_grader(cases, ModelSpec("judge", "judge", FakeProvider(reply)), repeats=2, log=lambda *_: None)
    assert result["cohens_kappa"] == 1.0
    assert result["consistency_across_repeats"] == 1.0
    assert result["disagreements"] == []
