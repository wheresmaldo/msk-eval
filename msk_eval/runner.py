"""Run models on cases, grade them, and check the grader.

Two jobs:
- run_eval: every model answers every case (and its variants), then the grader scores each answer.
- validate_grader: the grader re-grades answers humans already graded, so we can measure agreement.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass

from . import grader
from .agreement import cohens_kappa, consistency, percent_agreement
from .providers import Provider
from .schema import Case, Usage
from .scoring import score_case


@dataclass(frozen=True)
class ModelSpec:
    name: str          # label shown in results, e.g. "Llama 3.3 70B"
    model: str         # endpoint or API model id
    provider: Provider
    usd_per_m_in: float = 0.0   # price per million input tokens
    usd_per_m_out: float = 0.0  # price per million output tokens

    def cost(self, u: Usage) -> float:
        return (u.input_tokens * self.usd_per_m_in + u.output_tokens * self.usd_per_m_out) / 1_000_000


def run_eval(cases: list[Case], models: list[ModelSpec], judge: ModelSpec, include_variants: bool = True,
             log=print) -> list[dict]:
    rows = []
    for case in cases:
        prompts = {"original": case.prompt}
        if include_variants:
            prompts.update(case.variants)
        for variant, prompt in prompts.items():
            for m in models:
                answer, a_usage = m.provider.complete(m.model, grader.answer_prompt(prompt))
                grades, g_usage = grader.grade(judge.provider, judge.model, case, prompt, answer)
                s = score_case(case, grades)
                log(f"{case.id:<14} {variant:<10} {m.name:<22} {s.pct:>5}%  {'SAFETY FAIL' if s.safety_fail else 'pass'}")
                rows.append({
                    "case_id": case.id, "risk": case.risk, "variant": variant, "model": m.name,
                    "answer": answer, "points": s.points, "total": s.total, "pct": s.pct,
                    "safety_fail": s.safety_fail, "missed_must_pass": list(s.missed_must_pass),
                    "needs_review": list(s.needs_review), "grades": [asdict(g) for g in grades],
                    "answer_tokens_in": a_usage.input_tokens, "answer_tokens_out": a_usage.output_tokens,
                    "answer_seconds": a_usage.seconds,
                    "cost_usd": round(m.cost(a_usage) + judge.cost(g_usage), 6),
                })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """One line per model: average score, safety fails, and how variants changed things."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = []
    for model, rs in by_model.items():
        orig = [r for r in rs if r["variant"] == "original"]
        var = [r for r in rs if r["variant"] != "original"]
        out.append({
            "model": model,
            "avg_score_pct": round(sum(r["pct"] for r in orig) / len(orig), 1) if orig else None,
            "safety_fails": sum(r["safety_fail"] for r in orig),
            "cases": len(orig),
            "variant_safety_fails": sum(r["safety_fail"] for r in var),
            "variants": len(var),
            "sent_to_review": sum(len(r["needs_review"]) for r in rs),
            "cost_usd": round(sum(r["cost_usd"] for r in rs), 4),
        })
    return sorted(out, key=lambda x: (x["safety_fails"], -(x["avg_score_pct"] or 0)))


def validate_grader(cases: list[Case], judge: ModelSpec, repeats: int = 3, log=print) -> dict:
    """Compare AI grades with human reference grades, and check the grader is consistent."""
    human, ai, human_must, ai_must = [], [], [], []
    stable = []
    disagreements = []
    for case in cases:
        must_ids = {i.id for i in case.rubric if i.must_pass}
        for ref in case.reference_answers:
            runs = []
            for _ in range(repeats):
                grades, _ = grader.grade(judge.provider, judge.model, case, case.prompt, ref.text)
                runs.append([g.met for g in grades])
            stable.append(consistency(runs))
            first = runs[0]
            for item, ai_met in zip(case.rubric, first):
                h = ref.grades[item.id]
                human.append(h)
                ai.append(ai_met)
                if item.id in must_ids:
                    human_must.append(h)
                    ai_must.append(ai_met)
                if h != ai_met:
                    disagreements.append({"case": case.id, "answer": ref.label, "item": item.id,
                                          "human": h, "ai": ai_met, "must_pass": item.id in must_ids})
            log(f"{case.id:<14} answer {ref.label}: checked")
    return {
        "items": len(human),
        "percent_agreement": round(percent_agreement(human, ai), 3),
        "cohens_kappa": round(cohens_kappa(human, ai), 3),
        "must_pass_agreement": round(percent_agreement(human_must, ai_must), 3),
        "consistency_across_repeats": round(sum(stable) / len(stable), 3),
        "repeats": repeats,
        "disagreements": disagreements,
    }


def write_jsonl(rows: list[dict], path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
