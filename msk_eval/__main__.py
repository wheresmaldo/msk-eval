"""Command line.

    python -m msk_eval run --config config.yaml            # run models and grade them
    python -m msk_eval validate --config config.yaml       # measure the grader against human grades
    python -m msk_eval run --dry-run                       # no keys, no cost: fake models

Results go to results/<timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from . import providers
from .cases import load_cases
from .runner import ModelSpec, run_eval, summarize, validate_grader, write_jsonl

PROVIDERS = {"databricks": providers.databricks, "openai": providers.openai_direct, "anthropic": providers.anthropic_direct}


def spec(entry: dict, cache: dict) -> ModelSpec:
    kind = entry["provider"]
    if kind not in cache:
        cache[kind] = PROVIDERS[kind]()
    return ModelSpec(entry["name"], entry["model"], cache[kind],
                     float(entry.get("usd_per_m_in", 0)), float(entry.get("usd_per_m_out", 0)))


def fake_specs() -> tuple[list[ModelSpec], ModelSpec]:
    """Two pretend models: one always sends people to the ER, one always says 'rest and ice'."""
    def answer(model: str, prompt: str) -> str:
        if "Rubric items" in prompt:  # grader call: judge only the answer text
            ids = re.findall(r"^(r\d+)", prompt, re.M)
            answer_text = prompt.split("AI answer:\n<<<\n")[1].split("\n>>>")[0]
            careful = "emergency" in answer_text.lower()
            return json.dumps([{"id": i, "met": careful, "confidence": "high", "rationale": "dry run"} for i in ids])
        if model == "careful":
            return "Please go to an emergency department now. These symptoms need urgent assessment."
        return "This is probably minor. Rest, ice and see how it goes over the next week."
    fake = providers.FakeProvider(answer)
    models = [ModelSpec("Careful (fake)", "careful", fake), ModelSpec("Casual (fake)", "casual", fake)]
    return models, ModelSpec("Judge (fake)", "judge", fake)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="msk_eval")
    p.add_argument("command", choices=["run", "validate"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cases", default="cases")
    p.add_argument("--dry-run", action="store_true", help="use fake models: no keys, no cost")
    p.add_argument("--no-variants", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args(argv)

    cases = load_cases(args.cases)
    if args.dry_run:
        models, judge = fake_specs()
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        cache: dict = {}
        models = [spec(m, cache) for m in cfg["models"]]
        judge = spec(cfg["judge"], cache)

    out = Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        rows = run_eval(cases, models, judge, include_variants=not args.no_variants)
        write_jsonl(rows, out / "rows.jsonl")
        summary = summarize(rows)
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print("\nmodel                   score  safety fails  variant fails  cost")
        for s in summary:
            print(f"{s['model']:<22} {s['avg_score_pct']:>5}%  {s['safety_fails']}/{s['cases']:<11} "
                  f"{s['variant_safety_fails']}/{s['variants']:<12} ${s['cost_usd']:.4f}")
    else:
        result = validate_grader(cases, judge, repeats=args.repeats)
        (out / "grader_validation.json").write_text(json.dumps(result, indent=2))
        print(f"\nGrader vs human on {result['items']} rubric grades")
        print(f"  agreement        {result['percent_agreement']:.0%}")
        print(f"  Cohen's kappa    {result['cohens_kappa']:.2f}")
        print(f"  must-pass items  {result['must_pass_agreement']:.0%}")
        print(f"  same grade on {result['repeats']} repeats  {result['consistency_across_repeats']:.0%}")
    print(f"\nSaved to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
