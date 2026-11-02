# MSK Eval

Test AI models on musculoskeletal (MSK) patient questions and grade their answers against clinician-written rubrics. Built as the technical proof of concept for the HSS Atlas commercial case.

> Concept work for a job interview, not an HSS product. The five cases are illustrative and not HSS-approved clinical content.

## What it does

```
cases/*.yaml ──► models answer ──► AI grader scores each rubric item ──► must-pass rule ──► results
  (clinician       (Model Serving     (LLM judge, JSON out,                (missed safety      (Delta table,
   rubrics)         or any API)        confidence per item)                  item = fail)        MLflow run)
                                            │
                                            └── low-confidence grades ──► clinician review
```

1. **Cases** are YAML files: a patient message, a weighted rubric, at least one **must-pass** safety item, optional **stress-test variants** (the same emergency, downplayed or buried in other questions), and reference answers with human grades.
2. **Models** answer each case. Databricks Model Serving, OpenAI and Anthropic all use one client.
3. **The AI grader** grades every rubric item in one call and returns JSON. Anything it skips or garbles becomes "not met, low confidence", so it goes to a human instead of passing silently.
4. **Scoring**: points earned ÷ points possible. Missing any must-pass item is a safety fail regardless of score.
5. **Grader validation**: the grader re-grades answers humans already graded. We report agreement, **Cohen's kappa** (agreement corrected for chance), agreement on must-pass items, and consistency across repeated runs.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                                  # 12 tests, no network
python -m msk_eval run --dry-run           # fake models, no keys, no cost
```

With real models, copy `config.example.yaml` to `config.yaml`, set the keys as environment variables, then:

```bash
python -m msk_eval run                     # models answer and get graded
python -m msk_eval validate --repeats 3    # how far to trust the grader
```

## On Databricks

Clone this repo into a Databricks Git folder and run the notebooks in `databricks/`:

| Notebook | Does |
| --- | --- |
| `01_run_eval` | Writes cases and rubrics to Unity Catalog, runs models through Model Serving, grades them, logs to MLflow, appends to `eval_results` |
| `02_validate_grader` | Measures grader vs human agreement and kappa, logs to MLflow, saves disagreements |

Inside Databricks the notebooks use the workspace's own login, so no keys are needed.

## Layout

| Path | Purpose |
| --- | --- |
| `msk_eval/schema.py` | Data shapes: case, rubric item, grade |
| `msk_eval/cases.py` | Load and validate case files |
| `msk_eval/providers.py` | Model clients (Databricks, OpenAI, Anthropic, fake) |
| `msk_eval/grader.py` | Grader prompt and tolerant JSON parsing |
| `msk_eval/scoring.py` | Weighted score, must-pass rule, review routing |
| `msk_eval/agreement.py` | Percent agreement, Cohen's kappa, consistency |
| `msk_eval/runner.py` | Run models, summarize, validate the grader |
| `cases/` | Five illustrative cases with variants and reference grades |
| `tests/` | Unit and end-to-end tests with fake models |

## Design choices

- **One grader call per answer**, not per rubric item: about 5x cheaper, and the grader sees the whole rubric.
- **Fail safe**: unparseable or missing grades never count as met.
- **Kappa, not just accuracy**: a grader that always says "met" can agree 90% of the time and still have no skill.
- **Variants test robustness**: a model that catches the red flag only when it is stated plainly is not safe.
- **Keys never in files**: environment variables locally, workspace login on Databricks.
