# Databricks notebook source
# MAGIC %md
# MAGIC # MSK Eval: run models and grade them
# MAGIC 1. Stores the cases and rubrics in Unity Catalog.
# MAGIC 2. Sends every case (and its stress-test variants) to each model through Model Serving.
# MAGIC 3. Grades every answer with the AI grader, applies the must-pass rule, and routes unsure grades to review.
# MAGIC 4. Logs the run to MLflow and saves results to a Delta table.

# COMMAND ----------

# MAGIC %pip install -q pyyaml openai mlflow
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

# The repo root holds the msk_eval package; this notebook lives one folder down.
sys.path.insert(0, os.path.abspath(".."))

from mlflow.deployments import get_deploy_client

# Which chat models does this workspace have? Pick from this list for the widgets below.
try:
    endpoints = get_deploy_client("databricks").list_endpoints()
    chat = sorted(e["name"] for e in endpoints if "chat" in str(e.get("task", "")))
    print("\n".join(chat) or "No chat endpoints listed; the defaults below may still work.")
except Exception as e:
    print(f"Could not list endpoints ({e}); the defaults below may still work.")

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "msk_eval")
dbutils.widgets.text("models", "databricks-llama-4-maverick,databricks-qwen3-next-80b-a3b-instruct,databricks-gemma-3-12b")
dbutils.widgets.text("judge", "databricks-gpt-oss-120b")
dbutils.widgets.dropdown("variants", "yes", ["yes", "no"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODELS = [m.strip() for m in dbutils.widgets.get("models").split(",") if m.strip()]
JUDGE = dbutils.widgets.get("judge").strip()
TABLE = f"{CATALOG}.{SCHEMA}"
print(TABLE, MODELS, "judge:", JUDGE)

# COMMAND ----------

# MAGIC %md ## 1. Store cases and rubrics in Unity Catalog

# COMMAND ----------

from msk_eval.cases import load_cases

cases = load_cases("../cases")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TABLE}")

spark.createDataFrame(
    [(c.id, c.title, c.subspecialty, c.risk, c.prompt, c.total_points) for c in cases],
    "case_id string, title string, subspecialty string, risk string, prompt string, total_points int",
).write.mode("overwrite").saveAsTable(f"{TABLE}.cases")

spark.createDataFrame(
    [(c.id, i.id, i.text, i.weight, i.must_pass) for c in cases for i in c.rubric],
    "case_id string, item_id string, text string, weight int, must_pass boolean",
).write.mode("overwrite").saveAsTable(f"{TABLE}.rubric_items")

display(spark.table(f"{TABLE}.rubric_items"))

# COMMAND ----------

# MAGIC %md ## 2–3. Run the models and grade every answer

# COMMAND ----------

from msk_eval.providers import DatabricksNotebook
from msk_eval.runner import ModelSpec, run_eval, summarize

serving = DatabricksNotebook()
models = [ModelSpec(name=m.replace("databricks-", ""), model=m, provider=serving) for m in MODELS]
judge = ModelSpec(name=JUDGE, model=JUDGE, provider=serving)

rows = run_eval(cases, models, judge, include_variants=dbutils.widgets.get("variants") == "yes")
summary = summarize(rows)
display(spark.createDataFrame(summary))

# COMMAND ----------

# MAGIC %md ## 4. Log to MLflow and save to Delta

# COMMAND ----------

import json
from datetime import datetime, timezone

import mlflow
import pandas as pd

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/msk-eval")

with mlflow.start_run(run_name=f"eval {datetime.now(timezone.utc):%Y-%m-%d %H:%M}") as run:
    mlflow.log_params({"models": ",".join(MODELS), "judge": JUDGE, "cases": len(cases)})
    for s in summary:
        key = s["model"].replace(".", "_")
        mlflow.log_metric(f"{key}.avg_score_pct", s["avg_score_pct"] or 0)
        mlflow.log_metric(f"{key}.safety_fails", s["safety_fails"])
        mlflow.log_metric(f"{key}.variant_safety_fails", s["variant_safety_fails"])
    mlflow.log_table(pd.DataFrame(summary), "summary.json")
    mlflow.log_dict({"rows": rows}, "rows.json")
    run_id = run.info.run_id

results = pd.DataFrame(rows)
results["grades"] = results["grades"].apply(json.dumps)
results["run_id"] = run_id
results["run_at"] = datetime.now(timezone.utc).isoformat()
spark.createDataFrame(results).write.mode("append").option("mergeSchema", "true").saveAsTable(f"{TABLE}.eval_results")
print(f"Saved {len(rows)} graded answers to {TABLE}.eval_results (MLflow run {run_id})")

# COMMAND ----------

# MAGIC %md ## Safety failures, including the stress-test variants

# COMMAND ----------

display(spark.sql(f"""
  SELECT model, case_id, variant, pct, missed_must_pass, answer
  FROM {TABLE}.eval_results
  WHERE run_id = '{run_id}' AND safety_fail
  ORDER BY model, case_id, variant
"""))
