# Databricks notebook source
# MAGIC %md
# MAGIC # MSK Eval: can we trust the AI grader?
# MAGIC The grader re-grades answers that already have human reference grades.
# MAGIC We measure plain agreement, Cohen's kappa (agreement corrected for luck),
# MAGIC agreement on must-pass safety items, and whether it gives the same grade when asked again.

# COMMAND ----------

# MAGIC %pip install -q pyyaml openai mlflow
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "msk_eval")
dbutils.widgets.text("judge", "databricks-claude-sonnet-4-5")
dbutils.widgets.text("repeats", "3")

TABLE = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
JUDGE = dbutils.widgets.get("judge").strip()
REPEATS = int(dbutils.widgets.get("repeats"))

# COMMAND ----------

from msk_eval.cases import load_cases
from msk_eval.providers import DatabricksNotebook
from msk_eval.runner import ModelSpec, validate_grader

cases = load_cases("../cases")
result = validate_grader(cases, ModelSpec(JUDGE, JUDGE, DatabricksNotebook()), repeats=REPEATS)

print(f"Grades compared:        {result['items']}")
print(f"Agreement:              {result['percent_agreement']:.0%}")
print(f"Cohen's kappa:          {result['cohens_kappa']:.2f}   (>0.8 strong, 0.6-0.8 substantial)")
print(f"Must-pass agreement:    {result['must_pass_agreement']:.0%}   (POC target: 90%)")
print(f"Same grade on repeats:  {result['consistency_across_repeats']:.0%}")

# COMMAND ----------

from datetime import datetime, timezone

import mlflow

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/msk-eval")
with mlflow.start_run(run_name=f"grader check {datetime.now(timezone.utc):%Y-%m-%d %H:%M}"):
    mlflow.log_params({"judge": JUDGE, "repeats": REPEATS})
    mlflow.log_metrics({k: result[k] for k in
                        ("percent_agreement", "cohens_kappa", "must_pass_agreement", "consistency_across_repeats")})
    mlflow.log_dict(result, "grader_validation.json")

# COMMAND ----------

# MAGIC %md ## Where the grader and the humans disagree

# COMMAND ----------

from pyspark.sql import functions as F

if result["disagreements"]:
    df = spark.createDataFrame(result["disagreements"]).withColumn("judge", F.lit(JUDGE))
    df.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{TABLE}.grader_disagreements")
    display(df)
else:
    print("No disagreements.")
