import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
client = MlflowClient()

experiment = mlflow.get_experiment_by_name("/Shared/cake_training")
runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                          filter_string="attributes.status='FINISHED'",
                          max_results=5000)

rows = []
for r in runs:
    p = r.data.params; m = r.data.metrics
    if "class_name" in p and "val_auc_pr" in m:
        rows.append({"class_name": p["class_name"], "val_auc_pr": m["val_auc_pr"], "val_auc_roc": m.get("val_auc_roc", None)})
df = pd.DataFrame(rows).sort_values("class_name")

plt.figure(figsize=(10,6))
plt.barh(df["class_name"], df["val_auc_pr"])
plt.xlabel("Validation AUC-PR")
plt.title("Per-class Validation AUC-PR (latest batch)")
out = "/dbfs/FileStore/cake_plots/val_auc_per_class.png"
plt.savefig(out, bbox_inches="tight")
mlflow.log_artifact(out, artifact_path="aggregate_plots")

spark_df = spark.createDataFrame(df)
spark_df.write.mode("overwrite").format("delta").save("/Volumes/cb_prod/comp9300-9900-f18b-cake/9900-f18b-cake/metrics/val_auc_latest")
