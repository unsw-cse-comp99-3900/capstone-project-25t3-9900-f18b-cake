# /Workspace/9900-f18b-cake/capstone-project-25t3-9900-f18b-cake/src/jobs/02_train_per_class.py
# Random Forest per-class trainer with YAML hyperparams + MLflow + robustness

# --------------------
# Widgets (Job/Notebook parameters)
# --------------------
dbutils.widgets.text("class_name", "")                 # 必填：12 大类之一，如 'Leukaemia'
dbutils.widgets.text("use_yaml", "true")               # 是否使用 YAML 集中调参（true/false）
# （可选）Standalone 调试时传入这三个路径；在 Job 里会自动从 task values 读取
dbutils.widgets.text("train_path", "")
dbutils.widgets.text("val_path", "")
dbutils.widgets.text("test_path", "")
# （可选）不开 YAML 时的手动超参（用作兜底）
dbutils.widgets.text("numTrees", "300")
dbutils.widgets.text("maxDepth", "14")
dbutils.widgets.text("maxBins", "64")
dbutils.widgets.text("subsamplingRate", "0.9")
dbutils.widgets.text("featureSubsetStrategy", "sqrt")
dbutils.widgets.text("minInstancesPerNode", "1")
# （可选）错误日志 Delta 表路径（提供则在异常时写入一条错误记录）
dbutils.widgets.text("error_table_path", "")

# --------------------
# Imports
# --------------------
import os, yaml, hashlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlflow, mlflow.spark
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession
from pyspark.sql import functions as F, types as T
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# --------------------
# Spark / MLflow setup
# --------------------
spark = SparkSession.builder.getOrCreate()

# 固定一个实验路径（如需变更可改为 widgets 或环境变量）
try:
    mlflow.set_experiment("/Shared/cake_training")
except Exception:
    pass

# --------------------
# Resolve inputs & paths
# --------------------
cname = dbutils.widgets.get("class_name").strip()
if not cname:
    raise ValueError("必须传入 class_name（12 大类之一，例如 'Leukaemia'）。")

use_yaml = dbutils.widgets.get("use_yaml").lower() == "true"

# 从上游 data_prep 读取 task values；如为空，再用 widgets 兜底（便于单脚本调试）
train_path = dbutils.jobs.taskValues.get(taskKey="data_prep", key="train_path", default="")
val_path   = dbutils.jobs.taskValues.get(taskKey="data_prep", key="val_path",   default="")
test_path  = dbutils.jobs.taskValues.get(taskKey="data_prep", key="test_path",  default="")
if not train_path:
    train_path = dbutils.widgets.get("train_path").strip()
    val_path   = dbutils.widgets.get("val_path").strip()
    test_path  = dbutils.widgets.get("test_path").strip()
if not (train_path and val_path and test_path):
    raise ValueError("未能获取 train/val/test 路径。请通过 data_prep 的 task values 或 widgets 传入。")

error_table_path = dbutils.widgets.get("error_table_path").strip()

# --------------------
# Load splits
# --------------------
train = spark.read.format("delta").load(train_path)
val   = spark.read.format("delta").load(val_path)
test  = spark.read.format("delta").load(test_path)

if "super_label" not in train.columns:
    raise ValueError("缺少列 'super_label'。请确认 01_data_prep.py 已正确生成该列。")

# --------------------
# Binarize (one-vs-rest)
# --------------------
def binarize(df):
    return df.withColumn("y", F.when(F.col("super_label") == F.lit(cname), F.lit(1)).otherwise(F.lit(0)))

train_b, val_b, test_b = map(binarize, [train, val, test])

# 样本数与跳过保护
pos = train_b.filter(F.col("y")==1).count()
neg = train_b.filter(F.col("y")==0).count()
if pos == 0 or neg == 0:
    with mlflow.start_run(run_name=f"rf_class_{cname}_SKIPPED"):
        mlflow.log_param("class_name", cname)
        mlflow.log_param("skipped_due_to_no_samples", True)
        mlflow.log_metric("train_pos_count", pos)
        mlflow.log_metric("train_neg_count", neg)
    dbutils.notebook.exit(f"跳过训练：{cname}（train 正类={pos} / 负类={neg}）")

# 类不平衡权重
total = pos + neg
w_pos = total / (2 * pos)
w_neg = total / (2 * neg)
train_b = train_b.withColumn("w", F.when(F.col("y")==1, F.lit(float(w_pos))).otherwise(F.lit(float(w_neg))))

# --------------------
# Feature assembly
# --------------------
non_feature = {
    "label", "fine_label", "fine_label_str", "super_label",
    "y", "w"
}
feature_cols = [c for c in train_b.columns if c not in non_feature]
if not feature_cols:
    raise ValueError("未检测到特征列。请确认数据表中除 super_label/y/w 外的列为特征。")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# --------------------
# Hyperparams: YAML (preferred) or widgets fallback
# --------------------
def load_yaml_params(class_name: str):
    yaml_path = "/Workspace/9900-f18b-cake/capstone-project-25t3-9900-f18b-cake/src/configs/hyperparams.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        all_params = yaml.safe_load(f)
    default_params = all_params.get("default", {})
    class_params = all_params.get(class_name, {})
    merged = {**default_params, **class_params}
    yaml_hash = hashlib.md5(str(all_params).encode("utf-8")).hexdigest()
    return merged, yaml_path, yaml_hash

try:
    if use_yaml:
        rf_params, yaml_path, yaml_hash = load_yaml_params(cname)
        numTrees = int(rf_params.get("numTrees", 300))
        maxDepth = int(rf_params.get("maxDepth", 14))
        maxBins = int(rf_params.get("maxBins", 64))
        subsamplingRate = float(rf_params.get("subsamplingRate", 0.9))
        featureSubsetStrategy = rf_params.get("featureSubsetStrategy", "sqrt")
        minInstancesPerNode = int(rf_params.get("minInstancesPerNode", 1))
    else:
        numTrees = int(dbutils.widgets.get("numTrees"))
        maxDepth = int(dbutils.widgets.get("maxDepth"))
        maxBins = int(dbutils.widgets.get("maxBins"))
        subsamplingRate = float(dbutils.widgets.get("subsamplingRate"))
        featureSubsetStrategy = dbutils.widgets.get("featureSubsetStrategy")
        minInstancesPerNode = int(dbutils.widgets.get("minInstancesPerNode"))
        rf_params = {
            "numTrees": numTrees,
            "maxDepth": maxDepth,
            "maxBins": maxBins,
            "subsamplingRate": subsamplingRate,
            "featureSubsetStrategy": featureSubsetStrategy,
            "minInstancesPerNode": minInstancesPerNode
        }
        yaml_path, yaml_hash = "", ""
except Exception as e:
    # 如果 YAML 读取失败，自动降级到 widgets 参数
    numTrees = int(dbutils.widgets.get("numTrees"))
    maxDepth = int(dbutils.widgets.get("maxDepth"))
    maxBins = int(dbutils.widgets.get("maxBins"))
    subsamplingRate = float(dbutils.widgets.get("subsamplingRate"))
    featureSubsetStrategy = dbutils.widgets.get("featureSubsetStrategy")
    minInstancesPerNode = int(dbutils.widgets.get("minInstancesPerNode"))
    rf_params = {
        "numTrees": numTrees,
        "maxDepth": maxDepth,
        "maxBins": maxBins,
        "subsamplingRate": subsamplingRate,
        "featureSubsetStrategy": featureSubsetStrategy,
        "minInstancesPerNode": minInstancesPerNode
    }
    yaml_path, yaml_hash = f"[YAML load failed: {e}]", ""

# --------------------
# Train / Evaluate / Log
# --------------------
try:
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="y",
        weightCol="w",
        numTrees=numTrees,
        maxDepth=maxDepth,
        maxBins=maxBins,
        subsamplingRate=subsamplingRate,
        featureSubsetStrategy=featureSubsetStrategy,
        minInstancesPerNode=minInstancesPerNode,
        seed=42,
        probabilityCol="prob",
        rawPredictionCol="rawPred",
        predictionCol="prediction"
    )

    pipe = Pipeline(stages=[assembler, rf])

    with mlflow.start_run(run_name=f"rf_class_{cname}") as run:
        # 记录参数
        mlflow.log_param("class_name", cname)
        mlflow.log_param("use_yaml", use_yaml)
        if yaml_path:
            mlflow.log_param("hyperparams_yaml_path", yaml_path)
        if yaml_hash:
            mlflow.log_param("hyperparams_yaml_hash", yaml_hash)
        # 展开记录合并后的 rf_params（加前缀避免与其它 param 冲突）
        for k, v in rf_params.items():
            mlflow.log_param(f"rf_{k}", v)
        # 记录样本数
        mlflow.log_metric("train_pos_count", pos)
        mlflow.log_metric("train_neg_count", neg)

        # 训练
        model = pipe.fit(train_b)

        # 验证指标（AUC-ROC/AUC-PR）
        pred_val = model.transform(val_b)
        evaluator_roc = BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="rawPred", metricName="areaUnderROC")
        evaluator_pr  = BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="rawPred", metricName="areaUnderPR")
        auc_roc = evaluator_roc.evaluate(pred_val)
        auc_pr  = evaluator_pr.evaluate(pred_val)
        mlflow.log_metric("val_auc_roc", auc_roc)
        mlflow.log_metric("val_auc_pr",  auc_pr)

        # 混淆矩阵
        pdf = pred_val.select(F.col("y").alias("y_true"), F.col("prediction").cast("int").alias("y_pred")).toPandas()
        cm = pd.crosstab(pdf["y_true"], pdf["y_pred"], rownames=["True"], colnames=["Pred"], dropna=False)
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"RF Confusion Matrix - {cname}")
        plt.xlabel("Pred"); plt.ylabel("True")
        for (i, j), v in np.ndenumerate(cm.values):
            plt.text(j, i, str(v), ha='center', va='center')
        cm_path = f"/tmp/rf_cm_{cname}.png"
        plt.savefig(cm_path, bbox_inches="tight")
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # 特征重要性
        rf_stage = model.stages[-1]
        importances = rf_stage.featureImportances.toArray()
        imp_pdf = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
        imp_csv = f"/tmp/rf_feature_importance_{cname}.csv"
        imp_pdf.to_csv(imp_csv, index=False)
        mlflow.log_artifact(imp_csv, artifact_path="feature_importance")

        # 注册模型（每类一个注册名）
        registered_name = f"cake_rf_cls_{cname}"
        mlflow.spark.log_model(model, artifact_path="model", registered_model_name=registered_name)

except Exception as e:
    if error_table_path:
        err_df = spark.createDataFrame(
            [(cname, str(e)[:2000], F.current_timestamp())],
            schema=T.StructType([
                T.StructField("class_name", T.StringType(), True),
                T.StructField("error_msg",  T.StringType(), True),
                T.StructField("ts",         T.TimestampType(), True),
            ])
        )
        (err_df.write.mode("append").format("delta").save(error_table_path))
    # 抛出让 Job 重试
    raise

