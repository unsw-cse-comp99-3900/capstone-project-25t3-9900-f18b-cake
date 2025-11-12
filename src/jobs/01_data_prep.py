# 01_ingest_existing_splits_from_labels.py  — fast/robust ingest

import time, re, yaml
from pyspark.sql import SparkSession, functions as F, types as T

# ======== 路径区（按实际情况改）========
TRAIN_CSV = "/Volumes/cb_prod/comp9300-9900-f18b-cake/9900-f18b-cake/data/splits/train.csv"
VAL_CSV   = "/Volumes/cb_prod/comp9300-9900-f18b-cake/9900-f18b-cake/data/splits/val.csv"
TEST_CSV  = "/Volumes/cb_prod/comp9300-9900-f18b-cake/9900-f18b-cake/data/splits/test.csv"
LABELS_CSV= "/Volumes/cb_prod/comp9300-9900-f18b-cake/9900-f18b-cake/data/splits/labels.csv"

ID_COL = "biosample_id"
FINE_LABEL_COL = "label"         # labels.csv 中细粒度标签列
SPLIT_COL = "split"              # labels.csv 中split列
OUT_BASE = "/Volumes/cb_prod/comp9300-9900-f18b-cake/9900-f18b-cake/data/zh-cs"
MAPPING_YAML = "/Workspace/9900-f18b-cake/capstone-project-25t3-9900-f18b-cake/src/configs/label_mapping.yaml"

# ======== 快速验证开关 ========
FAST_MODE = False          # 先设 True 快速验证；跑通后改为 False 跑全量
FAST_N = 2000             # 每个 split 取前 N 行
# ==============================

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.files.maxPartitionBytes", 134217728)  # 128MB
spark.conf.set("spark.sql.csv.parser.columnPruning.enabled", "false")

def tlog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def read_split_csv(path: str):
    tlog(f"Reading CSV: {path}")
    df = (spark.read
          .option("header", True)
          .option("inferSchema", False)   # 关键：不推断，快
          .option("maxColumns", 2000000)
          .csv(path))
    if FAST_MODE:
        df = df.limit(FAST_N)
        tlog(f"FAST_MODE: limit {FAST_N}")
    tlog(f"Read done: {path}, rows≈{df.count()}")
    return df

# 1) 读三份特征 CSV
df_train = read_split_csv(TRAIN_CSV)
df_val   = read_split_csv(VAL_CSV)
df_test  = read_split_csv(TEST_CSV)

# 2) 读 labels.csv（只取必要列）
tlog("Reading labels.csv")
labels = (spark.read.option("header", True).option("inferSchema", True).csv(LABELS_CSV)
          .select(ID_COL, F.col(FINE_LABEL_COL).alias("fine_label"),
                  F.col(SPLIT_COL).alias("split_flag"), "disease_status"))
for need in [ID_COL, "fine_label", "split_flag"]:
    if need not in labels.columns:
        raise ValueError(f"labels.csv 缺少列：{need}")
tlog("labels.csv loaded")

# 3) 加载 YAML 映射（细→总），构造 UDF
with open(MAPPING_YAML, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
mapping = cfg["mapping"]
normalize = cfg.get("normalize", {"lower": True, "strip": True, "replace": []})

def _norm_py(s: str):
    if s is None: return s
    t = s.strip() if normalize.get("strip", True) else s
    if normalize.get("lower", True): t = t.lower()
    for a, b in normalize.get("replace", []): t = t.replace(a, b)
    return t

norm_map = {_norm_py(k): v for k, v in mapping.items()}
bmap = spark.sparkContext.broadcast(norm_map)

@F.udf(T.StringType())
def to_super_label(fine: T.StringType()):
    key = _norm_py(fine) if fine else None
    return bmap.value.get(key, "Other_benign_lowgrade")

def attach_labels_cast_and_write(split_name: str, split_df):
    tlog(f"[{split_name}] join with labels")
    joined = (split_df.join(labels, on=ID_COL, how="left")
              .withColumn("fine_label_str", F.col("fine_label").cast("string"))
              .withColumn("super_label", to_super_label(F.col("fine_label_str"))))

    # 将特征列统一转 double（除去 ID 和标签列）
    non_feature = {ID_COL, "fine_label", "fine_label_str", "super_label", "split_flag", "disease_status"}
    feature_cols = [c for c in joined.columns if c not in non_feature]
    tlog(f"[{split_name}] casting {len(feature_cols)} feature columns to double")
    for c in feature_cols:
        joined = joined.withColumn(c, F.col(c).cast("double"))

    # 统计信息
    mismatch_cnt = joined.filter((F.col("split_flag").isNotNull()) & (F.col("split_flag") != F.lit(split_name))).count()
    if mismatch_cnt > 0:
        tlog(f"⚠️  {split_name}: {mismatch_cnt} 行在 labels.csv 的 split_flag != '{split_name}'（以当前文件为准）")

    missing_label_cnt = joined.filter(F.col("fine_label").isNull()).count()
    if missing_label_cnt > 0:
        tlog(f"⚠️  {split_name}: {missing_label_cnt} 行未在 labels.csv 命中 fine_label → 将映射为 Other_benign_lowgrade")

    # 写 Delta
    out_path = f"{OUT_BASE}/{split_name}"
    tlog(f"[{split_name}] writing Delta to {out_path}")
    (joined.write.mode("overwrite").format("delta").save(out_path))

    # 打印分布
    tlog(f"[{split_name}] super_label distribution")
    joined.groupBy("super_label").count().orderBy("super_label").show(truncate=False)

    return out_path

out_train = attach_labels_cast_and_write("train", df_train)
out_val   = attach_labels_cast_and_write("val",   df_val)
out_test  = attach_labels_cast_and_write("test",  df_test)

# 4) 传给下游
dbutils.jobs.taskValues.set(key="train_path", value=out_train)
dbutils.jobs.taskValues.set(key="val_path",   value=out_val)
dbutils.jobs.taskValues.set(key="test_path",  value=out_test)

tlog(f"✅ Done. Delta under {OUT_BASE}. FAST_MODE={FAST_MODE}")
