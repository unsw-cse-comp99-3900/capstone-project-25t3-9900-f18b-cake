# -*- coding: utf-8 -*-


from pathlib import Path
import sys, os, time, json

SRC_DIR = Path(__file__).resolve().parents[2]  # .../hc_3900/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ---------------------------------------------------

print(">>> RUNNING FILE:", __file__)
print(">>> CWD:", os.getcwd())
print(">>> FIRST sys.path:", sys.path[:5])

import numpy as np
import pandas as pd
from joblib import dump

# sklearn & plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

try:
    from differentialMethylationClassifier import DifferentialMethylation
    _DM_AVAILABLE = True
except Exception as e:
    print("[WARN] DifferentialMethylation Unavailable (rpy2/R may be missing). The DM feature screening will be skipped", e)
    _DM_AVAILABLE = False


# Utility Functions

def _load_splits_and_labels(csv_dir: str,
                            labels_filename: str = "labels_with_healthy.csv",
                            id_col: str = "biosample_id",
                            label_col: str = "label"):

    base = Path(csv_dir)
    train_csv  = base / "train.csv"
    val_csv    = base / "val.csv"
    labels_csv = base / labels_filename
    for p in [train_csv, val_csv, labels_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing data file：{p}")

    train  = pd.read_csv(train_csv, low_memory=False)
    val    = pd.read_csv(val_csv,   low_memory=False)
    labels = pd.read_csv(labels_csv)

    if id_col not in train.columns or id_col not in val.columns:
        raise ValueError(f'train/val The "ID" column is missing: {id_col}')

    if id_col not in labels.columns or label_col not in labels.columns:
        raise ValueError(f"labels Missing necessary columns：{id_col}, {label_col}")

    train = train.merge(labels[[id_col, label_col]], on=id_col, how="left").dropna(subset=[label_col])
    val   = val.merge(labels[[id_col, label_col]],   on=id_col, how="left").dropna(subset=[label_col])

    feature_cols = [c for c in train.columns if c not in [id_col, label_col]]
    X_train = train[feature_cols]
    y_train = train[label_col]
    X_val   = val[feature_cols]
    y_val   = val[label_col]
    return X_train, y_train, X_val, y_val, feature_cols


def _clean_X(df: pd.DataFrame) -> pd.DataFrame:
    """统一清洗（inf→NaN→to_numeric→fillna(0)→float32）"""
    return (df.replace([np.inf, -np.inf], np.nan)
              .apply(pd.to_numeric, errors="coerce")
              .fillna(0.0)
              .astype("float32"))


# Main training function

def demoA_train_from_csv(csv_dir: str = "data/splits_concat",
                         labels_filename: str = "labels_with_healthy.csv",
                         id_col: str = "biosample_id",
                         label_col: str = "label",
                         top_k: int | None = 10_000,
                         n_estimators: int = 200,
                         seed: int = 42,
                         out_dir: str | None = None,
                         eval_test: bool = True):
    # A) output directory
    proj_root = Path(__file__).resolve().parents[3]   # .../hc_3900
    out = Path(out_dir) if out_dir else (proj_root / "results" / "result-training")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO]output directory: {out}")

    # B) read data
    print("[INFO] read CSV ...")
    X_train, y_train, X_val, y_val, feature_cols = _load_splits_and_labels(
        csv_dir, labels_filename, id_col, label_col
    )
    print(f"[INFO] Train/validate the shape: {X_train.shape} / {X_val.shape}；characteristic number: {len(feature_cols)}")

    # C) cleanning
    X_train = _clean_X(X_train)
    X_val   = _clean_X(X_val)

    # D) Feature selection (Prioritize using DM; otherwise, SelectKBest)
    if _DM_AVAILABLE and top_k:
        selector = DifferentialMethylation(top_k=top_k)
        X_train = selector.fit_transform(X_train, y_train)
        X_val   = selector.transform(X_val)
        sel_name = "DifferentialMethylation"
    elif top_k:
        selector = SelectKBest(f_classif, k=min(int(top_k), X_train.shape[1]))
        X_train = selector.fit_transform(X_train, y_train)
        X_val   = selector.transform(X_val)
        sel_name = "SelectKBest(f_classif)"
    else:
        selector = None
        sel_name = "None"
    print(f"[INFO] feature selection: {sel_name}；Dimensions after screening: {X_train.shape} / {X_val.shape}")

    # E) training model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = round(time.time() - t0, 3)

    # F) demonstration and evaluation
    y_pred = clf.predict(X_val)
    metrics_val = {
        "accuracy": round(accuracy_score(y_val, y_pred), 6),
        "f1_macro": round(f1_score(y_val, y_pred, average="macro"), 6),
        "train_time_sec": train_time
    }
    print("✅ Demo A Training Finished", metrics_val)

    # visualization versus prediction

    # 1) Confusion matrix (validation set）
    cm = confusion_matrix(y_val, y_pred, labels=np.unique(y_val))
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Val)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(ticks=range(len(np.unique(y_val))), labels=np.unique(y_val), rotation=45, ha="right")
    plt.yticks(ticks=range(len(np.unique(y_val))), labels=np.unique(y_val))
    plt.colorbar(); plt.tight_layout()
    fig.savefig(out / "confusion_matrix_val.png", dpi=150); plt.close(fig)

    # 2) F1 bar charts of various categories
    rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    cls = [k for k in rep.keys() if k not in ("accuracy","macro avg","weighted avg")]
    f1s = [rep[k]["f1-score"] for k in cls]
    fig = plt.figure(figsize=(max(6, len(cls)*0.5), 4))
    plt.bar(range(len(cls)), f1s)
    plt.xticks(range(len(cls)), cls, rotation=60, ha="right")
    plt.ylabel("F1-score"); plt.title("Per-class F1 (Val)")
    plt.tight_layout()
    fig.savefig(out / "per_class_f1_val.png", dpi=150); plt.close(fig)

    # 3) Top 20 RF Feature Importance
    try:
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        fig = plt.figure(figsize=(8, 5))
        plt.bar(range(len(idx)), importances[idx])
        plt.xticks(range(len(idx)), [str(i) for i in idx], rotation=60, ha="right")
        plt.ylabel("Importance"); plt.title("RF Feature Importances (Top 20)")
        plt.tight_layout()
        fig.savefig(out / "rf_feature_importance_top20.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("[WARN] 无法绘制特征重要性：", e)

    # 4) Training/validation category distribution
    fig = plt.figure(figsize=(6,4))
    pd.Series(y_train).value_counts().sort_index().plot(kind="bar")
    plt.title("Train Class Distribution"); plt.ylabel("Count"); plt.tight_layout()
    fig.savefig(out / "class_distribution_train.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(6,4))
    pd.Series(y_val).value_counts().sort_index().plot(kind="bar")
    plt.title("Val Class Distribution"); plt.ylabel("Count"); plt.tight_layout()
    fig.savefig(out / "class_distribution_val.png", dpi=150); plt.close(fig)

    # 5) ROC and PR curves
    try:
        classes = np.unique(y_val)
        if len(classes) >= 2 and hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_val)
            if y_score.ndim == 2 and y_score.shape[1] == len(classes):
                y_val_bin = label_binarize(y_val, classes=classes)

                # ROC
                fig = plt.figure(figsize=(7,6))
                for i, cls_name in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_score[:, i])
                    auc_i = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"{cls_name} (AUC={auc_i:.3f})")
                plt.plot([0,1], [0,1], "--", lw=1)
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.title("One-vs-Rest ROC (Val)"); plt.legend(fontsize=8); plt.tight_layout()
                fig.savefig(out / "roc_ovr_val.png", dpi=150); plt.close(fig)

                # PR
                fig = plt.figure(figsize=(7,6))
                for i, cls_name in enumerate(classes):
                    prec, rec, _ = precision_recall_curve(y_val_bin[:, i], y_score[:, i])
                    auc_pr = auc(rec, prec)
                    plt.plot(rec, prec, label=f"{cls_name} (AP={auc_pr:.3f})")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title("One-vs-Rest Precision-Recall (Val)"); plt.legend(fontsize=8); plt.tight_layout()
                fig.savefig(out / "pr_ovr_val.png", dpi=150); plt.close(fig)
            else:
                print("[WARN] predict_proba 形状与类别数不匹配，跳过 ROC/PR。")
        else:
            print("[INFO] 类别<2 或模型不支持 predict_proba，跳过 ROC/PR。")
    except Exception as e:
        print("[WARN] 绘制 ROC/PR 出错：", e)

    # 6) Validation set prediction comparison
    pd.DataFrame({
        "y_true": y_val.reset_index(drop=True),
        "y_pred": pd.Series(y_pred)
    }).to_csv(out / "val_predictions.csv", index=False)

    # 7)Complete Feature Importance CSV
    try:
        importances = clf.feature_importances_
        feat_idx = np.arange(importances.shape[0])
        imp_df = pd.DataFrame({"feature_index": feat_idx, "importance": importances})
        imp_df.sort_values("importance", ascending=False).to_csv(out / "rf_feature_importance_full.csv", index=False)
    except Exception as e:
        print("[WARN] 无法导出完整特征重要性：", e)

    # save the model and metrics/metadata
    dump(clf, out / "rf_model.joblib")
    pd.DataFrame([metrics_val]).to_csv(out / "metrics.csv", index=False)

    selector_info = {"name": "None", "top_k": None}
    if _DM_AVAILABLE and top_k:
        selector_info = {"name": "DifferentialMethylation", "top_k": int(top_k)}
    elif top_k:
        selector_info = {"name": "SelectKBest(f_classif)", "top_k": int(top_k)}

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "entry": "training.py:demoA_train_from_csv",
            "id_column": id_col,
            "label_column": label_col,
            "selector": selector_info,
            "model": {"name": "RandomForestClassifier",
                      "params": {"n_estimators": n_estimators, "random_state": seed, "class_weight": "balanced"}},
            "n_features_after_select": int(X_train.shape[1]),
            "metrics_val": metrics_val
        }, f, ensure_ascii=False, indent=2)

    #One-time assessment test
    if eval_test:
        test_csv   = Path(csv_dir) / "test.csv"
        labels_csv = Path(csv_dir) / labels_filename
        if test_csv.exists() and labels_csv.exists():
            test   = pd.read_csv(test_csv, low_memory=False)
            labels = pd.read_csv(labels_csv)
            test   = test.merge(labels[[id_col, label_col]], on=id_col, how="left").dropna(subset=[label_col])

            X_test = test[[c for c in test.columns if c not in [id_col, label_col]]]
            y_test = test[label_col]
            X_test = _clean_X(X_test)

            if selector is not None:
                try:
                    X_test = selector.transform(X_test)
                except Exception as e:
                    print("[WARN] selector.transform(test) fail：", e)
                    pass
            else:
                if hasattr(X_train, "columns"):
                    X_test = X_test.reindex(columns=list(X_train.columns), fill_value=0.0)

            y_pred_test = clf.predict(X_test)
            metrics_test = {
                "accuracy": round(accuracy_score(y_test, y_pred_test), 6),
                "f1_macro": round(f1_score(y_test, y_pred_test, average="macro"), 6)
            }
            pd.DataFrame([metrics_test]).to_csv(out / "metrics_test.csv", index=False)
            print("ℹ️ Test(one-time) indicators:", metrics_test)



if __name__ == "__main__":
    print(">>> ENTER main()")
    demoA_train_from_csv(
        csv_dir="data/splits_concat",
        labels_filename="labels_with_healthy.csv",
        id_col="biosample_id",
        label_col="label",
        top_k=10000,     
        n_estimators=200,
        seed=42,
        out_dir=None,     # results/result-training/
        eval_test=True    # One-time assessment test
    )
    print(">>> DONE main()")

