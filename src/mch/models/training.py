from __future__ import annotations
from pathlib import Path
import sys, json, time, warnings
import numpy as np
import pandas as pd

from joblib import dump, load

from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.exceptions import UndefinedMetricWarning  

#  plotting & metrics for curves 
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="No positive class found in y_true.*")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(cm, classes, out_path: Path, title="Confusion Matrix", normalize=False, dpi=220):
    cm_disp = cm.astype(np.float64)
    if normalize:
        row_sum = cm_disp.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_disp = cm_disp / row_sum

    plt.figure(figsize=(6.4, 5.4))
    plt.imshow(cm_disp, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=13)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30, ha="right", fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)

    fmt = ".2f" if normalize else ".0f"
    thresh = cm_disp.max() / 2.0 if cm_disp.size else 0.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            val = cm_disp[i, j]
            plt.text(
                j, i, format(val, fmt),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_roc_pr_curves(y_true, y_proba, classes, out_dir: Path):
   
    try:
        y_true = np.asarray(y_true)
        y_bin = label_binarize(y_true, classes=classes)
    except Exception:
        return

    # ROC
    plotted_any = False
    try:
        plt.figure(figsize=(6,5))
        for i, c in enumerate(classes):
            pos = int(y_bin[:, i].sum())
            neg = int(y_bin.shape[0] - pos)
            if pos == 0 or neg == 0:
                continue 
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.8, label=f"{c} (AUC={roc_auc:.2f})")
            plotted_any = True
        if plotted_any:
            plt.plot([0,1], [0,1], "k--", lw=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curves (Val)")
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout(); plt.savefig(out_dir/"val_roc_curve.png", dpi=220)
        plt.close()
    except Exception:
        plt.close()

    # PR
    plotted_any = False
    try:
        plt.figure(figsize=(6,5))
        for i, c in enumerate(classes):
            pos = int(y_bin[:, i].sum())
            neg = int(y_bin.shape[0] - pos)
            if pos == 0 or neg == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            plt.plot(rec, prec, lw=1.8, label=f"{c}")
            plotted_any = True
        if plotted_any:
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title("Precision–Recall Curves (Val)")
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout(); plt.savefig(out_dir/"val_pr_curve.png", dpi=220)
        plt.close()
    except Exception:
        plt.close()

def get_feature_names_from_pipeline(fitted_pipe: Pipeline, original_feat_cols):
    try:
        dm = fitted_pipe.named_steps.get("dm", None)
        if dm is not None and getattr(dm, "selected_cols_", None):
            return list(dm.selected_cols_)
    except Exception:
        pass
    return list(original_feat_cols)

def plot_feature_importance(fitted_pipe: Pipeline, feat_cols, out_path: Path, topn=30):
    try:
        clf = fitted_pipe.named_steps["clf"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return
        importances = np.asarray(importances)
        idx = np.argsort(importances)[::-1][:topn]
        names = np.array(feat_cols)[idx]

        plt.figure(figsize=(8,6))
        plt.barh(range(len(idx))[::-1], importances[idx][::-1])
        plt.yticks(range(len(idx))[::-1], names[::-1], fontsize=8)
        plt.xlabel("Importance"); plt.title(f"Top-{topn} Feature Importances")
        plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()
    except Exception:
        plt.close()

def plot_cv_curve_if_any(cv_csv_path: Path, out_path: Path):
    try:
        if not cv_csv_path.exists():
            return
        df = pd.read_csv(cv_csv_path)
        if "mean_test_score" in df.columns and "param_clf__n_estimators" in df.columns:
            grouped = df.groupby("param_clf__n_estimators")["mean_test_score"].mean().reset_index()
            plt.figure(figsize=(5.6,4.2))
            plt.plot(grouped["param_clf__n_estimators"], grouped["mean_test_score"], marker="o")
            plt.xlabel("n_estimators"); plt.ylabel("mean_test_score")
            plt.title("CV: F1_macro vs n_estimators")
            plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()
        elif "val_score" in df.columns and "clf__n_estimators" in df.columns:
            plt.figure(figsize=(5.6,4.2))
            plt.plot(df["clf__n_estimators"], df["val_score"], marker="o")
            plt.xlabel("n_estimators"); plt.ylabel("val_score (F1_macro)")
            plt.title("Val: F1_macro vs n_estimators")
            plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()
    except Exception:
        plt.close()



TASKS = [
    {"name": "global_multiclass", "csv_dir": "/Volumes/cb_prod/comp9300-9900-f18b-cake/9900-f18b-cake/data/splits"},
]
USE_PRECOMPUTED_DM = True
LABELS_FILENAME = "labels.csv"
ID_COL   = "biosample_id"
Y_COL    = "label"

SEED = 42
TOP_K = 5000
N_EST_GRID = [200, 400]
SCORING = "f1_macro"

# =========================
# I/O & helpers
# =========================
def _load_splits_and_labels(csv_dir: str,
                            labels_filename: str = "labels.csv",
                            id_col: str = "biosample_id",
                            label_col: str = "label"):
    base = Path(csv_dir)
    train_csv  = base / "train.csv"
    val_csv    = base / "val.csv"
    labels_csv = base / labels_filename
    for p in (train_csv, val_csv, labels_csv):
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {p}")

    train  = pd.read_csv(train_csv, low_memory=False)
    val    = pd.read_csv(val_csv,   low_memory=False)
    labels = pd.read_csv(labels_csv)

    for df in (train, val, labels):
        df[id_col] = df[id_col].astype(str).str.strip().str.upper()

    train = train.merge(labels[[id_col, label_col]], on=id_col, how="left").dropna(subset=[label_col])
    val   = val.merge(labels[[id_col, label_col]],   on=id_col, how="left").dropna(subset=[label_col])

    feat_cols = [c for c in train.columns if c not in [id_col, label_col]]
    X_train = train[feat_cols]
    y_train = train[label_col].astype(str)
    X_val   = val[feat_cols]
    y_val   = val[label_col].astype(str)
    return X_train, y_train, X_val, y_val, feat_cols

def _clean_X(df: pd.DataFrame) -> pd.DataFrame:
    return (df.replace([np.inf, -np.inf], np.nan)
              .apply(pd.to_numeric, errors="coerce")
              .fillna(0.0)
              .astype("float32"))

def _fmt_min(seconds: float) -> str:
    return f"{seconds/60.0:.2f} min"

# =========================
# Transformers
# =========================
class CleanNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame): return _clean_X(X)
        X = np.asarray(X, dtype="float32")
        return np.nan_to_num(X, copy=False)

class InFoldDMSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = TOP_K):
        self.k = int(k) if k is not None else None
        self._kbest = None
        self.selected_cols_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            k = min(self.k, X.shape[1]) if self.k else "all"
        else:
            X = np.asarray(X)
            k = min(self.k, X.shape[1]) if self.k else "all"
        self._kbest = SelectKBest(score_func=f_classif, k=k)
        self._kbest.fit(X, y)
        if isinstance(X, pd.DataFrame) and hasattr(self._kbest, "get_support"):
            mask = self._kbest.get_support()
            self.selected_cols_ = list(X.columns[mask])
        return self

    def transform(self, X):
        return self._kbest.transform(X)

class PrecomputedDMSelector(BaseEstimator, TransformerMixin):
    def __init__(self, csv_dir: str, top_k: int | None = TOP_K,
                 id_candidates=("biosample_id", "sample_id")):
        self.csv_dir = csv_dir
        self.top_k = top_k
        self.id_candidates = id_candidates
        self.selected_cols_ = None

    def fit(self, X, y=None):
        base = Path(self.csv_dir)
        sel_path   = base / "dm_selected_probes.csv"
        stats_path = base / "dm_stats.csv"
        selected = None

        if self.top_k and stats_path.exists():
            try:
                dm_stats = pd.read_csv(stats_path, low_memory=False)
                probe_col = next((c for c in ["probe_id","probe","feature","id"] if c in dm_stats.columns), None)
                pval_col  = next((c for c in ["p_value","pvalue","pval"] if c in dm_stats.columns), None)
                if probe_col and pval_col:
                    ranked = (dm_stats[[probe_col, pval_col]].dropna()
                              .sort_values(pval_col, ascending=True)[probe_col]
                              .astype(str).tolist())
                    selected = ranked[: int(self.top_k)]
            except Exception as e:
                print(f"[WARN] parse dm_stats.csv failed: {e}")

        if sel_path.exists():
            try:
                dm_sel = pd.read_csv(sel_path, low_memory=False)
                idc = next((c for c in self.id_candidates if c in dm_sel.columns), None)
                if idc:
                    cols_from_file = [c for c in dm_sel.columns if c != idc]
                else:
                    name_col = next((c for c in ["probe_id","probe","feature","id"] if c in dm_sel.columns),
                                    dm_sel.columns[0])
                    cols_from_file = dm_sel[name_col].astype(str).tolist()

                if selected is None:
                    selected = cols_from_file
                else:
                    sset = set(cols_from_file)
                    selected = [c for c in selected if c in sset]
            except Exception as e:
                print(f"[WARN] parse dm_selected_probes.csv failed: {e}")

        if isinstance(X, pd.DataFrame) and selected:
            selected = [c for c in selected if c in set(X.columns)]

        self.selected_cols_ = selected
        return self

    def transform(self, X):
        if self.selected_cols_ and isinstance(X, pd.DataFrame):
            return X[self.selected_cols_]
        return X


def _core_metrics(y_true, y_pred):
    return {
        "accuracy":          round(accuracy_score(y_true, y_pred), 6),
        "f1_macro":          round(f1_score(y_true, y_pred, average="macro"), 6),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 6),
    }

# =========================
# Train a task (node）
# =========================
def train_one_task(task_name: str, csv_dir: str, out_root: Path,
                   top_k: int = TOP_K, n_est_grid=None, scoring=SCORING, seed=SEED):
    n_est_grid = n_est_grid or N_EST_GRID
    out_dir = out_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Timing: Loading
    t_all0 = time.perf_counter()
    t0 = time.perf_counter()
    X_train, y_train, X_val, y_val, feat_cols = _load_splits_and_labels(csv_dir, LABELS_FILENAME, ID_COL, Y_COL)
    time_load = time.perf_counter() - t0

    # Adaptive folding
    cls_counts = y_train.value_counts()
    min_count = int(cls_counts.min())
    auto_splits = int(min(5, max(2, min_count)))

    selector_step = PrecomputedDMSelector(csv_dir=csv_dir, top_k=top_k) if USE_PRECOMPUTED_DM \
                    else InFoldDMSelector(k=top_k)

    # Pipeline：Cleaning → Selector → RF
    base_pipe = Pipeline(steps=[
        ("clean", CleanNumeric()),
        ("dm", selector_step),
        ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=seed)),
    ])

    # Grid
    param_grid = [{"clf__n_estimators": n_est_grid}]

    print(f"[INFO] Train/val shape: {X_train.shape} / {X_val.shape}; classes={len(cls_counts)}, min_count={min_count}")

    # train
    t0 = time.perf_counter()
    best_params_line = ""
    if min_count >= 2:
        skf = StratifiedKFold(n_splits=auto_splits, shuffle=True, random_state=seed)
        gs = GridSearchCV(estimator=base_pipe, param_grid=param_grid, scoring=scoring,
                          cv=skf, n_jobs=1, refit=True, return_train_score=False, verbose=1)
        gs.fit(X_train, y_train)
        time_fit = time.perf_counter() - t0
        best_est = gs.best_estimator_
        cv_mean = float(gs.best_score_)
        pd.DataFrame(gs.cv_results_).to_csv(out_dir / "cv_results.csv", index=False)
        best_params_line = f"[INFO] Selected by CV: score={cv_mean:.4f}, params={gs.best_params_}"
        print(best_params_line)
    else:
        print("[WARN] Min class <2 -> disable CV; select hyperparams on validation set.")
        records = []
        best_score, best_est, best_param = -1.0, None, None
        for n in n_est_grid:
            pipe = Pipeline(steps=[
                ("clean", CleanNumeric()),
                ("dm", selector_step),
                ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=seed, n_estimators=n)),
            ])
            pipe.fit(X_train, y_train)
            s = f1_score(y_val, pipe.predict(X_val), average="macro") if scoring=="f1_macro" \
                else accuracy_score(y_val, pipe.predict(X_val))
            records.append({"clf__n_estimators": n, "val_score": s})
            if s > best_score:
                best_score, best_est, best_param = s, pipe, {"clf__n_estimators": n}
        time_fit = time.perf_counter() - t0
        cv_mean = float(best_score)
        pd.DataFrame(records).to_csv(out_dir / "cv_results.csv", index=False)
        best_params_line = f"[INFO] Selected by val: score={cv_mean:.4f}, params={best_param}"
        print(best_params_line)

    # Prediction and Metrics （Val）
    t0 = time.perf_counter()
    y_pred = best_est.predict(X_val)
    time_pred = time.perf_counter() - t0

    ALL_CLASSES = np.unique(y_train).tolist()
    cm = confusion_matrix(y_val, y_pred, labels=ALL_CLASSES)  # 仅保存，不打印

    metrics_val = _core_metrics(y_val, y_pred)
    metrics_val.update({
        "cv_best": round(cv_mean, 6),
        "time_load_min": _fmt_min(time_load),
        "time_fit_min": _fmt_min(time_fit),
        "time_pred_min": _fmt_min(time_pred),
        "time_total_min": _fmt_min(time.perf_counter() - t_all0),
    })

    print("\nValidation (Val) :")
    print(f"- Accuracy          : {metrics_val['accuracy']}")
    print(f"- F1 Macro          : {metrics_val['f1_macro']}")
    print(f"- Balanced Accuracy : {metrics_val['balanced_accuracy']}")
    print(f"- CV Best ({SCORING}): {metrics_val['cv_best']}")

    _ensure_dir(out_dir)
    # 1) Confusion matrix
    plot_confusion_matrix(
        cm, classes=ALL_CLASSES,
        out_path=out_dir / "val_confusion_matrix.png",
        title="Confusion Matrix (Val)", normalize=False, dpi=220
    )
    plot_confusion_matrix(
        cm, classes=ALL_CLASSES,
        out_path=out_dir / "val_confusion_matrix_norm.png",
        title="Normalized Confusion Matrix (Val)", normalize=True, dpi=220
    )

    # 2) ROC / PR
    try:
        y_proba = best_est.predict_proba(X_val)
        if isinstance(y_proba, list):
            y_proba = np.column_stack([p[:,1] if (p.ndim == 2 and p.shape[1] > 1) else p for p in y_proba])
        clf_classes = best_est.named_steps["clf"].classes_.tolist()
        plot_roc_pr_curves(y_val, y_proba, classes=clf_classes, out_dir=out_dir)
    except Exception:
        pass

    # 3) Feature importance（Top-30）
    feat_names = get_feature_names_from_pipeline(best_est, feat_cols)
    plot_feature_importance(best_est, feat_names, out_path=out_dir / "feature_importance_top30.png", topn=30)

    # 4) CV/Val curve
    plot_cv_curve_if_any(out_dir / "cv_results.csv", out_path=out_dir / "cv_score_curve.png")
    # ==== END NEW ====

    # Test set
    metrics_test = None
    test_csv = Path(csv_dir) / "test.csv"
    labels_csv = Path(csv_dir) / LABELS_FILENAME
    if test_csv.exists() and labels_csv.exists():
        t_test0 = time.perf_counter()

        test   = pd.read_csv(test_csv, low_memory=False)
        labels = pd.read_csv(labels_csv)
        for df in (test, labels):
            df[ID_COL] = df[ID_COL].astype(str).str.strip().str.upper()
        test = test.merge(labels[[ID_COL, Y_COL]], on=ID_COL, how="left").dropna(subset=[Y_COL])
        X_test = test[[c for c in test.columns if c not in [ID_COL, Y_COL]]]
        y_test = test[Y_COL].astype(str)

        y_pred_test = best_est.predict(X_test)
        time_pred_test = time.perf_counter() - t_test0
        metrics_test = _core_metrics(y_test, y_pred_test)
        metrics_test.update({"time_total_min": _fmt_min(time_pred_test)})

        print("Test (One-time) :")
        print(f"- Accuracy          : {metrics_test['accuracy']}")
        print(f"- F1 Macro          : {metrics_test['f1_macro']}")
        print(f"- Balanced Accuracy : {metrics_test['balanced_accuracy']}")

    final_total_sec = time.perf_counter() - t_all0
    final_total_min = _fmt_min(final_total_sec)

    metrics_val["time_total_min"] = final_total_min
    metrics_val["time_total_sec"] = round(final_total_sec, 3)

    print(f"total time: {final_total_min}")

    summary_rows = []
    summary_rows.append({
        "set": "val",
        "accuracy": metrics_val["accuracy"],
        "f1_macro": metrics_val["f1_macro"],
        "balanced_accuracy": metrics_val["balanced_accuracy"],
        "cv_best": metrics_val["cv_best"],
    })
    if metrics_test is not None:
        summary_rows.append({
            "set": "test",
            "accuracy": metrics_test["accuracy"],
            "f1_macro": metrics_test["f1_macro"],
            "balanced_accuracy": metrics_test["balanced_accuracy"],
            "cv_best": np.nan,
        })
    print("Metrics Summary")
    summary_df = pd.DataFrame(summary_rows)
    with pd.option_context('display.float_format', '{:.6f}'.format):
        print(summary_df.to_string(index=False))
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    pd.DataFrame(metrics_val, index=[0]).to_csv(out_dir / "metrics_val.csv", index=False)
    pd.DataFrame(cm, index=ALL_CLASSES, columns=ALL_CLASSES).to_csv(out_dir / "val_confusion_matrix.csv")
    rep = classification_report(y_val, y_pred, labels=ALL_CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame(rep).to_csv(out_dir / "val_classification_report.csv")

    dump(best_est, out_dir / "model.joblib")

    model_package = {
        "model": best_est,
        "meta": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn_version,
            "seed": SEED,
            "task_name": task_name,
            "classes": ALL_CLASSES,
            "scoring": SCORING,
            "top_k": TOP_K,
            "n_estimators_grid": list(n_est_grid),
            "use_precomputed_dm": bool(USE_PRECOMPUTED_DM),
        },
        "info_lines": {
            "shape_line": f"[INFO] Train/val shape: {X_train.shape} / {X_val.shape}; classes={len(cls_counts)}, min_count={min_count}",
            "best_params_line": best_params_line,
        },
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "confusion_matrix_val": cm,
    }
    dump(model_package, out_dir / "model_package.joblib")


if __name__ == "__main__":
    out_root = Path("results") / "result-training"
    out_root.mkdir(parents=True, exist_ok=True)

    for t in TASKS:
        train_one_task(
            task_name=t["name"],
            csv_dir=t["csv_dir"],
            out_root=out_root,
            top_k=TOP_K,
            n_est_grid=N_EST_GRID,
            scoring=SCORING,
            seed=SEED
        )

    print("\n>>> All tasks done.")
