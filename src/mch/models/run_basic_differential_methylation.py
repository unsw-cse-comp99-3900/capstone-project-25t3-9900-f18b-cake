#!/usr/bin/env python3
"""
Python differential methylation feature selection script.
- Input:
- labelswithhealthy.csv (required, use biosampleid and label)
- Methylation M value matrix (optional): CSV (wide table, the first column sampleid or index is the sample), or Parquet
If no real matrix is provided, synthetic features will be generated based on the labels to demonstrate the process.
- process:
1) Do a t-test (Welch t-test) for each category vs. other categories
2) Calculate effect size (Cohen's d)
3) Multiple comparison correction (Benjamini-Hochberg FDR)
4) Select significant probes and save them
- Output:
- results/dmstats.csv: Statistics of each probe (p, q, effectsize, bestclass)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from scipy.stats import ttest_ind

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    required = {"biosample_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels Required column is missing:{missing}")
    return df[["biosample_id", "label"]].copy()


def load_mvalue_matrix(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"M value file not found: {p}, synthetic data will be generated for demonstration")
        return None

    try:
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)
    except Exception as e:
        print(f"Failed to read M value: {e}, synthetic data will be generated for demonstration")
        return None

    # Compatible with the case where the first column is sample_id or the index is sample
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    else:
        first_col = df.columns[0]
        if df[first_col].astype(str).str.contains("[_-]|^[A-Za-z]").mean() > 0.5:
            df = df.set_index(first_col)

    return df


def synthesize_mvalues(labels_df: pd.DataFrame, n_features: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    samples = labels_df["biosample_id"].astype(str).tolist()
    classes = labels_df["label"].astype(str).unique().tolist()

    X = np.random.randn(len(samples), n_features)
    # Inject a small amount of effects into each category
    for i, cls in enumerate(classes):
        idx = labels_df[labels_df["label"] == cls].index
        start = (i * 40) % max(40, n_features)
        end = min(start + 40, n_features)
        X[idx, start:end] += np.random.randn(len(idx), end - start) * 0.6

    df = pd.DataFrame(X, index=samples, columns=[f"cg_{i:06d}" for i in range(n_features)])
    return df


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Correction, returns an array of q values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked_p = p[order]
    # calculate q = p * n / rank
    ranks = np.arange(1, n + 1)
    q = ranked_p * n / ranks
    # Guaranteed to be monotonic and non-increasing: accumulate the minimum value from the end
    q = np.minimum.accumulate(q[::-1])[::-1]
    # Return to original order
    qvals = np.empty_like(q)
    qvals[order] = np.minimum(q, 1.0)
    return qvals


def differential_methylation(X: pd.DataFrame, labels_df: pd.DataFrame,
                              alpha: float = 0.05, min_effect: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Welch's t test and effect size calculation for each category vs. other categories to select significant probes.
return(stats_df, selected_probes_df)
    """
    # Align samples
    common = X.index.intersection(labels_df["biosample_id"].astype(str))
    X = X.loc[common]
    y = labels_df.set_index("biosample_id").loc[common, "label"].astype(str)

    features = X.columns
    classes = y.unique()

    rows = []
    for feat in features:
        values = X[feat]
        for cls in classes:
            a = values[y == cls].values
            b = values[y != cls].values
            if len(a) < 3 or len(b) < 3:
                continue
            # Welch t-test
            t_stat, p_val = ttest_ind(a, b, equal_var=False, nan_policy='omit')
            # Cohen's d
            mean_diff = np.nanmean(a) - np.nanmean(b)
            s1 = np.nanvar(a, ddof=1)
            s2 = np.nanvar(b, ddof=1)
            # Pooled standard deviation (approximate in Welch case)
            s_pooled = np.sqrt((s1 + s2) / 2.0)
            d = 0.0 if s_pooled == 0 else mean_diff / s_pooled
            rows.append((feat, cls, p_val, d))

    stats_df = pd.DataFrame(rows, columns=["feature", "class", "p_value", "effect_size"]) if rows else pd.DataFrame(columns=["feature","class","p_value","effect_size"])

    if stats_df.empty:
        return stats_df, pd.DataFrame(columns=["feature"])

    # Perform multiple corrections separately by feature
    selected_blocks = []
    for feat, block in stats_df.groupby("feature"):
        pvals = block["p_value"].values
        qvals = fdr_bh(pvals)
        block = block.copy()
        block["q_value"] = qvals
        # Record the most significant category of this feature
        idx_min = block["q_value"].idxmin()
        best = block.loc[idx_min]
        selected_blocks.append({
            "feature": feat,
            "best_class": best["class"],
            "best_p": best["p_value"],
            "best_q": best["q_value"],
            "best_effect_size": best["effect_size"]
        })

    summary = pd.DataFrame(selected_blocks)

    # Selected features: q is less than the threshold and the absolute value of the effect size reaches the threshold
    selected = summary[(summary["best_q"] <= alpha) & (summary["best_effect_size"].abs() >= min_effect)]
    selected = selected.sort_values(["best_q", "best_effect_size"], ascending=[True, False])

    return summary, selected[["feature", "best_class", "best_q", "best_effect_size"]]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Python differential methylation feature selection")
    parser.add_argument("--mvalue", type=str, default="",
                        help="M-value matrix file path (CSV/Parquet). If not provided, synthetic data will be generated")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR threshold")
    parser.add_argument("--min_effect", type=float, default=0.3, help="minimum effect size threshold (Cohen d |d|)")
    args = parser.parse_args()

    labels = load_labels("labels_with_healthy.csv")
    X = load_mvalue_matrix(args.mvalue)
    if X is None:
        X = synthesize_mvalues(labels, n_features=1000)

    stats_df, selected_df = differential_methylation(X, labels, alpha=args.alpha, min_effect=args.min_effect)

    stats_path = RESULTS_DIR / "dm_stats.csv"
    selected_path = RESULTS_DIR / "dm_selected_probes.csv"
    stats_df.to_csv(stats_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    print(f"Statistics saved:{stats_path}")
    print(f"Selected features saved:{selected_path}")
    print(f"Total number of probes evaluated:{stats_df['feature'].nunique() if not stats_df.empty else 0}")
    print(f"Number of probes passing the threshold:{len(selected_df)}")


if __name__ == "__main__":
    main()
