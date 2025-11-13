"""
Enhanced label generation pipeline with concurrency, retries, and metrics logging.

This script reads train/val/test split CSV files and a disease tree to
generate `labels.csv`. It satisfies acceptance criteria by providing:

1. Parallel processing with configurable worker count (>=2).
2. Runtime comparison between sequential baseline (sum of per-task durations)
   and actual parallel wall-clock time.
3. Retry logic with structured JSONL logs for failures.

Usage examples:

```
python label_generation_pipeline.py \
  --tree_path data/freeze0525/diseaseTree_mapped.joblib \
  --split_dir splits_concat \
  --output labels.csv

python label_generation_pipeline.py \
  --tree_path data/freeze0525/diseaseTree_mapped.joblib \
  --split_dir splits_concat \
  --concurrency 4 \
  --retries 3
```
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import joblib
import pandas as pd

# Compatibility shim for loading legacy diseaseTree pickle files
@dataclass
class DiseaseTree:
    name: str
    children: list
    samples: list
    training_samples: list = field(default_factory=list)
    validation_samples: list = field(default_factory=list)
    selected_features: list = field(default_factory=list)

    def is_leaf(self):
        return len(self.children) == 0

sys.modules['mch'] = type(sys)('mch')
sys.modules['mch.core'] = type(sys)('mch.core')
sys.modules['mch.core.disease_tree'] = sys.modules[__name__]


# ---------------------------------------------------------------------------
# Helper functions for multiprocessing
# ---------------------------------------------------------------------------


def _process_split_task(
    split_name: str,
    csv_path: str,
    sample_to_label: Dict[str, str],
    include_healthy: bool,
    encoding: str = "utf-8",
) -> Dict:
    """Process a single split CSV file.

    Returns a dictionary containing records, stats, and duration.
    """

    start = time.perf_counter()
    records: List[Dict[str, str]] = []
    total = disease = healthy = 0

    with open(csv_path, "r", encoding=encoding, errors="ignore") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"Empty CSV file: {csv_path}")
        for line in handle:
            line = line.strip()
            if not line:
                continue
            biosample_id = line.split(",", 1)[0].strip()
            total += 1
            if biosample_id in sample_to_label:
                label = sample_to_label[biosample_id]
                records.append(
                    {
                        "biosample_id": biosample_id,
                        "label": label,
                        "disease_status": "disease",
                        "split": split_name,
                    }
                )
                disease += 1
            elif include_healthy:
                records.append(
                    {
                        "biosample_id": biosample_id,
                        "label": "Healthy",
                        "disease_status": "healthy",
                        "split": split_name,
                    }
                )
                healthy += 1

    duration = time.perf_counter() - start
    stats = {
        "split": split_name,
        "total": total,
        "disease": disease,
        "healthy": healthy,
        "matched_pct": (disease / total * 100.0) if total else 0.0,
    }

    return {
        "split": split_name,
        "records": records,
        "stats": stats,
        "duration": duration,
    }


def _process_split_with_retry(args: Tuple) -> Dict:
    """Wrapper that retries `_process_split_task` based on provided settings."""

    (
        split_name,
        csv_path,
        sample_to_label,
        include_healthy,
        retries,
        backoff,
        encoding,
    ) = args

    attempt = 0
    while True:
        try:
            return _process_split_task(
                split_name,
                csv_path,
                sample_to_label,
                include_healthy,
                encoding=encoding,
            )
        except Exception as exc:  # pragma: no cover - pure error handling
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(backoff * attempt)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def build_sample_to_label_mapping(tree) -> Dict[str, str]:
    """Traverse disease tree and create sample -> label mapping."""

    mapping: Dict[str, str] = {}

    def traverse(node) -> None:
        if getattr(node, "samples", None):
            label = getattr(node, "name", "")
            for sample_id in node.samples:
                if sample_id not in mapping:
                    mapping[sample_id] = label
        for child in getattr(node, "children", []) or []:
            traverse(child)

    traverse(tree)
    return mapping


@dataclass
class LabelGenerationPipeline:
    tree_path: str
    split_dir: str
    output_path: Optional[str] = None
    include_healthy: bool = True
    concurrency: int = 2
    retries: int = 2
    backoff: float = 0.5
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.concurrency < 2:
            raise ValueError("concurrency must be >= 2 to satisfy AC requirements")

        self.split_dir_path = Path(self.split_dir)
        if not self.split_dir_path.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        if self.output_path:
            self.output_path_path = Path(self.output_path)
        else:
            self.output_path_path = self.split_dir_path / (
                "labels_with_healthy.csv" if self.include_healthy else "labels.csv"
            )

        self.output_path_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_path = self.output_path_path.parent / "labels_logs.jsonl"
        self.metrics_path = self.output_path_path.parent / "labels_metrics.json"

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------

    def log_event(self, event: str, payload: Optional[Dict] = None) -> None:
        record = {
            "event": event,
            "ts": time.time(),
        }
        if payload:
            record.update(payload)
        with open(self.logs_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{event}] {json.dumps(payload or {}, ensure_ascii=False)}")

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        self.log_event(
            "pipeline_start",
            {
                "tree_path": self.tree_path,
                "split_dir": str(self.split_dir_path),
                "output_path": str(self.output_path_path),
                "include_healthy": self.include_healthy,
                "concurrency": self.concurrency,
                "retries": self.retries,
                "backoff": self.backoff,
            },
        )

        # Load disease tree
        self.log_event("loading_tree")
        tree = joblib.load(self.tree_path)
        sample_to_label = build_sample_to_label_mapping(tree)
        self.log_event("tree_loaded", {"mapped_samples": len(sample_to_label)})

        # Prepare tasks
        tasks = []
        for split_name in ("train", "val", "test"):
            csv_path = self.split_dir_path / f"{split_name}.csv"
            if csv_path.exists():
                tasks.append((split_name, str(csv_path)))
            else:
                self.log_event("split_missing", {"split": split_name, "path": str(csv_path)})

        if not tasks:
            raise FileNotFoundError("No split CSV files found in the provided directory")

        # Execute tasks concurrently with retries
        args_list = [
            (
                split_name,
                csv_path,
                sample_to_label,
                self.include_healthy,
                self.retries,
                self.backoff,
                self.encoding,
            )
            for split_name, csv_path in tasks
        ]

        sequential_time = 0.0
        all_records: List[Dict[str, str]] = []
        split_stats: Dict[str, Dict] = {}

        t_parallel_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.concurrency) as pool:
            future_to_split = {
                pool.submit(_process_split_with_retry, args): args[0]
                for args in args_list
            }

            for future in as_completed(future_to_split):
                split_name = future_to_split[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self.log_event(
                        "split_failed",
                        {"split": split_name, "error": str(exc)},
                    )
                    raise

                self.log_event(
                    "split_processed",
                    {
                        "split": split_name,
                        "duration_sec": result["duration"],
                        "stats": result["stats"],
                    },
                )

                sequential_time += result["duration"]
                all_records.extend(result["records"])
                split_stats[split_name] = result["stats"]

        parallel_time = time.perf_counter() - t_parallel_start

        metrics = {
            "sequential_time_sec": sequential_time,
            "parallel_time_sec": parallel_time,
            "speedup": (sequential_time / parallel_time) if parallel_time else float("inf"),
            "concurrency": self.concurrency,
            "retries": self.retries,
            "backoff_sec": self.backoff,
        }

        with open(self.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        self.log_event("metrics_recorded", metrics)

        if not all_records:
            raise RuntimeError("No labels generated. Check if splits contain biosample IDs present in the tree.")

        labels_df = pd.DataFrame(all_records)
        labels_df.sort_values(by=["split", "biosample_id"], inplace=True)
        labels_df.to_csv(self.output_path_path, index=False)

        summary = {
            "output_path": str(self.output_path_path),
            "total_rows": len(labels_df),
            "unique_labels": int(labels_df["label"].nunique()),
            "include_healthy": self.include_healthy,
            "split_stats": split_stats,
        }
        self.log_event("labels_generated", summary)

        return {
            "metrics": metrics,
            "summary": summary,
            "logs_path": str(self.logs_path),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate labels from disease tree and split CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tree_path", required=True, help="Path to diseaseTree_mapped.joblib")
    parser.add_argument("--split_dir", required=True, help="Directory containing split CSV files")
    parser.add_argument("--output", default=None, help="Output CSV path (default: within split_dir)")
    parser.add_argument("--no_healthy", action="store_true", help="Exclude healthy labels for unmatched samples")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent workers (>=2)")
    parser.add_argument("--retries", type=int, default=2, help="Retries per split on failure")
    parser.add_argument("--backoff", type=float, default=0.5, help="Retry backoff in seconds")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding")

    args = parser.parse_args()

    pipeline = LabelGenerationPipeline(
        tree_path=args.tree_path,
        split_dir=args.split_dir,
        output_path=args.output,
        include_healthy=not args.no_healthy,
        concurrency=args.concurrency,
        retries=args.retries,
        backoff=args.backoff,
        encoding=args.encoding,
    )

    result = pipeline.run()
    print("\nSummary:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


