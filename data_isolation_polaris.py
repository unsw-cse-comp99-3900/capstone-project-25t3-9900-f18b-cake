import os, json, time, random, sys, traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------- Basic: Read IDs and Split ---------

def split_ids(ids: List[str], ratios=(0.7, 0.15, 0.15), seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Reproducibly split into three non-overlapping sets (train/val/test) by ratio."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    ids = list(ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(n * ratios[0]))
    n_val   = max(0, int(n * ratios[1]))
    used = n_train + n_val
    n_test  = max(0, n - used)
    if n_train + n_val + n_test != n:
        n_test = n - (n_train + n_val)
    return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]

def read_ids_only(csv_path: str, id_col_name="biosample_id") -> List[str]:
    """Read only the first column IDs to avoid loading the entire wide matrix into memory."""
    ids = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
        if header[0] != id_col_name:
            raise ValueError(f"Expected first column '{id_col_name}', got '{header[0]}'")
        for line in f:
            if not line.strip():
                continue
            ids.append(line.split(",", 1)[0])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs detected in input CSV first column.")
    if len(ids) == 0:
        raise ValueError("No IDs found in input CSV.")
    return ids

def stream_write_splits(csv_path: str, out_dir: Path,
                        train_ids: set, val_ids: set, test_ids: set,
                        id_col_name="biosample_id") -> Dict[str, str]:
    """Stream through input CSV once and write three subset CSVs (same header), memory efficient."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_dir / "train.csv"
    out_val   = out_dir / "val.csv"
    out_test  = out_dir / "test.csv"
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_train, "w", encoding="utf-8", newline="") as ft, \
         open(out_val,   "w", encoding="utf-8", newline="") as fv, \
         open(out_test,  "w", encoding="utf-8", newline="") as fs:
        header = fin.readline()
        for h in (ft, fv, fs):
            h.write(header)
        for line in fin:
            if not line.strip():
                continue
            sid = line.split(",", 1)[0]
            if sid in train_ids:
                ft.write(line)
            elif sid in val_ids:
                fv.write(line)
            elif sid in test_ids:
                fs.write(line)
    return {"train": str(out_train), "val": str(out_val), "test": str(out_test)}

# --------- AC: Retry / Logging / Parallel vs Sequential Time Comparison ---------

def _retry(call, *, retries=2, backoff=0.5, **kw):
    import time as _t
    k = 0
    while True:
        try:
            return call(**kw)
        except Exception as e:
            k += 1
            if k > retries:
                raise
            _t.sleep(backoff * k)

def _log_json(log_path: Path, record: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _process_split_csv(split_path: str, skip_stats: bool = False) -> Dict[str, float]:
    """Process CSV and optionally compute statistics."""
    if skip_stats:
        with open(split_path, 'r') as f:
            n_lines = sum(1 for _ in f) - 1
        return {"n_rows": n_lines, "skipped": True}
    
    df = pd.read_csv(split_path)
    num = df.select_dtypes(include=["number"])
    stats = {}
    if not num.empty:
        desc = num.describe().to_dict()
        means = [v.get("mean") for v in desc.values() if "mean" in v]
        stds  = [v.get("std")  for v in desc.values() if "std"  in v]
        means = [m for m in means if m is not None]
        stds  = [s for s in stds  if s is not None]
        stats = {
            "n_numeric_cols": int(num.shape[1]),
            "mean_of_means": float(sum(means)/len(means)) if means else float("nan"),
            "mean_of_stds": float(sum(stds)/len(stds))   if stds  else float("nan"),
        }
    return stats

def isolate_polaris(
    input_csv: str,
    output_dir: str = "./splits",
    ratios=(0.7, 0.15, 0.15),
    seed: int = 42,
    id_col_name: str = "biosample_id",
    write_subset_csv: bool = True,
    # -- AC-related parameters --
    concurrency: int = 2,     # Number of concurrent workers (>=2)
    retries: int = 2,         # Number of retries on task failure
    backoff: float = 0.5,     # Retry backoff base (seconds)
    skip_stats: bool = False  # Skip statistics computation (faster)
) -> Dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_path = out_dir / "logs.jsonl"
    metrics_path = out_dir / "metrics.json"

    # 1) Read IDs and split (minimum leak prevention: non-overlapping + fixed random seed)
    ids = read_ids_only(input_csv, id_col_name=id_col_name)
    train_ids, val_ids, test_ids = split_ids(ids, ratios=ratios, seed=seed)

    # 2) Write manifest (traceable)
    manifest = {
        "input_csv": input_csv,
        "ratios": ratios,
        "seed": seed,
        "id_col": id_col_name,
        "counts": {"total": len(ids), "train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "ids": {"train": train_ids, "val": val_ids, "test": test_ids}
    }
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _log_json(logs_path, {"event":"manifest_written","ts":time.time(),"counts":manifest["counts"]})

    # 3) Optional: Write three subset CSVs (streaming, memory efficient)
    csv_paths = None
    if write_subset_csv:
        csv_paths = stream_write_splits(input_csv, out_dir, set(train_ids), set(val_ids), set(test_ids), id_col_name=id_col_name)
        _log_json(logs_path, {"event":"subset_csv_written","ts":time.time(),"paths":csv_paths})

    # 4) AC #1 & #2: Sequential vs Parallel downstream task comparison + timing logs
    tasks = [("train", csv_paths["train"]), ("val", csv_paths["val"]), ("test", csv_paths["test"])] if csv_paths else []
    def run_one(name, path):
        _log_json(logs_path, {"event":"process_start","split":name,"ts":time.time()})
        t0 = time.perf_counter()
        res = _retry(_process_split_csv, retries=retries, backoff=backoff, split_path=path, skip_stats=skip_stats)
        dt = time.perf_counter() - t0
        _log_json(logs_path, {"event":"process_done","split":name,"duration_sec":dt,"ts":time.time(),"stats":res})
        return name, dt, res

    # Sequential
    t_seq0 = time.perf_counter()
    seq_results = [run_one(n, p) for (n,p) in tasks]
    t_seq = time.perf_counter() - t_seq0

    # Parallel (>=2 workers)
    t_par0 = time.perf_counter()
    par_results = []
    with ProcessPoolExecutor(max_workers=max(2, int(concurrency))) as ex:
        futs = {ex.submit(_process_split_csv, p, skip_stats): (n,p) for (n,p) in tasks}
        for fut in as_completed(futs):
            n, p = futs[fut]
            try:
                # Add another retry wrapper to meet "retry on failure + error logging"
                t1 = time.perf_counter()
                res = _retry(lambda split_path, skip: _process_split_csv(split_path, skip), retries=retries, backoff=backoff, split_path=p, skip=skip_stats)
                dt = time.perf_counter() - t1
                _log_json(logs_path, {"event":"process_done_parallel","split":n,"duration_sec":dt,"ts":time.time(),"stats":res})
                par_results.append((n, dt, res))
            except Exception as e:
                _log_json(logs_path, {"event":"process_error","split":n,"error":str(e),"ts":time.time(),
                                      "traceback": traceback.format_exc()})
    t_par = time.perf_counter() - t_par0

    # Record metrics (including sequential/parallel duration and speedup)
    metrics = {
        "sequential_time_sec": t_seq,
        "parallel_time_sec": t_par,
        "speedup": (t_seq / t_par) if t_par > 0 else float("inf"),
        "concurrency": max(2, int(concurrency)),
        "retries": int(retries),
        "backoff_sec": float(backoff)
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    _log_json(logs_path, {"event":"summary","metrics":metrics,"ts":time.time()})

    return {"manifest": manifest, "metrics": metrics, "logs_path": str(logs_path)}

# --------- CLI ---------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Path to very-wide CSV (first col is biosample_id).")
    ap.add_argument("--output_dir", default="./splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=str, default="0.7,0.15,0.15", help="train,val,test ratios")
    ap.add_argument("--id_col", default="biosample_id")
    ap.add_argument("--no_write_csv", action="store_true", help="Only write manifest, skip train/val/test CSV.")
    # -- AC: concurrency, retry, backoff
    ap.add_argument("--concurrency", type=int, default=2, help=">=2 to satisfy AC for concurrent nodes.")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=0.5)
    ap.add_argument("--skip_stats", action="store_true", help="Skip statistics computation for faster processing.")

    args = ap.parse_args()
    ratios = tuple(float(x) for x in args.ratios.split(","))
    result = isolate_polaris(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        ratios=ratios,
        seed=args.seed,
        id_col_name=args.id_col,
        write_subset_csv=not args.no_write_csv,
        concurrency=args.concurrency,
        retries=args.retries,
        backoff=args.backoff,
        skip_stats=args.skip_stats
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
