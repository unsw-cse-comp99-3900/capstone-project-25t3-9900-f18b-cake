import os
import sys
import json
import time
import random
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
import pandas as pd
import methylcheck


class CompleteMethylationPipeline:
    """
    Complete methylation data processing pipeline with:
    1. Data filtering and quality control
    2. Feature selection
    3. Train/val/test splitting
    4. Parallel processing
    5. Comprehensive logging
    6. Memory-efficient processing for large files (20GB+)
    
    Satisfies AC requirements:
    - 2+ concurrent nodes
    - Runtime comparison logging
    - Retry mechanism and error logging
    """
    
    def __init__(
        self,
        input_csv: str,
        output_dir: str = "./pipeline_output",
        probe_metadata_file: Optional[str] = None,
        seed: int = 42,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        concurrency: int = 2,
        retries: int = 2,
        backoff: float = 0.5,
        enable_filtering: bool = True,
        skip_stats: bool = True,
        use_streaming: bool = True,
        chunk_size: int = 10000
    ):
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.probe_metadata_file = probe_metadata_file
        self.seed = seed
        self.ratios = ratios
        self.concurrency = max(2, concurrency)
        self.retries = retries
        self.backoff = backoff
        self.enable_filtering = enable_filtering
        self.skip_stats = skip_stats
        self.use_streaming = use_streaming
        self.chunk_size = chunk_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_path = self.output_dir / "logs.jsonl"
        self.metrics_path = self.output_dir / "metrics.json"
        
        self.log_event("pipeline_init", {
            "input_csv": input_csv,
            "output_dir": str(output_dir),
            "enable_filtering": enable_filtering,
            "seed": seed,
            "ratios": ratios,
            "concurrency": concurrency
        })
    
    def log_event(self, event: str, data: dict = None):
        """Log events to JSONL file."""
        record = {
            "event": event,
            "ts": time.time(),
            **(data or {})
        }
        with open(self.logs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{event}] {json.dumps(data or {}, ensure_ascii=False)}")
    
    def retry_wrapper(self, func, *args, **kwargs):
        """Retry mechanism for operations."""
        for attempt in range(self.retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt >= self.retries:
                    self.log_event("operation_failed", {
                        "function": func.__name__,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    raise
                time.sleep(self.backoff * (attempt + 1))
    
    # ========== STEP 1: DATA FILTERING ==========
    
    def filter_data_streaming(self, input_path: str, output_path: str) -> str:
        """Apply filtering using streaming/lazy evaluation for large files."""
        if not self.enable_filtering:
            self.log_event("filtering_skipped")
            return input_path
        
        self.log_event("filtering_start_streaming", {"input": input_path})
        
        try:
            # Use Polars lazy API for memory-efficient processing
            df_lazy = pl.scan_csv(input_path)
            
            # Get initial count
            initial_count = df_lazy.select(pl.count()).collect().item()
            initial_cols = len(df_lazy.collect_schema())
            
            self.log_event("initial_data_size", {
                "rows": initial_count,
                "cols": initial_cols
            })
            
            # Load probe metadata if provided
            if self.probe_metadata_file and os.path.exists(self.probe_metadata_file):
                probe_metadata = pl.read_csv(self.probe_metadata_file)
                
                # Gene-associated probes
                gene_probes = probe_metadata.filter(
                    pl.col("gencodebasic_name").is_not_null()
                ).select("probe_id").to_series().to_list()
                
                # XYM probes
                xym_probes = probe_metadata.filter(
                    pl.col("chr_hg38").is_in(["chrX", "chrY", "chrM"])
                ).select("probe_id").to_series().to_list()
                
                # Get ID column name
                all_cols = df_lazy.collect_schema().names()
                id_col = "biosample_id" if "biosample_id" in all_cols else "sampleId"
                
                # Build list of columns to keep
                available_gene_probes = [p for p in gene_probes if p in all_cols]
                columns_to_keep = [id_col] + [p for p in available_gene_probes if p not in xym_probes]
                
                # Apply column selection using lazy evaluation
                df_lazy = df_lazy.select(columns_to_keep)
                
                self.log_event("probe_filtering_applied", {
                    "gene_probes_kept": len(available_gene_probes),
                    "xym_probes_removed": len([p for p in xym_probes if p in all_cols])
                })
            
            # Apply methylcheck filters using lazy evaluation
            # Note: Problem sample filtering requires row-level computation
            # For very large files, we'll apply only column-based filters in streaming mode
            
            # SNP-adjacent probes
            try:
                sketchy_probes_snp = set(methylcheck.list_problem_probes('epic', ["Polymorphism"]))
                current_cols = df_lazy.collect_schema().names()
                cols_after_snp = [c for c in current_cols if c not in sketchy_probes_snp]
                df_lazy = df_lazy.select(cols_after_snp)
                self.log_event("snp_filter_applied", {"removed": len(sketchy_probes_snp)})
            except Exception as e:
                self.log_event("snp_filter_skipped", {"error": str(e)})
            
            # Cross-hybridizing probes
            try:
                sketchy_probes_cross = set(methylcheck.list_problem_probes('epic', ["CrossHybridization"]))
                current_cols = df_lazy.collect_schema().names()
                cols_after_cross = [c for c in current_cols if c not in sketchy_probes_cross]
                df_lazy = df_lazy.select(cols_after_cross)
                self.log_event("cross_hybrid_filter_applied", {"removed": len(sketchy_probes_cross)})
            except Exception as e:
                self.log_event("cross_hybrid_filter_skipped", {"error": str(e)})
            
            # Write to output using streaming (sink)
            df_lazy.sink_csv(output_path)
            
            # Get final stats
            final_df_lazy = pl.scan_csv(output_path)
            final_count = final_df_lazy.select(pl.count()).collect().item()
            final_cols = len(final_df_lazy.collect_schema())
            
            self.log_event("filtering_complete_streaming", {
                "output": output_path,
                "final_rows": final_count,
                "final_cols": final_cols,
                "rows_removed": initial_count - final_count,
                "cols_removed": initial_cols - final_cols
            })
            
            return output_path
            
        except Exception as e:
            self.log_event("filtering_error_streaming", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            # Fallback to original file
            return input_path
    
    def filter_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all filtering steps to the data (in-memory version)."""
        if not self.enable_filtering:
            self.log_event("filtering_skipped")
            return df
        
        self.log_event("filtering_start", {"initial_shape": df.shape})
        initial_shape = df.shape
        
        # Load probe metadata if provided
        if self.probe_metadata_file and os.path.exists(self.probe_metadata_file):
            probe_metadata = pl.read_csv(self.probe_metadata_file)
            
            # Gene-associated probes
            gene_probes = probe_metadata.filter(
                pl.col("gencodebasic_name").is_not_null()
            ).select("probe_id").to_series().to_list()
            
            # XYM probes
            xym_probes = probe_metadata.filter(
                pl.col("chr_hg38").is_in(["chrX", "chrY", "chrM"])
            ).select("probe_id").to_series().to_list()
            
            # Filter to gene-associated probes
            id_col = "biosample_id" if "biosample_id" in df.columns else "sampleId"
            available_gene_probes = [p for p in gene_probes if p in df.columns]
            df = df.select([id_col] + available_gene_probes)
            self.log_event("gene_probe_filter", {
                "kept": len(available_gene_probes),
                "shape": df.shape
            })
            
            # Remove XYM probes
            available_xym = [p for p in xym_probes if p in df.columns]
            cols_to_keep = [c for c in df.columns if c not in available_xym]
            df = df.select(cols_to_keep)
            self.log_event("xym_probe_filter", {
                "removed": len(available_xym),
                "shape": df.shape
            })
        
        # Filter problem samples
        df = self._filter_problem_samples(df)
        
        # Filter SNP-adjacent probes
        df = self._filter_snp_adjacent(df)
        
        # Filter cross-hybridizing probes
        df = self._filter_cross_hybrid(df)
        
        self.log_event("filtering_complete", {
            "initial_shape": initial_shape,
            "final_shape": df.shape,
            "samples_removed": initial_shape[0] - df.shape[0],
            "probes_removed": initial_shape[1] - df.shape[1]
        })
        
        return df
    
    def _filter_problem_samples(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove samples with too many missing values."""
        initial_shape = df.shape
        na_counts = df.select([pl.sum_horizontal(pl.all().is_null()).alias("na_count")])
        samples_to_keep = df.with_row_index("idx").filter(
            na_counts.select("na_count").to_series() <= 50000
        ).select("idx").to_series().to_list()
        filtered_df = df.filter(pl.arange(0, df.height).is_in(samples_to_keep))
        
        self.log_event("problem_sample_filter", {
            "removed": initial_shape[0] - filtered_df.shape[0],
            "shape": filtered_df.shape
        })
        return filtered_df
    
    def _filter_snp_adjacent(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove probes adjacent to SNPs."""
        initial_shape = df.shape
        sketchy_probes = methylcheck.list_problem_probes('epic', ["Polymorphism"])
        good_probes = [col for col in df.columns if col not in sketchy_probes]
        filtered_df = df.select(good_probes)
        
        self.log_event("snp_adjacent_filter", {
            "removed": len(sketchy_probes),
            "shape": filtered_df.shape
        })
        return filtered_df
    
    def _filter_cross_hybrid(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove cross-hybridizing probes."""
        initial_shape = df.shape
        sketchy_probes = methylcheck.list_problem_probes('epic', ["CrossHybridization"])
        good_probes = [col for col in df.columns if col not in sketchy_probes]
        filtered_df = df.select(good_probes)
        
        self.log_event("cross_hybrid_filter", {
            "removed": len(sketchy_probes),
            "shape": filtered_df.shape
        })
        return filtered_df
    
    # ========== STEP 2: DATA SPLITTING ==========
    
    def split_ids(self, ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Split IDs into train/val/test sets."""
        assert abs(sum(self.ratios) - 1.0) < 1e-6, "ratios must sum to 1"
        ids = list(ids)
        rnd = random.Random(self.seed)
        rnd.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(n * self.ratios[0]))
        n_val = max(0, int(n * self.ratios[1]))
        used = n_train + n_val
        n_test = max(0, n - used)
        if n_train + n_val + n_test != n:
            n_test = n - (n_train + n_val)
        return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]
    
    def read_ids_only(self, csv_path: str, id_col: str = "biosample_id") -> List[str]:
        """Read only first column IDs from CSV."""
        ids = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split(",")
            if header[0] != id_col:
                raise ValueError(f"Expected first column '{id_col}', got '{header[0]}'")
            for line in f:
                if not line.strip():
                    continue
                ids.append(line.split(",", 1)[0])
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs detected")
        return ids
    
    def stream_write_splits(
        self,
        csv_path: str,
        train_ids: set,
        val_ids: set,
        test_ids: set,
        id_col: str = "biosample_id"
    ) -> Dict[str, str]:
        """Stream write train/val/test CSV files."""
        out_train = self.output_dir / "train.csv"
        out_val = self.output_dir / "val.csv"
        out_test = self.output_dir / "test.csv"
        
        self.log_event("split_write_start", {"csv_path": csv_path})
        start_time = time.time()
        
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as fin, \
             open(out_train, "w", encoding="utf-8", newline="") as ft, \
             open(out_val, "w", encoding="utf-8", newline="") as fv, \
             open(out_test, "w", encoding="utf-8", newline="") as fs:
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
        
        elapsed = time.time() - start_time
        self.log_event("split_write_complete", {"duration_sec": elapsed})
        
        return {
            "train": str(out_train),
            "val": str(out_val),
            "test": str(out_test)
        }
    
    # ========== STEP 3: PARALLEL PROCESSING ==========
    
    def _process_split_csv(self, split_path: str, skip_stats: bool = False) -> Dict:
        """Process a split CSV file (for parallel benchmark)."""
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
            stds = [v.get("std") for v in desc.values() if "std" in v]
            means = [m for m in means if m is not None]
            stds = [s for s in stds if s is not None]
            stats = {
                "n_numeric_cols": int(num.shape[1]),
                "mean_of_means": float(sum(means)/len(means)) if means else float("nan"),
                "mean_of_stds": float(sum(stds)/len(stds)) if stds else float("nan"),
            }
        return stats
    
    def parallel_benchmark(self, csv_paths: Dict[str, str]):
        """Benchmark sequential vs parallel processing."""
        tasks = [(name, path) for name, path in csv_paths.items()]
        
        def run_one(name, path):
            self.log_event("process_start", {"split": name})
            t0 = time.perf_counter()
            res = self.retry_wrapper(self._process_split_csv, path, self.skip_stats)
            dt = time.perf_counter() - t0
            self.log_event("process_done", {
                "split": name,
                "duration_sec": dt,
                "stats": res
            })
            return name, dt, res
        
        # Sequential
        t_seq0 = time.perf_counter()
        seq_results = [run_one(n, p) for (n, p) in tasks]
        t_seq = time.perf_counter() - t_seq0
        
        # Parallel
        t_par0 = time.perf_counter()
        par_results = []
        with ProcessPoolExecutor(max_workers=self.concurrency) as ex:
            futs = {ex.submit(self._process_split_csv, p, self.skip_stats): (n, p) for (n, p) in tasks}
            for fut in as_completed(futs):
                n, p = futs[fut]
                try:
                    t1 = time.perf_counter()
                    res = self.retry_wrapper(lambda sp, sk: self._process_split_csv(sp, sk), p, self.skip_stats)
                    dt = time.perf_counter() - t1
                    self.log_event("process_done_parallel", {
                        "split": n,
                        "duration_sec": dt,
                        "stats": res
                    })
                    par_results.append((n, dt, res))
                except Exception as e:
                    self.log_event("process_error", {
                        "split": n,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
        t_par = time.perf_counter() - t_par0
        
        metrics = {
            "sequential_time_sec": t_seq,
            "parallel_time_sec": t_par,
            "speedup": (t_seq / t_par) if t_par > 0 else float("inf"),
            "concurrency": self.concurrency,
            "retries": self.retries,
            "backoff_sec": self.backoff
        }
        
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        self.log_event("benchmark_complete", metrics)
        return metrics
    
    # ========== MAIN PIPELINE ==========
    
    def run(self) -> Dict:
        """Execute the complete pipeline."""
        print("=" * 70)
        print("COMPLETE METHYLATION DATA PROCESSING PIPELINE")
        print("=" * 70)
        print(f"Input: {self.input_csv}")
        print(f"Output: {self.output_dir}")
        print(f"Filtering: {'Enabled' if self.enable_filtering else 'Disabled'}")
        print(f"Concurrency: {self.concurrency} workers")
        print("=" * 70)
        
        pipeline_start = time.time()
        
        # STEP 1: Optional filtering (if enabled and probe metadata provided)
        if self.enable_filtering and self.probe_metadata_file:
            print("\n[STEP 1/4] Data Filtering & Quality Control...")
            if self.use_streaming:
                print("  Using STREAMING mode for memory-efficient processing...")
                filtered_path = self.output_dir / "filtered_data.csv"
                working_csv = self.retry_wrapper(
                    self.filter_data_streaming,
                    self.input_csv,
                    str(filtered_path)
                )
            else:
                print("  Using IN-MEMORY mode...")
                try:
                    df = pl.read_csv(self.input_csv)
                    df = self.filter_data(df)
                    filtered_path = self.output_dir / "filtered_data.csv"
                    df.write_csv(filtered_path)
                    self.log_event("filtered_data_saved", {"path": str(filtered_path)})
                    working_csv = str(filtered_path)
                except Exception as e:
                    self.log_event("filtering_error", {"error": str(e)})
                    print(f"Warning: Filtering failed, using original data. Error: {e}")
                    working_csv = self.input_csv
        else:
            print("\n[STEP 1/4] Data Filtering - SKIPPED")
            working_csv = self.input_csv
        
        # STEP 2: Read IDs and split
        print("\n[STEP 2/4] Data Splitting...")
        ids = self.read_ids_only(working_csv)
        train_ids, val_ids, test_ids = self.split_ids(ids)
        
        manifest = {
            "input_csv": self.input_csv,
            "filtered_csv": working_csv if working_csv != self.input_csv else None,
            "ratios": self.ratios,
            "seed": self.seed,
            "counts": {
                "total": len(ids),
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids)
            },
            "ids": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids
            }
        }
        
        manifest_path = self.output_dir / "split_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        self.log_event("manifest_written", manifest["counts"])
        print(f"  Total: {len(ids)} | Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
        
        # STEP 3: Write split CSV files
        print("\n[STEP 3/4] Writing Split Files...")
        csv_paths = self.stream_write_splits(
            working_csv,
            set(train_ids),
            set(val_ids),
            set(test_ids)
        )
        self.log_event("split_csv_written", {"paths": csv_paths})
        for split, path in csv_paths.items():
            print(f"  {split}: {path}")
        
        # STEP 4: Parallel benchmark
        print("\n[STEP 4/4] Parallel Processing Benchmark...")
        metrics = self.parallel_benchmark(csv_paths)
        print(f"  Sequential: {metrics['sequential_time_sec']:.2f}s")
        print(f"  Parallel: {metrics['parallel_time_sec']:.2f}s")
        print(f"  Speedup: {metrics['speedup']:.2f}x")
        
        pipeline_elapsed = time.time() - pipeline_start
        
        print("\n" + "=" * 70)
        print(f"✅ PIPELINE COMPLETE in {pipeline_elapsed:.2f}s")
        print("=" * 70)
        print(f"\nGenerated files:")
        print(f"  - {manifest_path}")
        print(f"  - {self.logs_path}")
        print(f"  - {self.metrics_path}")
        for split, path in csv_paths.items():
            print(f"  - {split}: {path}")
        
        return {
            "manifest": manifest,
            "metrics": metrics,
            "csv_paths": csv_paths,
            "logs_path": str(self.logs_path),
            "total_time_sec": pipeline_elapsed
        }


def main():
    """CLI entry point."""
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Complete Methylation Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (no filtering):
  python complete_data_pipeline.py --input_csv data.csv --output_dir output

  # With filtering:
  python complete_data_pipeline.py --input_csv data.csv --output_dir output \\
    --enable_filtering --probe_metadata probe_meta.csv

  # Custom parameters:
  python complete_data_pipeline.py --input_csv data.csv --output_dir output \\
    --seed 42 --ratios 0.8,0.1,0.1 --concurrency 4
        """
    )
    
    # Required
    ap.add_argument("--input_csv", required=True, help="Input CSV file path")
    ap.add_argument("--output_dir", default="./pipeline_output", help="Output directory")
    
    # Optional: Filtering
    ap.add_argument("--enable_filtering", action="store_true", help="Enable data filtering & QC")
    ap.add_argument("--probe_metadata", default=None, help="Probe metadata CSV file (required for filtering)")
    
    # Splitting
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--ratios", type=str, default="0.7,0.15,0.15", help="Train/val/test ratios")
    
    # AC requirements
    ap.add_argument("--concurrency", type=int, default=2, help="Number of concurrent workers (>=2)")
    ap.add_argument("--retries", type=int, default=2, help="Number of retries on failure")
    ap.add_argument("--backoff", type=float, default=0.5, help="Retry backoff time (seconds)")
    
    # Performance
    ap.add_argument("--skip_stats", action="store_true", help="Skip statistics computation for faster processing")
    ap.add_argument("--no_streaming", action="store_true", help="Disable streaming mode (use in-memory processing)")
    ap.add_argument("--chunk_size", type=int, default=10000, help="Chunk size for processing")
    
    args = ap.parse_args()
    
    # Parse ratios
    ratios = tuple(float(x) for x in args.ratios.split(","))
    
    # Validate
    if args.enable_filtering and not args.probe_metadata:
        print("Warning: --enable_filtering requires --probe_metadata, filtering will be skipped")
        args.enable_filtering = False
    
    # Run pipeline
    pipeline = CompleteMethylationPipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        probe_metadata_file=args.probe_metadata,
        seed=args.seed,
        ratios=ratios,
        concurrency=args.concurrency,
        retries=args.retries,
        backoff=args.backoff,
        enable_filtering=args.enable_filtering,
        skip_stats=args.skip_stats,
        use_streaming=not args.no_streaming,
        chunk_size=args.chunk_size
    )
    
    result = pipeline.run()
    
    print("\n📋 Summary:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

