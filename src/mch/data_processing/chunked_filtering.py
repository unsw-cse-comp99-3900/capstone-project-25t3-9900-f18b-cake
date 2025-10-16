import polars as pl

# Assuming you have your probe lists ready
# xym_probes = probe_metadata.filter(...).select('probe_id').to_series().to_list()
# gene_associated_probes = probe_metadata.filter(...).select('probe_id').to_series().to_list()

def process_large_file_chunked(
    input_file: str,
    output_file: str,
    xym_probes: list,
    gene_associated_probes: list,
    chunk_size: int = 50000,
    file_format: str = "parquet"  # or "csv"
):
    """
    Process large file in chunks, filtering probe_id column
    """
    
    # Convert lists to sets for faster lookup
    xym_probes_set = set(xym_probes)
    gene_associated_set = set(gene_associated_probes)
    
    print(f"Excluding {len(xym_probes_set)} XYM probes")
    print(f"Keeping only {len(gene_associated_set)} gene-associated probes")
    
    first_chunk = True
    total_rows_processed = 0
    total_rows_kept = 0
    
    
        # For parquet, we need to read in batches differently
    df_lazy = pl.scan_parquet(input_file)
    
    # Apply filters using lazy evaluation
    filtered_lazy = (df_lazy
                    .filter(~pl.col('probe_id').is_in(xym_probes))
                    .filter(pl.col('probe_id').is_in(gene_associated_probes)))
    
    # Write directly - Polars will handle memory efficiently
    filtered_lazy.sink_parquet(output_file)
    
    # Get stats
    total_kept = filtered_lazy.select(pl.len()).collect().item()
    print(f"Total rows kept: {total_kept}")
    
    
    print(f"\nProcessing complete!")
    if file_format == "csv":
        print(f"Total rows processed: {total_rows_processed:,}")
        print(f"Total rows kept: {total_rows_kept:,}")
        print(f"Reduction: {(1 - total_rows_kept/total_rows_processed)*100:.1f}%")

# Alternative approach using lazy evaluation (more memory efficient)
def process_with_lazy_evaluation(
    input_file: str,
    output_file: str,
    xym_probes: list,
    gene_associated_probes: list,
    file_format: str = "parquet"
    ):
    """
    Most memory-efficient approach using lazy evaluation
    """
    
    print("Using lazy evaluation for maximum memory efficiency...")
    
    lazy_df = pl.scan_parquet(input_file)
    
    # Chain operations without loading into memory
    filtered_lazy = (lazy_df
                    .filter(~pl.col('probe_id').is_in(xym_probes))
                    .filter(pl.col('probe_id').is_in(gene_associated_probes)))
    
    # Write directly to output
    if output_file.endswith('.parquet'):
        filtered_lazy.sink_parquet(output_file)
    else:
        filtered_lazy.sink_csv(output_file)
    
    # Get final count
    final_count = filtered_lazy.select(pl.len()).collect().item()
    print(f"Final dataset contains {final_count:,} rows")

# Usage example:
if __name__ == "__main__":
    print("Loading probe metadata...")
    probe_metadata = pl.read_csv(probe_metadata_file) #, schema_overrides=schema, null_values=["NA"])
    print("Identifying gene-associated probes...")
    gene_associated_probes = probe_metadata.filter(pl.col("gencodebasic_name").is_not_null()).select("probe_id").to_series().to_list()
    print("Identifying XYM probes...")
    xym_probes = probe_metadata.filter(pl.col("chr_hg38").is_in(["chrX", "chrY", "chrM"])).select("probe_id").to_series().to_list()
    
    
    process_with_lazy_evaluation(
        input_file="large_data.parquet",
        output_file="filtered_data.parquet",
        xym_probes=xym_probes,
        gene_associated_probes=gene_associated_probes,
        file_format="parquet"
    )
    
    pass