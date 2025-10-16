import os
import polars as pl
import pyarrow.parquet as pq
import methylcheck
import yaml

from mch.utils.logging_utils import load_config


def report_filtering_step(step_name, before_shape, after_shape, details=None):
    """Consistent reporting function for all filtering steps."""
    print(f"Filtering step: {step_name}")
    print(f"Before filtering: {before_shape[0]} samples × {before_shape[1]} probes")
    print(f"After filtering:  {after_shape[0]} samples × {after_shape[1]} probes")
    print(f"Removed: {before_shape[0] - after_shape[0]} samples, {before_shape[1] - after_shape[1]} probes")
    
    if details:
        for key, value in details.items():
            print(f"{key}: {value}")
    

def filter_snp_adjacent(df: pl.DataFrame) -> pl.DataFrame:
    """Remove probes adjacent to SNPs."""
    initial_shape = df.shape
    # Get the list of problem probes
    sketchy_probe_list = methylcheck.list_problem_probes('epic', ["Polymorphism"])
    # Create a mask - keep only columns that are not in the sketchy probe list
    good_probes = [col for col in df.columns if col not in sketchy_probe_list]
    # Filter the dataframe to exclude the problem probes
    filtered_df = df.select(good_probes)

    report_filtering_step("SNP Adjacent Probe Filtering", initial_shape, filtered_df.shape, {"SNP adjacent probes identified": len(sketchy_probe_list)})
    
    return filtered_df


def filter_problem_samples(df: pl.DataFrame) -> pl.DataFrame:
    """Remove samples with too many missing values."""
    initial_shape = df.shape
    # Count NAs per row and filter out rows with more than 50000 NAs
    na_counts = df.select([pl.sum_horizontal(pl.all().is_null()).alias("na_count")])
    # Get indices of samples to keep
    samples_to_keep = df.with_row_index("idx").filter(na_counts.select("na_count").to_series() <= 50000).select("idx").to_series().to_list()
    # Filter the dataframe
    filtered_df = df.select(pl.all()).filter(pl.arange(0, df.height).is_in(samples_to_keep))
    
    report_filtering_step("Problem Sample Filtering", initial_shape, filtered_df.shape, {"Samples with >50k missing values": initial_shape[0] - filtered_df.shape[0]})
    
    return filtered_df


def filter_problem_probes(df: pl.DataFrame) -> pl.DataFrame:
    """Remove probes with any missing values."""
    initial_shape = df.shape
    # Keep only columns with no NA values
    cols_to_keep = [col for col in df.columns if not df.select(pl.col(col).is_null().any()).item()]
    filtered_df = df.select(cols_to_keep)
    
    #report_filtering_step("Problem Probe Filtering", initial_shape, filtered_df.shape, {"Probes with ≥1 missing value": initial_shape[1] - filtered_df.shape[1]})
    
    return filtered_df


def filter_cross_hybrid(df: pl.DataFrame) -> pl.DataFrame:
    """Remove cross-hybridizing probes."""
    initial_shape = df.shape
    # Get the list of cross-hybridizing probes
    sketchy_probe_list = methylcheck.list_problem_probes('epic', ["CrossHybridization"])
    # Create a mask - keep only columns that are not in the sketchy probe list
    good_probes = [col for col in df.columns if col not in sketchy_probe_list]
    # Filter the dataframe to exclude the problem probes
    filtered_df = df.select(good_probes)
    
    report_filtering_step("Cross-Hybridization Probe Filtering", initial_shape, filtered_df.shape, {"Cross-hybridizing probes": len(sketchy_probe_list)})
    
    return filtered_df


def main():
    """Main function to process the data."""
    print("\n📊 Starting Methylation Data Filtering Process 📊\n")
    config_file = "filtering_config.yaml"
    config = load_config(config_file)
    
    # Data locations
    m_values_file_path = config["base_m_value_file"]
    m_values_filtered_data_file = config["filtered_m_value_file"]
    probe_metadata_file = config["probe_metadata_file"]
    
    # Check if the base beta values file exists
    if not os.path.isfile(m_values_file_path):
        print(f"Error: Base m values file not found at {m_values_file_path}")
        return
    
    schema={
        "Manifest_probe_match": pl.String,
        "Coordinate_36": pl.String,
        "AddressA_ID": pl.String,
        "AddressB_ID": pl.String
    }
    # Load probe metadata
    print("Loading probe metadata...")
    probe_metadata = pl.read_csv(probe_metadata_file) #, schema_overrides=schema, null_values=["NA"])
    
    # Find gene-associated probes
    print("Identifying gene-associated probes...")
    gene_associated_probes = probe_metadata.filter(pl.col("gencodebasic_name").is_not_null()).select("probe_id").to_series().to_list()
    
    # Find XYM probes
    print("Identifying XYM probes...")
    xym_probes = probe_metadata.filter(pl.col("chr_hg38").is_in(["chrX", "chrY", "chrM"])).select("probe_id").to_series().to_list()
    
    # Read in data
    print("Reading base data table...")
    df = pl.read_csv(m_values_file_path)
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape[0]} samples × {initial_shape[1]} probes")
    
    # Subset to gene-associated probes
    print("\nSubsetting to gene-associated probes...")
    available_gene_probes = [probe for probe in gene_associated_probes if probe in df.columns]
    df = df.select(["sampleId"] + available_gene_probes) if "sampleId" in df.columns else df.select(available_gene_probes)
    report_filtering_step("Gene-Associated Probe Filtering", initial_shape, df.shape, {"Gene-associated probes kept": len(available_gene_probes)})
    
    # Remove XYM probes
    print("Removing XYM probes...")
    available_xym_probes = [probe for probe in xym_probes if probe in df.columns]
    columns_to_keep = [col for col in df.columns if col not in available_xym_probes]
    df = df.select(columns_to_keep)
    report_filtering_step("XYM Probe Filtering", (initial_shape[0], len(available_gene_probes)), df.shape, {"XYM probes removed": len(available_xym_probes)})
    
    # Apply filtering steps
    df = filter_problem_samples(df)
    df = filter_snp_adjacent(df)
    # df = filter_problem_probes(df)
    df = filter_cross_hybrid(df)
    
    # Save beta values
    print("\nSaving filtered m values...")
    df.write_parquet(m_values_filtered_data_file)
    print(f"✅ M values saved to {m_values_filtered_data_file}")
    
    print("\n🎉 Filtering process completed successfully! 🎉")


if __name__ == "__main__":
    main()