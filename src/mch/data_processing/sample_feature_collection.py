import os
import sys
import logging
import argparse
import yaml
import polars as pl

from pathlib import Path
from typing import Optional, List
from datetime import datetime

from ..db.sql_connector import SQLConnector, Sample
from ..utils.file_utils import find_methylation_file_name, extract_patient_id
from ..utils.logging_utils import setup_logging, load_config


def add_new_row(file_path: str, sample_id: str, meth_sample_id: str, df: pl.DataFrame) -> tuple:
    """
    Add a new sample row to the existing polars DataFrame.

    Args:
        file_path (str): Path to the CSV file with sample data.
        sample_id (str): Sample identifier.
        meth_sample_id (str): sample's corresponding meth_sample_id
        df (pl.DataFrame): Existing DataFrame with feature values.

    Returns:
        Tuple containing the updated DataFrame and a list of missing features.
    """   
    try:
        # find the features in the current dataframe.
        features = df.columns
        logging.info(f"Reading new sample from: {file_path}")

        new_sample = pl.read_csv(file_path)
        new_sample = new_sample.filter(new_sample["probeName"].is_in(features))
 
        if "sample_id" == None:
            raise ValueError("A sample_id must be provdied")
        if "meth_sample_id" == None:
            raise ValueError("A meth_sample_id must be provided")

        # effectively transpose the new data to match the format of the base dataframe, and add the sample_id
        new_sample = new_sample.transpose(column_names="probeName")
        new_sample = new_sample.insert_column(0, pl.Series("sample_id", [sample_id] ))
        new_sample = new_sample.insert_column(1, pl.Series("meth_sample_id", [meth_sample_id] ))
        # Ensure the 'sample_id' column exists in the new sample

        # Extract probe (feature) columns from the original DataFrame, probe (feature) columns from the new sample, and identify missing probes
        original_probes = set(df.columns) - {"sample_id"}
        new_sample_probes = set(new_sample.columns) - {"sample_id"}
        missing_probes = list(original_probes - new_sample_probes)

        # Convert the new sample to a dictionary (for a single row)
        new_sample_dict = new_sample.to_dicts()[0]  # Assume single row in CSV
        
        # Create a new row DataFrame from the dictionary
        new_row = pl.DataFrame([new_sample_dict])

        # Align new_row columns with the existing DataFrame (fill missing with NaN)
        for col in df.columns:
            if col not in new_row.columns:
                # Add missing column with the appropriate type (match df or use Null)
                col_type = df.schema[col]
                new_row = new_row.with_columns(pl.lit(None, dtype=col_type).alias(col))
        for col in new_row.columns:
            if col not in df.columns:
                # Add new columns to df with Null values of consistent type
                new_row_type = new_row.schema[col]
                df = df.with_columns(pl.lit(None, dtype=new_row_type).alias(col))

        # Ensure column order matches the original DataFrame
        new_row = new_row.select(df.columns)

        # Concatenate the new row to the existing DataFrame
        ogShape = df.shape
        df = pl.concat([df, new_row], how="vertical")
        
        logging.info(f"Original shape {ogShape}, Updated DataFrame shape: {df.shape}")
 
        return df, missing_probes

    except Exception as e:
        logging.error(f"Error adding new row: {e}")
        raise
    
    return df

#def find_new_samples(config: dict) -> pl.DataFrame:
def select_base_file(output_file, extant_m_values_file):
    """
    Figure out if there is already an updated file holding values for new samples. Theoretically should only use the 
    extant_m_values file once for each build. i.e. once every three months or so.
    
    Args:
        extant_m_values_file (str): file name, file contains m value data for the base samples/features used for current iteration of models
        output_file (str): file name, file contains extant_m_values data plus new samples
    
    Returns:
        base_file (str): file name  for which ever file should be being added to.
    """

    if Path(output_file).exists():
        # Use output file as base if it exists
        base_file = output_file
        logging.info(f"Using existing output file: {output_file}")
    else:
        # Use existing M values file if output file doesn't exist
        base_file = extant_m_values_file
        logging.info(f"Using existing M values file: {extant_m_values_file}")
    return base_file


def find_new_samples(config: dict) -> List:
    """
    Find new samples. Looks for files with given suffixes in the raw data directory. Extracts the meth_sample_id
    from the file name. Compares that with the meth_sample_id's in the file names in the directory with extracted 
    data and returns the difference.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        list of meth_sample_id's for any new samples.
    """
    # Extract configuration
    extant_sample_directory = config['methylation']['data']['sample_mValues_directory']
    extant_m_values_file = config['methylation']['data']['existing_m_values_file']
    output_file = config['methylation']['data']['output_file']

    base_file = select_base_file(output_file, extant_m_values_file)
    
    # Load existing M-values, rename sampleId if I haven't finished changeing it everywhere. <le sigh> 
    mValuedf = pl.read_csv(base_file)
    if 'sampleId' in mValuedf.columns:
        mValuedf = mValuedf.rename({'sampleId': 'sample_id'})
    
    # Get list of files in upload directory
    files = [f for f in os.listdir(extant_sample_directory) if os.path.isfile(os.path.join(extant_sample_directory, f))]
    logging.info(f"Found {len(files)} files in upload directory, {mValuedf.shape[0]} samples in the existing dataframe")
    meth_sample_ids = [file.removesuffix('.csv') for file in files]
    current_samples = mValuedf.select([pl.col("sample_id"),pl.col("meth_sample_id")])
    new_samples = set(meth_sample_ids) - set(current_samples['meth_sample_id'].to_list())

    return new_samples

def add_new_samples(config, new_samples):
    """
    Update M-values DataFrame with new samples from a directory.
    
    Args:
        config (dict): Configuration dictionary
        new_samples (pl.DataFrame): polars dataframe containing sample_id, meth_sample_id and patient_id for each new sample
    
    Returns:
        Nothing. Writes updated dataframe to file
    """
    logging.info(f"Adding mValue data for {len(new_samples)} new samples to the classifier")
    extant_sample_directory = config['methylation']['data']['sample_mValues_directory']
    extant_m_values_file = config['methylation']['data']['existing_m_values_file']
    output_file = config['methylation']['data']['output_file']

    base_file = select_base_file(output_file, extant_m_values_file)
    missing_probes = []
    
    mValuedf = pl.read_csv(base_file)
    if 'sampleId' in mValuedf.columns:
        mValuedf = mValuedf.rename({'sampleId': 'sample_id'})
    updated_mValues = mValuedf

    for row in new_samples.iter_rows(named=True):
        meth_sample_id = row['meth_sample_id']
        sample_id = row['sample_id']
    
        filePath = os.path.join(extant_sample_directory, f"{meth_sample_id}.csv")  # Assuming file naming convention
    
        if ((mValuedf["meth_sample_id"] == meth_sample_id).any()):
            logging.info(f"{meth_sample_id} already found in dataframe. This shouldn't happen")
            continue

        try:
            updated_mValues, new_missing_probes = add_new_row(filePath, sample_id, meth_sample_id, updated_mValues )
            missing_probes.extend(new_missing_probes)
            logging.info(f"Added new features for sample_id: {sample_id}, meth_sample_id: {meth_sample_id}, {len(new_missing_probes)} probes missing from this sample")
            if len(new_missing_probes) > 0:
                logging.info(f"sample {meth_sample_id} missing probes: {new_missing_probes}")

        except Exception as e:
            logging.error(f"Failed to process file {filePath}: {e}")

    logging.info(f"(Not) Writing updated M-values to {output_file}")
    logging.info(f"Processed {len(new_samples)} new files, added to original dataframe of {mValuedf.shape}")
    #print(updated_mValues.select([pl.col("sample_id"), pl.col("meth_sample_id")]))
    updated_mValues.write_csv(output_file)
    

def main():
    """
    Main entry point for standalone script execution
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Methylation Sample Loader')
    parser.add_argument(
        '-c', '--config', 
        help='Path to configuration file', 
        default=None
    )
    parser.add_argument(
        '-l', '--log-level', 
        help='Logging level', 
        default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)

    # Setup logging
    try:
        base_log_file = config['methylation']['logging'].get('feature_collection_log')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = f'{base_log_file}_{timestamp}.log'
        setup_logging(
            log_file, 
            args.log_level
        )
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

    # Main processing
    try:
        logging.info("Starting sample feature collection for new samples")
        new_samples = find_new_samples(config)
        connector = SQLConnector()
        new_samples = pl.DataFrame(connector.get_sample_data(data_elements = ["sample_id","patient_id", "meth_sample_id"], key_value = "meth_sample_id",  ids = new_samples))
        add_new_samples(config, new_samples)
        logging.info("sample feature collection for new samples completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in sample loader: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()