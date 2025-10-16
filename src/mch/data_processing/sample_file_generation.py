import os
import sys
import logging
import argparse
import yaml
import polars as pl

from pathlib import Path
from typing import Optional
from datetime import datetime

from ..db.sql_connector import SQLConnector #, Sample
from ..utils.file_utils import find_methylation_file_name, extract_patient_id, get_files_with_suffix
from ..utils.logging_utils import setup_logging, load_config

# code to extract Mvalues and Beta Values from the R drive into separate folders where they can be exported to an s3 bucket for compilation by BIX
# should be fairly simple. ... ... ... ha. 

def probe_filter(duplicates, v2Manifest, duplicate_probes):
    """ 
    Finds and adjusts EPICv2 probes the now have multiple EPICv2 probes mapping to them. 
    
    Args:
        duplicates (): <hrmmm>
        v2Manifest (str): path pointing towards the aletnative v2Manifest put ut by Tim Peters group. 
        duplicate_probes (list): a list of probes with more than one v2 probe mapping to a v1 probe <rolls eyes>

    Returns a dataframe containing the mean of the probes to be used, or the value from teh best probe, depending on the work out of Tim Peters group. 
    """
    duplicatesSummary = v2Manifest.filter(v2Manifest["EPICv1locmatch"].is_in(duplicate_probes)).select(['IlmnID', 'Name', 'EPICv1locmatch', 'UCSC_RefGene_Name', 'GencodeV41_Name', 'Rep_results_by_NAME'])
    
    superiorProbe = (
        duplicatesSummary.filter(pl.col("Rep_results_by_NAME") == "Superior probe")
        .group_by("EPICv1locmatch")
        .agg(pl.col("IlmnID"))
    )
    toRemove = superiorProbe["EPICv1locmatch"].to_list()
    duplicatesSummary = duplicatesSummary.filter(~pl.col("EPICv1locmatch").is_in(toRemove))

    best_precision_by_group_mean = (
       duplicatesSummary.filter((pl.col("Rep_results_by_NAME").str.contains("Best")) | (pl.col("Rep_results_by_NAME").str.contains("Best precision")))
        .group_by("EPICv1locmatch")
        .agg(pl.col("IlmnID"))
    )
    toRemove = best_precision_by_group_mean["EPICv1locmatch"].to_list()
    duplicatesSummary = duplicatesSummary.filter(~pl.col("EPICv1locmatch").is_in(toRemove))

    df = pl.concat([best_precision_by_group_mean, superiorProbe])
    
    return(df, duplicatesSummary)

def replace_with_mean(mappingdf, sampledf, dataType):
    """
    Using the illumina_probe_map previously generated, find the mean of all all duplicate probes in the mapping that are
    in the new sample, and replace those with the mean of the probes in the mapping. 
    Not a super clear explanation but ... <rolls eyes hard at illumina>
        
    Args:
        mappingdf(pl.DataFrame): mapping dataframe.
        sampledf(pl.DataFrame): new sample to be mapped.
        dataType: (str): M value or Beta value.
               
    Returns:
        sampledf (pl.DataFrame): data frame containing sample with v2 probes reduced to v1
    """
    # The mapping df holds a mapping between the Illumina Id (multiple probes) and an EPICv1locmatch. 
    # The sample  df should hold three columns, the Illumina Id in a column called "Name", an Epic v1 loci and an M/beta value.

    sampledf = sampledf.filter(pl.col("EPICv1_Loci").is_not_null())

    for row in mappingdf.iter_rows(named=True):
        epic_match = row["EPICv1locmatch"]
        ilmn_ids = row["IlmnID"]
        # Filter sample_df where the Name column from the sample matches and IlmnID's that are in the list
        filtered = sampledf.filter(pl.col("Name").is_in(ilmn_ids)) 
        
        # this replaces all values for a given v1 locus with the mean derived from the relevant v2 probes in the mapping df.
        if not filtered.is_empty():
            # Calculate the mean MValue
            mean_mvalue = filtered[dataType].mean()
            # Replace MValue in sample_df for the given EPICv1locmatch
            sampledf = sampledf.with_columns(
                pl.when((pl.col("EPICv1_Loci") == epic_match))
                .then(mean_mvalue)
                .otherwise(pl.col(dataType))
                .alias(dataType)
            )
    # remove the Illumina Id's, I don't need them anymore. 
    sampledf = sampledf[["EPICv1_Loci", dataType]]
    sampledf = sampledf.unique()  # or keep="last"

    # find duplicate entries in the EPIC v1 locations. 
    duplicateLoci = sampledf.filter(sampledf["EPICv1_Loci"].is_duplicated())
    duplicateLoci = duplicateLoci["EPICv1_Loci"].to_list()
 
    # Remove all remaining rows with duplicate 'EPICv1_Loci', having checked that none of them are in the mapping dataframe. 
    # Where these probes are present in Tim's manifest, they are inferior probes, have insufficient evidence or have been assesed in comparison to WGBS due to insuffficient v1 data. 
    sampledf = sampledf.filter(~pl.col("EPICv1_Loci").is_in(duplicateLoci))

    return sampledf

def get_meth_sample_id(file_name, suffixes):
    """
    Extracts the sample_id from a file name by removing one of the specified suffixes. Based solely on the idea that the file name contains the meth_sample_id.
   
    Args:
        file_name (str): The name of the file.
        suffixes (list): A list of suffixes to check and remove from the file name.
    
    Returns:
        str: The sample ID if a matching suffix is found; otherwise, "MysterySample".
    """
    for suffix in suffixes:
        if file_name.endswith(suffix):
            return file_name.rstrip(suffix)
        else:
            print(file_name)

    return "MysterySample"
    
def map_illumina_probes(config):
    """
    Using the alternate epic V2 Manifest as provided by Tim Peters group at Garvan, idenitfy probes that do not have a 
    many to one mapping in the Epic V2-V1 arrays. Filter using probe_filter() to identify best selection strategy. 
        
    Args:
        config(dict): contents of a config file
               
    Returns:
        finalMatches (pl.DataFrame): data frame containing mapped probes that should be retained. 
    """
    
    # read in the alternative manifest
    dtypes={"Manifest_probe_match": str, "RMSE_with_WGBS": pl.Float64	}
    v2Manifest = pl.read_csv("/data/projects/classifiers/methylation/data/epicV2ManifestAlternate.csv", schema_overrides=dtypes, null_values="NA")

    # find the probes in EPIC v2 that need special handling. These are probes that have multipe EPICv2 probes mapped to a single EPICv1 probe. 
    # if a v2 probe needs to be replaced by a single probe, that will be recorded in this dataframe. 
    # also recording the v1 probes that are missing from v2. I'm sure I had a good reason for doing this. 
    probeCounts = v2Manifest.group_by("EPICv1locmatch").agg(pl.count("EPICv1locmatch").alias("count"))
    summary = probeCounts.group_by("count").agg(pl.count("EPICv1locmatch").alias("num_names"))

    finalMatches = pl.DataFrame()
    finalMissing = pl.DataFrame()
    for reps in [2,3,4,5,6,10]:
        duplicates = probeCounts.filter(pl.col("count") == reps)
        duplicate_probes = duplicates["EPICv1locmatch"]
        matches, missing = probe_filter(duplicates, v2Manifest, duplicate_probes)

        finalMatches = pl.concat([finalMatches, matches])
        finalMissing = pl.concat([finalMissing, missing])

    return finalMatches

def create_data_file(config, new_samples, illumina_probe_map):
    """
    Reads in new methylation data files. Identify whether it's v1 or v2, if v2, translate v2 probes to v1.
    Extracts M and Beta values and writes them to their own file where they can be used as the base data for 
    each sample.
        
    Args:
        config(dict): contents of a config file
        new_samples (list): list of sample files that have been identified as not already processed. 
        illumina_probe_map(dict): mapping from v2 probes with more than one probe for some v1 probes.
               
    Returns:
        None: sample is written to file, no output returned.
    """
    dtypes = {"Manifest_probe_match": str, "Start_hg38": pl.Float64}
    rawDataDirectory = config["methylation"]["data_generation"]["raw_methylation_data_directory"]
    baseMValueFilesLocation = config['methylation']['data']['sample_mValues_directory']
    baseBetaValueFilesLocation = config['methylation']['data']['sample_betaValues_directory']

    for new_meth_sample in new_samples.iter_rows():
        meth_sample_id = new_meth_sample[new_samples.columns.index("meth_sample_id")]
        file_name = new_meth_sample[new_samples.columns.index("file_name")]
        logging.info(f"attempting to create M and Beta value files for {meth_sample_id}")
        try:
            sampleData = pl.read_csv(file_name, separator="\t", null_values=["NA"], schema_overrides=dtypes)
            if sampleData.is_empty():
                logging.warn(f"{dataFile} is empty.")
            else:
                if 'EPICv1_Loci' in sampleData.columns:
                    # If it's a v2 file, extract the v2 probe names, the v1 probe name and the mValue, 
                    # Take the probes that need to be altered, found earlier and replace them with the mean value of the set of v2 probes found for each v1 probe. 
                    mValueData = sampleData[["Name", "EPICv1_Loci", "MValue"]]
                    mValueData = replace_with_mean(illumina_probe_map, mValueData, "MValue")
                    mValueData = mValueData.rename({'EPICv1_Loci': 'probeName'})
                    mValueData.write_csv(f"{baseMValueFilesLocation}/{meth_sample_id}.csv")
        
                    # do the same for the beta value
                    betaValueData = sampleData[["Name", "EPICv1_Loci", "Beta"]]
                    betaValueData = replace_with_mean(illumina_probe_map, betaValueData, "Beta")
                    betaValueData = betaValueData.rename({'EPICv1_Loci': 'probeName'})
                    betaValueData.write_csv(f"{baseBetaValueFilesLocation}/{meth_sample_id}.csv")
                else:
                    # If it's a v1 file, extract the v1 name and the mValue, and write that to a file. 
                    # There a number of ... odd files in the directory, hence the if statement.
                    if "MValue" in sampleData.columns:
                        mValueData = sampleData[["Name", "MValue"]]
                        mValueData = mValueData.rename({'Name': 'probeName'})
                        mValueData.write_csv(f"{baseMValueFilesLocation}/{meth_sample_id}.csv")
                        # ditto for the beta value.
                        betaValueData = sampleData[["Name", "Beta"]]
                        betaValueData = betaValueData.rename({'Name': 'probeName'})
                        betaValueData.write_csv(f"{baseBetaValueFilesLocation}/{meth_sample_id}.csv")
            logging.info(f"files created for {meth_sample_id}")
                #pl.DataFrame(sampleIds).write_csv(f"/data/projects/classifiers/methylation/data/{freeze}/samplesWithFiles.csv")
        
        except pl.exceptions.NoDataError:
            logging.warn(f"There is a problem with {dataFile}. Possibly empty, or the data not in the expected format")
            print(f"Error: {dataFile} is empty or contains no valid data.")
    

def check_sample_processed(meth_sample_ids, baseMValueFilesLocation, baseBetaValueFilesLocation ):
        """
        Check if a sample has already been processed by checking file existence.
        
        Args:
            meth_sample_id (str): Sample ID to check
        
        Returns:
            bool: True if sample is processed (files exist), False otherwise
        """
        processed = []
        for meth_sample_id in meth_sample_ids:
            mvalue_file = os.path.join(baseMValueFilesLocation, f"{meth_sample_id}.csv")
            beta_file = os.path.join(baseBetaValueFilesLocation, f"{meth_sample_id}.csv")
            if (os.path.exists(mvalue_file) and os.path.exists(beta_file)):
                processed.append(meth_sample_id)
        
        return processed


def find_new_samples(config):
    """
    Find samples that have an file in the raw data directory, but no corresponding Beta and M Value file.
    Extract meth_sample_ids for new files and return along with the file location - it's a pain to rebuild
    the file location later. Though now that I think about it, I probably could have extracted the meth_sample_id
    later. ... No I couldn't, I need to check the name of the generated files, which are all stored as <meth_sample_id>.csv
    
    Args:
        config (str): contents of config file. 
    
    Returns:
        pl.DataFrame: meth_sample_id and filename for each new sample  
    """
    # Next, we find all files with the correct suffix in the R_drive. 
    rawDataDirectory = config["methylation"]["data_generation"]["raw_methylation_data_directory"]
    baseMValueFilesLocation = config['methylation']['data']['sample_mValues_directory']
    baseBetaValueFilesLocation = config['methylation']['data']['sample_betaValues_directory']
    suffixes = [methylationFileSuffix1] = config['methylation']['data_generation']['file_suffixes']

    methylationFileSuffix1 = "_methylation_anno_hg38.txt"
    methylationFileSuffix2 = "_methylation_anno.txt"
    suffixes = [methylationFileSuffix1]
    
    methylationFiles = get_files_with_suffix(rawDataDirectory, suffixes )
    
    logging.info(f"There are {len(methylationFiles)} files in the raw methyatlion data directory structure with a matching suffix")

    # loop through all files and get a meth sample id from the file name
    meth_sample_ids = []

    for dataFile in methylationFiles:
        fileName = os.path.basename(dataFile)
        meth_sample_id = get_meth_sample_id(fileName, suffixes)
        meth_sample_ids.append(meth_sample_id)

    df = pl.DataFrame({
        "file_name": methylationFiles,
        "meth_sample_id": meth_sample_ids
    })

    ## skip files that have already been created. 
    processed_samples = check_sample_processed(meth_sample_ids, baseMValueFilesLocation, baseBetaValueFilesLocation)
    df = df.filter(~pl.col("meth_sample_id").is_in(processed_samples))
    
    logging.info(f"There are {len(df['meth_sample_id'])} new meth_sample_ids where the M and Beta values have not already been extracted, when looking at the extracted files directory")
 
    return df
    

def main():
    """
    Main entry point for standalone script execution
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Extraction of M and Beta values from the results of Methylation Analysis')
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
        base_log_file = config['methylation']['logging'].get('data_generation_log')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = f'{base_log_file}_{timestamp}.log'
        print(log_file)
        setup_logging(
            log_file, 
            args.log_level
        )
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

    # Main processing
    try:
        logging.info("Starting file generation for new samples")
        probe_map = map_illumina_probes(config)
        new_samples = find_new_samples(config)
        if new_samples.is_empty():
            logging.info("No new samples found")
            return
        create_data_file(config, new_samples, probe_map)

        logging.info("Base sample File generation completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in base sample file generation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()