import yaml
from pathlib import Path

# Load variables from refinement.yaml
def load_refinement_config():
    yaml_file_path = Path(__file__).resolve().parent / 'modelTrainingConfig.yaml'
    
    try:
        with open(yaml_file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
    except FileNotFoundError:
        raise Exception(f"Configuration file not found: {yaml_file_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")
    
    return config

# Load the configuration at module import time
refinement_config = load_refinement_config()

# Expose variables for easy access
resultsDirectory = refinement_config.get('resultsDirectory')
parameter_grid = refinement_config.get('parameter_grid')


#filteredMValueFile: /media/storage/bcurran/classifiers/methylation/data/methylationMValuesFiltered.parquet
#datafile = "/media/storage/bcurran/classifiers/methylation/data/methylationHG38FilteredMValue.h5ad"
#filteredDataFile: /data/projects/classifiers/methylation/data/baseMethylationBetaValuesFiltered.parquet


