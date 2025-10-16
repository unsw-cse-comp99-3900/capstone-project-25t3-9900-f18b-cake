import warnings
from pathlib import Path
import pandas as pd
import joblib
import json
import polars as pl
import polars as pl

#from mch.core.disease_tree import DiseaseTree
from mch.core.diseaseTree import DiseaseTree
from mch.config.base_config import FREEZE, FREEZE_NUMBER, WORKING_DIRECTORY

# Database settings
TYPEDB_URI = "localhost:1729"
TYPEDB_DATABASE = "your_database_name"

DATA_DIR = Path(f"{WORKING_DIRECTORY}/data/")
FREEZE_DIR = DATA_DIR / FREEZE

# Constants
#FREEZE_NUMBER = "0525"
#FREEZE = f"freeze{FREEZE_NUMBER}"

def safe_file_exists(file_path):
    """Check if a file exists and return Path object or None."""
    path = Path(file_path)
    if path.exists():
        return path
    else:
        warnings.warn(f"File not found: {file_path}")
        return None

def safe_load_csv(file_path, default=None):
    """Safely load a CSV file, return default if file doesn't exist."""
    path = safe_file_exists(file_path)
    if path:
        try:
            return pl.read_csv(path)
        except Exception as e:
            warnings.warn(f"Error loading CSV {file_path}: {e}")
            return default
    return default

def safe_load_joblib(file_path, default=None):
    """Safely load a joblib file, return default if file doesn't exist."""
    path = safe_file_exists(file_path)
    if path:
        try:
            with open(path, "rb") as f:
                return joblib.load(f)
        except Exception as e:
            warnings.warn(f"Error loading joblib {file_path}: {e}")
            return default
    return default

def safe_load_json(file_path, default=None):
    """Safely load a JSON file, return default if file doesn't exist."""
    path = safe_file_exists(file_path)
    if path:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Error loading JSON {file_path}: {e}")
            return default
    return default


# Load data files
#def load_data():
#    """Load and cache all necessary data files."""
#    mvalue_path = FREEZE_DIR / f"featureValuesWithNewSamples{FREEZE_NUMBER}.csv"
#    tree_path = FREEZE_DIR / "diseaseTree.joblib"
#    color_path = DATA_DIR / "colorProfiles.json"
#
#    mvalue_df = pl.read_csv(mvalue_path)
#    
#    tree_path = f"{WORKING_DIRECTORY}/data/{FREEZE}/diseaseTree.joblib"
#
#    with open(tree_path, "rb") as f:
#        main_tree = joblib.load(f)
#    with open(color_path, "r") as f:
#        color_profiles = json.load(f)
#
#    return mvalue_df, main_tree, color_profiles, main_tree


def load_data():
    """Load and cache all necessary data files with existence checks."""
    #mvalue_path = FREEZE_DIR / f"featureValuesWithNewSamples{FREEZE_NUMBER}.csv"
    mvalue_path = FREEZE_DIR / f"methylation_HG38_m_value_feature_values.csv"
    tree_path = FREEZE_DIR / "diseaseTree.joblib"
    color_path = DATA_DIR / "colorProfiles.json"

    # Load files safely with null defaults
    mvalue_df = safe_load_csv(mvalue_path)
    main_tree = safe_load_joblib(tree_path)
    color_profiles = safe_load_json(color_path)

    return mvalue_df, main_tree, color_profiles, main_tree
    return mvalue_df, main_tree, color_profiles, main_tree

# Create global variables
mvalue_df, main_tree, color_profiles, disease_tree = load_data()

model_directory = f"{FREEZE_DIR}/models/"
model_parameter_directory = f"{FREEZE_DIR}/model_parameters/"
full_model_directory = f"{FREEZE_DIR}/full_models/"
tree_directory = f"{FREEZE_DIR}/trees/"
embedding_directory = f"{FREEZE_DIR}/embeddings/"
who_book_file = f"{FREEZE_DIR}/who_book.json"
cancer_type_file = f"{FREEZE_DIR}/cancer_types.csv"
# Load additional CSV with safety check
#base_mvalue_df = safe_load_csv(f"{FREEZE_DIR}/featureValues{FREEZE_NUMBER}.csv")
base_mvalue_df = safe_load_csv(f"{FREEZE_DIR}/methylation_HG38_m_value_feature_values.csv")
#base_mvalue_df = pl.read_csv(f"{FREEZE_DIR}/featureValues{FREEZE_NUMBER}.csv")

def validate_critical_data():
    """Validate that critical data was loaded successfully."""
    critical_missing = []
    
    if mvalue_df is None:
        critical_missing.append("mvalue_df")
    if main_tree is None:
        critical_missing.append("main_tree")
    if color_profiles is None:
        critical_missing.append("color_profiles")
    if base_mvalue_df is None:
        critical_missing.append("base_mvalue_df")
    
    if critical_missing:
        warnings.warn(f"Critical data missing: {', '.join(critical_missing)}")
        return False
    return True

# Run validation
data_validation_passed = validate_critical_data()
