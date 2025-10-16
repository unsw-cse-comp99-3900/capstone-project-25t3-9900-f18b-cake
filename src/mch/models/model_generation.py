"""
Model generation utilities for methylation classification models.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
import logging
from datetime import datetime
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.preprocessing import StandardScaler

import pyarrow.parquet as pq
import pickle
import time

from mch.models.differentialMethylationClassifier import DifferentialMethylation
from mch.data_processing.dataset_filtering import filter_problem_probes
from mch.utils.logging_utils import load_config, logging

from mch.config.settings import (
    FREEZE_NUMBER, 
    FREEZE,
    FREEZE_DIR,
    main_tree, 
    model_directory,
    tree_directory
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a unique log file for each run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = Path('/data/projects/classifiers/methylation/data/logs/model_generation') / f'model_generation_{timestamp}.log'

# Create file handler and set format
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_model_config() -> dict:
    """Load model configuration from YAML file."""
    #config_path = Path(__file__).parent.parent / 'config' / 'modelTrainingConfig.yaml'
    #with open(config_path) as f:
    #    config = yaml.safe_load(f)
    config = load_config("model_training_config.yaml")
    return config

def make_dataset(filtered_data: pd.DataFrame, disease_tree) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Create a dataset from filtered methylation data and a disease tree.
    
    This function processes methylation data and categorizes samples based on their cancer types
    as defined in the disease tree. It performs several quality checks:
    - Removes samples not present in the disease tree
    - Ensures there are at least 10 samples total
    - Ensures there are at least 2 distinct cancer type groups
    
    Args:
        filtered_data: DataFrame containing methylation data with a sample_id column
        disease_tree: Tree structure containing cancer type hierarchy and sample mappings
        
    Returns:
        tuple: (filtered_data, design) where design contains cancer type labels,
                or None if quality checks fail
    """
    logger.info(f"Constructing dataset for {disease_tree.name}")

    filtered_data = filter_problem_probes(filtered_data)

    truth_values = pd.Series(["otherCancerType"] * len(filtered_data["sample_id"]))
    design = pd.DataFrame({"sample_id": filtered_data.sample_id, "cancerType": truth_values.values})

    for cancer in disease_tree.get_child_names():
        cancer_tree = disease_tree.find_node_by_name(cancer)
        samples = cancer_tree.get_samples_recursive()
        if len(samples) >= 3:
            design.loc[design['sample_id'].isin(samples), 'cancerType'] = cancer

    # Remove samples that don't have a cancer type - this will be samples outside the current cohort for this model. 
    filtered_data = filtered_data[design.cancerType != "otherCancerType"]
    design = design[design.cancerType != "otherCancerType"]

    # An entire category with only ten samples isn't sufficiently informative
    if len(filtered_data["sample_id"]) < 10:
        logger.warning(f"Skipping, {disease_tree.name} has fewer than 10 samples")
        return None
    
    # checking to make sure there are no null values
    filtered_data = filtered_data.dropna(axis="columns")

    # Make sure there are at least two groups being passed back
    if len(design["cancerType"].unique()) < 2:
        logger.warning(f"Skipping, there is only one subgroup of {disease_tree.name}")
        return None

    return filtered_data, pd.DataFrame({"cancerType": design.cancerType})

def get_model_config(model_type: str) -> tuple[BaseEstimator, dict]:
    """Get model and its parameter grid based on model type."""
    configs = load_model_config()['model_configs']
    if model_type not in configs:
        msg = f"Unsupported model type: {model_type}. Supported types are: {list(configs.keys())}"
        logger.error(msg)
        raise ValueError(msg)
    
    model_config = configs[model_type]
    
    if model_type == 'svm':
        model = svm.SVC(**model_config['parameters'])
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**model_config['parameters'])
    else:
        msg = f"Model type {model_type} is in config but not implemented"
        logger.error(msg)
        raise ValueError(msg)
    
    return model, model_config['parameter_grid']

def run_level(all_data: pd.DataFrame, tree, results_dir: Path | None = None) -> None:
    """Run model generation for a specific level in the disease tree.
        - Creates a dataset for the current tree level
        - Runs grid search to find the best model
        - Saves the model and validation samples
    Args:
        all_data: DataFrame containing methylation data
        tree: Disease tree node to process
        results_dir: Directory to save results. If None, uses directory from config
    """
    logger.info(f"Running model generation for {tree.name}")
    
    # Get results directory from config if not provided
    if results_dir is None:
        config = load_model_config()
        results_dir = Path(config['resultsDirectory'])
    
    # Create dataset for this level
    data_construction = make_dataset(all_data, tree)
    if data_construction is None:
        logger.warning(f"Skipping as no dataset was constructed for {tree.name}")
        return
    
    dataset, design = data_construction
    unique_types = design.cancerType.unique()
    logger.info(f"Dataset created: {dataset.shape}")
    logger.info(f"There are {len(unique_types)} sub categories in {tree.name} that will be used")
    logger.info(f"Categories: {unique_types}")
    
    # Get model configuration
    config = load_model_config()
    model_type = config.get('default_model_type', 'svm')
    
    # Run grid search
    model, validation_samples = run_grid_search(
        filtered_data=dataset,
        design=design,
        model_type=model_type,
        name=tree.name
    )
    
    # Store validation samples in tree
    tree.validation_samples = validation_samples
    
    # Create output directories if they don't exist
    model_dir = results_dir / 'trees'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tree with validation samples, including model type in filename
    model_path = model_dir / f"model-{model_type}-{tree.name}.joblib"
    tree_path = model_dir / f"diseaseTree-{model_type}-{tree.name}.joblib"
    
    logger.info(f"Saving {model_type} model to {model_path}")
    joblib.dump(model, model_path)
    
    logger.info(f"Saving tree with validation samples to {tree_path}")
    joblib.dump(tree, tree_path)
    
    logger.info(f"Completed model generation for {tree.name}")

def run_grid_search(filtered_data: pd.DataFrame, design: pd.DataFrame, model_type: str, custom_param_grid: dict | None = None, name: str = "") -> tuple[Pipeline, pd.Series]: 
    """Run grid search to find the best model parameters.
        - Prepares the data by dropping null values and ensuring column names are strings
        - Splits the data into training and test sets
        - Creates a pipeline with differential methylation, scaling, and the selected model type
        - Performs grid search with cross-validation to find optimal parameters
    Args:
        filtered_data: DataFrame containing methylation features
        design: DataFrame containing cancer type labels in 'cancerType' column
        model_type: Type of model to use ('svm' or 'random_forest')
        custom_param_grid: Optional custom parameter grid. If None, uses default for model type
        name: Optional name identifier for the model
        
    Returns:
        tuple: (best_model, y_test) where best_model is the fitted Pipeline with best parameters
               and y_test is the held-out test set labels
    """
    logger.info(f"Starting grid search for model type: {model_type}")
    logger.info(f"Initial training dataset shape: {filtered_data.shape}")
    
    df = filtered_data.dropna(axis="columns")
    df.columns = df.columns.astype(str)
    logger.info(f"Actual training dataset shape: {df.shape}")

    logger.info("Splitting data for training/validation")
    X_train, X_test, y_train, y_test = train_test_split(
        df, 
        design["cancerType"], 
        test_size=0.2, 
        random_state=42
    )
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    logger.info(f"Setting up {model_type} model")
    model, default_param_grid = get_model_config(model_type)
    param_grid = custom_param_grid if custom_param_grid is not None else default_param_grid

    logger.info("Defining pipeline components")
    differential_methylation = DifferentialMethylation()
    
    # Get grid search parameters from config
    config = load_model_config()
    grid_search_params = config['grid_search_config']
    stratified_cv = StratifiedKFold(n_splits=grid_search_params['n_splits'], shuffle=True, random_state=42)

    logger.info("Defining pipeline")
    pipeline = Pipeline([
        ("differentialMethylation", differential_methylation),
        ('scaling', StandardScaler()),
        ('modelGeneration', model)
    ])
    
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=grid_search_params['scoring'],
        cv=stratified_cv,
        verbose=grid_search_params['verbose'],
        n_jobs=grid_search_params['n_jobs'],
        error_score=grid_search_params['error_score']
    )
    
    logger.info(f"Fitting {model_type} models")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    logger.info(f"Best model parameters: {search.best_params_}")
    logger.info(f"Best cross-validation score: {search.best_score_:.3f}")

    return best_model, y_test
