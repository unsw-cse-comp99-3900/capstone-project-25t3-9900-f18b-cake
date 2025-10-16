"""Tests for the model generation functionality."""

import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import yaml
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from mch.models.model_generation import make_dataset, run_grid_search, run_level

@dataclass
class MockDiseaseNode:
    """Mock disease tree node for testing."""
    name: str
    samples: List[str]
    children: List['MockDiseaseNode']

    def get_child_names(self):
        return [child.name for child in self.children]
    
    def find_node_by_name(self, name):
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_node_by_name(name)
            if found:
                return found
        return None
    
    def get_samples_recursive(self):
        all_samples = self.samples.copy()
        for child in self.children:
            all_samples.extend(child.get_samples_recursive())
        return all_samples

def test_make_dataset_basic():
    """Test basic functionality of make_dataset with valid input."""
    # Create mock methylation data
    data = {
        'sample_id': ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10'],
        'feature1': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        'feature2': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
    }
    filtered_data = pd.DataFrame(data)
    
    # Create mock disease tree
    tree = MockDiseaseNode(
        name="root",
        samples=[],
        children=[
            MockDiseaseNode(
                name="typeA",
                samples=['sample1', 'sample2', 'sample3', 'sample9', 'sample10'],
                children=[]
            ),
            MockDiseaseNode(
                name="typeB",
                samples=['sample4', 'sample5', 'sample6', 'sample7', 'sample8'],
                children=[]
            )
        ]
    )
    
    # Test the function
    result = make_dataset(filtered_data, tree)
    assert result is not None
    filtered_result, design = result
    
    # Check that all samples were assigned to correct groups
    assert len(design) == 10
    assert set(design['cancerType'].unique()) == {'typeA', 'typeB'}
    assert len(design[design['cancerType'] == 'typeA']) == 5
    assert len(design[design['cancerType'] == 'typeB']) == 5

def test_make_dataset_with_nulls():
    """Test that make_dataset properly handles null values."""
    # Create mock methylation data with null values
    data = {
        'sample_id': ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10'],
        'feature1': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        'feature2': [0.5, None, 0.7, 0.6, 0.78, 1.0, 1.4, 1.2, 1.7, 1.4],
        'feature3': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
    }
    filtered_data = pd.DataFrame(data)
    
    # Create mock disease tree
    tree = MockDiseaseNode(
        name="root",
        samples=[],
        children=[
            MockDiseaseNode(
                name="typeA",
                samples=['sample1', 'sample2', 'sample3', 'sample9', 'sample10'],
                children=[]
            ),
            MockDiseaseNode(
                name="typeB",
                samples=['sample4', 'sample5', 'sample6', 'sample7', 'sample8'],
                children=[]
            )
        ]
    )
    # Test the function
    result = make_dataset(filtered_data, tree)
    assert result is not None
    filtered_result, design = result
    
    # Check that the column with null values was dropped
    assert 'feature2' not in filtered_result.columns
    assert set(filtered_result.columns) == {'sample_id', 'feature1', 'feature3'}

def test_make_dataset_insufficient_samples():
    """Test that make_dataset returns None when there are insufficient samples."""
    # Create mock methylation data
    data = {
        'sample_id': ['sample1', 'sample2'],
        'feature1': [0.5, 0.6],
        'feature2': [1.5, 1.6]
    }
    filtered_data = pd.DataFrame(data)
    
    # Create mock disease tree with insufficient samples
    tree = MockDiseaseNode(
        name="root",
        samples=[],
        children=[
            MockDiseaseNode(
                name="typeA",
                samples=['sample1'],
                children=[]
            ),
            MockDiseaseNode(
                name="typeB",
                samples=['sample2'],
                children=[]
            )
        ]
    )
    
    # Test the function
    result = make_dataset(filtered_data, tree)
    assert result is None

def test_make_dataset_single_group():
    """Test that make_dataset returns None when there's only one group."""
    # Create mock methylation data
    data = {
        'sample_id': ['sample1', 'sample2', 'sample3', 'sample4'],
        'feature1': [0.5, 0.6, 0.7, 0.8],
        'feature2': [1.5, 1.6, 1.7, 1.8]
    }
    filtered_data = pd.DataFrame(data)
    
    # Create mock disease tree with only one group
    tree = MockDiseaseNode(
        name="root",
        samples=[],
        children=[
            MockDiseaseNode(
                name="typeA",
                samples=['sample1', 'sample2', 'sample3', 'sample4'],
                children=[]
            )
        ]
    )
    
    # Test the function
    result = make_dataset(filtered_data, tree)
    assert result is None

@pytest.fixture
def mock_methylation_data():
    """Create mock methylation data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Create sample IDs as strings
    sample_ids = pd.Series([f'sample{i:03d}' for i in range(n_samples)], dtype='string')
    
    # Create random methylation features
    features = {f'feature{i}': np.random.rand(n_samples) for i in range(n_features)}
    
    # Add sample IDs as a column
    df = pd.DataFrame(features)
    df['sample_id'] = sample_ids
    
    return df

@pytest.fixture
def mock_design_data():
    """Create mock design data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create balanced classes
    cancer_types = ['typeA'] * 50 + ['typeB'] * 50
    np.random.shuffle(cancer_types)
    
    return pd.DataFrame({'cancerType': cancer_types})

@pytest.fixture
def mock_config(tmp_path):
    """Create a temporary config file for testing."""
    config = {
        'grid_search_config': {
            'scoring': 'accuracy',
            'n_splits': 2,
            'n_jobs': 1,
            'verbose': 0,
            'error_score': 'raise'
        },
        'model_configs': {
            'svm': {
                'parameters': {
                    'decision_function_shape': 'ovo',
                    'random_state': 42,
                    'cache_size': 500
                },
                'parameter_grid': {
                    'modelGeneration__C': [0.1, 1],
                    'modelGeneration__kernel': ['linear']
                }
            },
            'random_forest': {
                'parameters': {
                    'random_state': 42,
                    'n_jobs': 1
                },
                'parameter_grid': {
                    'modelGeneration__n_estimators': [10, 20],
                    'modelGeneration__max_depth': [2, 4]
                }
            }
        }
    }
    
    config_dir = tmp_path / "mch" / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "modelTrainingConfig.yaml"
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file

def test_run_grid_search_svm(mock_methylation_data, mock_design_data, monkeypatch, mock_config):
    """Test grid search with SVM model."""
    # Monkeypatch the config file path
    def mock_load_config():
        with open(mock_config) as f:
            return yaml.safe_load(f)
    
    # Mock the R script functionality
    def mock_run_r_script(self, data_filename, design_filename):
        return mock_methylation_data.columns[:5].tolist()  # Return first 5 features as "significant"
    
    monkeypatch.setattr('mch.models.model_generation.load_model_config', mock_load_config)
    monkeypatch.setattr('mch.models.differentialMethylationClassifier.DifferentialMethylation.run_r_script', mock_run_r_script)
    
    # Run grid search
    best_model, y_test = run_grid_search(
        filtered_data=mock_methylation_data,
        design=mock_design_data,
        model_type='svm'
    )
    
    # Check the results
    assert isinstance(best_model, Pipeline)
    assert isinstance(best_model.named_steps['modelGeneration'], SVC)
    assert isinstance(y_test, pd.Series)
    assert len(y_test) == 20  # 20% of 100 samples
    assert set(y_test.unique()) == {'typeA', 'typeB'}

def test_run_grid_search_random_forest(mock_methylation_data, mock_design_data, monkeypatch, mock_config):
    """Test grid search with Random Forest model."""
    # Monkeypatch the config file path
    def mock_load_config():
        with open(mock_config) as f:
            return yaml.safe_load(f)
    
    # Mock the R script functionality
    def mock_run_r_script(self, data_filename, design_filename):
        return mock_methylation_data.columns[:5].tolist()  # Return first 5 features as "significant"
    
    monkeypatch.setattr('mch.models.model_generation.load_model_config', mock_load_config)
    monkeypatch.setattr('mch.models.differentialMethylationClassifier.DifferentialMethylation.run_r_script', mock_run_r_script)
    
    # Run grid search
    best_model, y_test = run_grid_search(
        filtered_data=mock_methylation_data,
        design=mock_design_data,
        model_type='random_forest'
    )
    
    # Check the results
    assert isinstance(best_model, Pipeline)
    assert isinstance(best_model.named_steps['modelGeneration'], RandomForestClassifier)
    assert isinstance(y_test, pd.Series)
    assert len(y_test) == 20  # 20% of 100 samples
    assert set(y_test.unique()) == {'typeA', 'typeB'}

def test_run_grid_search_custom_params(mock_methylation_data, mock_design_data, monkeypatch, mock_config):
    """Test grid search with custom parameter grid."""
    # Monkeypatch the config file path
    def mock_load_config():
        with open(mock_config) as f:
            return yaml.safe_load(f)
    
    # Mock the R script functionality
    def mock_run_r_script(self, data_filename, design_filename):
        return mock_methylation_data.columns[:5].tolist()  # Return first 5 features as "significant"
    
    monkeypatch.setattr('mch.models.model_generation.load_model_config', mock_load_config)
    monkeypatch.setattr('mch.models.differentialMethylationClassifier.DifferentialMethylation.run_r_script', mock_run_r_script)
    
    # Custom parameter grid
    custom_grid = {
        'modelGeneration__n_estimators': [5],
        'modelGeneration__max_depth': [3]
    }
    
    # Run grid search
    best_model, y_test = run_grid_search(
        filtered_data=mock_methylation_data,
        design=mock_design_data,
        model_type='random_forest',
        custom_param_grid=custom_grid
    )
    
    # Check the results
    assert isinstance(best_model, Pipeline)
    assert best_model.named_steps['modelGeneration'].n_estimators == 5
    assert best_model.named_steps['modelGeneration'].max_depth == 3
    assert isinstance(y_test, pd.Series)
    assert len(y_test) == 20  # 20% of 100 samples
    assert set(y_test.unique()) == {'typeA', 'typeB'}

def test_run_grid_search_invalid_model_type(mock_methylation_data, mock_design_data, monkeypatch, mock_config):
    """Test grid search with invalid model type."""
    # Monkeypatch the config file path
    def mock_load_config():
        with open(mock_config) as f:
            return yaml.safe_load(f)
    
    # Mock the R script functionality
    def mock_run_r_script(self, data_filename, design_filename):
        return mock_methylation_data.columns[:5].tolist()  # Return first 5 features as "significant"
    
    monkeypatch.setattr('mch.models.model_generation.load_model_config', mock_load_config)
    monkeypatch.setattr('mch.models.differentialMethylationClassifier.DifferentialMethylation.run_r_script', mock_run_r_script)
    
    # Run grid search with invalid model type
    with pytest.raises(ValueError, match="Unsupported model type"):
        run_grid_search(
            filtered_data=mock_methylation_data,
            design=mock_design_data,
            model_type='invalid_model'
        )

def test_run_level(mock_methylation_data, mock_design_data, tmp_path, monkeypatch):
    """Test the run_level function for model generation and saving."""
    from mch.models.model_generation import run_level
    
    # Create a mock tree with sufficient samples for training
    tree = MockDiseaseNode(
        name="test_cancer",
        samples=[],
        children=[
            MockDiseaseNode(
                name="typeA",
                samples=[f'sample{i:03d}' for i in range(50)],
                children=[]
            ),
            MockDiseaseNode(
                name="typeB",
                samples=[f'sample{i:03d}' for i in range(50, 100)],
                children=[]
            )
        ]
    )
    
    # Mock the config to use the temporary directory
    def mock_config():
        return {
            'resultsDirectory': str(tmp_path),
            'default_model_type': 'svm',
            'model_configs': {
                'svm': {
                    'parameters': {
                        'decision_function_shape': 'ovo',
                        'random_state': 42,
                        'cache_size': 500
                    },
                    'parameter_grid': {
                        'modelGeneration__C': [1],
                        'modelGeneration__kernel': ['linear']
                    }
                }
            },
            'grid_search_config': {
                'scoring': 'accuracy',
                'n_splits': 2,
                'n_jobs': 1,
                'verbose': 0,
                'error_score': 'raise'
            }
        }
    
    # Mock the R script functionality
    def mock_run_r_script(self, data_filename, design_filename):
        return mock_methylation_data.columns[:5].tolist()  # Return first 5 features as "significant"

    monkeypatch.setattr('mch.models.model_generation.load_model_config', mock_config)
    monkeypatch.setattr('mch.models.differentialMethylationClassifier.DifferentialMethylation.run_r_script', mock_run_r_script)
    
    # Run the function
    run_level(mock_methylation_data, tree)
    
    # Check that model and tree files were created
    model_file = Path(tmp_path) / 'trees' / f'model-svm-{tree.name}.joblib'
    tree_file = Path(tmp_path) / 'trees' / f'diseaseTree-svm-{tree.name}.joblib'
    assert model_file.exists(), "Model file was not created"
    assert tree_file.exists(), "Tree file was not created"
    
    # Load and verify the saved model
    model = joblib.load(model_file)
    assert isinstance(model, Pipeline), "Saved model is not a Pipeline"
    assert isinstance(model.named_steps['modelGeneration'], SVC), "Model is not an SVM"
    
    # Load and verify the saved tree
    saved_tree = joblib.load(tree_file)
    assert hasattr(saved_tree, 'validation_samples'), "Tree does not have validation samples"
    assert isinstance(saved_tree.validation_samples, pd.Series), "Validation samples not properly saved"

def test_run_level_insufficient_samples(mock_methylation_data, tmp_path, monkeypatch):
    """Test run_level with insufficient samples returns early."""
    from mch.models.model_generation import run_level
    
    # Create a mock tree with insufficient samples
    tree = MockDiseaseNode(
        name="small_cancer",
        samples=[],
        children=[
            MockDiseaseNode(
                name="typeA",
                samples=['sample1', 'sample2'],
                children=[]
            ),
            MockDiseaseNode(
                name="typeB",
                samples=['sample3'],
                children=[]
            )
        ]
    )
    
    # Mock the config
    monkeypatch.setattr('mch.models.model_generation.load_model_config', 
                       lambda: {'resultsDirectory': str(tmp_path)})
    
    # Run the function
    run_level(mock_methylation_data, tree)
    
    # Check that no files were created
    model_dir = tmp_path / 'trees'
    if model_dir.exists():
        assert len(list(model_dir.glob('*.joblib'))) == 0, "Files were created for insufficient samples"
