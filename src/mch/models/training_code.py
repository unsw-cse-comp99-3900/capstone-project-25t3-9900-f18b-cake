import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import joblib
from typing import List, Dict, Optional, Tuple, Any, Union
import os

class NodeTrainer:
    """
    Trainer class for a specific node in the disease tree.
    Handles model training, calibration, evaluation and persistence.
    """
    def __init__(self, 
                 disease_tree=None,
                 mvalue_df: pl.DataFrame = None,
                 X_train: Union[pl.DataFrame, np.ndarray] = None,
                 y_train: Union[pl.Series, np.ndarray] = None,
                 X_test: Union[pl.DataFrame, np.ndarray] = None,
                 y_train: Union[pl.Series, np.ndarray] = None,
                 param_grid: Dict = None,
                 n_jobs: int = -1,
                 cv: int = 5,
                 min_samples: int = 10,
                 calibration_method: str = 'sigmoid'):
        """
        Initialize a trainer for a specific node in the disease tree.
        
        Args:
            node_name: Name of the node being trained
            disease_tree: DiseaseTree instance 
            mvalue_df: Polars DataFrame with measurement values and sample IDs
            X_train: Training features (if not using disease_tree approach)
            y_train: Training labels (if not using disease_tree approach)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            param_grid: Grid search parameters for RandomForest
            n_jobs: Number of jobs for parallel processing
            cv: Number of cross-validation folds
            min_samples: Minimum number of samples required for training
            calibration_method: Calibration method ('sigmoid' or 'isotonic')
        """
        self.disease_tree = disease_tree
        self.node_name = disease_tree.name
        self.mvalue_df = mvalue_df
        
        # If disease_tree and mvalue_df are provided, prepare node-specific data
        if disease_tree is not None and mvalue_df is not None:
            self._prepare_node_data()
            if self.X_train is None:
                # Node data preparation failed (e.g., too few samples)
                return
        
        # Default parameter grid if none provided
        self.param_grid = param_grid or {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        self.n_jobs = n_jobs
        self.cv = cv
        self.calibration_method = calibration_method
        self.min_samples = min_samples
        
        # Model attributes to be created during training
        self.base_model = None
        self.best_model = None
        self.calibrated_model = None
        self.feature_importances = None
        
        # Metrics storage
        self.metrics = {}
        
        # Flag to indicate if node data preparation was successful
        self.data_prepared = (self.X_train is not None and self.y_train is not None)
    
    def _prepare_node_data(self):
        """
        Prepare training data for this specific node using disease tree information.
        This adapts your original _prepare_node_data method to work with Polars.
        """
        if self.disease_tree is None or self.mvalue_df is None:
            raise ValueError("Both disease_tree and mvalue_df must be provided")
        
        # Create truth values series - equivalent to your pd.Series with "otherCancerType"
        sample_ids = self.mvalue_df.select("sample_id")
        truth_values = pl.DataFrame({
            "sample_id": sample_ids,
            "cancerType": ["otherCancerType"] * len(sample_ids)
        })
        
        # Find the node in the tree corresponding to this cancer type
        node_tree = self.disease_tree.find_node_by_name(self.node_name)
        if node_tree is None:
            print(f"Node {self.node_name} not found in disease tree")
            self.X_train = None
            self.y_train = None
            return
            
        disease_samples = node_tree.get_samples_recursive()
        
        # Update truth values for each child cancer type
        for cancer in node_tree.get_child_names():
            cancer_tree = node_tree.find_node_by_name(cancer)
            samples = cancer_tree.get_samples_recursive()
            
            if len(samples) >= 3:
                # Update only rows where sample_id is in samples
                truth_values = truth_values.with_columns(
                    pl.when(pl.col("sample_id").is_in(samples))
                    .then(pl.lit(cancer))
                    .otherwise(pl.col("cancerType"))
                    .alias("cancerType")
                )
        
        # Filter data to include only samples from this disease
        filtered_data = self.mvalue_df.filter(pl.col("sample_id").is_in(disease_samples))
        truth_values = truth_values.filter(pl.col("sample_id").is_in(filtered_data.select("sample_id")))
        
        # Check if we have enough samples
        if len(filtered_data) < self.min_samples:
            print(f"Skipping, {self.node_name} has fewer than {self.min_samples} samples")
            self.X_train = None
            self.y_train = None
            return
        
        # Drop columns with NA values
        filtered_data = filtered_data.drop_nulls(subset=filtered_data.columns)
        
        # Check if we have at least two different cancer types
        unique_cancer_types = truth_values.select("cancerType").unique()
        if len(unique_cancer_types) < 2:
            print(f"Skipping, there is only one subgroup of {self.node_name}")
            self.X_train = None
            self.y_train = None
            return
        
        # Store the processed data
        # Remove sample_id from X_train
        self.X_train = filtered_data.drop("sample_id")
        self.y_train = truth_values.select("cancerType")
        
        print(f"Prepared data for node {self.node_name}: {len(self.X_train)} samples, {len(unique_cancer_types)} classes")
        return
    
    def _to_numpy(self, data):
        """
        Convert Polars data to numpy arrays only when needed.
        """
        if data is None:
            return None
        return data.to_numpy() if isinstance(data, (pl.DataFrame, pl.Series)) else data
    
    def train(self, verbose: int = 1) -> Dict:
        """
        Train the model with GridSearchCV and calibration.
        
        Args:
            verbose: Verbosity level
            
        Returns:
            Dictionary containing training metrics
        """
        if not self.data_prepared:
            raise ValueError(f"Data for node {self.node_name} has not been properly prepared")
            
        if verbose:
            print(f"Training model for node: {self.node_name}")
        
        # Convert data to numpy just before passing to sklearn
        X_train_np = self._to_numpy(self.X_train)
        y_train_np = self._to_numpy(self.y_train)
        
        # Initialize base random forest classifier
        base_rf = RandomForestClassifier(random_state=42)
        
        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            verbose=verbose
        )
        
        # Fit the grid search
        grid_search.fit(X_train_np, y_train_np)
        
        # Save the best model from grid search
        self.base_model = grid_search
        self.best_model = grid_search.best_estimator_
        
        # Extract and save feature importances
        self.feature_importances = {
            'mean_importance': self.best_model.feature_importances_
        }
        
        # Calibrate the best model
        self.calibrated_model = CalibratedClassifierCV(
            base_estimator=self.best_model,
            method=self.calibration_method,
            cv='prefit'
        )
        
        # Fit the calibration model
        self.calibrated_model.fit(X_train_np, y_train_np)
        
        # Calculate metrics
        self._evaluate()
        
        if verbose:
            print(f"Training complete for node: {self.node_name}")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Training metrics: {self.metrics}")
        
        return self.metrics
    
    def _evaluate(self) -> Dict:
        """
        Evaluate the model on training and validation data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert data to numpy just before passing to sklearn
        X_train_np = self._to_numpy(self.X_train)
        y_train_np = self._to_numpy(self.y_train)
        
        # Training metrics
        train_pred_proba = self.calibrated_model.predict_proba(X_train_np)[:, 1]
        train_pred = (train_pred_proba >= 0.5).astype(int)
        
        self.metrics = {
            'train': {
                'accuracy': accuracy_score(y_train_np, train_pred),
                'roc_auc': roc_auc_score(y_train_np, train_pred_proba),
                'brier_score': brier_score_loss(y_train_np, train_pred_proba)
            }
        }
        
        # Validation metrics if validation data is available
        if self.X_val is not None and self.y_val is not None:
            X_val_np = self._to_numpy(self.X_val)
            y_val_np = self._to_numpy(self.y_val)
            
            val_pred_proba = self.calibrated_model.predict_proba(X_val_np)[:, 1]
            val_pred = (val_pred_proba >= 0.5).astype(int)
            
            self.metrics['val'] = {
                'accuracy': accuracy_score(y_val_np, val_pred),
                'roc_auc': roc_auc_score(y_val_np, val_pred_proba),
                'brier_score': brier_score_loss(y_val_np, val_pred_proba)
            }
        
        return self.metrics
    
    def predict(self, X: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make binary predictions using the calibrated model.
        
        Args:
            X: Features to predict on (Polars DataFrame or numpy array)
            
        Returns:
            Binary predictions
        """
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Convert Polars DataFrame to numpy just before prediction
        X_np = self._to_numpy(X)
        
        # Return binary predictions
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def predict_proba(self, X: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make probability predictions using the calibrated model.
        
        Args:
            X: Features to predict on (Polars DataFrame or numpy array)
            
        Returns:
            Probability predictions for the positive class
        """
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Convert Polars DataFrame to numpy just before prediction
        X_np = self._to_numpy(X)
        
        # Return probability for positive class
        return self.calibrated_model.predict_proba(X_np)[:, 1]
    
    def save_model(self, directory: str) -> str:
        """
        Save the calibrated model to disk.
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Define model path
        model_path = os.path.join(directory, f"{self.node_name}_model.joblib")
        
        # Save the calibrated model
        joblib.dump(self.calibrated_model, model_path)
        
        # Save feature importances as well
        if self.feature_importances is not None:
            importances_path = os.path.join(directory, f"{self.node_name}_importances.joblib")
            joblib.dump(self.feature_importances, importances_path)
        
        return model_path
    
    def load_model(self, model_path: str) -> None:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        self.calibrated_model = joblib.load(model_path)
        
        # Try to load feature importances if available
        importances_path = model_path.replace('_model.joblib', '_importances.joblib')
        if os.path.exists(importances_path):
            self.feature_importances = joblib.load(importances_path)


class TreeModelTrainer:
    """
    Manages training models for an entire DiseaseTree structure.
    """
    def __init__(self, 
                 disease_tree,
                 mvalue_df: pl.DataFrame,
                 X_val: pl.DataFrame = None,
                 y_val: pl.DataFrame = None,
                 param_grid: Dict = None,
                 min_samples: int = 10,
                 n_jobs: int = -1,
                 cv: int = 5,
                 output_dir: str = './models'):
        """
        Initialize the tree model trainer.
        
        Args:
            disease_tree: Instance of DiseaseTree class
            mvalue_df: Measurement values DataFrame with sample_id column
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            param_grid: Grid search parameters for RandomForest
            min_samples: Minimum number of samples required per node
            n_jobs: Number of jobs for parallel processing
            cv: Number of cross-validation folds
            output_dir: Directory to save trained models
        """
        self.disease_tree = disease_tree
        self.mvalue_df = mvalue_df
        self.X_val = X_val
        self.y_val = y_val
        self.param_grid = param_grid
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.cv = cv
        self.output_dir = output_dir
        
        # Dictionary to store trainers for each node
        self.node_trainers = {}
        
        # Dictionary to store results for each node
        self.results = {}
    
    def train_node(self, node, verbose: int = 1) -> Dict:
        """
        Train a model for a specific node.
        
        Args:
            node: Node from the disease tree
            verbose: Verbosity level
            
        Returns:
            Training metrics for the node or None if training was skipped
        """
        node_name = node.name
        
        if verbose:
            print(f"Processing node: {node_name}")
        
        # Create and configure node trainer - it will prepare its own data
        trainer = NodeTrainer(
            node_name=node_name,
            disease_tree=self.disease_tree,
            mvalue_df=self.mvalue_df,
            X_val=self.X_val,
            y_val=self.y_val,
            param_grid=self.param_grid,
            min_samples=self.min_samples,
            n_jobs=self.n_jobs,
            cv=self.cv
        )
        
        # Check if data preparation was successful
        if not trainer.data_prepared:
            if verbose:
                print(f"Skipping node {node_name} due to insufficient data")
            return None
        
        # Train the model
        metrics = trainer.train(verbose=verbose)
        
        # Save the model
        model_path = trainer.save_model(self.output_dir)
        
        # Store the trainer and results
        self.node_trainers[node_name] = trainer
        self.results[node_name] = {
            'metrics': metrics,
            'model_path': model_path
        }
        
        return metrics
    
    def train_tree(self, verbose: int = 1) -> Dict:
        """
        Train models for all nodes in the disease tree.
        
        Args:
            verbose: Verbosity level
            
        Returns:
            Dictionary of results for all nodes
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Recursively train models for all nodes
        self._train_node_recursive(self.disease_tree, verbose)
        
        return self.results
    
    def _train_node_recursive(self, node, verbose: int = 1) -> None:
        """
        Recursively train models for a node and all its children.
        
        Args:
            node: Current node to train
            verbose: Verbosity level
        """
        # Train model for current node
        self.train_node(node, verbose)
        
        # Train models for all children nodes
        for child in node.children:
            self._train_node_recursive(child, verbose)
    
    def get_trainer(self, node_name: str) -> Optional[NodeTrainer]:
        """
        Get the trainer for a specific node.
        
        Args:
            node_name: Name of the node
            
        Returns:
            NodeTrainer instance or None if not found
        """
        return self.node_trainers.get(node_name)