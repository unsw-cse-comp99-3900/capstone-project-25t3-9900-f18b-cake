import pandas as pd

from typing import Dict, List, Optional
from pathlib import Path
import logging
import yaml

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from differentialMethylationClassifier import DifferentialMethylation

from mch.config.settings import mvalue_df, main_tree, FREEZE, DATA_DIR
from mch.config.modelTrainingParameters import parameter_grid, resultsDirectory

class BatchModelTrainer:
    """Handles training of multiple models across the disease tree."""
    
    def __init__(self, tree=main_tree):
        self.tree = tree
        self.models: Dict[str, RandomForestClassifier] = {}
        self.training_stats: Dict[str, Dict] = {}

        self.dataDirectory = DATA_DIR
        self.resultsDirectory = resultsDirectory
        self.filteredMValueFile = mvalue_df
    
        # get gridsearch parameter grid from config file
                
    def train_all_models(self, save_dir: Optional[Path] = None) -> Dict:
        """
        Trains models for all nodes in the disease tree.
        
        Args:
            save_dir: Optional directory to save models. If None, models are kept in memory.
            
        Returns:
            Dictionary of training statistics for each model
        """
        nodes = self.tree.get_child_names()
        
        for node in nodes:
            try:
                logging.info(f"Training model for node: {node}")
                
                # Prepare data for this node
                nodeData, design = self._prepare_node_data(node)
 
                if type(nodeData)== None:
                    pass
                
                X_train, X_test, y_train, y_test = train_test_split(nodeData,  design["cancerType"], test_size=0.2, random_state=42)
                
                differentialMethylation = DifferentialMethylation()
                randomForest = RandomForestClassifier(random_state=42, n_jobs=5)
                stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

                pipeline = Pipeline([
                    ("differentialMethylation", differentialMethylation),
                    #("featureSelection", rfeFeatureSelection),
                    ('modelGeneration', randomForest)
                    #('scaling', StandardScaler()),
                    #('modelGeneration', refinementSVM)
                ])                

                search = GridSearchCV(pipeline,
                        param_grid=parameter_grid,
                        scoring='accuracy',
                        cv=stratified_cv,
                        verbose=2,
                        n_jobs=10,
                        error_score='raise',
                        )
                
                # Train model
                search.fit(X_train, y_train)
                
                # Store model
                self.models[node] = search.best_estimator_

                
                # Calculate and store metrics
                #self.training_stats[node] = {
                #    'accuracy': model.score(X_test, y_test),
                #    'samples_trained': len(y_train),
                #    'features': X_train.shape[1]
                #}
                
                # Save if directory provided
                if save_dir:
                    print(node, save_dir)
                else:
                    print("model not saved")
                #    self._save_model(node, model, save_dir)
                    
            except Exception as e:
                logging.error(f"Error training model for {node}: {str(e)}")
                self.training_stats[node] = {'error': str(e)}
                
        return self.training_stats
    
    def _prepare_node_data(self, node: str):
        """Prepare training data for a specific node."""
        #print(f"Constructing dataset for {node}")
        truthValues = pd.Series(["otherCancerType"] * len(mvalue_df["sample_id"]))
        design = pd.DataFrame({"sample_id": mvalue_df.sample_id, "cancerType": truthValues.values})

        # find the node in the main tree corresponding to this particular cancer type.
        diseaseTree = main_tree.find_node_by_name(node)
        diseaseSamples = diseaseTree.get_samples_recursive()

        # get the names of the children of that cancer type and the samples associated with each of them
        for cancer in diseaseTree.get_child_names():
            cancerTree = diseaseTree.find_node_by_name(cancer)
            samples = cancerTree.get_samples_recursive()

            #print(len(samples), cancer)
            # if the child node has at least 3 samples (this should probaby be 4?), then change the cancertype in the 
            # design dataframe to be the child node name. Otherwise it get's left as "otherCancerType"
            # this means that the samples of the child node remain in the training data, but it will never be used 
            # as a comparator in the differential methylation step. 
            if len(samples) >= 3:
                design.loc[design['sample_id'].isin(samples), 'cancerType'] = cancer

        # this records the samples that we have data for, but are missing from the oncotree. I don't think we need this. 
        # it will only record those few samples that have been missed on the latest update of the oncotree
        # with open("/media/storage/bcurran/classifiers/methylation/data/samplesMissingFromOncotree.csv", "wt") as file:
        #    filteredData[design.cancerType == "otherCancerType"].to_csv(file)

        # remove samples that don't have a cancer type - this prepares the dataset for differential methylation. 
        # actually. Hold on. I want the ones with "otherCancerType", but not the ones that are not in the sample list. 
        # this should be filtering on the sample_id. 
        # filteredData = filteredData[design.cancerType != "otherCancerType"]
        # design = design[design.cancerType != "otherCancerType"]

        filteredData = mvalue_df[mvalue_df["sample_id"].isin(diseaseSamples)]
        design = design[design["sample_id"].isin(filteredData["sample_id"])]

        print(node, filteredData.shape, design.shape)

        # an entire category with only ten samples isn't sufficiently informative.
        # this should filter any small cancer types and cancers/oncotree nodes with no children types.
        if len(filteredData["sample_id"]) < 10:
            print(f"Skipping, {node} has fewer than 10 samples")
            return None

        
        filteredDataset = filteredData.dropna(axis="columns")

        # there was another filter here ...
        # making sure there were at least nfolds number of samples for each child category?
        valueCounts = design.cancerType.value_counts()

        # make sure there are at least two groups being passed back
        if len(design["cancerType"].unique()) <2:
            print(f"Skipping, there is only one subgroup of {diseaseTree.name}")
            return None

        return filteredData, pd.DataFrame({"cancerType": design.cancerType})
            # Implementation from your existing code
    
    def _save_model(self, node: str, model: RandomForestClassifier, save_dir: Path):
        """Save a single model to disk."""
        model_path = save_dir / f"{node}_model.joblib"
        joblib.dump(model, model_path)
        
    def get_training_summary(self) -> pd.DataFrame:
        """Returns a DataFrame summarizing all model training results."""
        return pd.DataFrame.from_dict(self.training_stats, orient='index') 