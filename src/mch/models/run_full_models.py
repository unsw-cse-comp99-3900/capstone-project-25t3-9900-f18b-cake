import pandas as pd
import polars as pl
import numpy as np
import joblib
import os
import umap.umap_ as umap
import pickle

#from methylationMethods import makeUmapPlot3D, createPlotdf
#from mch.core.disease_tree import DiseaseTree
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay

import os
import sys
import json
import re

from ..config.settings import (
    FREEZE_NUMBER, 
    FREEZE, 
    mvalue_df, 
    main_tree, 
    color_profiles,
    model_parameter_directory,
    tree_directory,
    full_model_directory, 
    embedding_directory,
    base_mvalue_df
)

from mch.utils.logging_utils import setup_logging, load_config



def create_umap_embedding(X_data, feature_importances, model_name, embedding_directory):
    """
    Create a UMAP embedding using features with importance > 0 and save it.
    
    Parameters:
    -----------
    X_data : pandas.DataFrame
        The feature data, with sample_id as index
    feature_importances : array-like
        Array of feature importance values
    model_name : str
        Name of the model (used for file naming)
    embedding_directory : str
        Directory to save the embedding files
        
    Returns:
    --------
    bool
        True if embedding was created and saved, False otherwise
    """
    # Create feature importance dictionary
    print(X_data.shape)

    feature_names = X_data.columns
    feature_importance = dict(zip(feature_names, feature_importances))
    
    # Filter features with importance > 0
    important_features = {k: v for k, v in feature_importance.items() if v > 0}
    
    if not important_features:
        print(f" No features with importance > 0 found for {model_name}")
        return False
    
    # Select only important features
    X_important = X_data[list(important_features.keys())]
    
    # Create UMAP embedding
    reducer = umap.UMAP(random_state=42, n_components=3)
    embedding = reducer.fit_transform(X_important)
    
    # Create output directory if it doesn't exist
    os.makedirs(embedding_directory, exist_ok=True)
    
    sample_ids = X_data.index.values
    print(len(sample_ids.tolist()))

    meth_sample_ids = base_mvalue_df[["sample_id", "meth_sample_id"]]
    meth_sample_ids = meth_sample_ids.filter(pl.col("sample_id").is_in(sample_ids))["meth_sample_id"]
    #print(meth_sample_ids.shape)
    #print(meth_sample_ids.head())
    
    # Create output directory if it doesn't exist
    os.makedirs(embedding_directory, exist_ok=True)
    
    # Save as CSV
    embedding_df = pd.DataFrame(
        embedding, 
        columns=[f'umap_dim_{i+1}' for i in range(embedding.shape[1])]
    )
    embedding_df['sample_id'] = sample_ids
    embedding_df["meth_sample_id"] = meth_sample_ids

    embedding_df = embedding_df.rename(columns={'umap_dim_1': 'x', 'umap_dim_2': 'y', "umap_dim_3": "z"})

    csv_file = os.path.join(embedding_directory, f"{model_name}_umap_embedding.csv")
    embedding_df.to_csv(csv_file, index=False)
    
    # Save as pickle with additional metadata
    embedding_data = {
        'embedding': embedding,
        'sample_ids': sample_ids,
        'important_features': list(important_features.keys()),
        'feature_importance': {k: v for k, v in feature_importance.items() if k in important_features}
    }


    pickle_file = os.path.join(embedding_directory, f"{model_name}_umap_embedding.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(embedding_data, f)

    pickle_file = os.path.join(embedding_directory, f"{model_name}_embedding.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(embedding, f)
    
    print(f" Created UMAP embedding with {len(important_features)} important features")
    print(f" Saved embedding as CSV: {csv_file}")
    print(f" Saved embedding as pickle: {pickle_file}")
    
    return True











#file_paths = ["{}/{}".format(model_directory, file_name) for file_name in os.listdir(model_directory)]

#figureDataDirectory = f"/data/projects/classifiers/methylation/data/{freeze}/figureData/"
#figureDirectory = f"/data/projects/classifiers/methylation/data/{freeze}/figures/"

def main():
    mValueData = base_mvalue_df #pd.read_csv(dataFile, index_col=None)

    os.makedirs(full_model_directory, exist_ok=True)
    parameter_files = [f for f in os.listdir(model_parameter_directory) if f.endswith('.json')]

    for param_file in parameter_files:
        # Full path to parameter file
        param_path = os.path.join(model_parameter_directory, param_file)
        filename = os.path.basename(param_file)
        match = re.match(r'model-(.*?)\.json$', filename)
        model_name = match.group(1)

        # Output path for the rebuilt model
        output_path = os.path.join(full_model_directory, f"{model_name}.joblib")
        print(f"Processing model: {model_name}")
        
        tree = main_tree.find_node_by_name(model_name)
        sample_type = tree.get_samples_at_level(2)
        try:
            # Load model parameters
            with open(param_path, 'r') as f:
                model_info = json.load(f)
                model_module = __import__(model_info['model_module'], fromlist=[model_info['model_type']])
                ModelClass = getattr(model_module, model_info['model_type'])
                
                # Create model with saved parameters
                model = ModelClass(**model_info['parameters'])
                features = ["sample_id"] + model_info["feature_names"]
                
                features = [col for col in features if col in base_mvalue_df.columns]
                X_train = base_mvalue_df.select(features)
                X_train = X_train.filter(pl.col("sample_id").is_in(sample_type["sample_id"])).to_pandas()
                y_train = sample_type.filter(pl.col("sample_id").is_in(X_train["sample_id"])).to_pandas()

                X_train = X_train.set_index("sample_id")
                y_train = y_train.set_index("sample_id")
                model.fit(X_train, y_train["cancerType"])

                #print(X_train.shape)
                #print(X_train.loc[:, X_train.isnull().any()])

                if hasattr(model, 'feature_importances_'):
                    create_umap_embedding(X_train, model.feature_importances_, model_name, embedding_directory)
                else:
                    print(f" Model does not have feature_importances_ attribute")

                
                # Save the rebuilt model
                joblib.dump(model, output_path)
                print(f"  Successfully saved to: {output_path}")
        except Exception as e:
            print(f"  Error processing {model_name}: {str(e)}")

if __name__ == "__main__":
    main()



"""

            print(f"Shape of X_train: {X_train.shape}")
            print(f"Shape of y_train: {y_train.shape}")
            print(y_train["cancerType"].value_counts())
            print(X_train.head())

            # If you have training data, you could fit the model here
models = {}
sampleCohorts = {}
for file_name in file_paths:
    if file_name.endswith('.joblib')  and 'SVM' not in file_name:
        # this bit collects the model. Once we're done with that, we should be referencing tree_instance. 
        model = joblib.load(file_name)
        print(model)
        rfModel = model.named_steps["modelRefinement"]
        model_params = rfModel.get_params()
        features = rfModel.feature_names_in_

        # so here's a thing. This is pulling out the features that go into the RandomForest Classifier. One of the cross validation parameters though is max_features. 
        # max_features implies that maybe, the model isn't using all available features. So what happens when I just select the features with feature_importances_ greater than zero? 
        # if the feature importance is zero, then it's not used in the model surely. 
        # Testing this:
        importances = rfModel.feature_importances_
        featuredf = pl.DataFrame({
            "feature_name": features,
            "importance": importances
        })
        print(featuredf.shape)
        featuredf= featuredf.filter(featuredf["importance"]>0)
        print(featuredf.shape)
        features = featuredf["feature_name"].to_list()

        features = ["sampleId"] + list(features)

        print(f"There are {len(features)} for this model")

        modelName = os.path.basename(file_name)
        modelName = modelName[len("model-"):-len(".joblib")]
        treeFileName = f"diseaseTree-{modelName}.joblib"
        treeFile = os.path.join(tree_directory, treeFileName)
        
        # now I want all the samples, here I am referncing the main_tree
        sampleTree = joblib.load(treeFile)
        samples = sampleTree.get_samples_recursive()
        
        sampleIntersection = list(set(samples).intersection(set(mValueData.sampleId)))
        
        # here i pick up the main data set to run on. 


        X_train = mValueData[features]
        X_train = X_train[X_train["sampleId"].isin(sampleIntersection)]
        X_train = X_train.set_index("sampleId")

        #y_train = tree_instance.get_samples_at_level(2)
        print(sampleTree.name)
        y_train = sampleTree.get_samples_at_level(2)
        print("Y_train:")
        print(y_train.head())
        print("Non Leaf samples: ")       
        nonLeafSamples = set(X_train.index).difference(set(y_train['sampleId']))
        
        print(nonLeafSamples)
        print(y_train)
        for sample in nonLeafSamples:
            y_train.loc[len(y_train.index)] = [sample, sampleTree.name]

        y_train = y_train.set_index("sampleId")
        y_train = y_train.loc[X_train.index]

        print(len(X_train.index))
        print(len(y_train))
        
        final_model = RandomForestClassifier(**model_params)
        final_model.fit(X_train, y_train.cancerType)

        joblib.dump(final_model, f"{full_model_directory}/{sampleTree.name}_full_model.joblib")

        ############# Umap ####################
        reducer = umap.UMAP(n_components=3, random_state=42)
        #embedding = reducer.fit_transform(X_train)
        reducer.fit(X_train)
        #joblib.dump(embedding, f"{embedding_directory}/{sampleTree.name}_umap_embeddings.joblib")
        joblib.dump(reducer, f"{embedding_directory}/{sampleTree.name}_umap_embeddings.joblib")

        df = createPlotdf(X_train, y_train.cancerType, reducer.embedding_)
        df.to_csv(f"{figureDataDirectory}/{sampleTree.name}.csv")
        
        figure = makeUmapPlot3D(df, "cancerType" )
        figure.write_html(f"{figureDirectory}/{sampleTree.name}.html", full_html=False)

        ############### ROC ###################
        testSamples = sampleTree.validationSamples
        print(testSamples)
        X_test = mValueData[features]
        X_test = X_test[X_test["sampleId"].isin(testSamples)]
        X_test = X_test.set_index("sampleId")

        y_test = sampleTree.get_samples_at_level(2)
        y_test = y_test.set_index("sampleId")
        y_test = y_test.loc[X_train.index]


        print(y_test)

        roc = RocCurveDisplay.from_estimator(final_model, X_test, y_test, ax=ax, alpha=0.8)

        print(f"{sampleTree.name} done")
        #df = createPlotdf(X_train, y_train.cancerType, embedding)
        #print(df)
        #figure = makeUmapPlot3D(df, "cancerType" )

"""