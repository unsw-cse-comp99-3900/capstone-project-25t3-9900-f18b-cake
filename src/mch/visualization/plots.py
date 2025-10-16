import joblib
import logging
import os

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, classification_report
#from mch.core.disease_tree import DiseaseTree
from mch.core.diseaseTree import DiseaseTree
# can get rid of this in the next iteration - storing models will be much simpler hopefully.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from differentialMethylationClassifier import DifferentialMethylation
#from mch.analysis.differentialMethylationClassifier import DifferentialMethylation

import numpy as np
import pandas as pd
import polars as pl

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mch.db.database import Database
from mch.utils.model_utils import load_model

from ..config.settings import (
    FREEZE_NUMBER, 
    FREEZE, 
    mvalue_df, 
    main_tree, 
    color_profiles,
    embedding_directory, 
    model_directory
)

from mch.utils.logging_utils import setup_logging, load_config


def project_missing_samples(umapReducer, df, cohort_data, tree_samples):
    """
    Find samples in tree_samples that are not in df, project them onto the UMAP embedding,
    and add them to the DataFrame.
    
    Parameters:
    -----------
    umapReducer : UMAP object
        The fitted UMAP object used to create the original embedding.
    df : polars.DataFrame
        DataFrame containing the existing embedding coordinates and sample IDs.
    cohort_data : polars.DataFrame
        DataFrame containing the raw data for all samples.
    tree_samples : list
        List of sample IDs to check against df.
        
    Returns:
    --------
    polars.DataFrame
        Updated DataFrame with the new samples added.
    """
    # Convert sample_id column to a Python list
    existing_samples = df["sample_id"].to_list()
    
    # Find samples that are in tree_samples but not in df
    missing_samples = [sample for sample in tree_samples if sample not in existing_samples]
    
    if not missing_samples:
        print("All samples from tree_samples are already in the DataFrame.")
        return df
    
    print(f"Found {len(missing_samples)} samples to project onto the UMAP embedding.")
    
    # Initialize list to store new rows
    new_rows = []
    
    # Process each missing sample
    for sample_id in missing_samples:
        # Extract the sample data from cohort data
        sample_data = cohort_data.filter(pl.col("sample_id") == sample_id)
        
        if len(sample_data) == 0:
            print(f"Warning: Sample {sample_id} not found in cohort_data, skipping.")
            continue
        
        # Extract features (all columns except sample_id)

        feature_columns = [col for col in cohort_data.columns if col != "sample_id" and col != "meth_sample_id"]
        print(len(feature_columns))
        
        # Convert to numpy array for UMAP transform
        sample_features = sample_data.select(feature_columns).to_numpy()
        sample_features = impute_sample_values(cohort_data, sample_id)
        
        # Project the sample onto the UMAP embedding
        projection = umapReducer.transform(sample_features)
        
        # Create a new row for each sample
        for i in range(len(sample_data)):
            new_row = {
                "sample_id": sample_id,
                "x": projection[i, 0],
                "y": projection[i, 1]
            }
            
            # If the original embedding was 3D, add the z coordinate
            if "z" in df.columns:
                new_row["z"] = projection[i, 2]
                
            new_rows.append(new_row)
    
    # Convert new rows to a Polars DataFrame
    if new_rows:
        new_rows_df = pl.DataFrame(new_rows)
        
        # Append the new rows to the existing DataFrame
        updated_df = pl.concat([df, new_rows_df])
        return updated_df
    else:
        print("No new samples were projected (they may be missing from cohort data).")
        return df


def create_plot_df(model_name, sample_id=None, prediction=None,  cohort_tree=None):
    setup_logging()
    logging.info(f"prediction: {prediction}")
    """
    Okay, I think I need to re-write all of this. Preferably using polars. 
    And just to make things extra special dandy, I should add the sample_ids' into the embedding when I load it to file, 
    which is a data genreation thing. 

    I don't know that I've brought the umap generation across yet. le sigh. 
    """

    """Creates a dataframe for UMAP plotting."""
    # find the relevant node in the oncotree, and the samples in it. 
    # subset out the relevant samples from the main dataset - which is a reduced version of the raw data, containing only features/probes used
    # across all model. 
    # get the relevant embedding from the model and create a dataframe from it
    
    #meth_sample_id = mvalue_df.filter(mvalue_df["sample_id"]==sample_id).select("meth_sample_id")

    db = Database()

    if sample_id:
        #meth_sample_id = mvalue_df.filter(pl.col("sample_id") == sample_id)["meth_sample_id"][0]
        meth_sample_id = db.convert_sample_ids([sample_id], "wgs_id", "methylation_id")
    else:
        meth_sample_id = "" 

    tree_samples = main_tree.find_node_by_name(model_name).get_samples_recursive()
    #validation_samples = main_tree.find_node_by_name(model_name).get_samples_recursive()
    #tree_samples = [sample for sample in tree_samples if sample not in validation_samples]

    if isinstance(sample_id, list):
        tree_samples.extend(sample_id)
    elif isinstance(sample_id, str):
        tree_samples.append(sample_id)
        
    logging.info(f"number of samples from the main tree for this cohort:{len(tree_samples)}")
    cohort_data = mvalue_df.filter(pl.col("sample_id").is_in(tree_samples))
    
    logging.info(f"Shape of cohort data: {cohort_data.shape}")
    print(f"{embedding_directory}/{model_name}_umap_embedding.joblib")
    with open(f"{embedding_directory}/{model_name}_umap_embedding.joblib", "rb") as file:
        umapReducer = joblib.load(file)
        embedding = umapReducer["embedding"]#.embedding_
        
    with open(f"{embedding_directory}/{model_name}_umap_embedding.csv", "rb") as file:
        df = pl.read_csv(file)
    
    logging.info(f"plot dataframe creation for {sample_id}, dataframe shape: {df.shape}")

    sample_diagnoses = main_tree.find_node_by_name(model_name).get_samples_at_level(2)
    logging.info("sample_diagnoses found")

    if not isinstance(sample_diagnoses, pl.DataFrame):
        sample_diagnoses = pl.from_pandas(sample_diagnoses)
        sample_diagnoses = sample_diagnoses.rename({"sampleId": "sample_id"})
    
    logging.info("instance test done")
    df = df.join(sample_diagnoses, on="sample_id", how="inner")

    logging.info("datframe joined")

    if sample_id is not None and sample_id not in df["sample_id"]:
        logging.info(f"projecting new data onto embedding")
        #logging.info(f"mvalue_df shape: {mvalue_df.shape}")
        
        exists = sample_id in mvalue_df["sample_id"].to_list()
        #print(f"Sample ID '{sample_id}' exists: {exists}")

        sample_data = mvalue_df.filter(pl.col("sample_id") == sample_id)

        logging.info(f"sample data found {sample_data}")

        model = load_model(model_name)
        #model_path = f"{model_directory}/model-{model_name}.joblib"
        #if os.path.exists(model_path):
        #    with open(model_path, "rb") as model_file:
        #        model = joblib.load(model_file)
        #    model = model.named_steps["modelRefinement"]
        #    feature_names = model.feature_names_in_
        #    feature_importances = model.feature_importances_

        #feature_importance = dict(zip(feature_names, feature_importances))
        #important_features = {k: v for k, v in feature_importance.items() if v > 0}
        #sample_data = sample_data[list(important_features.keys())]

        #logging.info(f"Finding features")
        #feature_Imporatnces = featuredf.filter(featuredf["importance"]>0)
        #features = featuredf["feature_name"].to_list()
    
        #sample_data = sample_data[features]
        print(type(sample_data))
        #sample_data = sample_data.to_numpy()
        sample_data = sample_data.drop('sample_id').to_numpy()

        print(type(sample_data))
        print(type(umapReducer))
        if np.any(np.isnan(sample_data)):
            imputer = SimpleImputer(strategy='mean')
            imputed_data = imputer.fit_transform(sample_data)
            #imputed_data = impute_sample_values(mValuedf=mvalue_df, sample_id=sample_id)
            sample_embedding = embedding.transform(imputed_data)
        else:
            sample_embedding = embedding.transform(sample_data)

        sample_embedding = sample_embedding[0]
        dfLength = len(df["sample_id"])
        new_row = pl.DataFrame({
            "x": [sample_embedding[0]],
            "y": [sample_embedding[1]],
            "z": [sample_embedding[2]],
            "sample_id": [sample_id],
            "meth_sample_id": [meth_sample_id],
            "cancerType": prediction,
        }).with_columns([
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("z").cast(pl.Float64),
        ])
        print(new_row)
        #print(df.head())
        df = pl.concat([df, new_row], how="vertical")

        #df.loc[len(df["sample_id"])] = [sample_embedding[0], sample_embedding[1], sample_embedding[2], sample_id, meth_sample_id, prediction] 
        logging.info(f"Final dataframe shape: {df.shape}")

        #with open(f"/data/projects/classifiers/methylation/results/{FREEZE}/models/model-{diagnosis}.joblib", "rb") as file:
        #    model = joblib.load(file)
        
        #model = model.named_steps["modelGeneration"]
    #df = df.filter(df["sample_id"].is_in(tree_samples))

    #df = project_missing_samples(umapReducer, df, cohort_data, tree_samples)

    #print(df.loc[df.sample_id == sample_id])
    #df = df.rename({"column_0":"x", "column_1": "y", "column_2": "z"})
    #df = df.with_columns(cohort_data.select("sample_id").get_column("sample_id").alias("sample_id"))
    
    #dflen = len(df)
    #logging.info(f"Dataframe shape: {df.shape}")
    # Convert the sample diagnoses to a polars DataFrame if it isn't already
    #cohort_tree = main_tree.find_node_by_name(diagnosis)

    #sample_diagnoses = cohort_tree.get_samples_at_level(2)
    #print(type(sample_diagnoses))
    #print(type(df))
    #if not isinstance(sample_diagnoses, pl.DataFrame):
    #    sample_diagnoses = pl.from_pandas(sample_diagnoses)
    #    sample_diagnoses = sample_diagnoses.rename({"sampleId": "sample_id"})
        
    #print(f"plot dataframe creation debug 1: {df['sample_id'].dtype}, {diagnosis}")
    
    #logging.info(type(df))
    #logging.info(type(sample_diagnoses))
    # Join the DataFrames
    #df = df.join(sample_diagnoses, on="sample_id", how="inner")
    
    # Filter for the specific sample if provided
    #if sample_id:
    #    test_df = df.filter(pl.col("sample_id") == sample_id)
    logging.info("returning plot dataframe")   
    return df



    """
    if sample_id is not None and sample_id not in df["sample_id"].values:
        print(f"Sample id is: {sample_id}, i.e. not none")
        #print(f"but also not in df['sample_id'] though, which is odd, becuase:")
        #print(df[df["sample_id"]==sample_id])
        #meth_sample_id = mValuedf.filter(mValuedf["sample_id"]==sample_id).select("meth_sample_id")

        # print(mValuedf[mValuedf["sample_id"]==sample_id]["meth_sample_id"])
        meth_sample_id = mValuedf[mValuedf["sample_id"]==sample_id]["meth_sample_id"].values[0]

        # print(f"Single sample meth sample id: {meth_sample_id}")
        sampleData = mValuedf[mValuedf["sample_id"]==sample_id]
               
        with open(f"/data/projects/classifiers/methylation/results/{freeze}/models/model-{diagnosis}.joblib", "rb") as file:
            model = joblib.load(file)
        model = model.named_steps["modelGeneration"]
        features = model.feature_names_in_
        importances = model.feature_importances_
        featuredf = pl.DataFrame({
            "feature_name": features,
            "importance": importances
        })
        
        featuredf= featuredf.filter(featuredf["importance"]>0)
        features = featuredf["feature_name"].to_list()

        #print(sampleData)

        #sampleEmbedding = sampleEmbedding[0]
        #dfLength = len(df["sample_id"])
        #print(f"Data frame length: {dfLength}")
        #print(type(df))
        #df.loc[len(df["sample_id"])] = [sampleEmbedding[0], sampleEmbedding[1], sampleEmbedding[2], sample_id, meth_sample_id, prediction] 
        print(mValuedf.shape, type(mValuedf))
        print(sampleData.shape, type(sampleData))
        print((sampleData.head()))

        sampleData = impute_sample_values(mValuedf, sample_id)
        sampleData = sampleData[features]

        print(sampleData.shape, type(sampleData))
        print(sampleData.head())


        if not np.any(np.isnan(sampleData)):
            sampleEmbedding = umapReducer.transform(sampleData)
            sampleEmbedding = sampleEmbedding[0]
            dfLength = len(df["sample_id"])
            #print(f"Data frame length: {dfLength}")
            #print(type(df))

            df.loc[len(df["sample_id"])] = [sampleEmbedding[0], sampleEmbedding[1], sampleEmbedding[2], sample_id, meth_sample_id, prediction] 
           
    # additional hovertext data needs to specified here: 
    # df["final_diagnosis"] = adata_subset.obs["final_diagnosis"]
    print(f"returning {diagnosis} dataframe for {sample_id}")
    #print(df[df["sample_id"] == sample_id] )#df["meth_sample_id"]="wibble"
    """
    return df



def makeUmapPlot2D(df, colorField, sample_id=None, cancerType=None):
    colorSet = color_profiles[cancerType]
    df["size"] = [5] * len(df["x"])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Default plotly colors
    
    l2_samples = pl.read_csv("/Users/bcurran/workspace/landscape2/data/Cohort_metadata_14012025.moremeta.txt", separator="\t", null_values=["NULL"])
    l2_samples = l2_samples["sample_id"].to_list()
    logging.info(type(l2_samples))
    #logging.info(l2_samples)
    logging.info(type(df["sample_id"]))
    df = df[df["sample_id"].isin(l2_samples)]

    fig = px.scatter(df, x="x", y="y",
                            color=df[colorField], 
                            color_discrete_map=colorSet,
                            labels={'color': 'Diagnosis'}, 
                            hover_data=df[["sample_id",  "meth_sample_id" ]],
                            #hover_data=df[["sample_id", "patient_id", "z2_final_diagnosis", "z2_cancer_type", "z2_cancer_subtype", "purity"]],
                            width = 1000,
                            height = 600,
                            size = "size",
                            opacity = 1,
                            )
    fig.update_traces(marker=dict(
        size=8,
        line=dict(width=0)  # Setting the border width to 0
        )
    )    
         
    

    for species in df[colorField].unique(): 
        species_data = df[df[colorField] == species]
        centroid_x = species_data['x'].mean()
        centroid_y = species_data['y'].mean()

        y_range = species_data['y'].max() - species_data['y'].min()
        offset_y = 1
        offset_x = 1
        if species == "Peripheral nerve sheath tumour, Sarcoma":
            offset_y = -0.8
            offset_x = 7.5

        if species == "Sarcoma other":
            offset_y = 0.5
            offset_x = 3

        if species == "Kidney tumour, Sarcoma":
            offset_y = -0.5
            offset_x = -3.5

        if species == "Liver tumour, Sarcoma":
            offset_y = 0 #-0.5
            offset_x = -5 #-1.4

        if species == "Pineal tumour":
            offset_y = 0 #-0.5
            offset_x = 3 #-1.4

        if species == "CNS embryonal tumour":
            offset_y = -1 #-0.5
            offset_x = -6 #-1.4
        
        if species == "Paediatric-type diffuse glioma":
            offset_y = 2 #-0.5
            offset_x = 6 #-1.4
        
        if species == "Adult-type diffuse glioma":
            offset_y = 1.5 #-1.4
            offset_x = -5 #-1.4
        
        if species == "Choroid plexus tumour":
            offset_y = -1 #-0.5
            offset_x = -1 #-1.4
        
        if species == "Meningioma":
            offset_y = -0.8 #-0.5
            offset_x = 1.7 #-1.4
        
        if species == "Glioneuronal and neuronal tumour":
            offset_y = -1.8 #-0.5
            offset_x = -4 #-1.4

        if species == "Circumscribed astrocytic glioma":
            offset_y = 2 #-0.5
            offset_x = 6 #-1.4

        if species == "CNS other":
            #offset_y = -1.3 #-0.5
            offset_x = 0.5 #-1.4

        if species == "Melanocytic tumour, Solid":
            offset_y = -1.3 #-0.5
            offset_x = -0.5 #-1.4

        if species == "Solid other":
            offset_y = -0.5 #-0.5
            offset_x = -5 #-1.

        if species == "Adrenal cortical tumour":
            offset_y = 0.5 #-0.5
            offset_x = -6.3 #-1.
        
        if species == "Ependymoma":
            offset_y = 0 #-0.5
            offset_x = 3 #-1.

        if species == "Thyroid tumour":
            offset_y = 0 #-0.5
            offset_x = 4.5 #-1.
        
        if species == "Sarcoma other":
            offset_y = 1 #-0.5
            offset_x = 6 #-1.

        if species == "Germ cell tumour, Solid":
            offset_y = -0.50 #-0.5
            offset_x = 7 #-1.

        if species == "CNS mesenchymal tumour":
            offset_y = 2 #-0.5
            offset_x = -2 #-1.

        

        species_color = colorSet.get(species, '#1f77b4')  # Default color if species not in colorSet

        if species not in ["Kidney tumour, Sarcoma", "Choroid plexus tumour", "Germ cell tumour, CNS", "Sellar tumour", "CNS other", "Carcinoma"]:
            fig.add_annotation(
                x=centroid_x + offset_x, 
                y=centroid_y + offset_y,
                text=species,
                showarrow=False,
                font=dict(size=18, color='white', family='Arial'),  # White text
                bgcolor=species_color,  # Use the color from colorSet
                bordercolor=species_color,  # Same color border (or set to different color if you want)
                borderwidth=0
            )
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)

    fig.update_layout(
        showlegend = False,
        width = 1200,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig

def makeUmapPlot3D(df, colorField, sample_id=None, cancerType=None):
    #print(f"Making Umap plot for {sample_id}")
    # print(df.head())
    #print(df[df["sample_id"]==sample_id])
    #print(f"sample_id found in makeUmapPlot {sample_id}")
    
    df = df.to_pandas()
    colorSet = color_profiles[cancerType]
    
    df["size"] = [2] * len(df["x"])
    
    fig = px.scatter_3d(df, x="x", y="y", z="z",
                        color=df[colorField], 
                        color_discrete_map=colorSet,
                        labels={'color': 'Diagnosis'}, 
                        hover_data=df[["sample_id",  "meth_sample_id" ]],
                        #hover_data=df[["sample_id", "patient_id", "z2_final_diagnosis", "z2_cancer_type", "z2_cancer_subtype", "purity"]],
                        width = 1000,
                        height = 600,
                        size = "size",
                        opacity = 1,
                        )
    fig.update_traces(marker=dict(
                        size=2,
                        line=dict(width=0)  # Setting the border width to 0
                        )
                     )           

    fig.update_layout(
        legend=dict(
            font=dict(
                size=18,
            ),
            traceorder="normal",
            orientation="v",
            itemsizing="constant",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.5,
            bgcolor="#fff",
            valign="middle",
            #font=dict(size=11),
            
            title=dict(text=colorField, font=dict(size=11)),
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    if sample_id is not None:
        # print(df[df["sample_id"]==sample_id])
        #print(f"adding sample_id trace")
        meth_sample_id = df.loc[df["sample_id"] == sample_id, "meth_sample_id"],
        #print(f"meth sample id: {meth_sample_id}")
        #print(df.loc[df.sample_id == sample_id])
        fig.add_trace(go.Scatter3d(
            x=df.loc[df.sample_id == sample_id, "x"],
            y=df.loc[df.sample_id == sample_id, "y"],
            z=df.loc[df.sample_id == sample_id, "z"],
            mode='markers',
            hovertemplate=(
                f"Sample ID: {sample_id} <br>"
                f"Meth Sample ID: {meth_sample_id}<br>"
            ),
            
            marker=dict(
                size=4,
                color='black'
            ),
            #name=f'{sample_id}'
            name=sample_id
        ))
        fig.update_layout(width=1200)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        
        logging.info(f"successful plot creatioon for {sample_id} {cancerType}")

    umap_2d_fig = makeUmapPlot2D(df, colorField, cancerType=cancerType)
    umap_2d_fig.write_image(f"/Users/bcurran/workspace/methylationclassifier/results/freeze0525/umap_plots/umap_{cancerType}.pdf", width=800, height=800)
    logging.info(f"2d umap for {cancerType} should have been created")
    logging.info(f"2d umap for {cancerType} should have ... ")

    return fig.to_json()




def impute_sample_values(mValuedf, sample_id, n_neighbors=5):
    """
    Impute missing values for a specific sample in a dataframe using KNN imputation.
    
    Parameters:
    -----------
    mValuedf : pandas.DataFrame
        The input dataframe containing all samples
    sample_id : str
        The sample_id to identify the row to impute
    n_neighbors : int, optional (default=5)
        Number of neighbors to use for KNN imputation
        
    Returns:
    --------
    pandas.Series
        The imputed row with all numeric values filled
    """
    from sklearn.impute import KNNImputer
    
    logging.info(f"imputing values for {sample_id} for plot creation")
    # Identify numeric columns (exclude string columns)
    numeric_cols = [col for col in mValuedf.columns if col not in ["sample_id", "meth_sample_id"]]
    
    # Get numeric data for imputation
    numeric_data = mValuedf[numeric_cols]
    numeric_data_cleaned = numeric_data.with_columns([
        pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)
        for col in numeric_data.columns
    ])    

    numeric_data_np = numeric_data_cleaned.to_numpy()
    
    # Create and fit the imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    #imputed_data = imputer.fit_transform(numeric_data)
    imputed_np = imputer.fit_transform(numeric_data_np)
    imputed_df = pl.DataFrame(imputed_np, schema=numeric_cols)

    
    # Create a dataframe with imputed values
    #imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols)
    # Get the index of the target sample
    #sample_idx = mValuedf[mValuedf["sample_id"]==sample_id].index[0]
    sample_idx = mValuedf.select("sample_id").to_series().to_list().index(sample_id)
    
    # Get the imputed values for the specific sample
    #imputed_sample = imputed_df.iloc[sample_idx]
    imputed_row = imputed_df[sample_idx]

    # Create a complete row including string columns
    #complete_row = mValuedf.iloc[sample_idx].copy()
    #complete_row = mValuedf.iloc[[sample_idx]].copy()  # Note the double brackets to keep as DataFrame
    #complete_row[numeric_cols] = imputed_sample
    non_numeric = mValuedf.select(["sample_id", "meth_sample_id"])[sample_idx]
    complete_row = non_numeric.hstack(imputed_row)
    
    return complete_row



def distributionOfCorrectCalls(cancerType: str, model: RandomForestClassifier, tree: DiseaseTree):

    # print(f"Distribution type for: {cancerType}")
    # subset to the validation samples
    samples = tree.validationSamples
    y_test = tree.get_samples_at_level(2)
    y_test = y_test[y_test["sampleId"].isin(samples)]

    features = model.feature_names_in_
    columns = ["sample_id"] + list(features)
    
    X_test = mvalue_df.loc[mvalue_df["sample_id"].isin(y_test["sampleId"]), columns]
    X_test = X_test.set_index("sample_id")
    X_test = X_test[list(features)]

    y_test = y_test.set_index("sampleId")
    if set(X_test.index) == set(y_test.index):
        X_test = X_test.reindex(y_test.index)

    # Find predictions and their corresponding probabilities
    y_pred = model.predict(X_test)
    y_true = y_test.cancerType
    probabilities = model.predict_proba(X_test)

    # print(type(probabilities))
    probs = np.max(probabilities, axis=1)
    # print(type(probs))
    # print(probs)

    predictiondf = pd.DataFrame({"Actual": y_true, "Class": y_pred, "Probability": probs})
    df = predictiondf[predictiondf['Actual'] == predictiondf['Class']]
    df = df.drop(columns=["Actual"])

    # print(f"Correct calls: {df.shape}")

    #correct_predictions = y_pred == y_true
    #print(f"Correct Predictions ({correct_predictions.shape}): ")
    #print(correct_predictions)

    #correct_indices = np.where(correct_predictions)[0]
    #print(f"Number of correct indices: {len(correct_indices)}")
    #orrect_labels = y_pred[correct_indices]
    #print(f"Number of correct labels: {len(correct_labels)}")
    #print(f"Call Probabilities: ")
    #print(probs)
    #classes = model.classes_
    #labels = np.argmax(probabilities, axis=1)
    #labels = [classes[i] for i in labels]
    #print(accuracy_score(y_true, labels))
        


    
    #correct_probabilities = probabilities[correct_indices, correct_labels]

    #results = np.column_stack((correct_indices, correct_labels, correct_probabilities))
    #print(results)

    #correct_pred_probs = pred_probs[correct_predictions]
    #correct_pred_labels = pred_labels[correct_predictions]
    #probability = np.max(correct_pred_probs, axis=1)

    #df = pd.DataFrame({"Class": correct_labels, "Probability": correct_probabilities})
    fig = go.Figure()

    colorSet = color_profiles[cancerType]

    # Generate a ridgeline plot for each class
    classes = df['Class'].unique()
    for i, cls in enumerate(classes):
        #print(cls)
        class_data = df[df['Class'] == cls]['Probability']
        fig.add_trace(go.Violin(x=class_data, y=[cls]*len(class_data), line_color=colorSet[cls], side='positive', width=0.8, spanmode = 'hard'))

    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(
        title='Distribution of Predicted Probabilities for Correct Predictions by Class',
        xaxis_title='Predicted Probability',
        yaxis_title='Class',
        xaxis=dict(range=[0,1]),
        violingap=0, 
        violingroupgap=0, 
        violinmode='overlay',
    )
  

    return fig
