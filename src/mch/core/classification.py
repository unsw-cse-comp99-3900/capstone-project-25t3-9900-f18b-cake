import logging
import warnings
import os
import sys

from typing import Optional, List

# Configure rpy2 logging to only show WARNING and above
logging.getLogger('rpy2').setLevel(logging.WARNING)

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Trying to unpickle estimator.*')

import pandas as pd
import polars as pl
import numpy as np

#from typedb.client import *
#from .diseaseTree import DiseaseTree
from mch.core.disease_tree import DiseaseTree
from mch.utils.model_utils import load_model
#from methylationMethods import getMatch
#from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


#import plotly.express as px
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots


import joblib

from mch.config.settings import (
    FREEZE_NUMBER, 
    FREEZE, 
    mvalue_df, 
    main_tree, 
    color_profiles, 
    disease_tree,
    model_directory,
    disease_tree,
    model_directory
)

freeze = FREEZE

def zero2_final_diagnosis(sample_id):
    z2 = find_sample(main_tree, sample_id)
    return z2

def find_sample(tree: DiseaseTree, sample_id: str, path: Optional[List[str]] = None) -> Optional[List[str]]:
        if path is None:
            path = []
        path.append(tree.name)
        #print(f"visiting: {tree.name}, current path: {path}")
        if sample_id in tree.samples:
            #print(f"found sample in {path}")
            return path
        for child in tree.children:
            result = find_sample(child, sample_id)
            if result is not None:
                return result
        path.pop()
        return None


def makePredictions(sample_id, modelName, freeze, full = True):
    if full:
        modelPath = f"/data/projects/classifiers/methylation/data/{freeze}/fullModels/{modelName}_full_model.joblib"
    else:
        modelPath = f"{model_directory}/model-{modelName}.joblib"

    if not os.path.exists(modelPath):
        return {}  # Return an empty dictionary if the model file doesn't exist

    with open(modelPath, "rb") as file:
        model = joblib.load(file)

    selectedFeatures = model.feature_names_in_

    # Check if sample_id is a list, if not, convert it into a list
    is_single_sample = isinstance(sample_id, str)
    sample_ids = [sample_id] if is_single_sample else sample_id

def zero2_final_diagnosis(sample_id):
    z2 = find_sample(main_tree, sample_id)
    return z2

def find_sample(tree: DiseaseTree, sample_id: str, path: Optional[List[str]] = None) -> Optional[List[str]]:
        if path is None:
            path = []
        path.append(tree.name)
        #print(f"visiting: {tree.name}, current path: {path}")
        if sample_id in tree.samples:
            #print(f"found sample in {path}")
            return path
        for child in tree.children:
            result = find_sample(child, sample_id)
            if result is not None:
                return result
        path.pop()
        return None


def makePredictions(sample_id, modelName, freeze):
    
    modelPath = f"{model_directory}/{modelName}_model.joblib"

    if not os.path.exists(modelPath):
        return {}  # Return an empty dictionary if the model file doesn't exist

    #with open(modelPath, "rb") as file:
    #    model = joblib.load(file)
    model = load_model(modelName)
    selectedFeatures = model.feature_names_in_
    print(modelPath)
    print(model)

    # Check if sample_id is a list, if not, convert it into a list
    is_single_sample = isinstance(sample_id, str)
    sample_ids = [sample_id] if is_single_sample else sample_id

    predictions = {}

    #progress_bar = tqdm(total=len(sample_ids), desc="Making predictions")

    for sid in sample_ids:
        singleSample = mvalue_df.filter(pl.col("sample_id") == sid)

        #singleSample = mvalue_df[mvalue_df["sample_id"] == sid]
        
        if singleSample.is_empty():
            continue  # Skip if the sample is not found in the dataset

        singleSample = singleSample[selectedFeatures]
        singleSample = singleSample.to_pandas()

        predictedClass = model.predict(singleSample)
        predictedProba = model.predict_proba(singleSample)[0]
        #:w
        # print(f"Sample: {sid}, Predicted class:{ predictedClass}, model: {modelName}, classes: {model.classes_}")
        predictedClassIndex = model.classes_.tolist().index(predictedClass)
        predictedClassProba = predictedProba[predictedClassIndex]
        predictedClass = predictedClass.tolist()[0]

        predictions[sid] = {
            modelName: {
                "prediction": predictedClass,
                "probability": float(predictedClassProba),
            }
        }

        # Recursively call makePredictions for the predicted class
        if predictedClass != modelName and predictedClass in model.classes_:
            sub_predictions = makePredictions(sid, predictedClass, freeze)
            predictions[sid].update(sub_predictions)
        #progress_bar.update(1)

    #progress_bar.close()

    return predictions if not is_single_sample else predictions[sample_id]


    """
    singleSample = mvalue_df[mvalue_df.sample_id == sample_id]
    singleSample = singleSample[selectedFeatures]

    #progress_bar = tqdm(total=len(sample_ids), desc="Making predictions")

    for sid in sample_ids:
        singleSample = mvalue_df.filter(pl.col("sample_id") == sid)

        #singleSample = mvalue_df[mvalue_df["sample_id"] == sid]
        
        if singleSample.is_empty():
            continue  # Skip if the sample is not found in the dataset

        singleSample = singleSample[selectedFeatures]
        singleSample = singleSample.to_pandas()

        predictedClass = model.predict(singleSample)
        predictedProba = model.predict_proba(singleSample)[0]
        #:w
        # print(f"Sample: {sid}, Predicted class:{ predictedClass}, model: {modelName}, classes: {model.classes_}")
        predictedClassIndex = model.classes_.tolist().index(predictedClass)
        predictedClassProba = predictedProba[predictedClassIndex]
        predictedClass = predictedClass.tolist()[0]

        predictions[sid] = {
            modelName: {
                "prediction": predictedClass,
                "probability": float(predictedClassProba),
            }
        }

        # Recursively call makePredictions for the predicted class
        if predictedClass != modelName and predictedClass in model.classes_:
            sub_predictions = makePredictions(sid, predictedClass, freeze)
            predictions[sid].update(sub_predictions)
        #progress_bar.update(1)

    #progress_bar.close()

    return predictions if not is_single_sample else predictions[sample_id]


    """
    singleSample = mvalue_df[mvalue_df.sample_id == sample_id]
    singleSample = singleSample[selectedFeatures]

    # print(type(singleSample))
    # print(singleSample.head())
    predictedClass = model.predict(singleSample)
    predictedProba = model.predict_proba(singleSample)[0]
    # print(type(singleSample))
    # print(singleSample.head())
    predictedClass = model.predict(singleSample)
    predictedProba = model.predict_proba(singleSample)[0]

    # print(f"predicted Class: {predictedClass}")
    # print(f"predicted probabilities: {predictedProba}")
    predictedClassIndex = model.classes_.tolist().index(predictedClass)
    predictedClassProba = predictedProba[predictedClassIndex]
    # print(f"predicted Class: {predictedClass}")
    # print(f"predicted probabilities: {predictedProba}")
    predictedClassIndex = model.classes_.tolist().index(predictedClass)
    predictedClassProba = predictedProba[predictedClassIndex]

    predictedClass = predictedClass.tolist()[0]
    predictedClass = predictedClass.tolist()[0]

    # print(predictedClassIndex)
    # print(predictedProba)
    # print(predictedClassIndex)
    # print(predictedProba)

    predictions[modelName] = {
        "prediction": predictedClass,
        "probability": float(predictedClassProba),
    }
    
    #predictions[modelName] = predictedClass.tolist()
    predictions[modelName] = {
        "prediction": predictedClass,
        "probability": float(predictedClassProba),
    }
    
    #predictions[modelName] = predictedClass.tolist()

    #predictions[predictedClass] = makePredictions(sample_id, predictedClass, freeze)
    #predictions[predictedClass] = makePredictions(sample_id, predictedClass)
    # Check if further predictions are needed
    if predictedClass in model.classes_:
        # Recursively call makePredictions for the predicted class
        sub_predictions = makePredictions(sample_id, predictedClass, freeze)
        # Merge the sub-predictions into the main predictions dictionary
        predictions.update(sub_predictions)
    #predictions[predictedClass] = makePredictions(sample_id, predictedClass, freeze)
    #predictions[predictedClass] = makePredictions(sample_id, predictedClass)
    # Check if further predictions are needed
    if predictedClass in model.classes_:
        # Recursively call makePredictions for the predicted class
        sub_predictions = makePredictions(sample_id, predictedClass, freeze)
        # Merge the sub-predictions into the main predictions dictionary
        predictions.update(sub_predictions)

    return predictions

    """
    """

def validate_cancer_type(cancerType):
    logging.info("starting validation")
    treeFile = f"/data/projects/classifiers/methylation/results/{freeze}/trees/diseaseTree-{cancerType}.joblib"
    logging.info("tree loaded")
    modelFile = f"/data/projects/classifiers/methylation/results/{freeze}/models/model-{cancerType}.joblib"
    
    # print(modelFile)
    with open(treeFile, "rb") as file:
        tree = joblib.load(file)
    with open(modelFile, "rb") as file:
        model = joblib.load(file)

    #rfModel = model.named_steps["modelGeneration"]
    rfModel = model.named_steps["modelRefinement"]
    #rfModel = model.named_steps["modelGeneration"]
    rfModel = model.named_steps["modelRefinement"]
    features = rfModel.feature_names_in_
    columns = ["sample_id"] + list(features)

    #print(tree.name)
    validationSamples = tree.validation_samples
    #print(f"Number of validation samples: {len(validationSamples)}")

    #testdf = mvalue_df.loc[mvalue_df["sample_id"].isin(validationSamples), columns]
    testdf = mvalue_df.filter(pl.col("sample_id").is_in(validationSamples)).select(columns)
    testdf = testdf.to_pandas()
    #testdf = mvalue_df.loc[mvalue_df["sample_id"].isin(validationSamples), columns]
    testdf = mvalue_df.filter(pl.col("sample_id").is_in(validationSamples)).select(columns)
    testdf = testdf.to_pandas()
    #print(f"validation dataframe shape: {testdf.shape}")

    testdf = testdf.set_index("sample_id")
    testdf = testdf[list(features)]
    if set(testdf.index) == set(tree.validation_samples):
        testdf = testdf.reindex(tree.validation_samples)


    predictions = rfModel.predict(testdf)
    actual = tree.get_samples_at_level(2)
    logging.info(f"type of actual adataframe{type(actual)}")
    logging.info(f"type of predictions adataframe{type(predictions)}")
    logging.info(f"type of actual adataframe{type(actual)}")
    logging.info(f"type of predictions adataframe{type(predictions)}")
    actual = actual[actual["sampleId"].isin(testdf.index)]
    actual = actual.set_index('sampleId')
    actual = actual.loc[testdf.index]
    actual = actual.cancerType
    #print(f"actual values: {actual}")
    #print(f"length of actual values: {len(actual)}")
    #print(f"length of predictions: {len(predictions)}")
    accuracy = classification_report(actual, predictions, output_dict=True, zero_division=0)
    
    
    #print(accuracy)
    #correctCalls = distributionOfCorrectCalls(cancerType, rfModel, tree).to_json()
    #correctCalls = distributionOfCorrectCalls(cancerType, rfModel, tree).to_json()
    #correctCalls = []
    summary = {
        "name": tree.name,
        "numberOfSamples": len(tree.validation_samples),
        "report": accuracy,
        #"correctCallDistribution": correctCalls,
        #"correctCallDistribution": correctCalls,
    }
    return summary

