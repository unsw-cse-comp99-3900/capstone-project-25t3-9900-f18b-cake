import pandas as pd
import polars as pl
import numpy as np

import joblib
from typing import List

import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import sci_palettes

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import label_binarize

from classificationMethods import makePredictions
from ..core.disease_tree import DiseaseTree



from ..config.settings import (
    FREEZE_NUMBER, 
    FREEZE, 
    mvalue_df, 
    main_tree, 
    color_profiles
)


def depth_first_traversal(tree: DiseaseTree, withSamples=True) -> List[DiseaseTree]:
    result = []

    def dfs(node: DiseaseTree, withSamples):
        if withSamples == True:
            if node.samples:  # Only include nodes with samples
                result.append(node)
        else:
            result.append(node)    
        for child in node.children:
            dfs(child, withSamples)
    
    dfs(tree, withSamples)
    return result


def getFinalPrediction(sample_id, modelName, freeze):
    modelPath = f"/data/projects/classifiers/methylation/data/{freeze}/fullModels/{modelName}_full_model.joblib"
    
    if not os.path.exists(modelPath):
        #print(f"Model path does not exist: {modelPath}")
        # print(f"Bottoming out : {sample_id}, {modelName}")
        return modelName  # Return None or a suitable default value
    
    with open(modelPath, "rb") as file:
        model = joblib.load(file)

    selectedFeatures = model.feature_names_in_

    # Ensure the sample exists in the DataFrame
    singleSample = mvalue_df[mvalue_df.sample_id == sample_id]
    if singleSample.empty:
        return None  # Return None or handle as appropriate

    singleSample = singleSample[selectedFeatures]
    # Perform prediction
    predictedClass = model.predict(singleSample)[0]
    predictedProba = model.predict_proba(singleSample)[0]

    predictedClassIndex = model.classes_.tolist().index(predictedClass)
    predictedClassProba = predictedProba[predictedClassIndex]

    # there are two classes called undifferentiated sarcoma. That's why. As it comes back up out of the recursion, it finds itself and goes back down. 
    # looks to be only in an old version of the oncotree, should sort itself in the next iteration?
    if predictedClass == modelName:
        #print(f"Why on earth is this happening. The predicted class is the same as the model name, which unchecked leads to infintite recursion")
        print(f"{sample_id}, {modelName}")

        return modelName
    
    # Check if the predicted class is in the model's classes
    if predictedClass in model.classes_:
        # Recursively call getFinalPrediction for the predicted class
        # print(f"mid-prediction: {sample_id}, {predictedClass}")
        return getFinalPrediction(sample_id, predictedClass, freeze)
    else:
        # Terminal condition: predicted class is not in model.classes_
        return predictedClass
        
def youden_index_per_class(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    classes = np.unique(y_true)
    youden_indices = {}
    
    for i, class_label in enumerate(classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden_index = sensitivity + specificity - 1
        youden_indices[class_label] = youden_index
    
    return youden_indices

def proportion_above_threshold(numbers, threshold):
    count_above = sum(1 for num in numbers if num > threshold)
    total_count = len(numbers)
    return count_above / total_count if total_count > 0 else 0


def calculate_metrics_per_class(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    classes = np.unique(y_true)
    
    metrics = []
    
    for i, class_label in enumerate(classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden_index = sensitivity + specificity - 1
        
        metrics.append({
            'Class': class_label,
            'Youden Index': youden_index,
            'Balanced Accuracy': sensitivity  # In multi-class, balanced accuracy per class is just the sensitivity
        })
    
    df = pd.DataFrame(metrics)
    df.set_index('Class', inplace=True)
    return df

def roc_plot(model_name):
    with open(f"/data/projects/classifiers/methylation/data/{FREEZE}/trees/diseaseTree-{model_name}.joblib", "rb") as file:
        tree = joblib.load(file)
    print(tree.name)
    with open(f"/data/projects/classifiers/methylation/data/{FREEZE}/models/model-{model_name}.joblib", "rb") as file:
        model = joblib.load(file)

    model = model.named_steps["modelGeneration"]

    testSamples = tree.validationSamples
    model_params = model.get_params()
    features = model.feature_names_in_

    features = ["sample_id"] + list(features)
    X_test = mvalue_df[features]
    print(X_test["sample_id"])
    print(testSamples)
    X_test = X_test[X_test["sample_id"].isin(testSamples)]

    print(X_test)
    X_test = X_test.set_index("sample_id")

    y_test = tree.get_samples_at_level(2)
    y_test = y_test.set_index("sampleId")
    print(y_test)
    y_test = y_test.loc[X_test.index]
    
    n_classes = len(model.classes_)
    y_score = model.predict_proba(X_test)

    #print(y_test)
    y_predict = model.predict(X_test)
    metrics_df = calculate_metrics_per_class(y_test, y_predict)
    print(metrics_df)

    #print(balanced_accuracy_score(y_test, y_predict))
    #youden_indices = youden_index_per_class(y_test, y_predict)
    #print(youden_indices)
    #print(proportion_above_threshold(np.max(y_score, axis=1), 0.76))
    #le = LabelEncoder()
    #y_encoded = le.fit_transform(y_test)
    #print(y_encoded)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #colors = pc.qualitative.Plotly[:n_classes]
    colors = (pc.qualitative.Plotly * (n_classes // len(pc.qualitative.Plotly) + 1))[:n_classes]
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    
    if n_classes == 2:
        fpr[0], tpr[0], _ = roc_curve(y_test_bin[:, 0], y_score[:, 1])
        roc_auc[0] = auc(fpr[0], tpr[0])
        y_bin_inverted = 1- y_test_bin
        fpr[1], tpr[1], _ = roc_curve(y_bin_inverted[:, 0], y_score[:, 0])
        roc_auc[1] = auc(fpr[1], tpr[1])
    else:
        # Multi-class classification
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


    fig = go.Figure()

    if n_classes == 2:
        fig.add_trace(go.Scatter(x=fpr[0], y=tpr[0], mode='lines', name=f'Class {model.classes_[0]} (AUC = {roc_auc[1]:.2f})', line=dict(color=colors[0], width=2)))
        fig.add_trace(go.Scatter(x=fpr[1], y=tpr[1], mode='lines', name=f'Class {model.classes_[1]} (AUC = {roc_auc[1]:.2f})', line=dict(color=colors[1], width=2)))
    else:
        for i in range(n_classes):
            fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', name=f'{model.classes_[i]} (AUC = {roc_auc[i]:.2f})', line=dict(color=colors[i], width=2)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='navy', width=2, dash='dash')))

    fig.update_layout(title=f'{tree.name} ROC Curve',
                    titlefont=dict(size=26),
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    xaxis=dict(range=[0, 1], constrain='domain'),
                    yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
                    width=800, height=500)
    fig.update_xaxes(
        titlefont=dict(size=26)
    )
    fig.update_yaxes(
        titlefont=dict(size=26)

    )
    return fig
 