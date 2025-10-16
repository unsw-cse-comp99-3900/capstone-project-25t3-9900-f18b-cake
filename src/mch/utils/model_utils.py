#import polars as pl
import joblib

from mch.config.settings import (
    FREEZE_DIR,
    model_directory
)


def load_model(model_name):
    modelFile = f"{model_directory}/{model_name}_collection.joblib"
    with open(modelFile, "rb") as file:
        collection = joblib.load(file)
    try:
        model = collection["calibratedModel"]
    except Exception as e:
        print(f"File for model {model_name} does not appear to contain a calibrated model: {e}")
    return model