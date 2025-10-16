import logging 
import os
import pandas as pd
import numpy as np
import uuid

#import methylize


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

import os
os.environ["R_ENABLE_JIT"] = "0"

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings


import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path

# becasue stupid version things. iteritems has been removed from later versions of python, is stll required by rpy2 conversions. 
pd.DataFrame.iteritems = pd.DataFrame.items

class DifferentialMethylation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tempFileLocation = "/tmp/"
        self.result_ = None  # to store the combined result
        self.design = None

    def fit(self, X: pd.DataFrame, y: pd.Series):

        #cancerTypes = y.cancerType.unique()
        #print(y)
        cancerTypes = y.unique()
        combined_result = []
        n=1
        #print(X.columns)
        for cancerType in cancerTypes:
            print(f"probe identification for: {cancerType}, {n} of {len(cancerTypes)} cancer types")
            n = n+1
            condition = pd.Series(np.where(y == cancerType, cancerType, 'otherCancerType'))
            results = self.runDifferentialMethylation(X,condition)
            combined_result = combined_result + results
        #combined_result.append(results)
        combined_result = list(set(combined_result))
        #print(combined_result.head())
        print(f"Number of probes identified from differential methylation analyses: {len(combined_result)}")
        self.result = combined_result
        return self
        
    def transform(self, X):
        if self.result is None:
            raise NotFittedError("Transformer must be fitted before transforming data.")
        # Transform logic if needed
        print("transforming dataset to contain only differentially methylated features")
        #self.result = self.result.sort()
        #rint(X.columns[0:20])
        X.columns = X.columns.astype(str)
        return X[self.result]
        #return X.iloc[:,1:500]


    #def runDifferentialMethylation(self, X: pd.DataFrame, y:pd.Series):
    def runDifferentialMethylation(self, X: pd.DataFrame, y: pd.Series):
        print("Running analysis")

        # Prepare data
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis="columns")

        # Create design DataFrame and ensure sampleId is string type
        design = pd.DataFrame({
            "sampleId": pd.Series(X.index, dtype='string'),
            "cancerType": y.values
        })

        # Convert DataFrames to Arrow format
        arrowData = self.convert_to_arrow(X)
        arrowDesign = self.convert_to_arrow(design)

        # Save to temporary files
        dataFilename = self.save_to_file(arrowData)
        designFilename = self.save_to_file(arrowDesign)

        # Run R script
        result = self.run_r_script(dataFilename, designFilename)

        # Clean up temporary files
        os.remove(dataFilename)
        os.remove(designFilename)

        return result

    def run_r_script(self, data_filename, design_filename):
        r_script_path = Path(__file__).resolve().parent.parent / 'scripts' / 'differentialMethylation.R'
        r_environment = robjects.r('options(rpy2_quiet = TRUE)')  # This suppresses R startup messages

        r_environment.source(str(r_script_path))
        runDM = r_environment["runDM"]
        
        fit = runDM(data_filename, design_filename)

        with (robjects.default_converter + pandas2ri.converter).context():
            dm_probes = robjects.conversion.get_conversion().rpy2py(fit)

        return dm_probes.index.to_list()

    def convert_to_arrow(self, df):
        schema_fields = []
        for col in df.columns:
            if col == 'sampleId':
                schema_fields.append((col, pa.string()))
            elif col == 'cancerType':
                schema_fields.append((col, pa.string()))
            elif col == "Name":
                schema_fields.append((col, pa.string()))
            else:
                schema_fields.append((col, pa.float64()))
        
        schema = pa.schema(schema_fields)
        #data["sampleId"] = data["sampleId"].astype(str)
        arrowFrame = pa.Table.from_pandas(df, schema = schema )
        return arrowFrame

    def save_to_file(self, arrowFrame):
        # generate some random filenames 
        location = self.tempFileLocation
        randomElement = str(uuid.uuid4())
        fileName = os.path.join(location, randomElement + ".feather")
        feather.write_feather(arrowFrame, fileName)
        return fileName