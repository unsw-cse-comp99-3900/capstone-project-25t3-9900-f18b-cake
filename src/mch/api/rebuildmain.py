from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import polars as pl

from epicV1V2Processing import epicFilter

import os
import re
from io import BytesIO

from mnpMethods import mnpCoverage, mnpReportable, mnpPrediction
from classificationMethods import cohortSampleList, createPlotdf, makeUmapPlot3D, makePredictions, validateCancerType, zero2FinalDiagnosis
#from methylationMethods import createPlotdf

import pandas as pd
import random

app = FastAPI()

sampleUploadDirectory = "/data/projects/classifiers/methylation/data/uploadedSamples"
freeze = "freeze0824"
freezeNumber = "0824"

def findMethylationFileName(filePath: str)-> str:
    fileName = os.path.basename(filePath)
    pattern = r'^.*?_T_M'
    match = re.search(pattern, fileName)
    if match:
        return match.group(0)
    else:
        return fileName
    

def addNewRow(filePath: str, df: pl.DataFrame) -> pl.DataFrame:
    # Read the file into a dataframe
    newSample = pl.read_csv(filePath, separator="\t", ignore_errors=True)
    sampleId = findMethylationFileName(filePath)

    # print(newSample.shape)
    features = mValuedf.columns

    # print([i for i, x in enumerate(features) if features.count(x) > 1])

    newSample = newSample.filter(newSample["EPICv1_Loci"].is_in(features))
    newSample = newSample[["EPICv1_Loci", "MValue"]]
    ### Change this. This needs to have the epic filter applied. Which should be done when the file is read in. 
    #duplicates = newSample['EPICv1_Loci'].value_counts().filter(pl.col('count') > 1).select('EPICv1_Loci')
    #duplicates = [locus for locus in set(features) if features.count(locus) > 1]
    missingFeatures = list(set(features).difference(set(newSample['EPICv1_Loci'].to_list())))
    if len(missingFeatures) >0:
        print(f"There are {len(missingFeatures)} not present in this sample, which may result in the inability to make some predictions")
        print(missingFeatures)
    
    newSample = newSample.unique(('EPICv1_Loci'), keep='first', maintain_order=True)
 
    newSample = dict(zip(newSample['EPICv1_Loci'], newSample['MValue']))
    newRow = pl.DataFrame([newSample])
    newRow = newRow.with_columns(pl.lit(sampleId).alias('sample_id'))

    for col in mValuedf.columns:
        if col not in newRow.columns:
            #print(col)
            newRow = newRow.with_columns(pl.lit(None).alias(col))
    newRow = newRow.select(mValuedf.columns)       

    df = pl.concat([mValuedf, newRow], how='vertical')
    mValueWithNewSamplesFile = f"/data/projects/classifiers/methylation/data/{freeze}/featureValuesWithNewSamples{freezeNumber}.csv"
    df.write_csv(mValueWithNewSamplesFile)

    
    return df

mValueFile = f"/data/projects/classifiers/methylation/data/{freeze}/featureValues{freezeNumber}.csv"
mValuedf = pl.read_csv(mValueFile)

mValuedf = mValuedf.rename({'sampleId': 'sample_id'})

mValueFile = f"/data/projects/classifiers/methylation/data/{freeze}/featureValuesWithNewSamples{freezeNumber}.csv"
mValuedfUpToDate = pl.read_csv(mValueFile)


#files = [f for f in os.listdir(sampleUploadDirectory) if os.path.isfile(os.path.join(sampleUploadDirectory, f))]
#
#for file in files:
#    if file != 'previouslyUploadedFiles.csv':
#        sampleName = findMethylationFileName(file)
#        #print(mValuedf["sample_id"].str.contains(sampleName).any())
#        #print(sampleName)
#        if mValuedf["sample_id"].str.contains(sampleName).any():
#           print("alreadyLoaded")
#        else: 
#            #print("loading additonal sample")
#            file_path = os.path.join(sampleUploadDirectory, file)
#            mValuedf = addNewRow(file_path, mValuedf)

@app.get("/sample")
async def sample(sample_id, newSample=""):
    #mnpCoverage = mnpCoverage(sample_id)
    #print(f"Shape of mValue data: {mValuedf.shape}")
    #print(type(newSample))
    if newSample != "" and sample_id not in mValuedf["sample_id"].values :
        #print("New sample not in mValuedf. Obviously. Or at least, it should be obvious")
        newSample_df = newSample.to_pandas()
        #print(f"Shape of new sample data{newSample_df.shape}")
        filtered_df = newSample_df[newSample_df["EPICv1Loci"].isin(mValuedf.columns)]
        #print(f"New sample after filtering for EPICv1 loci{filtered_df.shape}")
        new_row = {"sample_id": sample_id}
        for _, row in filtered_df.iterrows():
            new_row[row["EPICv1Loci"]] = row["MValues"]
        mValuedf.loc[len(mValuedf)] = new_row

    #print(f"Shape of mValue data after adding a new sample (theoretically): {mValuedf.shape}")
    print(f"mValue data frame for sample id {sample_id} at api entry")
    print(mValuedfUpToDate.filter(mValuedfUpToDate["sample_id"] == sample_id))
    print(mValuedfUpToDate.filter(mValuedfUpToDate["sample_id"] == "01-0146_425440_T_W"))
        
    # print(mValuedfUpToDate.filter(pl.col("sample_id").str.contains("P001307")))
    sampleSummary = {}
    #print(sample_id)
    if sample_id in mValuedfUpToDate["sample_id"]:
        predictions = makePredictions(sample_id, "ZERO2", freeze)
        #print(sample_id)
        #print(predictions)
        samplePlots = {}
        for key in predictions.keys():
            #print(key)
            #print(predictions[key])
            levelPrediction = predictions[key]["prediction"]
            samplePlotdf = createPlotdf(mValuedf, key, sample_id, prediction=levelPrediction)
            samplePlots[key] = makeUmapPlot3D(samplePlotdf, "cancerType", sample_id = sample_id, cancerType=key)

        sampleSummary["predictions"] = predictions
        sampleSummary["plots"] = samplePlots
        #sampleSummary["mnp"] = mnpPrediction([sample_id]).to_json()
        sampleSummary["zero2_final_diagnosis"] = zero2FinalDiagnosis(sample_id)
        #print(sampleSummary["zero2_final_diagnosis"])

        return JSONResponse(content=sampleSummary)

    else:
        sampleSummary["predictions"] = {}
        sampleSummary["plots"] = {}
        #sampleSummary["mnp"] = pd.DataFrame().to_json()
        sampleSummary["zero2FinalDiagnosis"] = {}
        return JSONResponse(content=sampleSummary)


@app.get("/cancerType/{cancerType}")
async def cancer(cancerType):
    
    cohortSamples = cohortSampleList(cancerType)
    mnpPredictions = mnpCoverage(cohortSamples)
    #mnpPredictions = mnpPredictions.drop(columns=['reportable'])
    print("Coverage and samples done")

    mnpPredictions['classifier'] = mnpPredictions['name'] + '_' + mnpPredictions['version']
    mnpPredictions = mnpPredictions.pivot_table(index='sample_id', columns='classifier', aggfunc='first')
    mnpPredictions.columns = ['_'.join(col) for col in mnpPredictions.columns.values]
    mnpPredictions = mnpPredictions.fillna("-")
    mnpPredictions = mnpPredictions.reset_index()

    #reportableDetails = mnpReportable(cohortSamples)

    cohortUmapdf = createPlotdf(mValuedf, cancerType)
    cohortUmapPlot = makeUmapPlot3D(cohortUmapdf, "cancerType", cancerType=cancerType)

    cancerSummary = {}
    cancerSummary["mnpPredictions"] = mnpPredictions.to_dict()
    cancerSummary["cohortUmapPlot"] = cohortUmapPlot
    validation = validateCancerType(cancerType)
    cancerSummary["accuracy"] = validation["report"]
    cancerSummary["correctCallDistribution"] = validation["correctCallDistribution"]

    return JSONResponse(content=cancerSummary)

@app.get("/testValidation/{cancerType}")
def validate(cancerType):
    return validateCancerType(cancerType)

@app.post("/checkFileUpload")
async def checkFileUpload(fileHash):
    if os.path.exists(fileHash):
        filteredContent = pl.read_csv(fileHash)
        features = mValuedf.columns
        #print(features)
    return True

#@app.post("/uploadFile")
#async def upload(uploadFile):
#    return True

@app.get("/files/")
def list_files():
    try:
        files = os.listdir(sampleUploadDirectory)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"files": files}


@app.post("/upload/")
async def uploadNewSample(file: UploadFile = File(...)):
    fileLocation = os.path.join(sampleUploadDirectory, file.filename)

    if os.path.exists(fileLocation):
        filteredContent = pl.read_csv(fileLocation)
        #features = mValuedf.columns
        #print("Sample data:")
        #sample = filteredContent[features]
        #mValuedf = mValuedf.
        #print(f"reading new sample from file: {filteredContent.shape}")
    else:
        content = await file.read()
        filteredContent = epicFilter(content)
        filteredContent.write_csv(fileLocation)

    
    sampleId = ""
    pattern = r'.*?_T_M'
    match = re.search(pattern, file.filename)
    if match:
        #print(match)
        sampleId = match.group(0)

    #print(f"Type of filtered content: {type(filteredContent)}")
    sampleResponse = sample(sampleId, newSample=filteredContent)

    return {"info": f"file '{file.filename}' saved at '{fileLocation}'"}

#@app.get("/sampleList")
#async def getSampleList():
#    sampleList = mValuedf["sample_id"].to_list()
#    return sampleList