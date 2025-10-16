from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import polars as pl
import os
import re
from typing import Dict, Any, Optional, List

import logging
from mch.utils.logging_utils import setup_logging
from mch.utils.tree_utils import collect_tree_structure
from mch.visualization.plots import create_plot_df, makeUmapPlot3D
from mch.core.classification import validate_cancer_type, makePredictions, zero2_final_diagnosis
from mch.db.database import Database
from mch.config.settings import main_tree

from mch.api.who_book_processing import WHOBookOfTumourClient

import traceback

app = FastAPI(
    title="Methylation Classifier API",
    description="API for methylation data analysis",
    version="0.1.0"
)

from mch.config.settings import (
    FREEZE_NUMBER, 
    FREEZE, 
    mvalue_df, 
    main_tree, 
    color_profiles,
    disease_tree,
    who_book_file,
    cancer_type_file
)



setup_logging()

def find_methylation_filename(file_path: str) -> str:
    """Extract methylation filename pattern from path."""
    filename = os.path.basename(file_path)
    pattern = r'^.*?_T_M'
    match = re.search(pattern, filename)
    return match.group(0) if match else filename


@app.get("/sample/sample_list")
async def get_sample_list():
    db = Database()
    sample_list = db.get_sample_ids()
    return JSONResponse(content={"sample_list": sample_list}, status_code=200)

@app.get("/sample/{sample_id}")
async def get_sample(sample_id: Optional[str] = None, patient_id: Optional[str] = Query(None, description="Patient ID")) -> JSONResponse:
    """
    #Get sample analysis results.
    #Args:
        sample_id: return a prediction for this sample id 
        patient_id: return predictions for all samples for this patient
    """
    
    if not sample_id and not patient_id:
        raise HTTPException(status_code=400, detail="Either sample_id or patient_id must be provided")
    
    try:
        sql = Database()
        if not sample_id:
            sample_ids = sql.get_sample_ids(patient_id)
        else:
            sample_ids = [sample_id]
        
        
        response = {
            "patient_id": patient_id if patient_id else None,
            "sample_count": len(sample_ids),
            "samples": {}
        }

        for sample_id in sample_ids:
            print(f"Processing: {sample_id}, type: {type(sample_id)}")

            # Initialize response structure
            sample_summary = {
                "message": "",
                "predictions": {},
                "plots": {},
                "zero2_final_diagnosis": {}
            }
            # Check if sample exists in up-to-date data
            if sample_id not in mvalue_df["sample_id"].to_list():
                sample_summary = {"message": "sample_id not found in current dataset"}
                return JSONResponse(content=sample_summary)

            response["samples"][sample_id] = sample_summary

            predictions = makePredictions(sample_id, "ZERO2", FREEZE)
            sample_plots = {}
        
            for key in predictions:
                print(f"{key}:{predictions[key]}")
                level_prediction = predictions[key]["prediction"]
                sample_plot_df = create_plot_df(key, sample_id=sample_id, prediction=level_prediction)
                sample_plots[key] = makeUmapPlot3D(sample_plot_df, "cancerType", sample_id=sample_id, cancerType=key)

            sample_summary.update({
                "predictions": predictions,
                "plots": sample_plots,
                "zero2_final_diagnosis": zero2_final_diagnosis(sample_id)
            })
            response["samples"][sample_id] = sample_summary


        return JSONResponse(content= response)

    except Exception as e:
        print(traceback.format_exc())  # prints full traceback

        raise HTTPException(status_code=500, detail=f"Error processing sample: {str(e)}")

@app.get("/cancer_type/{cancer_type}")
async def get_cancer_type(cancer_type: str) -> JSONResponse:
    """
    Get cancer type analysis results.

    #Args:

        cancer_type: The type of cancer to analyze
    """
    try:
        print("aslkdfnalksdf")
        tree = main_tree.find_node_by_name(cancer_type)
        print("aasdasdslkdfnalksdf")
        cohortSamples =  tree.get_samples_recursive()
        print("aslkdfnfsdgdsfgdsfalksdf")
        logging.info(f"cancer type: {cancer_type}, number of samples: {len(cohortSamples)}")
 
        try:
            cohortUmapdf = create_plot_df(cancer_type, cohort_tree=tree)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'ffs {str(e)}')
        
        logging.info("plot dataframe recieved")
        cohortUmapPlot = makeUmapPlot3D(cohortUmapdf, "cancerType", cancerType=cancer_type)
        logging.info("plot created")
        cancerSummary = {}
        cancerSummary["plot"] = cohortUmapPlot
        #logging.info("validating cancer_type ...")
        #validation = validate_cancer_type(cancer_type)
        #logging.info("cancer_type validated...")
        #cancerSummary["accuracy"] = validation["report"]
        cancerSummary["accuracy"] = ""
        #cancerSummary["correctCallDistribution"] = validation["correctCallDistribution"]

        return JSONResponse(content=cancerSummary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing cancer type: {str(e)}")


@app.get("/cancer/cancer_structure")
async def get_list_of_cancer_types():
    df = collect_tree_structure(main_tree)
    df = {
        "cancer_types": [
            {"parent": p, "child": c, "count": cnt}
            for p, c, cnt in df
        ]
    }
    print(df)
    #df = pl.read_csv(cancer_type_file)
    #df = df.to_dicts()
    return JSONResponse(content=df, status_code = 200)



@app.get("/who_description/{cancer_type}")
async def get_who_description(cancer_type: str) -> JSONResponse:
    inspector = WHOBookOfTumourClient(who_book_file)
    description = inspector.search(book="", substring=cancer_type)
    if not description:
        raise HTTPException(status_code=404, detail="Cancer type not found")
    
    matching_titles = []
    if len(description) > 1 :
        results = [{'chapterTitle': item['chapterTitle'], 'chapterId': item['chapterId']} for item in description]
        return JSONResponse(content = results)

    bookId = description[0]["bookId"]
    bookTitle = description[0]["bookTitle"]
    chapterId = description[0]["chapterId"]
    chapterTitle = description[0]["chapterTitle"]
    description = inspector.get_book_chapter(bookId, chapterId, bookTitle, chapterTitle)
    
    return JSONResponse(description)



