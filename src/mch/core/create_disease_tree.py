# data structures
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# typedb
from typedb.client import *
from mch.core.disease_tree import DiseaseTree
#from methylationMethods import getMatch

#from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from mch.db.database import Database

import joblib

FREEZE_NUMBER = "0525"
FREEZE = f"freeze{FREEZE_NUMBER}"

# Base paths
DATA_DIR = Path("/data/projects/methylationclassifier/data/")
FREEZE_DIR = DATA_DIR / FREEZE

db = Database()

def get_disease_tree():
    disease_tree_df = db.get_disease_tree_nodes()
    # and now the samples associated with each disease classification
    sample_df = db.get_disease_samples()
   
    print(f" samples from typeDB: {sample_df.shape}")
   
    # Now remove samples with low purity. 
    purity_df = db.get_sample_purity(sample_df["sample_id"])
    purity_df = purity_df[["biosample_id", "purity"]]
   
    print(f"Samples with purity data before filtering: {purity_df.shape}")
    purity_df = purity_df.filter(purity_df["purity"] > 0.30)
    print(f"Samples with purity data after filtering: {purity_df.shape}")


    print(sample_df.shape)
    sample_df = sample_df[sample_df["sample_id"].isin(purity_df["biosample_id"])]
    print(sample_df.shape)

    # and finally, build the tree:
    root_node = 'ZERO2'
    root_disease_node = DiseaseTree(root_node, [], [])

    # Build the tree and pass diseaseTreedf and sampledf as arguments
    diseaseTree = root_disease_node.build_disease_tree(root_node, disease_tree_df, sample_df)

    print(len(diseaseTree.get_samples_recursive()))


    # write the tree to file for use elsewhere. 

    return diseaseTree

if __name__ == "__main__":
    tree = get_disease_tree()
    dataDirectory= f"{FREEZE_DIR}"
    diseaseTreeFile = dataDirectory + "/diseaseTree.joblib"
    with open(diseaseTreeFile, 'wb') as file:
        joblib.dump(tree, file)