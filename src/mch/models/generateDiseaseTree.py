# data structures
import numpy as np
import pandas as pd
import polars as pl

# typedb
from typedb.client import *
from mch.core.disease_tree import DiseaseTree
from mch.db.type_db_tools import getMatch

from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from mch.db.database_tables import Sample, Purity, AnalysisSet
from mch.db.sql_connector import SQLConnector

import joblib

freeze = "freeze0125"

def getDiseaseTree():
    #diseaseTreedf = pd.DataFrame(columns=["parent", "child"])
    query = """ match
                $d1 isa Disease, has name $parent;
                $d2 isa Disease, has name $child;
                (parent: $d1, child: $d2) isa disease-molecular-hierarchy; 
                get $parent, $child;
                """
    with TypeDB.core_client("localhost:1729") as client:
        with client.session("oncotree", SessionType.DATA) as session:
            options = TypeDBOptions.core()
            options.infer = False
            result = getMatch(query, session, options)
            columns = result[0].keys()
            diseaseTreedf = pd.DataFrame(result,columns=columns)
        pass
        #diseaseTreedf = pd.concat([diseaseTreedf, df])

    # and now the samples associated with each disease classification
    query = """ 
            match
            $sample isa Sample, has sample_id $sample_id, has zcc_id $zcc_id;
            $disease isa Disease, has name $diseaseName;
            (disease: $disease, sample: $sample) isa sample-molecular-diagnosis;
            get $diseaseName, $sample_id;
            """ 

    with TypeDB.core_client("localhost:1729") as client:
        with client.session("oncotree", SessionType.DATA) as session:
            options = TypeDBOptions.core()
            options.infer = False
            result = getMatch(query, session, options)
            columns = result[0].keys()
            sampledf = pd.DataFrame(result,columns=columns)
        pass

    #print(f" samples from typeDB: {sampledf.shape}")
    #print((sampledf.head()))

    # Now remove samples with low purity. 

    #connector = DatabaseConnector()
    connector = SQLConnector()
    with connector.get_session() as session:
        statement = select(Sample, AnalysisSet, Purity).where(
            Sample.patient_id == AnalysisSet.patient_id,
            AnalysisSet.analysis_set_id == Purity.analysis_set_id
        )
        result = session.execute(statement)
        rows = result.fetchall()
    purityData = []
    for row in rows:
        tableOne_dict = row.Sample.model_dump()
        tableTwo_dict = row.Purity.model_dump()
        tableThree_dict = row.AnalysisSet.model_dump()
        combined_dict = {**tableOne_dict, **tableTwo_dict, **tableThree_dict}
        purityData.append(combined_dict)

    puritydf = pl.from_dicts(purityData)
    puritydf = puritydf[["sample_id", "purity"]]

    #print(f"Samples with purity data before filtering: {puritydf.shape}")
    puritydf = puritydf.filter(puritydf["purity"] > 0.30)
    sampledf = sampledf[sampledf["sample_id"].isin(puritydf["sample_id"])]
    
    # and finally, build the tree:
    root_node = 'ZERO2'
    root_disease_node = DiseaseTree(root_node, [], [])

    # Build the tree and pass diseaseTreedf and sampledf as arguments
    diseaseTree = root_disease_node.build_disease_tree(root_node, diseaseTreedf, sampledf)

    print(len(diseaseTree.get_samples_recursive()))

    return diseaseTree

if __name__ == "__main__":
    tree = getDiseaseTree()
    dataDirectory= f"/data/projects/methylationclassifier/data/{freeze}/"
    diseaseTreeFile = dataDirectory + "diseaseTree.joblib"
    with open(diseaseTreeFile, 'wb') as file:
        joblib.dump(tree, file)