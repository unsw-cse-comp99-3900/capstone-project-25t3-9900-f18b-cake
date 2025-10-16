#from mch.db.database_tables import Sample, DatabaseConnector
import yaml
import polars as pl


from typing import Optional
from datetime import datetime

from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from sqlalchemy.orm import aliased

from mch.config.base_config import credential_file
from mch.db.database_tables import Sample, CuratedMethylation, BioSample, MethylationGroup, MethylationClassifier, AnalysisSet, BioSample, Purity, AnalysisSetXRef
from contextlib import contextmanager

from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

class SQLConnector:
    def __init__(self):
        # Use the existing DatabaseConnector's engine
        # this needs to be changed to use doppler secrets
        credentialFile = credential_file
        databaseName = "zcc"
        with open(credentialFile, 'r') as file:
            credentials = yaml.safe_load(file)

        credentials = credentials[databaseName]
        host = credentials["host"]
        port = credentials["port"]
        user = credentials["user"]
        password = credentials["password"]
        database = credentials["database"]
        ssa_ca = credentials["ca-cert"]
        connection = URL.create(
            #drivername="mysql+mysqlconnector",
            #drivername="mysql+mysqldb",
            drivername="mysql+pymysql",
            host=host,
            username=user,
            password=password,
            database=database,
            port=port,
            query={'ssl_ca': ssa_ca}
        )

        self.engine = create_engine(connection, echo = False)
        self.Session = sessionmaker(bind=self.engine)

        
    @contextmanager
    def get_session(self):
        session = Session(self.engine)
        session = Session(self.engine)
        try:
            yield session
        finally:
            session.close()

    def sqlmodel_to_dataframe(self, result):
        rows = result.fetchall()
        combined_data = []
        for row in rows:
        # Initialize an empty dictionary for this row
            combined_dict = {}
        
            # Iterate through each table object in the tuple
            for table_obj in row:
                # Check if it's a SQLModel object (has model_dump method)
                if hasattr(table_obj, 'model_dump'):
                    # Add the table data to our combined dictionary
                    combined_dict.update(table_obj.model_dump())
            
            combined_data.append(combined_dict)
    
        # Convert to DataFrame
        return pl.DataFrame(combined_data)


    def get_sample_ids(self, patient_ids:Optional[str]=None):
        if patient_ids is None:
            with self.get_session() as session:
                statement = select(Sample.sample_id)
                result = session.execute(statement)
                sample_ids = result.fetchall()
                sample_ids = [row[0] for row in sample_ids]
                meth_sample_ids = self.convert_sample_ids(sample_ids, "wgs", "methylation")
                sample_ids = meth_sample_ids["methylation_id"].to_list()
                
                #print(meth_sample_ids)
                return sample_ids
        else:
            with self.get_session() as session:
                if not isinstance(patient_ids, list):
                    patient_ids = [patient_ids]

                statement = select(Sample.sample_id).where(
                    Sample.patient_id.in_(patient_ids)
                )
                result = session.execute(statement)
                sample_ids = result.fetchall()
                return sample_ids

    def get_sample_purity(self, sample_ids):
        with self.get_session() as session:
            statement = (select(AnalysisSetXRef, BioSample, Purity).join(
                BioSample, AnalysisSetXRef.biosample_id == BioSample.biosample_id).join(
                    Purity, Purity.analysis_set_id == AnalysisSetXRef.analysis_set_id).where(
                    BioSample.biosample_id.in_(sample_ids), 
                    BioSample.sample_type == 'wgs',
                    BioSample.biosample_status == 'tumour',
                    BioSample.biosample_type == 'dna'
                    )
            )
            result = session.execute(statement)
            df = self.sqlmodel_to_dataframe(result)
   
            return df
            

    def get_diagnoses(self):
        """Helper method to fetch diagnoses"""
        with self.get_session() as session:
            statement = select(Sample.zero2_final_diagnosis)
            result = session.execute(statement)
            samples = result.fetchall()
            return [dict(sample._mapping) for sample in samples] 

    def get_mnp_results(self, patient_id):
        print(f"In sql connetor: {patient_id}")
        with self.get_session() as session:
            statement = select(Sample, CuratedMethylation, MethylationGroup, MethylationClassifier).where(
                Sample.sample_id.in_(patient_id),
                Sample.meth_sample_id == CuratedMethylation.biosample_id,
                CuratedMethylation.meth_group_id == MethylationGroup.meth_group_id,
                MethylationClassifier.meth_classifier_id == MethylationGroup.meth_classifier_id
                )
            result = session.execute(statement)
            samples = result.fetchall()
            print(samples)

            flattened_dicts = []
            
            for sample in samples:
                # Convert row to dict
                row_dict = {}
                for table_name, obj in sample._mapping.items():
                    if obj is None:
                        continue
                    # Get all columns for this model
                    for key in obj.__dict__:
                        if not key.startswith('_'):  # Skip SQLAlchemy internal attributes
                            # Create keys with table prefix to avoid column name conflicts
                            column_name = f"{table_name}_{key}" if table_name else key
                            row_dict[column_name] = getattr(obj, key)
                
                flattened_dicts.append(row_dict)
            
            all_keys = set()
            for d in flattened_dicts:
                all_keys.update(d.keys())

            standardized_dicts = []
            for d in flattened_dicts:
                new_dict = {k: d.get(k, None) for k in all_keys}
                standardized_dicts.append(new_dict)
            
            for i, d in enumerate(standardized_dicts[:2]):  # Just print the first two for brevity
                print(f"Row {i} content types:")
                for k, v in d.items():
                    print(f"  {k}: {type(v)} = {v}")

            # 2. Try creating the DataFrame with explicit schema control
            try:
                # Option A: Create with inference of all rows
                df = pl.DataFrame(standardized_dicts, infer_schema_length=None)
            except Exception as e:
                print(f"Option A failed: {e}")
            # Then ensure each dictionary has all keys (with None for missing values)

            # Now create the DataFrame with the standardized dictionaries
            #df = pl.DataFrame(standardized_dicts)
            
        # Convert list of dicts to DataFrame
            #df = pl.DataFrame(flattened_dicts)
            return df
            #return [dict(sample._mapping) for sample in samples] 

    #def get_sample_data(self, data_elements):
    #    with self.get_session() as session:
    #        statement = select 

    #def get_sample_ids(self, patient_ids):
    #    with self.get_session() as session:
    #        # Check if patient_id is a list or a single value
    #        
    #        if not isinstance(patient_ids, list):
    #            patient_ids = [patient_ids]
    #        
    #        statement = select(Sample.sample_id).where(Sample.patient_id.in_(patient_ids))
                
    #        result = session.execute(statement)
    #        sample_ids = result.fetchall()
    #       sample_ids = [row[0] for row in sample_ids]
    #        return sample_ids

    def get_meth_ids(self, sample_ids):
        if not isinstance(sample_ids, list):
            sample_ids=[sample_ids]
        
        ASX1 = aliased(AnalysisSetXRef)
        ASX2 = aliased(AnalysisSetXRef)
        
        with self.get_session() as session:
            statement = (
                select(BioSample.biosample_id, ASX1, ASX2)
                .select_from(ASX1)
                .join(ASX2, ASX1.analysis_set_id == ASX2.analysis_set_id)
                .join(BioSample, BioSample.biosample_id == ASX2.biosample_id)
                .where(
                    ASX1.biosample_id.in_(sample_ids),
                    BioSample.sample_type == "methylation"
                )
            )
            result = session.execute(statement)
            meth_sample_ids = self.sqlmodel_to_dataframe(result)
            
            return meth_sample_ids["biosample_id"]

    def convert_sample_ids(self, sample_ids, input_type: str, output_type: str):
        if not isinstance(sample_ids, list):
            sample_ids = [sample_ids]

        ASX1 = aliased(AnalysisSetXRef)
        ASX2 = aliased(AnalysisSetXRef)
        BS1 = aliased(BioSample)
        BS2 = aliased(BioSample)

        with self.get_session() as session:
            statement = (
                select(BS1.biosample_id.label("input_id"), BS2.biosample_id.label("output_id"))
                .select_from(ASX1)
                .join(ASX2, ASX1.analysis_set_id == ASX2.analysis_set_id)
                .join(BS1, BS1.biosample_id == ASX1.biosample_id)
                .join(BS2, BS2.biosample_id == ASX2.biosample_id)
                .where(
                    ASX1.biosample_id.in_(sample_ids),
                    BS1.sample_type == input_type,
                    BS2.sample_type == output_type
                )
            )
            result = session.execute(statement).all()
            df = pl.DataFrame(result, schema=["input_id", "output_id"])
            df = df.rename({
                "input_id": f"{input_type}_id",
                "output_id": f"{output_type}_id"
            })
            return df
            #converted_sample_ids = self.sqlmodel_to_dataframe(result)
            #print(converted_sample_ids)
            #return converted_sample_ids #["biosample_id"]

    def get_sample_data(self, data_elements=None, sample_ids=None):
        """
        Helper method to fetch sample data for specific columns and sample IDs.
        
        Args:
            data_elements (list, optional): List of column names to retrieve. If None, retrieves all columns.
            sample_ids (list, optional): List of sample ID values to filter. If None, retrieves all samples.
        
        Returns:
            list: A list of dictionaries containing the requested sample data
        """
        print(f"there are {len(sample_ids) if sample_ids else 0} ids, and the columns returned should be {data_elements if data_elements else 'all'}")
        
        with self.get_session() as session:
            # If no columns specified, select all columns from Sample
            if data_elements is None:
                statement = select(Sample)
            else:
                # Dynamically select specified columns
                selected_columns = [getattr(Sample, col) for col in data_elements]
                statement = select(*selected_columns)
            
            # Apply sample_ids filter if provided
            if sample_ids is not None:
                # Assuming the primary key or filter column is 'id' - adjust as needed
                # Replace 'id' with your actual sample ID column name (e.g., 'meth_sample_id')
                filter_column = getattr(Sample, 'sample_id')  # Change 'id' to your actual column name
                statement = statement.where(filter_column.in_(sample_ids))
                
            result = session.execute(statement)
             # Convert results to Polars DataFrame
            if data_elements is None:
                # If all columns selected, convert full Sample objects
                data_dicts = [sample.__dict__ for sample in result.scalars().all()]
                # Remove any SQLAlchemy internal attributes that start with '_'
                if data_dicts:
                    clean_data = [{k: v for k, v in row.items() if not k.startswith('_')} 
                                for row in data_dicts]
                    return pl.DataFrame(clean_data)
                else:
                    return pl.DataFrame()
            else:
                # If specific columns selected, return those
                rows = result.all()
                if rows:
                    data_dicts = [dict(zip(data_elements, row)) for row in rows]
                    return pl.DataFrame(data_dicts)
                else:
                    # Return empty DataFrame with correct column names
                    return pl.DataFrame({col: [] for col in data_elements})
 