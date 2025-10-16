from .sql_connector import SQLConnector
#from .typedb_connection import TypeDBConnection
from typing import Optional

class Database:
    def __init__(self):
        self.sql = SQLConnector()
        #self.typedb = TypeDBConnection()
    
    def get_diagnoses(self):
        """Fetch diagnoses using SQL"""
        return self.sql.get_diagnoses()

    def get_sample_data(self, data_elements, sample_ids):
        return self.sql.get_sample_data(data_elements, sample_ids)
    
    def get_sample_ids(self, patient_ids:Optional[str]=None):
        return self.sql.get_sample_ids(patient_ids)

    def get_sample_purity(self, sample_ids):
        return self.sql.get_sample_purity(sample_ids)

    def get_meth_ids(self, sample_ids):
        return self.sql.get_meth_ids(sample_ids)

    def convert_sample_ids(self, sample_ids, input_type, output_type):
        return self.sql.convert_sample_ids(sample_ids, input_type, output_type)

    #def get_disease_tree_nodes(self):
    #    return self.typedb.get_disease_tree_nodes()

    #def get_disease_samples(self):
    #    return self.typedb.get_disease_samples()
        
    # Add other database operations here 