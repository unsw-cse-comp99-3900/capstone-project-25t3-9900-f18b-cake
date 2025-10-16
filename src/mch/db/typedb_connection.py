from contextlib import contextmanager
from typing import Generator, List, Dict, Any
import pandas as pd
import typedb.driver
#from typedb.client import TypeDB, TypeDBClient, TypeDBSession
#from typedb.driver import TypeDB, TypeDBClient, TypeDBSession
#from typedb.driver import TypeDB, TransactionType, TypeDBDriver
from typedb.driver import TypeDB, SessionType, TransactionType

from typedb.api.connection.session import SessionType
#from typedb.api.connection.transaction import TransactionType
#from typedb.api.connection.options import TypeDBOptions



class TypeDBConnection:
    _instance = None
    _client = None
    _driver: TypeDBDriver = None
    DEFAULT_URI = "localhost:1729"
    DEFAULT_DATABASE = "oncotree"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TypeDBConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        #if TypeDBConnection._client is None:
        if TypeDBConnection._driver is None:
            self._initialize_client()

    def _initialize_client(self, uri: str = DEFAULT_URI):
        TypeDBConnection._client = TypeDB.core_client(uri)

    def _initialize_driver(self, uri: str = DEFAULT_URI):
        # New API: use core_driver instead of core_client
        TypeDBConnection._driver = TypeDB.core_driver(uri)


    @property
    def client(self) -> TypeDBClient:
        #return TypeDBConnection._client
        return TypeDBConnection._driver

    #@contextmanager
    #def session(self, database_name: str = DEFAULT_DATABASE, 
    #            session_type: SessionType = SessionType.DATA) -> Generator[TypeDBSession, None, None]:
    #    """Creates a TypeDB session with proper cleanup"""
    #    session = self._client.session(database_name, session_type)
    #    try:
    #        yield session
    #    finally:
    #        session.close()
    

    @contextmanager
    def session(self, database_name: str = DEFAULT_DATABASE,
                session_type: SessionType = SessionType.DATA):
        """Context manager that yields a TypeDB session with cleanup"""
        with self._driver.session(database_name, session_type) as session:
            yield session

    @contextmanager
    def transaction(self, database_name: str = DEFAULT_DATABASE,
                    session_type: SessionType = SessionType.DATA,
                    transaction_type: TransactionType = TransactionType.READ):
        """Helper to open a transaction in one go"""
        with self.session(database_name, session_type) as session:
            with session.transaction(transaction_type) as tx:
                yield tx

    def get_match(self, query: str, session, options) -> dict:
        with conn.transaction(transaction_type=TransactionType.READ) as tx:
            read_transaction = tx.query.fetch("match $p isa person; fetch $p: name;")

        #read_transaction = session.transaction(TransactionType.READ, options) # use your typedb connection session
        answer_iterator = read_transaction.query().match(query)
        result = []
        for ans in answer_iterator:
            item = {k: v.get_value() if v.is_attribute() else v for k, v in ans.map().items()}
            result.append(item)
        read_transaction.close()

        return result

    
    def query_to_dataframe(self, query: str, database: str = DEFAULT_DATABASE, infer: bool = False) -> pd.DataFrame:
        """
        Execute a query and return results as a pandas DataFrame.
        
        Args:
            query: The TypeDB query string
            database: Database name (default: oncotree)
            infer: Whether to use inference (default: False)
            
        Returns:
            pandas DataFrame containing the query results
        """
        with self.session(database) as session:
            result = self.get_match(query, session, infer)
            if not result:
                return pd.DataFrame()
            return pd.DataFrame(result)


    def close(self):
        """Close the TypeDB client connection"""
        if TypeDBConnection._client:
            TypeDBConnection._client.close()
            TypeDBConnection._client = None


    def get_disease_tree_nodes(self):
        query = """ match
                $d1 isa Disease, has name $parent;
                $d2 isa Disease, has name $child;
                (parent: $d1, child: $d2) isa disease-molecular-hierarchy; 
                get $parent, $child;
                """
        return self.query_to_dataframe(query, self.DEFAULT_DATABASE, False)

    def get_disease_samples(self):
        query = """ 
            match
            $sample isa Sample, has sample_id $sample_id, has zcc_id $zcc_id;
            $disease isa Disease, has name $diseaseName;
            (disease: $disease, sample: $sample) isa sample-molecular-diagnosis;
            get $diseaseName, $sample_id;
            """ 
        return self.query_to_dataframe(query, self.DEFAULT_DATABASE, False) 




