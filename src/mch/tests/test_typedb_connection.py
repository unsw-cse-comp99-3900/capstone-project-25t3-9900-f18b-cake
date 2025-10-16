import warnings
import pytest
import pandas as pd
from typedb.api.connection.session import SessionType
from typedb.api.connection.transaction import TransactionType
from mch.db.typedb_connection import TypeDBConnection

# At the top of your test file
warnings.filterwarnings("ignore")
# Or for specific warnings:
warnings.filterwarnings("ignore", category=DeprecationWarning)

@pytest.fixture
def typedb_conn():
    """Fixture to provide a TypeDB connection"""
    conn = TypeDBConnection()
    yield conn
    conn.close()

def test_typedb_connection_singleton():
    """Test that we get the same connection instance"""
    conn1 = TypeDBConnection()
    conn2 = TypeDBConnection()
    assert conn1 is conn2

def test_can_create_session(typedb_conn):
    """Test session creation"""
    with typedb_conn.session() as session:
        assert session is not None

def test_get_match_returns_results(typedb_conn):
    """Test that get_match returns data in expected format"""
    query = """
     match
                $d1 isa Disease, has name $parent;
                get $parent;
                limit 1;
    """
    with typedb_conn.session() as session:
        results = typedb_conn.get_match(query, session)
        assert isinstance(results, list)
        if results:  # If database has data
            assert isinstance(results[0], dict)
            assert 'parent' in results[0]

def test_query_to_dataframe(typedb_conn):
    """Test conversion of query results to DataFrame"""
    query = """
    match 
    $d1 isa Disease, has name $parent;
    get $parent;
    limit 5;
    """
    df = typedb_conn.query_to_dataframe(query)
    assert isinstance(df, pd.DataFrame)
    
    if not df.empty:  # If database has data
        print(df)
        assert 'parent' in df.columns

@pytest.mark.parametrize("invalid_query", [
    "match $x isa NonExistentType; get $x;",
    "invalid query syntax",
    ""
])
def test_invalid_queries(typedb_conn, invalid_query):
    """Test handling of invalid queries"""
    with pytest.raises(Exception):  # You might want to catch specific TypeDB exceptions
        with typedb_conn.session() as session:
            typedb_conn.get_match(invalid_query, session)

def test_empty_result_handling(typedb_conn):
    """Test handling of queries that return no results"""
    query = """
    match
    $d isa Disease, has name "NonExistentDisease";
    get $d;
    """
    df = typedb_conn.query_to_dataframe(query)
    assert df.empty

def test_connection_cleanup(typedb_conn):
    """Test that resources are properly cleaned up"""
    with typedb_conn.session() as session:
        assert session is not None
    # Session should be closed after the with block
    # We could add more specific checks if TypeDB provides a way to check session state

def test_custom_uri_connection():
    """Test connection with custom URI"""
    conn = TypeDBConnection()
    conn._initialize_client("localhost:1729")  # or different port
    assert conn.client is not None
