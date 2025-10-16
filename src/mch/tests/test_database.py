import pytest
from mch.db.database import Database

@pytest.fixture
def db():
    return Database()

def test_database_initialization(db):
    """Test that Database class initializes correctly"""
    assert db.sql is not None
    assert db.typedb is not None

def test_get_diagnoses(db):
    """Test the get_diagnoses method"""
    diagnoses = db.get_diagnoses()
    assert isinstance(diagnoses, list)
    if len(diagnoses) > 0:
        assert isinstance(diagnoses[0], dict)

@pytest.fixture(scope="session")
def test_db():
    """Fixture for test database configuration"""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_password"
    }

def test_database_config(test_db):
    """Test database configuration"""
    # Add your configuration tests here
    pass 