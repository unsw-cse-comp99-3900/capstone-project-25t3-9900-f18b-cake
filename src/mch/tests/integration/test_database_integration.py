import pytest
from mch.db.database import Database

@pytest.mark.integration
def test_full_database_workflow():
    """Test complete database workflow"""
    db = Database()
    
    # Test SQL operations
    diagnoses = db.get_diagnoses()
    assert isinstance(diagnoses, list)
    
    # Test TypeDB operations
    # Add your TypeDB specific tests here 