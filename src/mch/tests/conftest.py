import pytest
import os
from pathlib import Path

@pytest.fixture(scope="session")
def database_url():
    """Get database URL from environment or use default"""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://default:default@localhost:5432/methylation"
    )

@pytest.fixture(scope="session")
def typedb_url():
    """Get TypeDB URL from environment or use default"""
    return os.getenv("TYPEDB_URL", "localhost:1729")

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "test_data" 