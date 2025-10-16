import os
import pytest
import polars as pl
from unittest.mock import Mock, patch

from mch.data_processing.sample_file_generation import get_files_with_suffix


@patch("os.walk")
def test_get_files_with_suffix(mock_os_walk):
    # Mock the os.walk output
    mock_os_walk.return_value = [
        ("/mocked_dir", ["subdir"], ["file1.txt", "file2_methylation_anno_hg38.txt", "file3.csv"]),
        ("/mocked_dir/subdir", [], ["file4_methylation_anno_hg38.txt", "file5.json"])
    ]
    
    # Define the directory and suffixes
    directory = "/mocked_dir"
    suffixes = ["_methylation_anno_hg38.txt", ".csv"]
    
    # Expected output
    expected_files = [
        "/mocked_dir/file2_methylation_anno_hg38.txt",
        "/mocked_dir/subdir/file4_methylation_anno_hg38.txt",
        "/mocked_dir/file3.csv"
    ]
    
    # Call the function
    result = get_files_with_suffix(directory, suffixes)
    
    # Assert the result
    assert sorted(result) == sorted(expected_files)