import os
import pytest
from mch.utils.file_utils import extract_patient_id, find_methylation_file_name

def test_extract_patient_id():
    """
    Test various scenarios for patient ID extraction
    """
    # Test P-number format
    assert extract_patient_id("P000101_244263_T_W") == "P000101"
    
    # Test numeric format
    assert extract_patient_id("06-0086_410877_T_W") == "06-0086"
    
    # Test no match
    assert extract_patient_id("random_sample_name") is None

def test_find_methylation_file_name(tmp_path):
    """
    Test file name extraction for methylation samples
    """
    # Create a mock file path
    test_file = tmp_path / "P000101_244263_T_M.csv"
    test_file.write_text("dummy content")
    
    # Test full path
    fileName, sampleName, methName = find_methylation_file_name(str(test_file))
    
    assert fileName == "P000101_244263_T_M.csv"
    assert sampleName == "P000101_244263_T_W"
    assert methName == "P000101_244263_T_M"

def test_find_methylation_file_name_no_match(tmp_path):
    """
    Test file name extraction for files without expected pattern
    """
    # Create a file without the expected pattern
    test_file = tmp_path / "random_file.csv"
    test_file.write_text("dummy content")
    
    # Test file without matching pattern
    fileName, sampleName, methName = find_methylation_file_name(str(test_file))
    
    assert fileName == "random_file.csv"
    assert sampleName == "random_file.csv"
    assert methName == "random_file.csv"