import os
import re
from typing import Union

from ..db.sql_connector import SQLConnector #, Sample


def extract_patient_id(sample_id: str) -> Union[str, None]:
    """
    Extract patient ID from a sample identifier.
    
    Args:
        sample_id (str): Sample identifier to extract patient ID from.
    
    Returns:
        str or None: Extracted patient ID or None if no match found.
    
    Examples:
        >>> extract_patient_id("P000101_244263_T_W")
        'P000101'
        >>> extract_patient_id("06-0086_410877_T_W")
        '06-0086'
    """
    pattern = r'(?:(P\d*))|\d{2}-\d{4}'
    match = re.search(pattern, sample_id)
    return match.group() if match else None

def find_methylation_file_name(meth_sample_id: str) -> tuple:
    """
    Extract file name components from a file path for methylation samples.
    
    Args:
        file_path (str): Path to the methylation file.
    
    Returns:
        tuple: (fileName, sample_name, meth_name)
    
    Examples:
        >>> find_methylation_file_name("/path/to/P000101_244263_T_M.csv")
        ('P000101_244263_T_M.csv', 'P000101_244263_T_W', 'P000101_244263_T_M')
    """
    
    fileName = f"{meth_sample_id}.csv"
    sample_id = ""
    #fileName = os.path.basename(file_path)
    #pattern = r'^(.*?)_T_M'
    #match = re.search(pattern, fileName)
    #if match:
    #    sample_name = f"{match.group(1)}_T_W"
    #    meth_name = f"{match.group(1)}_T_M"
    #    return fileName, sample_name, meth_name
    #else:
    #    return fileName, fileName, fileName

def get_files_with_suffix(directory, suffixes):
    """ 
    Recursively finds all files in a directory that end with any of the specified suffixes.
    
    Args:
        directory (str): The root directory to search.
        suffixes (list): A list of suffixes to match file names against.
        
    Returns:
        list: A list of full paths to all files that match any of the suffixes.
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(suffix) for suffix in suffixes):
                file_path = os.path.join(root, file)
                matching_files.append(file_path)
    return matching_files