import logging
from typing import Optional
import os
import yaml
import sys
import re

from pathlib import Path

from mch.config.base_config import FREEZE, FREEZE_NUMBER, WORKING_DIRECTORY

# Constants
#FREEZE_NUMBER = "0525"
#FREEZE = f"freeze{FREEZE_NUMBER}"

# Base paths
DATA_DIR = Path(f"{WORKING_DIRECTORY}/data/")
FREEZE_DIR = DATA_DIR / FREEZE

def setup_logging(log_file: Optional[str] = None, log_level: str = 'INFO'):
    """
    Configure logging for the sample loader.
    
    Args:
        log_file (str, optional): Path to log file
        log_level (str): Logging level
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # Console output
        ]
    )
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)


def interpolate_config_variables(config: dict) -> dict:
    """
    Process a configuration dictionary to replace variable references with their values.
    
    Args:
        config (dict): The configuration dictionary
        
    Returns:
        dict: The processed configuration with variables interpolated
    """
    processed_config = {k: v.replace("{FREEZE}", FREEZE) if isinstance(v, str) else v for k, v in config.items()}
    processed_config = {k: v.replace("{WORKING_DIRECTORY}", WORKING_DIRECTORY) if isinstance(v, str) else v for k, v in processed_config.items()}

    return processed_config

def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    # Default config path
    if not config_path:
        logging.error("No configuration path provided.")
        raise ValueError("A config file must be provided. Either the name of a yaml file in the config directory, or a path to a file")
    
    if not os.path.isabs(config_path):
        package_dir = os.path.dirname(__file__)  # Directory of the current script
        config_dir = os.path.join(package_dir, "..", "config")
        config_path = os.path.join(config_dir, config_path)
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return interpolate_config_variables(config)
            #return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise
