import yaml
from pathlib import Path

config_dir = Path(__file__).parent
config_path = config_dir / "base_config.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

FREEZE = config.get("freeze")
FREEZE_NUMBER = config.get("freeze_number")
WORKING_DIRECTORY = config.get("working_directory")
credential_file = config.get("credential_file")