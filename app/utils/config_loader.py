import os
import yaml
from typing import Any, Dict

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Loads configuration from YAML file, resolving paths relative to the project root."""
    if config_path is None:
        # Resolve path relative to this file's location: app/utils/config_loader.py -> app/config/config.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(current_dir), "config", "config.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Resolve all relative paths defined in config to absolute paths from project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Resolve paths in the configuration dictionary
    if "paths" in config:
        for key, val in config["paths"].items():
            if isinstance(val, str) and not os.path.isabs(val):
                config["paths"][key] = os.path.normpath(os.path.join(project_root, val))
                
    return config
