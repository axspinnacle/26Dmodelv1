"""
Utility functions for GBMFirst project.
"""
import yaml
import os
from pathlib import Path


def get_project_root() -> Path:
    """Find project root by searching up for .PC file. Usage: root = get_project_root()"""
    current = Path.cwd()
    
    # Search current directory and all parents
    for directory in [current] + list(current.parents):
        if (directory / '.PC').exists():
            return directory
    
    raise FileNotFoundError(
        "Could not find project root. No .PC file found in current directory or parents."
    )


def get_machine_id() -> int:
    """Read machine ID from .PC file (returns 1-4). Usage: pc_id = get_machine_id()"""
    root = get_project_root()
    with open(root / '.PC') as f:
        return int(f.read().strip())


def setup_notebook_environment():
    """Setup notebook: detect root, cd to it, return path. Usage: root = setup_notebook_environment()"""
    project_root = get_project_root()
    os.chdir(project_root)
    return project_root


def get_machine_config(config_path: str) -> dict:
    """Get machine-specific config from .PC file. Usage: cfg = get_machine_config('config/car_coll/v1/config.yaml')"""
    # Find project root (where .PC file is located)
    config_file = Path(config_path)
    project_root = config_file.parent
    while not (project_root / '.PC').exists() and project_root != project_root.parent:
        project_root = project_root.parent
    
    pc_file = project_root / '.PC'
    if not pc_file.exists():
        raise FileNotFoundError(
            f".PC file not found. Create a .PC file in the project root with a number (1-4)."
        )
    
    # Read PC number
    with open(pc_file, 'r') as f:
        pc_num = f.read().strip()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pc_key = f'PC{pc_num}'
    if pc_key not in config.get('machines', {}):
        raise ValueError(f"Machine '{pc_key}' not found in config. Available: {list(config['machines'].keys())}")
    
    return config['machines'][pc_key]


def load_config(config_path: str) -> dict:
    """Load full config + add current machine paths. Usage: cfg = load_config('config/car_coll/v1/config.yaml')"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add current machine config
    config['machine'] = get_machine_config(config_path)
    
    return config
