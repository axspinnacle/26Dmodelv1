"""
Utility functions for GBMFirst project.
"""
import yaml
from pathlib import Path


def get_machine_config(config_path: str) -> dict:
    """
    Read the .PC file and return the appropriate machine configuration.
    
    Args:
        config_path: Path to the config.yaml file
        
    Returns:
        dict: Machine-specific configuration including conda_env and paths
        
    Example:
        >>> machine = get_machine_config('config/car_coll/v1/config.yaml')
        >>> print(machine['paths']['path_prefix'])
        '/Users/Mach/dev/aps/data/2024_CX_Cmodel/v2/'
    """
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
    """
    Load the full config and add the current machine's paths.
    
    Args:
        config_path: Path to the config.yaml file
        
    Returns:
        dict: Full config with 'machine' key containing current machine settings
        
    Example:
        >>> config = load_config('config/car_coll/v1/config.yaml')
        >>> print(config['machine']['paths']['path_prefix'])
        >>> print(config['experiment']['name'])
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add current machine config
    config['machine'] = get_machine_config(config_path)
    
    return config
