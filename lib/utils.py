"""
Utility functions for GBMFirst project.
"""
import yaml
import os
from pathlib import Path


def get_project_root() -> Path:
    """Find project root by searching up for current.pc file. Usage: root = get_project_root()"""
    current = Path.cwd()
    
    # Search current directory and all parents
    for directory in [current] + list(current.parents):
        if (directory / 'current.pc').exists():
            return directory
    
    raise FileNotFoundError(
        "Could not find project root. No current.pc file found in current directory or parents."
    )


def get_machine_id() -> int:
    """Read machine ID from current.pc file (returns 1-4). Usage: pc_id = get_machine_id()"""
    root = get_project_root()
    with open(root / 'current.pc') as f:
        return int(f.read().strip())


def setup_notebook_environment():
    """Setup notebook: detect root, cd to it, return path. Usage: root = setup_notebook_environment()"""
    project_root = get_project_root()
    os.chdir(project_root)
    return project_root


def get_machine_config(config_path: str) -> dict:
    """Get machine-specific config from current.pc file. Usage: cfg = get_machine_config('config/car_coll/v1/config.yaml')"""
    # Find project root (where current.pc file is located)
    config_file = Path(config_path)
    project_root = config_file.parent
    while not (project_root / 'current.pc').exists() and project_root != project_root.parent:
        project_root = project_root.parent
    
    pc_file = project_root / 'current.pc'
    if not pc_file.exists():
        raise FileNotFoundError(
            f"current.pc file not found. Create a current.pc file in the project root with a number (1-4)."
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
    
    # Get machine-specific config
    machine_config = get_machine_config(config_path)
    config['machine'] = machine_config
    
    # Merge machine-specific paths with common paths
    # Machine-specific paths take precedence over common paths
    if 'paths' in machine_config and 'paths' in config:
        # Start with common paths
        merged_paths = config['paths'].copy()
        # Override with machine-specific paths
        for key, value in machine_config['paths'].items():
            if value is not None:  # Only override if machine path is not null
                merged_paths[key] = value
        config['paths'] = merged_paths
    elif 'paths' in machine_config:
        # Only machine paths exist
        config['paths'] = machine_config['paths']
    
    return config


def get_data_root(config_path: str = 'config/car_coll/v1/config.yaml') -> str:
    """Get data_root for current machine. Usage: data_root = get_data_root()"""
    config = load_config(config_path)
    data_root = config['paths']['data_root']
    
    if data_root is None:
        pc_num = get_machine_id()
        raise ValueError(
            f"data_root not configured for PC{pc_num}. "
            f"Please edit {config_path} and set machines.PC{pc_num}.paths.data_root"
        )
    
    return data_root


def plot_learning_curve(history_df, output_path, metric='rmse'):
    """Plot train/test learning curve. Usage: plot_learning_curve(df, 'path.png', 'tweedie-nloglik')"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Find train/test columns dynamically
    train_col = [c for c in history_df.columns if c.startswith('train_')][0]
    test_col = [c for c in history_df.columns if c.startswith('test_')][0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_df['iteration'], history_df[train_col], label='Train', color='blue', linewidth=2)
    ax.plot(history_df['iteration'], history_df[test_col], label='Test', color='orange', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'XGBoost Learning Curve ({metric})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def validate_critical_columns(df, cfg, stage_name, check_nulls=True):
    """Validate critical columns exist and optionally check for nulls. Usage: validate_critical_columns(df, cfg, 'Stage 01')"""
    critical = [
        cfg['data']['join_key'],
        cfg['data']['fold_column'],
        cfg['experiment']['target'],
        cfg['experiment']['exposure']
    ]
    
    missing = [c for c in critical if c not in df.columns]
    if missing:
        raise ValueError(f"[{stage_name}] Missing critical columns: {missing}")
    
    if check_nulls:
        for col in critical:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_pct = null_count / len(df) * 100
                print(f"[{stage_name}] WARNING: {col} has {null_count:,} nulls ({null_pct:.1f}%)")
    
    print(f"[{stage_name}] ✓ Critical columns validated: {', '.join(critical)}")
    return True
