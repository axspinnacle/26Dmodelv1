"""
Model utilities for XGBoost training with monotonicity constraints and GLM initialization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_monotonicity_constraints(
    config_path: str,
    vehicle_type: str,
    feature_list: List[str]
) -> Dict[str, int]:
    """
    Load monotonicity constraints from CSV and return dict for features in training set.
    
    Args:
        config_path: Path to config directory containing monotonicity.csv
        vehicle_type: Vehicle type column name (CAR, SUV, TRUCK, VAN)
        feature_list: List of features in the training dataset
        
    Returns:
        Dict mapping feature name to constraint (-1, 0, 1)
    """
    mono_file = Path(config_path) / "monotonicity.csv"
    if not mono_file.exists():
        print(f"Warning: Monotonicity file not found: {mono_file}")
        return {}
    
    mono_df = pd.read_csv(mono_file)
    
    # Build constraint dict for features that exist in training data
    constraints = {}
    for _, row in mono_df.iterrows():
        field = row['field']
        if field in feature_list:
            constraint_val = row[vehicle_type]
            # Handle both numeric and string representations
            if pd.isna(constraint_val) or constraint_val == 'x' or constraint_val == '':
                constraints[field] = 0
            else:
                constraints[field] = int(constraint_val)
    
    print(f"Loaded {len(constraints)} monotonicity constraints for {vehicle_type}")
    return constraints


def load_exclusions(
    config_path: str,
    vehicle_type: str
) -> List[str]:
    """
    Load feature exclusions from CSV.
    
    Args:
        config_path: Path to config directory containing exclusion.csv
        vehicle_type: Vehicle type column name (CAR, SUV, TRUCK, VAN)
        
    Returns:
        List of feature names to exclude
    """
    excl_file = Path(config_path) / "exclusion.csv"
    if not excl_file.exists():
        print(f"Warning: Exclusion file not found: {excl_file}")
        return []
    
    excl_df = pd.read_csv(excl_file)
    
    # Get features marked as 1 (exclude) for this vehicle type
    excluded = excl_df[excl_df[vehicle_type] == 1]['field'].tolist()
    
    print(f"Loaded {len(excluded)} exclusions for {vehicle_type}")
    return excluded


def load_glm_init(
    aux_data_path: str,
    control_file: str,
    data: pd.DataFrame,
    join_key: str,
    target_col: str,
    transform: str = "log"
) -> pd.Series:
    """
    Load GLM control model predictions and join to data.
    
    Args:
        aux_data_path: Path to directory containing control model file
        control_file: Filename of control model parquet
        data: DataFrame to join with (must have join_key column)
        join_key: Column name to join on (e.g., 'vin_date')
        target_col: Column name of GLM predictions (e.g., 'pred_pp_coll')
        transform: 'log' or 'raw' - whether to log-transform predictions
        
    Returns:
        Series of base_margin values (same index as data)
    """
    control_path = Path(aux_data_path) / control_file
    if not control_path.exists():
        raise FileNotFoundError(f"Control model file not found: {control_path}")
    
    # Load GLM predictions
    glm_preds = pd.read_parquet(control_path, columns=[join_key, target_col])
    
    # Join with data
    data_with_glm = data[[join_key]].merge(
        glm_preds,
        on=join_key,
        how='left'
    )
    
    # Handle missing GLM predictions
    missing_count = data_with_glm[target_col].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} rows missing GLM predictions, filling with median")
        median_pred = data_with_glm[target_col].median()
        data_with_glm[target_col].fillna(median_pred, inplace=True)
    
    # Transform if requested
    if transform == "log":
        # Ensure positive values for log
        min_val = data_with_glm[target_col].min()
        if min_val <= 0:
            print(f"Warning: Non-positive GLM predictions found (min={min_val}), adding offset")
            data_with_glm[target_col] = data_with_glm[target_col] + abs(min_val) + 1e-6
        base_margin = np.log(data_with_glm[target_col])
    else:
        base_margin = data_with_glm[target_col]
    
    print(f"Loaded GLM init: transform={transform}, mean={base_margin.mean():.4f}, std={base_margin.std():.4f}")
    
    return base_margin


def apply_feature_filters(
    features: List[str],
    exclusions: List[str],
    verbose: bool = True
) -> List[str]:
    """
    Filter out excluded features from feature list.
    
    Args:
        features: List of all features
        exclusions: List of features to exclude
        verbose: Print filtering results
        
    Returns:
        Filtered feature list
    """
    original_count = len(features)
    filtered = [f for f in features if f not in exclusions]
    
    if verbose and len(filtered) < original_count:
        excluded_found = set(features) & set(exclusions)
        print(f"Filtered {original_count - len(filtered)} features: {excluded_found}")
    
    return filtered


def build_xgb_params(
    base_params: Dict,
    monotonicity_dict: Dict[str, int],
    feature_names: List[str]
) -> Dict:
    """
    Build XGBoost parameters with monotonicity constraints.
    
    Args:
        base_params: Base XGBoost parameters from config
        monotonicity_dict: Dict of feature -> constraint value
        feature_names: Ordered list of feature names (must match training data)
        
    Returns:
        Updated parameters dict with monotone_constraints tuple
    """
    params = base_params.copy()
    
    # Build constraint tuple in same order as feature_names
    constraints = tuple(monotonicity_dict.get(f, 0) for f in feature_names)
    
    # Only add if there are non-zero constraints
    if any(c != 0 for c in constraints):
        params['monotone_constraints'] = constraints
        non_zero = sum(1 for c in constraints if c != 0)
        print(f"Applied {non_zero} monotonicity constraints")
    
    return params
