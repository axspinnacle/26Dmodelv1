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


def load_glm_predictions(
    aux_data_path: str,
    control_file: str,
    data: pd.DataFrame,
    join_key: str,
    target_col: str
) -> pd.Series:
    """
    Load raw GLM predictions and join to data.
    
    Args:
        aux_data_path: Path to directory containing control model file
        control_file: Filename of control model parquet
        data: DataFrame to join with (must have join_key column)
        join_key: Column name to join on (e.g., 'vin_date')
        target_col: Column name of GLM predictions (e.g., 'pred_pp_coll')
        
    Returns:
        Series of raw GLM predictions (same index as data)
    """
    control_path = Path(aux_data_path) / control_file
    if not control_path.exists():
        raise FileNotFoundError(f"Control model file not found: {control_path}")
    
    # Load GLM predictions
    glm_preds = pd.read_parquet(control_path, columns=[join_key, target_col])
    
    # Join with data - preserve original index
    data_with_glm = data[[join_key]].reset_index().merge(
        glm_preds,
        on=join_key,
        how='left'
    ).set_index('index')
    
    # Handle missing GLM predictions
    missing_count = data_with_glm[target_col].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} rows missing GLM predictions, filling with median")
        median_pred = data_with_glm[target_col].median()
        data_with_glm[target_col] = data_with_glm[target_col].fillna(median_pred)
    
    # Ensure positive values
    min_val = data_with_glm[target_col].min()
    if min_val <= 0:
        print(f"Warning: Non-positive GLM predictions found (min={min_val}), adding offset")
        data_with_glm[target_col] = data_with_glm[target_col] + abs(min_val) + 1e-6
    
    print(f"Loaded GLM predictions: mean={data_with_glm[target_col].mean():.4f}, std={data_with_glm[target_col].std():.4f}")
    
    # Return series with original index preserved
    return data_with_glm[target_col]


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
    DEPRECATED: Use load_glm_predictions() instead and transform as needed.
    
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
        data_with_glm[target_col] = data_with_glm[target_col].fillna(median_pred)
    
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
    target: str = None,
    exposure: str = None,
    join_key: str = None,
    verbose: bool = True
) -> List[str]:
    """
    Filter out excluded features from feature list.
    Automatically excludes target, exposure, and join_key to prevent data leakage.
    
    Args:
        features: List of all features
        exclusions: List of features to exclude
        target: Target column name (auto-excluded)
        exposure: Exposure column name (auto-excluded)
        join_key: Join key column name (auto-excluded)
        verbose: Print filtering results
        
    Returns:
        Filtered feature list
    """
    original_count = len(features)
    
    # Build complete exclusion list
    all_exclusions = set(exclusions)
    
    # Auto-exclude target, exposure, join_key to prevent data leakage
    auto_exclude = []
    if target:
        all_exclusions.add(target)
        auto_exclude.append(f"target={target}")
    if exposure:
        all_exclusions.add(exposure)
        auto_exclude.append(f"exposure={exposure}")
    if join_key:
        all_exclusions.add(join_key)
        auto_exclude.append(f"join_key={join_key}")
    
    # Filter features
    filtered = [f for f in features if f not in all_exclusions]
    
    if verbose:
        if auto_exclude:
            print(f"Auto-excluding: {', '.join(auto_exclude)}")
        if len(filtered) < original_count:
            excluded_found = set(features) & all_exclusions
            print(f"Filtered {original_count - len(filtered)} features: {sorted(excluded_found)}")
    
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


def transform_target(
    pp,  # Can be pd.Series or np.ndarray
    glm_pred,  # Can be pd.Series, np.ndarray, or None
    method: str
):
    """
    Transform pure premium target for GBM training.
    
    Args:
        pp: Pure premium (actual target values) - Series or array
        glm_pred: GLM predictions (baseline from carrier features) - Series, array, or None
        method: 'residual' (pp/glm) or 'direct' (pp as-is)
        
    Returns:
        Transformed target for GBM training (same type as input)
    """
    if method == "residual":
        if glm_pred is None:
            raise ValueError("glm_pred required for residual method")
        # GBM learns correction factor (ratio)
        return pp / glm_pred
    elif method == "direct":
        # GBM learns PP directly
        return pp
    else:
        raise ValueError(f"Unknown target_method: {method}. Must be 'residual' or 'direct'")


def inverse_transform(
    gbm_pred,  # Can be pd.Series or np.ndarray
    glm_pred,  # Can be pd.Series, np.ndarray, or None
    method: str
):
    """
    Convert GBM predictions back to pure premium space.
    
    Args:
        gbm_pred: Raw GBM predictions - Series or array
        glm_pred: GLM predictions (baseline from carrier features) - Series, array, or None
        method: 'residual' (multiply by glm) or 'direct' (use as-is)
        
    Returns:
        Final pure premium predictions (same type as input)
    """
    if method == "residual":
        if glm_pred is None:
            raise ValueError("glm_pred required for residual method")
        # GBM output is ratio, multiply by GLM to get PP
        return gbm_pred * glm_pred
    elif method == "direct":
        # GBM output is already PP
        return gbm_pred
    else:
        raise ValueError(f"Unknown target_method: {method}. Must be 'residual' or 'direct'")


def predict_with_transform(
    model,
    X: pd.DataFrame,
    glm_pred: Optional[pd.Series],
    target_method: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate predictions and apply inverse transform.
    
    Args:
        model: Trained XGBoost model
        X: Features for prediction
        glm_pred: GLM predictions (required for residual method)
        target_method: 'residual' or 'direct'
        
    Returns:
        Tuple of (gbm_pred_raw, pp_predicted)
        - gbm_pred_raw: Raw GBM output (ratio for residual, PP for direct)
        - pp_predicted: Final PP predictions (after inverse transform)
    """
    # Ensure X is a proper DataFrame with column names as list (not Index)
    # XGBoost requires feature_names attribute for validation
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X.columns = list(X.columns)
    
    gbm_pred_raw = pd.Series(model.predict(X, validate_features=False), index=X.index)
    pp_predicted = inverse_transform(gbm_pred_raw, glm_pred, target_method)
    return gbm_pred_raw, pp_predicted


def save_predictions(
    output_base: str,
    stage: str,
    split: str,
    y_actual: pd.Series,
    y_pred: pd.Series,
    exposure: pd.Series,
    **extra_cols
) -> str:
    """
    Save predictions in standard format.
    
    Args:
        output_base: Output directory path
        stage: Stage identifier (e.g., '05a', '05c', '08')
        split: Data split ('train', 'test', 'holdout')
        y_actual: Actual target values
        y_pred: Predicted values
        exposure: Exposure/weights
        **extra_cols: Additional columns to include (e.g., vin_date=..., fold=...)
        
    Returns:
        Path to saved file
    """
    import os
    
    results_dir = f"{output_base}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Build dataframe (use .values to avoid index alignment issues)
    df = pd.DataFrame({
        "actual": y_actual.values if isinstance(y_actual, pd.Series) else y_actual,
        "pred": y_pred.values if isinstance(y_pred, pd.Series) else y_pred,
        "exposure": exposure.values if isinstance(exposure, pd.Series) else exposure
    })
    
    # Add extra columns
    for col_name, col_data in extra_cols.items():
        df[col_name] = col_data
    
    # Save
    output_file = f"{results_dir}/{stage}_predictions_{split}.parquet"
    df.to_parquet(output_file, index=True)
    
    return output_file


def save_debug_output(
    output_base: str,
    stage: str,
    split: str,
    data_orig: pd.DataFrame,
    pp_actual: pd.Series,
    glm_pred: pd.Series,
    gbm_pred_raw: pd.Series,
    pp_predicted: pd.Series,
    exposure: pd.Series
) -> str:
    """
    Save full debug output with all columns for analysis.
    
    Args:
        output_base: Output directory path
        stage: Stage identifier (e.g., '05c', '08')
        split: Data split ('train', 'test', 'holdout')
        data_orig: Original dataframe containing source columns
        pp_actual: Actual pure premium
        glm_pred: GLM predictions
        gbm_pred_raw: Raw GBM output (ratio or PP depending on method)
        pp_predicted: Final PP predictions
        exposure: Exposure values
        
    Returns:
        Path to saved file
        
    Notes:
        Expected columns in data_orig:
        - vin_date, fold
        - incurred_raw_coll_imps (or similar loss column)
        - vc_msrp_impa, Dep_factor, veh_value_dep (if available)
    """
    import os
    
    results_dir = f"{output_base}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Build debug dataframe (use .values to avoid index alignment issues)
    debug_df = pd.DataFrame({
        "pp_actual": pp_actual.values if isinstance(pp_actual, pd.Series) else pp_actual,
        "glm_pred": glm_pred.values if isinstance(glm_pred, pd.Series) else glm_pred,
        "gbm_pred_raw": gbm_pred_raw.values if isinstance(gbm_pred_raw, pd.Series) else gbm_pred_raw,
        "pp_predicted": pp_predicted.values if isinstance(pp_predicted, pd.Series) else pp_predicted,
        "exposure": exposure.values if isinstance(exposure, pd.Series) else exposure
    })
    
    # Add columns from original data if available
    optional_cols = {
        'vin_date': 'vin_date',
        'fold': 'fold',
        'loss_raw': ['incurred_raw_coll_imps', 'incurred_raw_comp_imps', 'incurred_raw_bi_imps'],
        'msrp': 'vc_msrp_impa',
        'dep_factor': 'Dep_factor',
        'veh_value_dep': 'veh_value_dep'
    }
    
    for target_col, source_col in optional_cols.items():
        if isinstance(source_col, list):
            # Try multiple possible column names
            for col in source_col:
                if col in data_orig.columns:
                    debug_df[target_col] = data_orig[col]
                    break
        else:
            if source_col in data_orig.columns:
                debug_df[target_col] = data_orig[source_col]
    
    # Save
    output_file = f"{results_dir}/{stage}_debug_{split}.parquet"
    debug_df.to_parquet(output_file, index=True)
    
    print(f"  Debug output: {output_file} ({len(debug_df.columns)} columns)")
    return output_file

# New function
def load_glm_for_scoring(df,cfg,score_cfg,pc_id,target_method):
    if target_method!="residual": return None
    glm_source=score_cfg.get("glm_source","lookup")
    join_key=cfg["data"]["join_key"]
    if glm_source=="column":
        col=score_cfg.get("glm_column")
        if col not in df.columns: raise ValueError(f"Missing {col}")
        return df[col].copy()
    if join_key not in df.columns:
        raise ValueError(f"Missing {join_key}. Use glm_source=column")
    
    # Get GLM prediction column (required for residual method)
    glm_col = cfg.get("model", {}).get("glm_prediction_column")
    if not glm_col:
        raise ValueError("model.glm_prediction_column required for target_method=residual")
    
    from model_utils import load_glm_predictions
    paths=cfg["machines"][pc_id]["paths"]
    return load_glm_predictions(paths["aux_data_path"],cfg["data"]["control_model_file"],df,join_key,glm_col)
