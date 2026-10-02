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
    target_col: str,
    drop_missing: bool = False
) -> tuple:
    """
    Load raw GLM predictions and join to data.
    
    Args:
        aux_data_path: Path to directory containing control model file
        control_file: Filename of control model parquet
        data: DataFrame to join with (must have join_key column)
        join_key: Column name to join on (e.g., 'vin_date')
        target_col: Column name of GLM predictions (e.g., 'pred_pp_coll')
        drop_missing: If True, use inner join and return matched indices. If False, use left join and fill with median.
        
    Returns:
        If drop_missing=False: Series of raw GLM predictions (same index as data)
        If drop_missing=True: (Series of GLM predictions, matched_indices)
    """
    control_path = Path(aux_data_path) / control_file
    if not control_path.exists():
        raise FileNotFoundError(f"Control model file not found: {control_path}")
    
    # Load GLM predictions
    glm_preds = pd.read_parquet(control_path, columns=[join_key, target_col])
    
    # Join with data - preserve original index
    join_type = 'inner' if drop_missing else 'left'
    data_with_glm = data[[join_key]].reset_index().merge(
        glm_preds,
        on=join_key,
        how=join_type
    ).set_index('index')
    
    if drop_missing:
        # Report dropped rows
        matched_count = len(data_with_glm)
        dropped_count = len(data) - matched_count
        dropped_pct = 100 * dropped_count / len(data) if len(data) > 0 else 0
        print(f"  Removed {dropped_count:,} rows ({dropped_pct:.2f}%) without GLM predictions")
        matched_indices = data_with_glm.index
    else:
        # Handle missing GLM predictions with fillna
        missing_count = data_with_glm[target_col].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} rows missing GLM predictions, filling with median")
            median_pred = data_with_glm[target_col].median()
            data_with_glm[target_col] = data_with_glm[target_col].fillna(median_pred)
        matched_indices = None
    
    # Ensure positive values
    min_val = data_with_glm[target_col].min()
    if min_val <= 0:
        print(f"Warning: Non-positive GLM predictions found (min={min_val}), adding offset")
        data_with_glm[target_col] = data_with_glm[target_col] + abs(min_val) + 1e-6
    
    print(f"Loaded GLM predictions: mean={data_with_glm[target_col].mean():.4f}, std={data_with_glm[target_col].std():.4f}")
    
    # Return based on mode
    if drop_missing:
        return data_with_glm[target_col], matched_indices
    else:
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


def apply_target_cap(
    pp_train,
    pp_test, 
    target_cap,
    verbose=True
):
    """
    DEPRECATED: Use apply_loss_cap() instead.
    Apply cap to target variable to remove extreme outliers.
    
    Args:
        pp_train: Training target values (Series or array)
        pp_test: Test target values (Series or array)
        target_cap: Maximum allowed value (e.g., 100000)
        verbose: Whether to print summary
        
    Returns:
        Tuple of (capped_train, capped_test)
    """
    if target_cap is None:
        return pp_train, pp_test
    
    # Count values that will be capped
    n_cap_train = (pp_train > target_cap).sum()
    n_cap_test = (pp_test > target_cap).sum()
    
    # Apply cap
    if isinstance(pp_train, pd.Series):
        pp_train_capped = pp_train.clip(upper=target_cap)
        pp_test_capped = pp_test.clip(upper=target_cap)
    else:
        pp_train_capped = np.clip(pp_train, None, target_cap)
        pp_test_capped = np.clip(pp_test, None, target_cap)
    
    if verbose and (n_cap_train + n_cap_test > 0):
        print(f"  Capped target at {target_cap:,}:")
        print(f"    Train: {n_cap_train:,} values")
        print(f"    Test: {n_cap_test:,} values")
    
    return pp_train_capped, pp_test_capped


def apply_loss_cap(
    train_df,
    test_df,
    loss_cap,
    loss_type="bi",
    pp_bi_col="pp_bi",
    pp_pd_col="pp_pd",
    ee_bi_col="ee_bi_imps",
    ee_pd_col="ee_pd_imps",
    exposure_floor=None,
    verbose=True
):
    """
    Apply loss cap and filter records based on loss type.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        loss_cap: Maximum allowed loss value
        loss_type: "bi", "pd", or "bi+pd"
        pp_bi_col: Column name for BI pure premium
        pp_pd_col: Column name for PD pure premium
        ee_bi_col: Column name for BI exposure
        ee_pd_col: Column name for PD exposure
        exposure_floor: Minimum exposure value (e.g., 0.0833 for 1 month)
        verbose: Whether to print summary
        
    Returns:
        Tuple of (train_filtered, test_filtered)
    """
    if loss_cap is None:
        print("  No loss cap applied")
        return train_df.copy(), test_df.copy()
    
    print(f"\n=== Applying Loss Cap: {loss_cap:,} (type: {loss_type}) ===")
    
    # 1. Filter records based on loss_type
    if loss_type == "bi":
        train_mask = train_df[ee_bi_col] > 0
        test_mask = test_df[ee_bi_col] > 0
        print(f"  Filter: Dropping records where {ee_bi_col} = 0")
    elif loss_type == "pd":
        train_mask = train_df[ee_pd_col] > 0
        test_mask = test_df[ee_pd_col] > 0
        print(f"  Filter: Dropping records where {ee_pd_col} = 0")
    elif loss_type == "bi+pd":
        train_mask = (train_df[ee_bi_col] > 0) & (train_df[ee_pd_col] > 0)
        test_mask = (test_df[ee_bi_col] > 0) & (test_df[ee_pd_col] > 0)
        print(f"  Filter: Dropping records where {ee_bi_col} = 0 OR {ee_pd_col} = 0")
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'bi', 'pd', or 'bi+pd'")
    
    train_filtered = train_df[train_mask].copy()
    test_filtered = test_df[test_mask].copy()
    
    print(f"  Records dropped - Train: {(~train_mask).sum():,}, Test: {(~test_mask).sum():,}")
    print(f"  Records kept - Train: {len(train_filtered):,}, Test: {len(test_filtered):,}")
    
    # 2. Apply exposure floor (prevents extreme PP from tiny exposures)
    if exposure_floor:
        import numpy as np
        n_floor_train_bi = (train_filtered[ee_bi_col] < exposure_floor).sum() if ee_bi_col in train_filtered.columns else 0
        n_floor_test_bi = (test_filtered[ee_bi_col] < exposure_floor).sum() if ee_bi_col in test_filtered.columns else 0
        
        if ee_bi_col in train_filtered.columns:
            train_filtered[ee_bi_col] = np.maximum(train_filtered[ee_bi_col], exposure_floor)
        if ee_bi_col in test_filtered.columns:
            test_filtered[ee_bi_col] = np.maximum(test_filtered[ee_bi_col], exposure_floor)
        
        if loss_type in ["bi+pd", "pd"]:
            n_floor_train_pd = (train_filtered[ee_pd_col] < exposure_floor).sum() if ee_pd_col in train_filtered.columns else 0
            n_floor_test_pd = (test_filtered[ee_pd_col] < exposure_floor).sum() if ee_pd_col in test_filtered.columns else 0
            
            if ee_pd_col in train_filtered.columns:
                train_filtered[ee_pd_col] = np.maximum(train_filtered[ee_pd_col], exposure_floor)
            if ee_pd_col in test_filtered.columns:
                test_filtered[ee_pd_col] = np.maximum(test_filtered[ee_pd_col], exposure_floor)
        
        if verbose and (n_floor_train_bi + n_floor_test_bi > 0):
            print(f"  Exposure floor ({exposure_floor}):")
            print(f"    {ee_bi_col} - Train: {n_floor_train_bi:,}, Test: {n_floor_test_bi:,}")
            if loss_type in ["bi+pd", "pd"]:
                print(f"    {ee_pd_col} - Train: {n_floor_train_pd:,}, Test: {n_floor_test_pd:,}")
    
    # 3. Calculate loss
    if loss_type == "bi":
        loss_train = train_filtered[pp_bi_col] * train_filtered[ee_bi_col]
        loss_test = test_filtered[pp_bi_col] * test_filtered[ee_bi_col]
    elif loss_type == "pd":
        loss_train = train_filtered[pp_pd_col] * train_filtered[ee_pd_col]
        loss_test = test_filtered[pp_pd_col] * test_filtered[ee_pd_col]
    elif loss_type == "bi+pd":
        loss_bi_train = train_filtered[pp_bi_col] * train_filtered[ee_bi_col]
        loss_pd_train = train_filtered[pp_pd_col] * train_filtered[ee_pd_col]
        loss_train = loss_bi_train + loss_pd_train
        
        loss_bi_test = test_filtered[pp_bi_col] * test_filtered[ee_bi_col]
        loss_pd_test = test_filtered[pp_pd_col] * test_filtered[ee_pd_col]
        loss_test = loss_bi_test + loss_pd_test
    
    # 4. Apply cap
    n_cap_train = (loss_train > loss_cap).sum()
    n_cap_test = (loss_test > loss_cap).sum()
    
    if loss_type == "bi+pd":
        # Proportional scaling for combined loss
        cap_mask_train = loss_train > loss_cap
        cap_mask_test = loss_test > loss_cap
        
        if cap_mask_train.any():
            ratio_train = loss_cap / loss_train[cap_mask_train]
            train_filtered.loc[cap_mask_train, pp_bi_col] = train_filtered.loc[cap_mask_train, pp_bi_col] * ratio_train
            train_filtered.loc[cap_mask_train, pp_pd_col] = train_filtered.loc[cap_mask_train, pp_pd_col] * ratio_train
        
        if cap_mask_test.any():
            ratio_test = loss_cap / loss_test[cap_mask_test]
            test_filtered.loc[cap_mask_test, pp_bi_col] = test_filtered.loc[cap_mask_test, pp_bi_col] * ratio_test
            test_filtered.loc[cap_mask_test, pp_pd_col] = test_filtered.loc[cap_mask_test, pp_pd_col] * ratio_test
    else:
        # Single loss type - direct capping
        pp_col = pp_bi_col if loss_type == "bi" else pp_pd_col
        ee_col = ee_bi_col if loss_type == "bi" else ee_pd_col
        
        cap_mask_train = loss_train > loss_cap
        cap_mask_test = loss_test > loss_cap
        
        if cap_mask_train.any():
            train_filtered.loc[cap_mask_train, pp_col] = loss_cap / train_filtered.loc[cap_mask_train, ee_col]
        
        if cap_mask_test.any():
            test_filtered.loc[cap_mask_test, pp_col] = loss_cap / test_filtered.loc[cap_mask_test, ee_col]
    
    if verbose and (n_cap_train + n_cap_test > 0):
        print(f"  Loss capped at {loss_cap:,}:")
        print(f"    Train: {n_cap_train:,} records ({n_cap_train/len(train_filtered)*100:.4f}%)")
        print(f"    Test: {n_cap_test:,} records ({n_cap_test/len(test_filtered)*100:.4f}%)")
    
    return train_filtered, test_filtered


def generate_top_targets_report(
    df,
    target_col,
    pp_bi_col="pp_bi",
    pp_pd_col="pp_pd",
    ee_bi_col="ee_bi_imps",
    ee_pd_col="ee_pd_imps",
    incurred_bi_col="incurred_raw_bi_imps",
    incurred_pd_col="incurred_raw_pd_imps",
    glm_col=None,
    n=100
):
    """
    Generate report of top N records by target value (before capping).
    
    Args:
        df: Dataframe with all columns
        target_col: Target column name (e.g., "target" = pp/glm)
        pp_bi_col, pp_pd_col: PP column names
        ee_bi_col, ee_pd_col: Exposure column names
        incurred_bi_col, incurred_pd_col: Incurred loss columns
        glm_col: GLM prediction column (optional)
        n: Number of top records
        
    Returns:
        DataFrame with top N records
    """
    # Get top N by target
    top_n = df.nlargest(n, target_col).copy()
    
    # Calculate losses
    top_n['loss_bi'] = top_n[pp_bi_col] * top_n[ee_bi_col]
    if pp_pd_col in top_n.columns and ee_pd_col in top_n.columns:
        top_n['loss_pd'] = top_n[pp_pd_col] * top_n[ee_pd_col]
        top_n['loss_total'] = top_n['loss_bi'] + top_n['loss_pd']
    else:
        top_n['loss_pd'] = 0
        top_n['loss_total'] = top_n['loss_bi']
    
    # Select columns for report
    report_cols = [target_col, pp_bi_col, ee_bi_col, 'loss_bi']
    
    if pp_pd_col in top_n.columns:
        report_cols.extend([pp_pd_col, ee_pd_col, 'loss_pd', 'loss_total'])
    
    if incurred_bi_col in top_n.columns:
        report_cols.append(incurred_bi_col)
    if incurred_pd_col in top_n.columns:
        report_cols.append(incurred_pd_col)
    if glm_col and glm_col in top_n.columns:
        report_cols.append(glm_col)
    
    return top_n[report_cols]


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




def load_base_margin(output_base, use_base_margin=False, dataset="train", verbose=True):
    """
    Load base_margin file if enabled.
    
    Args:
        output_base: Path to output directory
        use_base_margin: Whether to load base_margin
        dataset: "train", "test", or score name (e.g., "holdout")
        verbose: Print status messages
    
    Returns:
        numpy array of base_margin values, or None if disabled/not found
    """
    import os
    import pandas as pd
    
    if not use_base_margin:
        if verbose:
            print(f"\n* Base margin disabled (use_base_margin=False)")
        return None
    
    # Determine file path based on dataset type
    if dataset in ["train", "test"]:
        file_path = f"{output_base}/data/04c_{dataset}_base_margin.parquet"
    else:
        # Scoring data
        file_path = f"{output_base}/data/04c_score_{dataset}_base_margin.parquet"
    
    if os.path.exists(file_path):
        if verbose:
            print(f"\n* Loading base_margin ({dataset})...")
        base_margin = pd.read_parquet(file_path)["base_margin"].values
        if verbose:
            print(f"  mean={base_margin.mean():.4f}, count={len(base_margin):,}")
        return base_margin
    else:
        if verbose:
            print(f"\n* Base margin file not found: {file_path}")
            print(f"  Continuing without base_margin")
        return None


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
    
    # Preserve index from y_actual if it has one (e.g., vin_date)
    if isinstance(y_actual, pd.Series) and y_actual.index.name is not None:
        df.index = y_actual.index
    
    # Add extra columns
    for col_name, col_data in extra_cols.items():
        df[col_name] = col_data
    
    # Save with reset_index to make index a regular column (for portability)
    output_file = f"{results_dir}/{stage}_predictions_{split}.parquet"
    df.reset_index().to_parquet(output_file, index=False)
    
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
    """
    Load GLM predictions for scoring. Returns (glm_preds, matched_indices).
    Drops rows without GLM predictions (inner join).
    """
    if target_method!="residual": return None, None
    glm_source=score_cfg.get("glm_source","lookup")
    join_key=cfg["data"]["join_key"]
    if glm_source=="column":
        col=score_cfg.get("glm_column")
        if col not in df.columns: raise ValueError(f"Missing {col}")
        return df[col].copy(), df.index  # Return all indices when using column
    if join_key not in df.columns:
        raise ValueError(f"Missing {join_key}. Use glm_source=column")
    
    # Get GLM prediction column (required for residual method)
    glm_col = cfg.get("model", {}).get("glm_prediction_column")
    if not glm_col:
        raise ValueError("model.glm_prediction_column required for target_method=residual")
    
    from model_utils import load_glm_predictions
    paths=cfg["machines"][pc_id]["paths"]
    # Use drop_missing=True to get matched indices
    return load_glm_predictions(paths["aux_data_path"],cfg["data"]["control_model_file"],df,join_key,glm_col,drop_missing=True)



def create_model_analysis_table(
    df,
    target,
    pred_gbm,
    glm_pred,
    exposure,
    pp_actual_from_capped,
    pred_pp,
    dataset_name="data"
):
    """
    Create analysis table with incurred calculations.
    
    Args:
        df: Original dataframe (for index)
        target: GBM target (pp/glm for residual)
        pred_gbm: GBM raw prediction
        glm_pred: GLM prediction
        exposure: Exposure values
        pp_actual_from_capped: PP from capped loss (capped_loss / ee)
        pred_pp: Predicted PP (pred_gbm * glm for residual)
    
    Returns:
        DataFrame with analysis columns (formulas in column names)
    """
    import pandas as pd
    
    analysis = pd.DataFrame(index=df.index)
    
    # Basic inputs
    analysis['exposure (ee)'] = exposure
    analysis['glm_pred (glm)'] = glm_pred
    analysis['pp_from_capped_loss (capped_loss/ee)'] = pp_actual_from_capped
    analysis['pred_pp (pred_gbm * glm)'] = pred_pp
    analysis['target (pp_capped / glm)'] = target
    analysis['pred_gbm (GBM output)'] = pred_gbm
    
    # Incurred calculations
    analysis['incurred_capped (pp_capped * ee)'] = pp_actual_from_capped * exposure
    analysis['incurred_pred (pred_pp * ee)'] = pred_pp * exposure
    analysis['incurred_from_target (target * glm * ee)'] = target * glm_pred * exposure
    analysis['incurred_from_pred_gbm (pred_gbm * glm * ee)'] = pred_gbm * glm_pred * exposure
    
    # Weighted by exposure
    analysis['target_weighted (target * ee)'] = target * exposure
    analysis['pred_gbm_weighted (pred_gbm * ee)'] = pred_gbm * exposure
    
    return analysis


def create_decile_analysis_table(
    analysis_df,
    pred_col='pred_pp (pred_gbm * glm)',
    exposure_col='exposure (ee)',
    n_deciles=10
):
    """
    Aggregate analysis table by decile.
    
    Args:
        analysis_df: Output from create_model_analysis_table()
        pred_col: Column for decile ranking
        exposure_col: Exposure column
        n_deciles: Number of deciles
        
    Returns:
        DataFrame with decile-level aggregations
    """
    import pandas as pd
    
    df = analysis_df.copy()
    df['decile'] = pd.qcut(df[pred_col].rank(method='first'), n_deciles, labels=range(1, n_deciles + 1))
    
    agg_cols = {
        'exposure (ee)': 'sum',
        'incurred_capped (pp_capped * ee)': 'sum',
        'incurred_pred (pred_pp * ee)': 'sum',
        'incurred_from_target (target * glm * ee)': 'sum',
        'incurred_from_pred_gbm (pred_gbm * glm * ee)': 'sum',
        'target_weighted (target * ee)': 'sum',
        'pred_gbm_weighted (pred_gbm * ee)': 'sum',
    }
    
    decile_table = df.groupby('decile').agg(
        n_records=('decile', 'count'),
        **{k: (k, v) for k, v in agg_cols.items()}
    ).reset_index()
    
    # Add averages
    decile_table['avg_target (target_wt / ee)'] = (
        decile_table['target_weighted (target * ee)'] / decile_table['exposure (ee)']
    )
    decile_table['avg_pred_gbm (pred_gbm_wt / ee)'] = (
        decile_table['pred_gbm_weighted (pred_gbm * ee)'] / decile_table['exposure (ee)']
    )
    decile_table['avg_pp_act (incurred_capped / ee)'] = (
        decile_table['incurred_capped (pp_capped * ee)'] / decile_table['exposure (ee)']
    )
    decile_table['avg_pp_pred (incurred_pred / ee)'] = (
        decile_table['incurred_pred (pred_pp * ee)'] / decile_table['exposure (ee)']
    )
    
    return decile_table
