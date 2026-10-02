"""
Data preparation functions for model training.
Handles data loading, loss capping, and filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def prepare_model_data(
    output_base,
    target_col,
    exposure_col,
    target_method,
    loss_cap=None,
    loss_type="bi",
    exposure_floor=None,
    verbose=True
):
    """
    Load and prepare all data for model training with loss cap filtering.
    This is the main orchestration function called from notebooks.
    
    Args:
        output_base: Output directory path
        target_col: Target column name (e.g., "pp_bi")
        exposure_col: Exposure column name (e.g., "ee_bi_imps")
        target_method: "residual" or "direct"
        loss_cap: Loss cap value (None to skip)
        loss_type: "bi", "pd", or "bi+pd"
        exposure_floor: Minimum exposure value (e.g., 0.0833)
        verbose: Print progress
        
    Returns:
        dict with keys:
            X_train, X_test: Feature matrices
            y_train, y_test: Transformed targets
            w_train, w_test: Weights (exposure)
            pp_train, pp_test: Original PP (after cap)
            glm_train, glm_test: GLM predictions (if residual method)
            train_orig, test_orig: Original dataframes (filtered)
            top_100_report: DataFrame with top 100 targets before cap
    """
    from model_utils import transform_target, apply_loss_cap, generate_top_targets_report
    
    print(f"\n{'='*60}")
    print(f"PREPARING MODEL DATA")
    print(f"{'='*60}")
    
    # 1. Load encoded features
    if verbose:
        print(f"\n* Loading encoded features...")
    X_train = pd.read_parquet(f"{output_base}/data/04c_train_encoded.parquet")
    X_test = pd.read_parquet(f"{output_base}/data/04c_test_encoded.parquet")
    
    # 2. Load original data for target/exposure
    train_orig = pd.read_parquet(f"{output_base}/data/04b_train.parquet")
    test_orig = pd.read_parquet(f"{output_base}/data/04b_test.parquet")
    
    if verbose:
        print(f"  Loaded - Train: {len(train_orig):,}, Test: {len(test_orig):,}")
    
    # Load GLM and apply loss cap (continued...)
    return _prepare_model_data_part2(
        X_train, X_test, train_orig, test_orig,
        output_base, target_col, exposure_col, target_method,
        loss_cap, loss_type, exposure_floor, verbose
    )


def _prepare_model_data_part2(
    X_train, X_test, train_orig, test_orig,
    output_base, target_col, exposure_col, target_method,
    loss_cap, loss_type, exposure_floor, verbose
):
    """Part 2 of prepare_model_data - handles GLM loading and loss capping."""
    from model_utils import transform_target, apply_loss_cap, generate_top_targets_report
    
    # 3. Load GLM predictions if using residual method
    glm_train = glm_test = None
    glm_train_df = glm_test_df = None
    
    if target_method == "residual":
        if verbose:
            print(f"\n* Loading GLM predictions...")
        glm_train_df = pd.read_parquet(f"{output_base}/data/04c_train_glm.parquet")
        glm_test_df = pd.read_parquet(f"{output_base}/data/04c_test_glm.parquet")
        
        # Filter out NaN GLM predictions
        glm_train_orig = len(glm_train_df)
        glm_test_orig = len(glm_test_df)
        glm_train_df = glm_train_df[glm_train_df['glm_pred'].notna()]
        glm_test_df = glm_test_df[glm_test_df['glm_pred'].notna()]
        glm_train_removed = glm_train_orig - len(glm_train_df)
        glm_test_removed = glm_test_orig - len(glm_test_df)
        
        if verbose and (glm_train_removed > 0 or glm_test_removed > 0):
            print(f"  Removed NaN GLM predictions:")
            print(f"    Train: {glm_train_removed:,} ({100*glm_train_removed/glm_train_orig:.2f}%)")
            print(f"    Test: {glm_test_removed:,} ({100*glm_test_removed/glm_test_orig:.2f}%)")
        
        # Use index intersection
        train_common_idx = X_train.index.intersection(glm_train_df.index)
        test_common_idx = X_test.index.intersection(glm_test_df.index)
        
        train_removed = len(X_train) - len(train_common_idx)
        test_removed = len(X_test) - len(test_common_idx)
        
        if verbose and (train_removed > 0 or test_removed > 0):
            print(f"  Rows without valid GLM:")
            print(f"    Train: {train_removed:,} ({100*train_removed/len(X_train):.2f}%)")
            print(f"    Test: {test_removed:,} ({100*test_removed/len(X_test):.2f}%)")
        
        # Filter all datasets to common index
        train_orig = train_orig.loc[train_common_idx]
        X_train = X_train.loc[train_common_idx]
        test_orig = test_orig.loc[test_common_idx]
        X_test = X_test.loc[test_common_idx]
        
        # Extract GLM predictions
        glm_train = glm_train_df.loc[train_common_idx, 'glm_pred'].values
        glm_test = glm_test_df.loc[test_common_idx, 'glm_pred'].values
        
        if verbose:
            print(f"  GLM loaded: train={len(glm_train):,}, test={len(glm_test):,}")
    
    # Continue with part 3...
    return _prepare_model_data_part3(
        X_train, X_test, train_orig, test_orig,
        glm_train, glm_test, glm_train_df, glm_test_df,
        output_base, target_col, exposure_col, target_method,
        loss_cap, loss_type, exposure_floor, verbose
    )


def _prepare_model_data_part3(
    X_train, X_test, train_orig, test_orig,
    glm_train, glm_test, glm_train_df, glm_test_df,
    output_base, target_col, exposure_col, target_method,
    loss_cap, loss_type, exposure_floor, verbose
):
    """Part 3 - Generate report, apply loss cap, and finalize."""
    from model_utils import transform_target, apply_loss_cap, generate_top_targets_report
    
    # 4. Generate top 100 report BEFORE capping
    top_100_report = None
    if loss_cap is not None and verbose:
        print(f"\n* Generating top 100 targets report (before cap)...")
        temp_train = train_orig.copy()
        temp_train['target'] = temp_train[target_col]
        if target_method == "residual" and glm_train is not None:
            temp_train['glm_pred'] = glm_train
            temp_train['target'] = temp_train[target_col] / temp_train['glm_pred']
        
        top_100_report = generate_top_targets_report(
            df=temp_train,
            target_col='target',
            pp_bi_col=target_col,
            pp_pd_col="pp_pd" if "pp_pd" in temp_train.columns else None,
            ee_bi_col=exposure_col,
            ee_pd_col="ee_pd_imps" if "ee_pd_imps" in temp_train.columns else None,
            incurred_bi_col="incurred_raw_bi_imps" if "incurred_raw_bi_imps" in temp_train.columns else None,
            incurred_pd_col="incurred_raw_pd_imps" if "incurred_raw_pd_imps" in temp_train.columns else None,
            glm_col='glm_pred' if target_method == "residual" else None,
            n=100
        )
        print(f"  Top 100 report generated ({len(top_100_report)} records)")
    
    # 5. Apply loss cap and filter records
    if loss_cap is not None:
        train_orig, test_orig = apply_loss_cap(
            train_df=train_orig,
            test_df=test_orig,
            loss_cap=loss_cap,
            loss_type=loss_type,
            pp_bi_col=target_col,
            pp_pd_col="pp_pd",
            ee_bi_col=exposure_col,
            ee_pd_col="ee_pd_imps",
            exposure_floor=exposure_floor,
            verbose=verbose
        )
        
        # Re-align all datasets
        X_train = X_train.loc[train_orig.index]
        X_test = X_test.loc[test_orig.index]
        if glm_train is not None:
            glm_train = glm_train_df.loc[train_orig.index, 'glm_pred'].values
            glm_test = glm_test_df.loc[test_orig.index, 'glm_pred'].values
    
    # 6. Extract final target and exposure
    pp_train = train_orig[target_col].values
    pp_test = test_orig[target_col].values
    w_train = train_orig[exposure_col].values
    w_test = test_orig[exposure_col].values
    
    # 7. Transform target
    if verbose:
        print(f"\n* Transforming target (method: {target_method})...")
    y_train = transform_target(pp_train, glm_train, target_method)
    y_test = transform_target(pp_test, glm_test, target_method)
    
    if verbose:
        print(f"\n* Final dataset sizes:")
        print(f"  Train: {X_train.shape[0]:,} records x {X_train.shape[1]} features")
        print(f"  Test: {X_test.shape[0]:,} records x {X_test.shape[1]} features")
        print(f"{'='*60}\n")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'w_train': w_train,
        'w_test': w_test,
        'pp_train': pp_train,
        'pp_test': pp_test,
        'glm_train': glm_train,
        'glm_test': glm_test,
        'train_orig': train_orig,
        'test_orig': test_orig,
        'top_100_report': top_100_report
    }
