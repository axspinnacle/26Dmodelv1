"""
Model validation utilities for feature matching and sanity checks
"""
import pandas as pd
import xgboost as xgb


def validate_features_match(model, data, data_name="data"):
    """
    Validate that data features match model expectations.
    
    Args:
        model: XGBRegressor or XGBClassifier model
        data: pd.DataFrame with features
        data_name: str, name for error messages
    
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If features don't match, with detailed message
    """
    # Get model's expected features
    booster = model.get_booster()
    model_num_features = booster.num_features()
    model_feature_names = booster.feature_names
    
    # Get data features
    data_num_features = data.shape[1]
    data_feature_names = list(data.columns)
    
    print(f"\n* Feature Validation:")
    print(f"  Model expects: {model_num_features} features")
    print(f"  {data_name} has: {data_num_features} features")
    
    # Check feature count
    if model_num_features != data_num_features:
        error_msg = (
            f"\n{'='*60}\n"
            f"FEATURE COUNT MISMATCH\n"
            f"{'='*60}\n"
            f"Model expects: {model_num_features} features\n"
            f"{data_name} has: {data_num_features} features\n"
            f"Difference: {abs(model_num_features - data_num_features)} features\n"
            f"\n"
            f"This will cause XGBoost prediction errors.\n"
            f"{'='*60}\n"
        )
        
        # If model has feature names, show which are missing/extra
        if model_feature_names:
            model_set = set(model_feature_names)
            data_set = set(data_feature_names)
            
            missing_in_data = model_set - data_set
            extra_in_data = data_set - model_set
            
            if missing_in_data:
                error_msg += f"\nMissing in {data_name} (model expects them):\n"
                for feat in sorted(missing_in_data)[:10]:
                    error_msg += f"  - {feat}\n"
                if len(missing_in_data) > 10:
                    error_msg += f"  ... and {len(missing_in_data) - 10} more\n"
            
            if extra_in_data:
                error_msg += f"\nExtra in {data_name} (model doesn't use):\n"
                for feat in sorted(extra_in_data)[:10]:
                    error_msg += f"  - {feat}\n"
                if len(extra_in_data) > 10:
                    error_msg += f"  ... and {len(extra_in_data) - 10} more\n"
        
        error_msg += f"\n{'='*60}\n"
        raise ValueError(error_msg)
    
    # Check feature names if available
    if model_feature_names:
        if model_feature_names != data_feature_names:
            # Same count but different names - warning only
            print(f"  ⚠ Warning: Feature names differ (but count matches)")
            print(f"    This may cause issues if feature order is wrong")
            
            # Show first few mismatches
            mismatches = [(m, d) for m, d in zip(model_feature_names, data_feature_names) if m != d]
            if mismatches:
                print(f"    First 3 mismatches:")
                for m, d in mismatches[:3]:
                    print(f"      Model: {m} | Data: {d}")
        else:
            print(f"  ✓ Feature names match")
    else:
        print(f"  ⚠ Model has no feature names stored (can't verify names)")
    
    print(f"  ✓ Validation passed\n")
    return True


def validate_predictions(predictions, data, check_range=True):
    """
    Validate model predictions are sensible.
    
    Args:
        predictions: array-like predictions
        data: pd.DataFrame, original data
        check_range: bool, check if predictions are in reasonable range
    
    Returns:
        bool: True if valid
    """
    import numpy as np
    
    print(f"\n* Prediction Validation:")
    print(f"  Shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
    print(f"  Min: {np.min(predictions):.2f}")
    print(f"  Max: {np.max(predictions):.2f}")
    print(f"  Mean: {np.mean(predictions):.2f}")
    print(f"  Median: {np.median(predictions):.2f}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(predictions)):
        n_nan = np.sum(np.isnan(predictions))
        raise ValueError(f"Predictions contain {n_nan} NaN values!")
    
    if np.any(np.isinf(predictions)):
        n_inf = np.sum(np.isinf(predictions))
        raise ValueError(f"Predictions contain {n_inf} Inf values!")
    
    # Check for negative predictions (for regression models predicting costs/amounts)
    if check_range and np.any(predictions < 0):
        n_negative = np.sum(predictions < 0)
        print(f"  ⚠ Warning: {n_negative} negative predictions (may be valid depending on target)")
    
    print(f"  ✓ Predictions valid\n")
    return True
