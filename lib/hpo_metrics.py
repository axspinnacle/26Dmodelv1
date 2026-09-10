"""
HPO Metrics for Actuarial Model Optimization

Functions for calculating fit quality, model power, and combined lift optimization score.
Based on actuarial best practices for hyperparameter optimization.
"""

import numpy as np
import pandas as pd


def calculate_model_metrics(data, weight_name, bins=10):
    """
    Calculate fit_quality and model_power from model predictions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Must contain: 'pred', 'act_weighted', weight_name
    weight_name : str
        Column name for exposure/weights
    bins : int
        Number of deciles (typically 10)
    
    Returns
    -------
    tuple
        (fit_quality, model_power)
        
    Notes
    -----
    - fit_quality: 1 - weighted average of |pred/act - 1| across deciles
                   Range: 0-1 (can be negative if very poor)
    - model_power: Weighted average of |pred/overall_mean - 1| across deciles
                   Range: 0-unbounded (higher = more separation)
    """
    test_data = data.copy()
    
    # Create deciles based on predictions (descending order)
    test_data['decile'] = (
        round(
            test_data.sort_values(by='pred', ascending=False)[weight_name].cumsum() / 
            test_data[weight_name].sum(), 
            2
        ) * bins
    ).apply(np.floor)
    
    test_data['decile'] = np.where(
        test_data['decile'] + 1 > bins, 
        bins, 
        test_data['decile'] + 1
    )
    
    # Aggregate by decile
    x = test_data.groupby(['decile'], dropna=False).agg({
        weight_name: 'sum',
        'act_weighted': 'sum',
        'pred_weighted': 'sum'
    }).reset_index()
    
    # Calculate per-policy values
    x['pp_act'] = x['act_weighted'] / x[weight_name]
    x['pp_pred'] = x['pred_weighted'] / x[weight_name]
    
    # FIT QUALITY: How well predicted matches actual per decile
    x['decile_error'] = (
        abs(x['pp_pred'] / x['pp_act'] - 1)
        .replace(-np.inf, np.nan)
        .replace(np.inf, np.nan)
        .fillna(1)
    )
    x['decile_error_sp'] = x['decile_error'] * x[weight_name]
    fit_quality = 1 - x['decile_error_sp'].sum() / x[weight_name].sum()
    
    # MODEL POWER: How much predictions deviate from overall mean
    tot_pp = x['act_weighted'].sum() / x[weight_name].sum()
    x['diff_unity'] = abs(x['pp_pred'] / tot_pp - 1)
    x['diff_unity'] = x['diff_unity'] * x[weight_name]
    model_power = x['diff_unity'].sum() / x[weight_name].sum()
    
    return fit_quality, model_power


def calculate_lift_opt_score(
    fit_quality: float,
    model_power: float,
    fit_threshold: float = 0.70,
    steepness: float = 20.0,
    power_weight: float = 0.25,
    power_norm_cap: float = 0.50
) -> float:
    """
    Computes an actuarial hyperparameter objective balancing calibration and separation.
    
    Parameters
    ----------
    fit_quality : float
        Bounded metric measuring bucket alignment/monotonicity (<= 1.0)
    model_power : float
        Unbounded metric measuring decile spread / lift differential
    fit_threshold : float, default=0.70
        Minimum acceptable fit quality before heavy penalization
    steepness : float, default=20.0
        Controls how sharply the penalty kicks in below threshold
    power_weight : float, default=0.25
        Relative importance of separation once fit is acceptable
    power_norm_cap : float, default=0.50
        Typical strong benchmark for model_power to scale to [0, 1]
    
    Returns
    -------
    float
        Combined optimization score (higher is better)
        
    Notes
    -----
    The sigmoid gate ensures that if fit drops below threshold, score degrades
    rapidly regardless of power. This prevents optimizer from chasing separation
    at the cost of calibration.
    
    Score = gate × (fit_quality + power_weight × norm_power)
    
    where:
    - gate: sigmoid penalty based on fit vs threshold
    - norm_power: model_power normalized to [0,1] using soft saturation
    """
    # 1. Normalize model_power smoothly using soft saturation
    # Normalized power reaches ~0.63 at cap, asymptoting toward 1.0
    norm_power = 1.0 - np.exp(-model_power / power_norm_cap)
    
    # 2. Compute smooth penalty gate based on fit quality
    # Gate evaluates near 1.0 when fit > threshold; collapses to 0.0 when fit drops
    gate = 1.0 / (1.0 + np.exp(-steepness * (fit_quality - fit_threshold)))
    
    # 3. Base composite score (fit-dominated)
    base_score = fit_quality + (power_weight * norm_power)
    
    # 4. Gated final evaluation score
    return float(gate * base_score)


def compute_fit_quality(y_true, y_pred, weights):
    """
    Compute fit quality from raw arrays (simpler interface for notebooks).
    
    Parameters
    ----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted values
    weights : array-like
        Sample weights (exposure)
    
    Returns
    -------
    float
        Fit quality score (0-1, higher is better)
    """
    # Create dataframe for calculate_model_metrics
    df = pd.DataFrame({
        'act_weighted': y_true * weights,
        'pred_weighted': y_pred * weights,
        'pred': y_pred,
        'weight': weights
    })
    
    fit_quality, _ = calculate_model_metrics(df, 'weight', bins=10)
    return fit_quality


def compute_model_power(y_true, y_pred, weights):
    """
    Compute model power from raw arrays (simpler interface for notebooks).
    
    Parameters
    ----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted values
    weights : array-like
        Sample weights (exposure)
    
    Returns
    -------
    float
        Model power score (0+, higher is better)
    """
    # Create dataframe for calculate_model_metrics
    df = pd.DataFrame({
        'act_weighted': y_true * weights,
        'pred_weighted': y_pred * weights,
        'pred': y_pred,
        'weight': weights
    })
    
    _, model_power = calculate_model_metrics(df, 'weight', bins=10)
    return model_power
