"""
Optuna-based HPO for XGBoost with actuarial metrics
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error
from hpo_metrics import calculate_model_metrics, calculate_lift_opt_score


def run_optuna_search(
    X_train, y_train, w_train,
    X_test, y_test, w_test,
    param_grid, base_xgb_params,
    monotone_constraints, exposure_col,
    scoring_config,
    n_trials=100,
    glm_pred_test=None,
    target_method="direct"
):
    """
    Run Optuna optimization over hyperparameters using actuarial lift metrics.
    
    Returns dict with: best_params, best_metrics, best_score, results_df
    """
    from model_utils import inverse_transform
    
    bins = scoring_config.get('bins', 10)
    fit_threshold = scoring_config.get('fit_threshold', 0.70)
    steepness = scoring_config.get('steepness', 20.0)
    power_weight = scoring_config.get('power_weight', 0.25)
    power_norm_cap = scoring_config.get('power_norm_cap', 0.50)
    penalty_type = scoring_config.get('penalty_type', 'linear')
    
    results = []
    best_score_so_far = -float('inf')
    
    print(f"  n_trials: {n_trials}")
    print(f"  penalty_type: {penalty_type}")
    print(f"  param_grid:")
    for k, v in param_grid.items():
        print(f"    {k}: {v}")
    
    def objective(trial):
        # Suggest parameters from ranges
        param_dict = {}
        for param_name, param_values in param_grid.items():
            param_dict[param_name] = trial.suggest_categorical(param_name, param_values)
        
        # Merge with base params
        full_params = base_xgb_params.copy()
        full_params.update(param_dict)
        
        # Fix eval_metric for tweedie if variance_power changed
        if "tweedie_variance_power" in param_dict and "eval_metric" in full_params:
            if "tweedie" in full_params["eval_metric"]:
                vp = param_dict["tweedie_variance_power"]
                full_params["eval_metric"] = f"tweedie-nloglik@{vp}"
        
        # Add monotonicity constraints
        if monotone_constraints:
            full_params['monotone_constraints'] = monotone_constraints
        
        # Train model
        model = xgb.XGBRegressor(**full_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            sample_weight_eval_set=[w_test],
            verbose=0
        )
        
        # Predict (raw GBM output)
        pred_test_raw = model.predict(X_test)
        
        # Apply inverse transform if using residual method
        # Maintain index alignment: w_test has vin_date index, use it for all Series
        if target_method == "residual":
            if glm_pred_test is None:
                raise ValueError("glm_pred_test required for residual method")
            # Create Series with proper index from w_test (which has vin_date from test_orig)
            pred_test_series = pd.Series(pred_test_raw, index=w_test.index)
            y_test_series = pd.Series(y_test, index=w_test.index)
            
            # inverse_transform preserves Series type and index
            pred_test_pp = inverse_transform(pred_test_series, glm_pred_test, target_method)
            y_test_pp = inverse_transform(y_test_series, glm_pred_test, target_method)
        else:
            pred_test_pp = pd.Series(pred_test_raw, index=w_test.index)
            y_test_pp = pd.Series(y_test, index=w_test.index)
        
        # Prepare data for metrics (using PP space)
        # All Series now have matching vin_date index - no alignment errors
        test_eval = pd.DataFrame({
            'pred': pred_test_pp,
            'act_weighted': y_test_pp * w_test,
            'pred_weighted': pred_test_pp * w_test,
            exposure_col: w_test
        })
        
        # Calculate actuarial metrics
        fit_quality, model_power = calculate_model_metrics(test_eval, exposure_col, bins=bins)
        score = calculate_lift_opt_score(
            fit_quality, model_power,
            fit_threshold=fit_threshold,
            steepness=steepness,
            power_weight=power_weight,
            power_norm_cap=power_norm_cap,
            penalty_type=penalty_type
        )
        
        # Calculate MAE for reference (in PP space)
        mae = mean_absolute_error(y_test_pp, pred_test_pp, sample_weight=w_test)
        
        # Store results
        result_row = param_dict.copy()
        result_row.update({
            'fit_quality': fit_quality,
            'model_power': model_power,
            'lift_opt_score': score,
            'mae': mae
        })
        results.append(result_row)
        
        # Track best score and print trial info
        nonlocal best_score_so_far
        if score > best_score_so_far:
            best_score_so_far = score
            improved = " ★ NEW BEST"
        else:
            improved = ""
        
        trial_num = len(results)
        print(f"  Trial {trial_num:3d}: score={score:.4f} (best={best_score_so_far:.4f}){improved}")
        print(f"    fit={fit_quality:.4f}, power={model_power:.4f}, mae={mae:.2f}")
        print(f"    params: {param_dict}")
        
        return score
    
    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('lift_opt_score', ascending=False).reset_index(drop=True)
    
    best_params = study.best_params
    best_score = study.best_value
    
    # Get best metrics from results
    best_metrics = results_df.iloc[0][['fit_quality', 'model_power', 'lift_opt_score', 'mae']].to_dict()
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_score': best_score,
        'results_df': results_df,
        'study': study
    }
