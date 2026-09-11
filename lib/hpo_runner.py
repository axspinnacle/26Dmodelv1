"""
HPO Grid Search Runner

Orchestrates hyperparameter optimization for XGBoost models using actuarial metrics.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import product
from sklearn.metrics import mean_absolute_error
from hpo_metrics import calculate_model_metrics, calculate_lift_opt_score


def run_grid_search(
    X_train, y_train, w_train,
    X_test, y_test, w_test,
    param_grid, base_xgb_params, n_estimators,
    monotone_constraints, exposure_col,
    scoring_config
):
    """
    Run grid search over hyperparameters using actuarial lift metrics.
    
    Returns dict with: best_params, best_metrics, best_score, results_df
    """
    bins = scoring_config.get('bins', 10)
    fit_threshold = scoring_config.get('fit_threshold', 0.70)
    steepness = scoring_config.get('steepness', 20.0)
    power_weight = scoring_config.get('power_weight', 0.25)
    power_norm_cap = scoring_config.get('power_norm_cap', 0.50)
    
    best_score = -float('inf')
    best_params = None
    best_metrics = None
    best_iteration = 0
    results = []
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"  Grid size: {total_combinations} combinations")
    print(f"  n_estimators: {n_estimators}")
    
    # Grid search
    for i, params in enumerate(product(*param_grid.values())):
        param_dict = dict(zip(param_grid.keys(), params))
        param_dict['n_estimators'] = n_estimators
        
        # Merge with base params
        full_params = base_xgb_params.copy()
        full_params.update(param_dict)
        
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
        
        # Predict
        pred_test = model.predict(X_test)
        
        # Prepare data for metrics
        test_eval = pd.DataFrame({
            'pred': pred_test,
            'act_weighted': y_test * w_test,
            'pred_weighted': pred_test * w_test,
            exposure_col: w_test
        })
        
        # Calculate actuarial metrics
        fit_quality, model_power = calculate_model_metrics(test_eval, exposure_col, bins=bins)
        score = calculate_lift_opt_score(
            fit_quality, model_power,
            fit_threshold=fit_threshold,
            steepness=steepness,
            power_weight=power_weight,
            power_norm_cap=power_norm_cap
        )
        
        # Calculate MAE for reference
        mae = mean_absolute_error(y_test, pred_test, sample_weight=w_test)
        
        # Store results
        result_row = {k: v for k, v in param_dict.items() if k in param_grid or k == 'n_estimators'}
        result_row.update({
            'fit_quality': fit_quality,
            'model_power': model_power,
            'lift_opt_score': score,
            'mae': mae
        })
        results.append(result_row)
        
        # Track best
        is_new_best = score > best_score
        if is_new_best:
            best_score = score
            best_iteration = i + 1
            best_params = {k: v for k, v in param_dict.items() if k in param_grid or k == "n_estimators"}
            best_metrics = {
                'fit_quality': fit_quality,
                'model_power': model_power,
                'lift_opt_score': score,
                'mae': mae
            }
            print(f"  [{i+1}/{total_combinations}] ★ NEW BEST: score={score:.4f} (fit={fit_quality:.4f}, power={model_power:.4f})")
        elif (i + 1) % 10 == 0:
            # Progress update every 10 iterations
            print(f"  [{i+1}/{total_combinations}] score={score:.4f} | best so far={best_score:.4f} ({best_iteration}/{total_combinations}, fit={best_metrics['fit_quality']:.2f}, power={best_metrics['model_power']:.2f})")
    
    print(f"\n* Grid search complete")
    print(f"  Best score: {best_score:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('lift_opt_score', ascending=False)
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_score': best_score,
        'results_df': results_df
    }


def save_hpo_results(output_base, best_params, best_metrics, results_df):
    """
    Save HPO results to disk.
    """
    import yaml
    import os
    
    results_dir = f'{output_base}/results'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f'\n* Saving results...')
    
    # Save best params as YAML
    best_params_file = f'{results_dir}/05b_best_params.yaml'
    with open(best_params_file, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=True)
    print(f'  Best params: {best_params_file}')
    
    # Save best metrics
    metrics_file = f'{results_dir}/05b_metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(best_metrics, f, default_flow_style=False)
    print(f'  Metrics: {metrics_file}')
    
    # Save all trial results
    results_file = f'{results_dir}/05b_hpo_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f'  All results: {results_file}')
    
    print(f'\n[OK] HPO results saved')
