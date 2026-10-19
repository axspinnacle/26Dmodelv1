"""
HPO Grid Search Runner

Orchestrates hyperparameter optimization for XGBoost models using actuarial metrics.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import product
from hpo_metrics import evaluate_split, score_from_config


def build_trial_params(param_dict, base_xgb_params, monotone_constraints):
    """Merge trial params into base params."""
    full_params = base_xgb_params.copy()
    full_params.update(param_dict)
    if "tweedie_variance_power" in param_dict and "tweedie" in str(full_params.get("eval_metric", "")):
        full_params["eval_metric"] = f"tweedie-nloglik@{param_dict['tweedie_variance_power']}"
    if monotone_constraints:
        full_params['monotone_constraints'] = monotone_constraints
    return full_params


def fit_trial_model(full_params, X_train, y_train, w_train, X_test, y_test, w_test):
    """Train XGBoost with early stopping on test."""
    model = xgb.XGBRegressor(**full_params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[w_test],
        verbose=0
    )
    return model


def get_best_iteration(model):
    """Trees used (best_iteration + 1), or n_estimators without early stopping."""
    try:
        return int(model.best_iteration) + 1
    except (AttributeError, ValueError, TypeError):
        return int(model.get_params().get('n_estimators') or 100)


def run_single_trial(
    param_dict, base_xgb_params, monotone_constraints,
    X_train, y_train, w_train, X_test, y_test, w_test,
    glm_pred_train, glm_pred_test, target_method, scoring_config
):
    """Train one trial; return params + train/test metrics."""
    bins = (scoring_config or {}).get('bins', 10)
    full_params = build_trial_params(param_dict, base_xgb_params, monotone_constraints)
    model = fit_trial_model(full_params, X_train, y_train, w_train, X_test, y_test, w_test)
    n_trees = get_best_iteration(model)
    rng = (0, n_trees)

    te = evaluate_split(model, X_test, y_test, w_test, glm_pred_test, target_method, bins, rng)
    if target_method == "residual" and glm_pred_train is None:
        tr = {'fit_quality': np.nan, 'model_power': np.nan, 'mae': np.nan}
    else:
        tr = evaluate_split(model, X_train, y_train, w_train, glm_pred_train, target_method, bins, rng)

    score = score_from_config(te['fit_quality'], te['model_power'], scoring_config)
    row = dict(param_dict)
    row.update({
        'fit_quality': te['fit_quality'],
        'model_power': te['model_power'],
        'lift_opt_score': score,
        'mae': te['mae'],
        'train_fit': tr['fit_quality'],
        'train_power': tr['model_power'],
        'test_fit': te['fit_quality'],
        'test_power': te['model_power'],
        'fit_gap': tr['fit_quality'] - te['fit_quality'],
        'power_gap': tr['model_power'] - te['model_power'],
        'best_iteration': n_trees
    })
    return row


def run_grid_search(
    X_train, y_train, w_train,
    X_test, y_test, w_test,
    param_grid, base_xgb_params,
    monotone_constraints, exposure_col,
    scoring_config,
    glm_pred_test=None,
    target_method="direct",
    glm_pred_train=None
):
    """
    Run grid search over hyperparameters using actuarial lift metrics.

    Returns dict with: best_params, best_metrics, best_score, results_df
    """
    total = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"  Grid size: {total} combinations")
    print(f"  penalty_type: {(scoring_config or {}).get('penalty_type', 'linear')}")
    print(f"  n_estimators cap: {base_xgb_params.get('n_estimators')}, early_stopping_rounds: {base_xgb_params.get('early_stopping_rounds')}")
    print(f"  param_grid:")
    for k, v in param_grid.items():
        print(f"    {k}: {v}")

    results = []
    best_score, best_idx = -float('inf'), 0
    for i, params in enumerate(product(*param_grid.values())):
        param_dict = dict(zip(param_grid.keys(), params))
        row = run_single_trial(
            param_dict, base_xgb_params, monotone_constraints,
            X_train, y_train, w_train, X_test, y_test, w_test,
            glm_pred_train, glm_pred_test, target_method, scoring_config
        )
        row['trial_number'] = i + 1
        results.append(row)
        if row['lift_opt_score'] > best_score:
            best_score, best_idx = row['lift_opt_score'], i
            print(f"  [{i+1}/{total}] ★ NEW BEST: score={best_score:.4f} (test_fit={row['test_fit']:.4f}, train_fit={row['train_fit']:.4f}, power={row['test_power']:.4f}, trees={row['best_iteration']})")
        elif (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] score={row['lift_opt_score']:.4f} | best so far={best_score:.4f} ({best_idx+1}/{total})")

    print(f"\n* Grid search complete")
    print(f"  Best score: {best_score:.4f}")
    return package_results(results, list(param_grid.keys()))


def _py(v):
    """numpy scalar -> python (for yaml)."""
    return v.item() if hasattr(v, 'item') else v


def package_results(results, param_names):
    """Build results_df (sorted by score), best_params, best_metrics."""
    results_df = pd.DataFrame(results).sort_values('lift_opt_score', ascending=False).reset_index(drop=True)
    best = results_df.iloc[0]
    best_row = max(results, key=lambda r: r['lift_opt_score'])
    best_params = {k: _py(best_row[k]) for k in param_names}
    metric_cols = ['fit_quality', 'model_power', 'lift_opt_score', 'mae', 'train_fit', 'train_power',
                   'test_fit', 'test_power', 'fit_gap', 'best_iteration', 'trial_number']
    best_metrics = {k: _py(best[k]) for k in metric_cols if k in best.index}
    return {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_score': float(best['lift_opt_score']),
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

    best_params_file = f'{results_dir}/05b_best_params.yaml'
    with open(best_params_file, 'w') as f:
        yaml.dump({k: _py(v) for k, v in best_params.items()}, f, default_flow_style=False, sort_keys=True)
    print(f'  Best params: {best_params_file}')

    metrics_file = f'{results_dir}/05b_metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump({k: _py(v) for k, v in best_metrics.items()}, f, default_flow_style=False)
    print(f'  Metrics: {metrics_file}')

    results_file = f'{results_dir}/05b_hpo_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f'  All results: {results_file}')

    print(f'\n[OK] HPO results saved')

