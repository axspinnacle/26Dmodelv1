"""
Optuna-based HPO for XGBoost with actuarial metrics
"""
import optuna
from hpo_runner import run_single_trial, package_results


def run_optuna_search(
    X_train, y_train, w_train,
    X_test, y_test, w_test,
    param_grid, base_xgb_params,
    monotone_constraints, exposure_col,
    scoring_config,
    n_trials=100,
    glm_pred_test=None,
    target_method="direct",
    glm_pred_train=None
):
    """
    Run Optuna optimization over hyperparameters using actuarial lift metrics.
    Duplicate param combos are not retrained (cached score reused, not logged twice).

    Returns dict with: best_params, best_metrics, best_score, results_df, study
    """
    print(f"  n_trials: {n_trials}")
    print(f"  penalty_type: {(scoring_config or {}).get('penalty_type', 'linear')}")
    print(f"  n_estimators cap: {base_xgb_params.get('n_estimators')}, early_stopping_rounds: {base_xgb_params.get('early_stopping_rounds')}")
    print(f"  param_grid:")
    for k, v in param_grid.items():
        print(f"    {k}: {v}")

    results = []
    cache = {}
    state = {'best': -float('inf'), 'dups': 0}

    def objective(trial):
        param_dict = {k: trial.suggest_categorical(k, v) for k, v in param_grid.items()}
        key = tuple(sorted(param_dict.items()))
        if key in cache:
            state['dups'] += 1
            return cache[key]

        row = run_single_trial(
            param_dict, base_xgb_params, monotone_constraints,
            X_train, y_train, w_train, X_test, y_test, w_test,
            glm_pred_train, glm_pred_test, target_method, scoring_config
        )
        row['trial_number'] = len(results) + 1
        results.append(row)
        score = row['lift_opt_score']
        cache[key] = score

        improved = ""
        if score > state['best']:
            state['best'] = score
            improved = " ★ NEW BEST"
        print(f"  Trial {row['trial_number']:3d}: score={score:.4f} (best={state['best']:.4f}){improved}")
        print(f"    test_fit={row['test_fit']:.4f}, train_fit={row['train_fit']:.4f}, power={row['test_power']:.4f}, trees={row['best_iteration']}, mae={row['mae']:.2f}")
        print(f"    params: {param_dict}")
        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n* Optuna complete: {len(results)} unique trials trained, {state['dups']} duplicates skipped")
    out = package_results(results, list(param_grid.keys()))
    out['study'] = study
    return out
