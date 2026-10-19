"""
HPO diagnostic plots (saved as PNG, displayed in notebook)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULT_COLS = {'fit_quality', 'model_power', 'lift_opt_score', 'mae', 'train_fit', 'train_power',
               'test_fit', 'test_power', 'fit_gap', 'power_gap', 'best_iteration', 'trial_number'}


def _param_cols(results_df):
    return [c for c in results_df.columns if c not in RESULT_COLS and results_df[c].nunique() > 1]


def _fmt(v):
    return int(v) if isinstance(v, (float, np.floating)) and float(v).is_integer() else v


def _save_show(fig, path, show=True):
    from IPython.display import Image, display
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    if show:
        display(Image(path))


def plot_optimization_history(results_df, path, show=True):
    df = results_df.sort_values('trial_number')
    best = df.loc[df['lift_opt_score'].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(df['trial_number'], df['lift_opt_score'], s=18, alpha=0.6, label='Trial score')
    ax.plot(df['trial_number'], df['lift_opt_score'].cummax(), color='#d62728', lw=2, label='Best so far')
    _mark_best(ax, best['trial_number'], best['lift_opt_score'], f"Best (trial {int(best['trial_number'])})")
    ax.set_xlabel('Trial')
    ax.set_ylabel('lift_opt_score (test)')
    ax.set_title('HPO Optimization History')
    ax.grid(alpha=0.3)
    ax.legend()
    _save_show(fig, path, show)


def param_importance(results_df, study=None):
    # Optuna importance; if that fails, spread of mean score across values
    params = _param_cols(results_df)
    if study is not None:
        try:
            import optuna
            imp = optuna.importance.get_param_importances(study)
            imp = {k: v for k, v in imp.items() if k in params}
            if imp:
                return pd.Series(imp).sort_values(), 'Optuna fANOVA'
        except Exception as e:
            print(f'  [WARN] Optuna importance failed ({e}); using fallback')
    rng = {p: results_df.groupby(p)['lift_opt_score'].mean().agg(lambda s: s.max() - s.min()) for p in params}
    s = pd.Series(rng)
    return (s / s.sum()).sort_values(), 'Range of mean score (normalized)'


def plot_param_importance(results_df, path, study=None, show=True):
    imp, method = param_importance(results_df, study)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(imp) + 1.5))
    ax.barh(imp.index, imp.values, color='#1f77b4')
    for i, v in enumerate(imp.values):
        ax.text(v, i, f' {v:.2f}', va='center')
    ax.set_xlabel('Importance')
    ax.set_title(f'Hyperparameter Importance ({method})')
    ax.grid(alpha=0.3, axis='x')
    _save_show(fig, path, show)
    return imp


def plot_score_by_param(results_df, path, show=True):
    params = _param_cols(results_df)
    best = results_df.loc[results_df['lift_opt_score'].idxmax()]
    n = len(params)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow), squeeze=False)
    for ax, p in zip(axes.flat, params):
        vals = sorted(results_df[p].unique())
        data = [results_df.loc[results_df[p] == v, 'lift_opt_score'].values for v in vals]
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticks(range(1, len(vals) + 1))
        ax.set_xticklabels([str(v) for v in vals])
        for v, box in zip(vals, bp['boxes']):
            box.set_facecolor('gold' if v == best[p] else '#c6dbef')
        for i, d in enumerate(data):
            ax.text(i + 1, ax.get_ylim()[0], f'n={len(d)}', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'{p} (best={_fmt(best[p])})')
        ax.set_ylabel('lift_opt_score')
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle('Score by Hyperparameter Value (highlighted = best trial)', fontweight='bold')
    fig.tight_layout()
    _save_show(fig, path, show)


def _mark_best(ax, x, y, label='Best'):
    ax.scatter([x], [y], marker='o', s=60, facecolors='none', edgecolors='red', linewidths=1.5, zorder=5, label=label)


def plot_fit_vs_power(results_df, path, show=True):
    best = results_df.loc[results_df['lift_opt_score'].idxmax()]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(results_df['test_power'], results_df['test_fit'], c=results_df['lift_opt_score'],
                    cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='lift_opt_score')
    _mark_best(ax, best['test_power'], best['test_fit'])
    ax.set_xlabel('Model Power (test)')
    ax.set_ylabel('Fit Quality (test)')
    ax.set_title('Fit Quality vs Model Power')
    ax.grid(alpha=0.3)
    ax.legend()
    _save_show(fig, path, show)


def plot_train_vs_test_fit(results_df, path, show=True):
    df = results_df.dropna(subset=['train_fit', 'test_fit'])
    if df.empty:
        print('  [WARN] No train_fit values; skipping train vs test plot')
        return
    best = df.loc[df['lift_opt_score'].idxmax()]
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(df['test_fit'], df['train_fit'], c=df['lift_opt_score'], cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='lift_opt_score')
    lo = min(df['test_fit'].min(), df['train_fit'].min()) - 0.01
    hi = max(df['test_fit'].max(), df['train_fit'].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.6, label='train = test')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # Lift fit drops on train when the model memorizes train claims, so below the line = overfit
    ax.text(0.97, 0.03, 'Below line: train fit < test fit\n(overfit to train claims)', transform=ax.transAxes,
            ha='right', fontsize=9)
    _mark_best(ax, best['test_fit'], best['train_fit'])
    ax.set_xlabel('Fit Quality (test)')
    ax.set_ylabel('Fit Quality (train)')
    ax.set_title('Train vs Test Fit Quality per Trial')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center')
    _save_show(fig, path, show)


def best_trial_learning_curve(best_params, base_xgb_params, monotone_constraints,
                              X_train, y_train, w_train, glm_train,
                              X_test, y_test, w_test, glm_test,
                              target_method, bins=10, eval_every=25, n_trees=None):
    # Refit best params without early stopping, score every eval_every trees
    import xgboost as xgb
    from hpo_runner import build_trial_params
    from hpo_metrics import evaluate_split

    clean = {k: (int(v) if isinstance(v, float) and v.is_integer() and k in ('max_depth', 'n_estimators') else v)
             for k, v in best_params.items()}
    params = build_trial_params(clean, base_xgb_params, monotone_constraints)
    params.pop('early_stopping_rounds', None)
    if n_trees:
        params['n_estimators'] = int(n_trees)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, sample_weight=w_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              sample_weight_eval_set=[w_train, w_test], verbose=0)
    ev = model.evals_result()
    metric = list(ev['validation_0'].keys())[0]
    loss = pd.DataFrame({'tree': np.arange(1, len(ev['validation_0'][metric]) + 1),
                         'train_loss': ev['validation_0'][metric],
                         'test_loss': ev['validation_1'][metric]})

    n = len(loss)
    steps = sorted(set(list(range(eval_every, n + 1, eval_every)) + [n]))
    rows = []
    for k in steps:
        tr = evaluate_split(model, X_train, y_train, w_train, glm_train, target_method, bins, (0, k))
        te = evaluate_split(model, X_test, y_test, w_test, glm_test, target_method, bins, (0, k))
        rows.append({'tree': k, 'train_fit': tr['fit_quality'], 'test_fit': te['fit_quality'],
                     'train_power': tr['model_power'], 'test_power': te['model_power']})
    return loss, pd.DataFrame(rows), metric


def plot_learning_curve(loss_df, metrics_df, metric_name, best_iteration, path, show=True):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ax = axes[0]
    ax.plot(loss_df['tree'], loss_df['train_loss'], label='Train')
    ax.plot(loss_df['tree'], loss_df['test_loss'], label='Test')
    ax.set_title(f'Loss ({metric_name})')
    for a, (col, ttl) in zip(axes[1:], [('fit', 'Fit Quality'), ('power', 'Model Power')]):
        a.plot(metrics_df['tree'], metrics_df[f'train_{col}'], marker='o', ms=3, label='Train')
        a.plot(metrics_df['tree'], metrics_df[f'test_{col}'], marker='o', ms=3, label='Test')
        a.set_title(ttl)
    for a in axes:
        if best_iteration:
            a.axvline(best_iteration, color='gray', ls='--', label=f'Early stop ({int(best_iteration)})')
        a.set_xlabel('Trees')
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle('Best Trial Learning Curve (no early stopping)', fontweight='bold')
    fig.tight_layout()
    _save_show(fig, path, show)


def print_best_summary(results_df, imp=None):
    df = results_df.sort_values('lift_opt_score', ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    params = _param_cols(df)
    print('\n* Best trial summary:')
    print(f"  Trial {int(best['trial_number'])}: score={best['lift_opt_score']:.4f}")
    print(f"  Test : fit={best['test_fit']:.4f}, power={best['test_power']:.4f}")
    print(f"  Train: fit={best['train_fit']:.4f}, power={best['train_power']:.4f}, fit_gap={best['fit_gap']:+.4f}")
    print(f"  Trees used: {int(best['best_iteration'])}")
    if len(df) > 1:
        print(f"  Margin over #2: {best['lift_opt_score'] - df.iloc[1]['lift_opt_score']:.4f}")
    order = list(imp.index[::-1]) if imp is not None else params
    print('  Param: best value -> mean score (best value vs others):')
    for p in order:
        if p not in params:
            continue
        m_best = df.loc[df[p] == best[p], 'lift_opt_score'].mean()
        m_oth = df.loc[df[p] != best[p], 'lift_opt_score'].mean()
        imp_s = f', importance={imp[p]:.2f}' if imp is not None and p in imp else ''
        print(f"    {p}={_fmt(best[p])}: {m_best:.4f} vs {m_oth:.4f}{imp_s}")



def run_hpo_plots(results_df, output_base, study=None,
                  best_params=None, base_xgb_params=None, monotone_constraints=None,
                  X_train=None, y_train=None, w_train=None, glm_train=None,
                  X_test=None, y_test=None, w_test=None, glm_test=None,
                  target_method='direct', bins=10, eval_every=25, show=True):
    # Saves 05b_plot_*.png to results/ and shows them
    out = f'{output_base}/results'
    print('\n* Creating HPO plots...')
    imp = None
    steps = [
        ('history', lambda: plot_optimization_history(results_df, f'{out}/05b_plot_history.png', show)),
        ('importance', lambda: plot_param_importance(results_df, f'{out}/05b_plot_importance.png', study, show)),
        ('score_by_param', lambda: plot_score_by_param(results_df, f'{out}/05b_plot_score_by_param.png', show)),
        ('fit_vs_power', lambda: plot_fit_vs_power(results_df, f'{out}/05b_plot_fit_vs_power.png', show)),
        ('train_vs_test', lambda: plot_train_vs_test_fit(results_df, f'{out}/05b_plot_train_vs_test.png', show)),
    ]
    for name, fn in steps:
        try:
            r = fn()
            if name == 'importance':
                imp = r
        except Exception as e:
            print(f'  [WARN] Plot {name} failed: {e}')

    if best_params is not None and X_train is not None:
        try:
            bi = int(results_df.loc[results_df['lift_opt_score'].idxmax(), 'best_iteration'])
            cap = int(base_xgb_params.get('n_estimators') or 3000)
            n_trees = min(cap, max(2 * bi, bi + 200))
            print(f'\n* Refitting best trial for learning curve ({n_trees} trees)...')
            loss, mdf, metric = best_trial_learning_curve(
                best_params, base_xgb_params, monotone_constraints,
                X_train, y_train, w_train, glm_train, X_test, y_test, w_test, glm_test,
                target_method, bins, eval_every, n_trees)
            mdf.to_csv(f'{out}/05b_learning_curve_best.csv', index=False)
            plot_learning_curve(loss, mdf, metric, bi, f'{out}/05b_plot_learning_curve.png', show)
        except Exception as e:
            print(f'  [WARN] Learning curve failed: {e}')

    try:
        print_best_summary(results_df, imp)
    except Exception as e:
        print(f'  [WARN] Summary failed: {e}')

