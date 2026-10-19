"""
Gini / Lorenz curve (adapted from edited_20240314_symbols_gbm_functions.ipynb)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lorenz_curve(actual, pred, weight=None):
    # Sort by pred (low to high). x = cumulative weight share, y = cumulative actual*weight share
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    w = np.ones_like(a) if weight is None else np.asarray(weight, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p) & np.isfinite(w)
    a, p, w = a[ok], p[ok], w[ok]
    o = np.argsort(p, kind='mergesort')
    a, w = a[o], w[o]
    x = np.r_[0.0, np.cumsum(w) / w.sum()]
    y = np.r_[0.0, np.cumsum(a * w) / (a * w).sum()]
    return x, y


def gini(actual, pred, weight=None):
    # 1 - 2 * area under Lorenz curve. weight=None gives the unweighted version
    x, y = lorenz_curve(actual, pred, weight)
    return float(1 - 2 * np.trapezoid(y, x))


def test_gini():
    actual = [1, 2, 4, 8]
    pred = [1, 2, 4, 8]
    print(gini(actual, pred))

    assert abs(gini(actual, pred) - 0.383333) < 1e-5
    assert abs(gini(actual, pred[::-1]) + 0.383333) < 1e-5
    assert abs(gini([1, 1, 1, 1], [4, 3, 2, 1])) < 1e-9
    # weight 2 on a row = that row duplicated
    assert abs(gini(actual, pred, [1, 1, 1, 2]) - gini([1, 2, 4, 8, 8], [1, 2, 4, 8, 8])) < 1e-9
    print('test_gini passed')


def _read_cols(path, wanted):
    import pyarrow.parquet as pq
    have = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in wanted if c in have])


def load_gini_inputs(output_base, cfg, pc_id, score_name='holdout'):
    # Train/test from 05c debug files, holdout from 08 predictions
    data = {}
    for s in ['train', 'test']:
        f = f'{output_base}/results/05c_debug_{s}.parquet'
        if os.path.exists(f):
            d = pd.read_parquet(f, columns=['pp_actual', 'pp_predicted', 'glm_pred', 'exposure'])
            data[s] = pd.DataFrame({'actual': d['pp_actual'].values, 'pred': d['pp_predicted'].values,
                                    'glm': d['glm_pred'].values, 'weight': d['exposure'].values})
        else:
            print(f'  [WARN] Missing {f}')

    f = f'{output_base}/scoring/{score_name}/predictions.parquet'
    if os.path.exists(f):
        target = cfg['experiment']['target']
        exposure = cfg['experiment']['exposure']
        join_key = cfg['data']['join_key']
        h = _read_cols(f, [target, exposure, 'prediction', 'glm_pred', join_key])
        if 'glm_pred' not in h.columns:
            from model_utils import load_glm_predictions
            print('  glm_pred not in holdout file; loading from control file')
            glm, idx = load_glm_predictions(cfg['machines'][pc_id]['paths']['aux_data_path'],
                                            cfg['data']['control_model_file'], h, join_key,
                                            cfg['model']['glm_prediction_column'], drop_missing=True)
            h = h.loc[idx]
            h['glm_pred'] = glm.loc[idx].values
        data[score_name] = pd.DataFrame({'actual': h[target].values, 'pred': h['prediction'].values,
                                         'glm': h['glm_pred'].values, 'weight': h[exposure].values})
    else:
        print(f'  [WARN] Missing {f}')
    return data


def gini_summary(data):
    # Results table: Gini of final predicted PP, exposure weighted
    rows = []
    for name, d in data.items():
        rows.append({'dataset': name, 'n_records': len(d), 'exposure': d['weight'].sum(),
                     'gini_predicted': gini(d['actual'], d['pred'], d['weight'])})
    return pd.DataFrame(rows)


def gini_debug(data):
    # Debug table: predicted vs GLM, perfect-model Gini, unweighted version
    rows = []
    for name, d in data.items():
        g_pred = gini(d['actual'], d['pred'], d['weight'])
        g_glm = gini(d['actual'], d['glm'], d['weight'])
        g_max = gini(d['actual'], d['actual'], d['weight'])
        rows.append({'dataset': name, 'gini_predicted': g_pred, 'gini_glm': g_glm,
                     'gini_predicted_minus_glm': g_pred - g_glm, 'gini_perfect': g_max,
                     'norm_gini_predicted': g_pred / g_max, 'norm_gini_glm': g_glm / g_max,
                     'gini_predicted_unweighted': gini(d['actual'], d['pred'])})
    return pd.DataFrame(rows)


def plot_lorenz(name, d, gini_value, path, n_points=2000, show=True):
    # One Lorenz chart per dataset: predicted vs random
    from IPython.display import Image, display
    x, y = lorenz_curve(d['actual'], d['pred'], d['weight'])
    i = np.linspace(0, len(x) - 1, n_points).astype(int)
    x, y = x[i], y[i]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(x, y, color='#1f77b4', lw=2, label=f'Predicted (Gini {gini_value:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.fill_between(x, y, x, color='#1f77b4', alpha=0.15)
    ax.set_title(f'{name.title()} - Gini {gini_value:.3f} (n={len(d):,})', fontweight='bold')
    ax.set_xlabel('Cumulative exposure (sorted by predicted, low to high)')
    ax.set_ylabel('Cumulative actual loss')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    if show:
        display(Image(path))


def _show_save(df, title, path):
    print(f'\n* {title}:')
    print(df.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    df.to_csv(path, index=False)
    print(f'  Saved: {path}')


def run_gini_analysis(data, output_base, stage='08b', show=True):
    out = f'{output_base}/results'
    summary = gini_summary(data)
    _show_save(summary, 'Gini summary (exposure weighted)', f'{out}/{stage}_gini_summary.csv')
    _show_save(gini_debug(data), 'Gini debug', f'{out}/{stage}_gini_debug.csv')

    old = f'{out}/{stage}_gini_lorenz.png'
    if os.path.exists(old):
        os.remove(old)
    g = summary.set_index('dataset')['gini_predicted']
    for name, d in data.items():
        plot_lorenz(name, d, g[name], f'{out}/{stage}_gini_lorenz_{name}.png', show=show)
    return summary
