"""
Stage 07a/07b SHAP analysis: loading, importance tables, per-feature plots.

07a (summary): importance per model feature and per original column (one-hot levels summed
               per record), train vs test, zero-SHAP features, coverage, beeswarms.
07b (feature plots): residual + SHAP range plots for the top N features, saved to plots/07b/.

All SHAP rows are aligned to the rows the model was trained/evaluated on (05c predictions),
weighted by exposure. Usage: see templates/07a_shap_summary.ipynb and 07b_shap_feature_plots.ipynb.
"""

import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from pathlib import Path

SHAP_EXTRA_COLS = ('base_value', 'weight')


def shap_feature_columns(shap_df):
    """Model feature columns in a SHAP frame (drops base_value / weight)."""
    return [c for c in shap_df.columns if c not in SHAP_EXTRA_COLS]


def load_shap_data(output_base, dataset, features=None, with_values=True):
    """
    Load SHAP values (+ encoded feature values) for 'train' or 'test', restricted to rows
    present in 05c predictions (rows the model used), with exposure as 'weight'.

    Args:
        features: subset of feature columns to load (None = all)
        with_values: also load encoded feature values (04c) and actual/pred
    Returns:
        shap_df (SHAP + 'weight'), data_df (features + actual, pred, weight) or None
    """
    out = Path(output_base)
    shap_file = out / 'results' / f'06_shap_{dataset}.parquet'
    cols = None if features is None else list(features) + ['base_value']
    shap_df = pd.read_parquet(shap_file, columns=cols)
    preds = pd.read_parquet(out / 'results' / f'05c_predictions_{dataset}.parquet')
    idx = shap_df.index.intersection(preds.index)
    shap_df = shap_df.loc[idx]
    shap_df['weight'] = preds.loc[idx, 'exposure'].values
    if not with_values:
        return shap_df, None

    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    feat = shap_feature_columns(shap_df)
    key = shap_df.index.name or 'vin_date'
    enc_file = out / 'data' / f'04c_{dataset}_encoded.parquet'
    # Stream-filter to SHAP rows only (never loads the full 11M-row train file)
    tbl = ds.dataset(str(enc_file)).to_table(columns=feat + [key], filter=pc.field(key).isin(list(idx)))
    data_df = tbl.to_pandas()
    if key in data_df.columns:
        data_df = data_df.set_index(key)
    data_df = data_df.loc[idx]
    data_df['actual'] = preds.loc[idx, 'actual'].values
    data_df['pred'] = preds.loc[idx, 'pred'].values
    data_df['weight'] = shap_df['weight'].values
    return shap_df, data_df


def shap_stats(shap_df, weight_col='weight'):
    """
    Per-column SHAP statistics (one row per SHAP column):
      mean_abs_shap             exposure-weighted mean |SHAP| per record  (importance)
      mean_abs_shap_unweighted  plain mean |SHAP| per record
      mean_shap                 exposure-weighted mean signed SHAP
      share_nonzero             exposure share of records with SHAP != 0
      n_rows, exposure
    """
    feat = shap_feature_columns(shap_df)
    w = shap_df[weight_col].to_numpy(dtype='float64')
    v = np.nan_to_num(shap_df[feat].to_numpy(dtype='float64'))
    a = np.abs(v)
    W = w.sum()
    return pd.DataFrame({
        'mean_abs_shap': a.T @ w / W,
        'mean_abs_shap_unweighted': a.mean(axis=0),
        'mean_shap': v.T @ w / W,
        'share_nonzero': (a > 0).T @ w / W,
        'n_rows': len(v),
        'exposure': W,
    }, index=feat)


def shap_importance(shap_df, weight_col='weight'):
    """Exposure-weighted mean |SHAP| per feature, sorted. Returns Series (name=feature)."""
    return shap_stats(shap_df, weight_col)['mean_abs_shap'].sort_values(ascending=False)


def feature_source_map(encoding_summary):
    """Encoded column -> original column, from 04c_encoding_summary.csv (one-hot -> source)."""
    src = {}
    for _, r in encoding_summary.iterrows():
        outs = str(r.get('output_columns', ''))
        if r.get('encoding_type') in ('skip', 'DROP') or outs in ('', 'SKIPPED', 'nan'):
            continue
        for c in outs.split(','):
            src[c] = r['original_column']
    return src


def encoding_type_map(encoding_summary):
    """Original column -> encoding type (from 04c_encoding_summary.csv)."""
    return dict(zip(encoding_summary['original_column'], encoding_summary['encoding_type']))


def column_shap(shap_df, source_map, weight_col='weight'):
    """
    Record-level SHAP per ORIGINAL column: one-hot levels are summed within each record
    (SHAP is additive, so the sum is that record's total effect of the column).
    Single-column features are copied unchanged. Returns DataFrame (+ weight column).
    """
    feat = shap_feature_columns(shap_df)
    groups = {}
    for c in feat:
        groups.setdefault(source_map.get(c, c), []).append(c)
    out = {}
    for col, members in groups.items():
        if len(members) == 1:
            out[col] = shap_df[members[0]].to_numpy()
        else:
            out[col] = shap_df[members].to_numpy(dtype='float64').sum(axis=1).astype('float32')
    res = pd.DataFrame(out, index=shap_df.index)
    res[weight_col] = shap_df[weight_col].to_numpy()
    return res


def _rank_table(stats_test, stats_train, key):
    """Join test/train stats, add pct / cum pct / rank (ranked by test importance)."""
    df = stats_test.add_prefix('test_').join(stats_train.add_prefix('train_'), how='outer')
    num = [c for c in df.columns if not c.endswith(('n_rows', 'exposure'))]
    df[num] = df[num].fillna(0.0)
    df = df.sort_values('test_mean_abs_shap', ascending=False)
    for s in ('test', 'train'):
        tot = df[f'{s}_mean_abs_shap'].sum()
        df[f'{s}_pct'] = 100 * df[f'{s}_mean_abs_shap'] / tot if tot else 0.0
        df[f'{s}_rank'] = df[f'{s}_mean_abs_shap'].rank(ascending=False, method='first').astype('Int64')
    df['test_cum_pct'] = df['test_pct'].cumsum()
    df['train_cum_pct'] = df.sort_values('train_rank')['train_pct'].cumsum()
    df['rank_change'] = df['train_rank'] - df['test_rank']
    df.loc[df['test_mean_abs_shap'] == 0, 'test_rank'] = pd.NA
    df.index.name = key
    return df.reset_index()


_STAT_ORDER = ['mean_abs_shap', 'pct', 'cum_pct', 'rank', 'mean_abs_shap_unweighted', 'mean_shap',
               'share_nonzero', 'n_rows', 'exposure']


def _ordered(df, lead):
    cols = lead + [f'{s}_{k}' for s in ('test', 'train') for k in _STAT_ORDER] + ['rank_change']
    return df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]


def importance_table(stats_train, stats_test, source_map=None, enc_map=None):
    """
    Feature (model column) importance, ranked by test importance.
    Args: stats_* from shap_stats(); source_map from feature_source_map(); enc_map from encoding_type_map().
    Key columns used downstream: feature, test_mean_abs_shap, test_pct, test_cum_pct, test_rank,
    train_pct, train_rank.
    """
    df = _rank_table(stats_test, stats_train, 'feature')
    df['source_column'] = df['feature'].map(source_map or {}).fillna(df['feature'])
    df['encoding_type'] = df['source_column'].map(enc_map or {}).fillna('')
    return _ordered(df, ['feature', 'source_column', 'encoding_type'])


def importance_by_column(col_stats_train, col_stats_test, feature_table, enc_map=None):
    """
    Original-column importance from RECORD-LEVEL column SHAP (column_shap -> shap_stats).
    Also keeps the per-level sum (test/train_pct_sum_of_levels) for comparison only.
    """
    df = _rank_table(col_stats_test, col_stats_train, 'source_column')
    lv = feature_table.groupby('source_column').agg(
        n_encoded=('feature', 'size'),
        test_pct_sum_of_levels=('test_pct', 'sum'), train_pct_sum_of_levels=('train_pct', 'sum'))
    df = df.merge(lv, left_on='source_column', right_index=True, how='left')
    df['encoding_type'] = df['source_column'].map(enc_map or {}).fillna('')
    return _ordered(df, ['source_column', 'encoding_type', 'n_encoded'])


def coverage_counts(table, levels=(50, 80, 90, 95, 99)):
    """Number of features needed to reach each cumulative test-importance level (%)."""
    cum = table['test_cum_pct'].to_numpy()
    return {f'{p}pct': int(np.searchsorted(cum, p) + 1) for p in levels}


# ---------------------------------------------------------------- 07a charts

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def plot_importance_bars(table, value_col, label_col, top_n, title, path):
    """Horizontal bar chart of top_n rows by value_col (%), labels from label_col. Saves + closes."""
    t = table.nlargest(top_n, value_col)
    fig, ax = plt.subplots(figsize=(10, max(6, len(t) * 0.32)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(t)))
    ax.barh(range(len(t)), t[value_col], color=colors, edgecolor='black', linewidth=0.4)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t[label_col], fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(t[value_col]):
        ax.text(v, i, f' {v:.1f}%', va='center', fontsize=7)
    ax.set_xlabel('Share of total |SHAP| (%)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    return _save(fig, path)


def plot_cumulative_importance(table, coverage, path):
    """Cumulative test importance vs number of features, with coverage markers. Saves + closes."""
    used = table.loc[table['test_mean_abs_shap'] > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, len(used) + 1), used['test_cum_pct'], linewidth=2)
    for k, n in coverage.items():
        p = int(k.replace('pct', ''))
        ax.axhline(p, color='grey', linestyle=':', linewidth=0.8)
        ax.plot(n, p, 'o', color='crimson')
        ax.annotate(f'{p}%: {n} features', (n, p), textcoords='offset points', xytext=(6, -12), fontsize=8)
    ax.set_xlabel('Number of features (ranked by test importance)')
    ax.set_ylabel('Cumulative importance (%)')
    ax.set_title(f'Cumulative SHAP importance ({len(used)} of {len(table)} features used)', fontweight='bold')
    ax.set_ylim(0, 101)
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_beeswarm(shap_df, values_df, order, title, path, max_rows=50000, grey_columns=(), seed=0):
    """
    SHAP beeswarm with a FIXED row order (same as the matching importance chart).

    Args:
        shap_df: SHAP per record (+ 'weight'), columns include `order`
        values_df: feature values per record (same index); columns in grey_columns may be absent
        order: columns to plot, top to bottom (e.g. top 40 by test importance)
        max_rows: dots per row; records drawn with probability proportional to exposure,
                  so dot density reflects exposure like the (exposure-weighted) bar chart
        grey_columns: columns without a numeric value (one-hot groups) -> plotted grey
    """
    import shap
    w = shap_df['weight'].to_numpy(dtype='float64')
    n = len(shap_df)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False, p=w / w.sum()) if n > max_rows else np.arange(n)
    grey = set(grey_columns)
    vals = np.column_stack([
        np.full(len(idx), np.nan) if c in grey else
        pd.to_numeric(values_df[c].iloc[idx], errors='coerce').to_numpy(dtype='float64')
        for c in order])
    expl = shap.Explanation(values=shap_df[order].to_numpy(dtype='float64')[idx], data=vals,
                            feature_names=list(order))
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')  # grey (one-hot) rows
        shap.plots.beeswarm(expl, max_display=len(order), order=list(range(len(order))),
                            plot_size=(11, max(6, 0.38 * len(order))), show=False)
    fig = plt.gcf()
    plt.title(title, fontweight='bold')
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close('all')
    return str(path)


def plot_train_vs_test(table, top_n, path, label_col='feature'):
    """Train vs test importance (%) for the top_n test rows; points off the diagonal differ."""
    t = table.head(top_n)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(t['test_pct'], t['train_pct'], s=20)
    m = max(t['test_pct'].max(), t['train_pct'].max()) * 1.05
    ax.plot([0, m], [0, m], 'k--', linewidth=0.8)
    for _, r in t.head(10).iterrows():
        ax.annotate(r[label_col], (r['test_pct'], r['train_pct']), fontsize=7, xytext=(3, 3),
                    textcoords='offset points')
    ax.set_xlabel('Test importance (%)')
    ax.set_ylabel('Train importance (%)')
    what = 'columns' if label_col == 'source_column' else 'features'
    ax.set_title(f'Train vs test SHAP importance (top {top_n} {what})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    return _save(fig, path)



# ---------------------------------------------------------------- 07b feature plots

def safe_name(name):
    """File-system safe version of a feature name (spaces, slashes, etc. -> '_')."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_')


def feature_plot_path(plots_dir, rank, feature, dataset, kind):
    """plots/07b/{rank:02d}_{feature}_{dataset}_{kind}.png (rank prefix sorts by importance)."""
    return Path(plots_dir) / f'{rank:02d}_{safe_name(feature)}_{dataset}_{kind}.png'


def _residual_table(tab, feature, rank, dataset):
    t = tab.rename(columns={feature: 'bin'})[['bin', 'weight', 'act_rate', 'pred_rate']].copy()
    t['bin'] = t['bin'].astype(str)
    t.insert(0, 'dataset', dataset)
    t.insert(0, 'feature', feature)
    t.insert(0, 'rank', rank)
    return t


def make_feature_plots(features, frames, plots_dir, resid_cfg=None, range_cfg=None):
    """
    Residual + SHAP range plots for each feature and dataset.

    Args:
        features: feature names in rank order (rank = position + 1)
        frames: {'train': (shap_df, data_df), 'test': (shap_df, data_df)}
    Returns:
        plots    DataFrame[rank, feature, dataset, kind, path]
        tables   residual tables DataFrame[rank, feature, dataset, bin, weight, act_rate, pred_rate]
        failures DataFrame[rank, feature, dataset, kind, error]
    """
    from shap_utils import create_residual_plot, create_shap_range_plot
    resid_cfg = {**(resid_cfg or {}), 'print_table': False}   # tables go to CSV, not the notebook
    range_cfg = dict(range_cfg or {})
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plots, tables, failures = [], [], []

    for rank, f in enumerate(features, 1):
        for ds, (shap_df, data_df) in frames.items():
            if f not in data_df.columns or f not in shap_df.columns:
                failures.append({'rank': rank, 'feature': f, 'dataset': ds, 'kind': 'all',
                                 'error': 'feature not found'})
                continue
            for kind in ('residual', 'shap_range'):
                path = feature_plot_path(plots_dir, rank, f, ds, kind)
                try:
                    if kind == 'residual':
                        fig, tab = create_residual_plot(data_df, f, 'weight', dataset_label=ds.capitalize(), **resid_cfg)
                        tables.append(_residual_table(tab, f, rank, ds))
                    else:
                        fig, _ = create_shap_range_plot(shap_df, data_df, f, 'weight',
                                                        dataset_label=ds.capitalize(), **range_cfg)
                    fig.savefig(path, dpi=100, bbox_inches='tight')
                    plots.append({'rank': rank, 'feature': f, 'dataset': ds, 'kind': kind, 'path': str(path)})
                except Exception as e:
                    failures.append({'rank': rank, 'feature': f, 'dataset': ds, 'kind': kind, 'error': repr(e)[:300]})
                finally:
                    plt.close('all')

    return (pd.DataFrame(plots, columns=['rank', 'feature', 'dataset', 'kind', 'path']),
            pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
            pd.DataFrame(failures, columns=['rank', 'feature', 'dataset', 'kind', 'error']))

