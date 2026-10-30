"""
Stage 09 value-adjusted lift: overall (09a), by company (09b), standard vs non-standard (09c),
by state (09d). All outputs in results/09/ (chart PNG and its table CSV side by side, same name);
09b-d: one chart + CSV per group in results/09/09x_{dataset}/.
CSV columns: documentation/ModelMath_v2.md, "Stage 09 Lift Tables".

Value-adjusted lift (documentation/ModelMath_v2.md):
    Upstream the loss was divided by the depreciation factor D (L_adj = L_raw / D) and the GLM
    carries D as an offset. Multiplying back by D shows actual vs predicted loss with state /
    year / model year (rest of the GLM) controlled:
        Actual    A = (PP / GLM) * D = y * D
        Predicted P = (PRED / GLM) * D = gbm * D
    Rows are sorted by P and cut into exposure-weighted ntiles. value_adjust=False drops D
    (plain y vs gbm ratio lift).

Datasets: 'train', 'test', 'holdout', 'test_holdout' (test + holdout stacked; disjoint folds).
    train/test: results/05c_debug_{train,test}.parquet; holdout: scoring/holdout/predictions.parquet.
    D (Dep_factor) and companyid are always joined from the source files on the join key.

Usage: see templates/09a_value_lift.ipynb, 09b_company_lift.ipynb, 09c_segment_lift.ipynb,
       09d_state_analysis.ipynb.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import matplotlib.pyplot as plt

FRAME_COLS = ['actual', 'pred', 'weight', 'source']


# ---------------------------------------------------------------- data

def _source_lookup(path, key, cols, keys):
    """Read `cols` for `keys` from a parquet file (stream-filtered); index = key, unique."""
    tbl = ds.dataset(str(path)).to_table(columns=[key] + cols, filter=pc.field(key).isin(list(keys)))
    return tbl.to_pandas().drop_duplicates(key).set_index(key)


def _load_dataset(output_base, cfg, dataset, score_name='holdout'):
    """One dataset as DataFrame(index=key): pp, pred, glm, weight. None if the file is missing."""
    key = cfg['data']['join_key']
    if dataset in ('train', 'test'):
        f = f'{output_base}/results/05c_debug_{dataset}.parquet'
        if not os.path.exists(f):
            print(f'  [WARN] Missing {f}')
            return None
        d = pd.read_parquet(f, columns=['pp_actual', 'pp_predicted', 'glm_pred', 'exposure'])
    else:
        f = f'{output_base}/scoring/{score_name}/predictions.parquet'
        if not os.path.exists(f):
            print(f'  [WARN] Missing {f}')
            return None
        t, e = cfg['experiment']['target'], cfg['experiment']['exposure']
        d = pd.read_parquet(f, columns=[key, t, 'prediction', 'glm_pred', e]).set_index(key)
    d.columns = ['pp', 'pred', 'glm', 'weight']
    d.index.name = key
    return d


MASTER_COLS = {'companyid': 'companyid', 'state': 'st_raw'}  # group name -> master file column


def lift_frame(output_base, cfg, pc_id, dataset, value_adjust=True, groups=()):
    """
    Record-level lift input for one dataset.
    Args:
        groups: group columns to join from the master file, names from MASTER_COLS ('companyid', 'state')
    Returns DataFrame(index=key): actual, pred, weight, source [, dep_factor] [, groups...].
        actual = (pp / glm) * D, pred = (pred / glm) * D   (D = 1 if value_adjust is False)
    Rows with missing / non-positive D or GLM, or zero exposure, are dropped (count printed).
    """
    parts = ['test', 'holdout'] if dataset == 'test_holdout' else [dataset]
    frames = []
    for p in parts:
        d = _load_dataset(output_base, cfg, p)
        if d is not None:
            d['source'] = p
            frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames)
    if not d.index.is_unique:
        raise ValueError(f'{dataset}: duplicate keys across {parts}')

    paths = cfg['machines'][pc_id]['paths']
    key = cfg['data']['join_key']
    if value_adjust:
        aux = _source_lookup(f"{paths['aux_data_path']}/{cfg['data']['aux_file']}", key, ['Dep_factor'], d.index)
        d['dep_factor'] = aux['Dep_factor'].reindex(d.index).to_numpy()
    if groups:
        src = [MASTER_COLS[g] for g in groups]
        m = _source_lookup(f"{paths['master_data_path']}/{cfg['data']['master_file']}", key, src, d.index)
        for g, c in zip(groups, src):
            d[g] = m[c].reindex(d.index).astype('string').fillna('MISSING').to_numpy()

    D = d['dep_factor'] if value_adjust else 1.0
    d['actual'] = d['pp'] / d['glm'] * D
    d['pred'] = d['pred'] / d['glm'] * D
    ok = np.isfinite(d['actual']) & np.isfinite(d['pred']) & (d['weight'] > 0) & (d['glm'] > 0)
    if value_adjust:
        ok &= d['dep_factor'] > 0
    n_bad = int((~ok).sum())
    if n_bad:
        print(f'  {dataset}: dropped {n_bad:,} rows (missing/invalid D, GLM or exposure)')
    keep = FRAME_COLS + [c for c in ('dep_factor', *groups) if c in d.columns]
    return d.loc[ok, keep]


def add_segment(df, segment_file):
    """Add 'segment' from companyid -> segment mapping CSV; unmapped companies -> 'unmapped'."""
    seg = pd.read_csv(segment_file, dtype=str)
    mp = dict(zip(seg['companyid'].str.strip(), seg['segment'].str.strip()))
    df['segment'] = df['companyid'].map(mp).fillna('unmapped')
    unmapped = sorted(df.loc[df['segment'] == 'unmapped', 'companyid'].unique())
    if unmapped:
        print(f'  [WARN] companies not in {segment_file}: {unmapped}')
    return df


# ---------------------------------------------------------------- lift tables

def ntile_lift(df, bins=10, actual='actual', pred='pred', weight='weight'):
    """
    Exposure-weighted ntile lift. Rows sorted by pred; a record goes to the ntile containing the
    midpoint of its exposure (no rounding, equal exposure per ntile up to one record).
    Columns: ntile, n_records, exposure, exposure_pct, actual, pred (exposure-weighted means),
             actual_rel, pred_rel (relative to overall pred mean), a_to_p,
             [actual_no_dep, pred_no_dep: same groups, means without D; only if df has dep_factor].
    Column definitions: documentation/ModelMath_v2.md, "Stage 09 Lift Tables".
    """
    a = df[actual].to_numpy('float64')
    p = df[pred].to_numpy('float64')
    w = df[weight].to_numpy('float64')
    o = np.argsort(p, kind='mergesort')
    a, p, w = a[o], p[o], w[o]
    cw = np.cumsum(w)
    g = np.minimum((bins * (cw - w / 2) / cw[-1]).astype(int), bins - 1) + 1
    cols = {'ntile': g, 'w': w, 'aw': a * w, 'pw': p * w}
    has_d = 'dep_factor' in df.columns
    if has_d:
        d = df['dep_factor'].to_numpy('float64')[o]
        cols.update(aw_nd=a / d * w, pw_nd=p / d * w)
    agg = dict(n_records=('w', 'size'), exposure=('w', 'sum'), aw=('aw', 'sum'), pw=('pw', 'sum'))
    if has_d:
        agg.update(aw_nd=('aw_nd', 'sum'), pw_nd=('pw_nd', 'sum'))
    t = pd.DataFrame(cols).groupby('ntile').agg(**agg).reset_index()
    t['exposure_pct'] = 100 * t['exposure'] / t['exposure'].sum()
    t['actual'] = t['aw'] / t['exposure']
    t['pred'] = t['pw'] / t['exposure']
    base = t['pw'].sum() / t['exposure'].sum()
    t['actual_rel'] = t['actual'] / base
    t['pred_rel'] = t['pred'] / base
    t['a_to_p'] = t['actual'] / t['pred']
    if has_d:
        t['actual_no_dep'] = t['aw_nd'] / t['exposure']
        t['pred_no_dep'] = t['pw_nd'] / t['exposure']
    return t.drop(columns=[c for c in ('aw', 'pw', 'aw_nd', 'pw_nd') if c in t.columns])


def lift_stats(table):
    """Top/bottom ntile ratio (actual, pred) and exposure-weighted mean |A/P - 1|."""
    w = table['exposure'] / table['exposure'].sum()
    a0 = table['actual'].iloc[0]
    return {'lift_top_bottom_actual': float(table['actual'].iloc[-1] / a0) if a0 > 0 else np.nan,
            'lift_top_bottom_pred': float(table['pred'].iloc[-1] / table['pred'].iloc[0]),
            'mean_abs_a_to_p_error': float((w * (table['a_to_p'] - 1).abs()).sum())}


def overall_summary(df, tables):
    """One-row-per-bins summary for an overall lift: rows, exposure, A/P, gini, lift stats."""
    from gini_utils import gini
    expo = df['weight'].sum()
    base = {'n_records': len(df), 'exposure': expo,
            'actual_mean': (df['actual'] * df['weight']).sum() / expo,
            'pred_mean': (df['pred'] * df['weight']).sum() / expo,
            'gini': gini(df['actual'], df['pred'], df['weight'])}
    base['a_to_p'] = base['actual_mean'] / base['pred_mean']
    return pd.DataFrame([{'bins': b, **base, **lift_stats(t)} for b, t in tables.items()])


def group_lift(df, group_col, bins=10, min_exposure=0.0, small_bins=5):
    """
    Lift per group (company / segment / state). Groups below min_exposure use small_bins.
    Returns (tables: {group: lift table}, summary DataFrame sorted by exposure).
    """
    from gini_utils import gini
    tables, rows = {}, []
    for gname, gd in df.groupby(group_col, observed=True):
        expo = gd['weight'].sum()
        b = bins if expo >= min_exposure else small_bins
        t = ntile_lift(gd, b)
        tables[gname] = t
        rows.append({group_col: gname, 'n_records': len(gd), 'exposure': expo,
                     'n_claims': int((gd['actual'] > 0).sum()), 'bins': b,
                     'actual_mean': (gd['actual'] * gd['weight']).sum() / expo,
                     'pred_mean': (gd['pred'] * gd['weight']).sum() / expo,
                     **lift_stats(t), 'gini': gini(gd['actual'], gd['pred'], gd['weight'])})
    s = pd.DataFrame(rows)
    s['a_to_p'] = s['actual_mean'] / s['pred_mean']
    s['exposure_pct'] = 100 * s['exposure'] / s['exposure'].sum()
    s = s.sort_values('exposure', ascending=False).reset_index(drop=True)
    return tables, s


# ---------------------------------------------------------------- plots

def _draw(ax, t, title):
    many = len(t) > 20
    lw = 1.5 if many else 2
    ax.plot(t['ntile'], t['actual_rel'], label='Actual', linewidth=lw, color='#1f77b4',
            **({} if many else {'marker': 'o'}))
    ax.plot(t['ntile'], t['pred_rel'], label='Predicted', linewidth=lw, color='#ff7f0e',
            **({} if many else {'marker': 's'}))
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    vals = np.r_[t['actual_rel'], t['pred_rel']]
    ax.set_ylim(0, max(1.5, 1.1 * np.nanpercentile(vals, 99 if many else 100)))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Percentile' if len(t) == 100 else ('Decile' if len(t) == 10 else f'Group (of {len(t)})'))
    ax.set_ylabel('Relativity')
    ax.grid(True, alpha=0.3)
    if not many:
        ax.set_xticks(t['ntile'])
    ax2 = ax.twinx()
    ax2.bar(t['ntile'], t['exposure_pct'], alpha=0.2, color='#2196F3', width=0.8 if many else 0.6)
    ax2.set_ylim(0, max(400 / len(t), 1.2 * t['exposure_pct'].max()))
    ax2.set_ylabel('Exposure %', color='#2196F3')
    return ax


def plot_lift(table, title, path):
    """Single lift chart (10 or 100 groups) with exposure bars. Saves PNG, returns path."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _draw(ax, table, title)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def stage_dir(output_base):
    """All stage 09 outputs (CSV + PNG side by side): {output_base}/results/09."""
    path = f'{output_base}/results/09'
    os.makedirs(path, exist_ok=True)
    return path


def reset_dir(path):
    """Create path and remove old PNG/CSV in it (group ranks/names change between runs)."""
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path):
        if f.endswith(('.png', '.csv')):
            os.remove(os.path.join(path, f))
    return path


def save_lift(table, title, path_stem):
    """Lift chart + its table: {path_stem}.png and {path_stem}.csv. Returns PNG path."""
    table.to_csv(f'{path_stem}.csv', index=False)
    return plot_lift(table, title, f'{path_stem}.png')


def save_group_lifts(tables, summary, group_col, title_prefix, out_dir, file_prefix):
    """
    One full-size lift chart + CSV per group (summary order = largest exposure first).
    Files: {out_dir}/{file_prefix}_{rank:02d}_{group}.png / .csv. Returns {group: png path}.
    """
    import re
    s = summary.set_index(group_col)
    paths = {}
    for rank, g in enumerate(summary[group_col], 1):
        r = s.loc[g]
        title = (f"{title_prefix} - {group_col} {g}\n"
                 f"exposure {r['exposure_pct']:.1f}% | claims {int(r['n_claims']):,} | "
                 f"A/P {r['a_to_p']:.2f} | gini {r['gini']:.3f}")
        name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(g))
        paths[g] = save_lift(tables[g], title, f'{out_dir}/{file_prefix}_{rank:02d}_{name}')
    return paths


def stack_tables(tables, key_col):
    """{key: lift table} -> one DataFrame with key_col first (for CSV output)."""
    out = pd.concat([t.assign(**{key_col: k}) for k, t in tables.items()], ignore_index=True)
    return out[[key_col] + [c for c in out.columns if c != key_col]]
