"""
Feature ablation experiments: one change vs a reference version, judged against seed noise.

An experiment = config/{model}/{name} copied from a reference version (e.g. v3) with ONE change
(feature rule or random_state). It reuses the reference 04b train/test files (symlinks), so only
04c -> 05c run (~20 min). Holdout (08) is not run; compare test gini (same rows for all runs).

Usage (setup_notebooks/run_experiments.ipynb):
    create_experiment(root, 'car_coll', 'x_seed7', ref_version='v3', random_state=7)
    create_experiment(root, 'car_coll', 'x_noraw', ref_version='v3', drop_raw_with_imputed=True)
    run_experiment(root, 'car_coll', 'x_seed7')
    compare_experiments(root, 'car_coll', ['v3', 'x_seed7', 'x_noraw'], ref='v3')
"""

import re
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from version_builder import VERSION_FILES, apply_feature_rules, _update_config_text

EXPERIMENT_STAGES = ['04c', '05c']
STAGE_TEMPLATES = {'04c': '04c_feature_encoding', '05c': '05c_production'}
REUSED_DATA = ['04b_train.parquet', '04b_test.parquet']


def _set_stages(text, on):
    """Turn every stage_XX off except those in `on`."""
    return re.sub(r'^(\s*stage_(\w+):\s*)\w+',
                  lambda m: f"{m.group(1)}{'true' if m.group(2) in on else 'false'}", text, flags=re.M)


def _set_random_state(text, seed):
    new, n = re.subn(r'^(\s*random_state:\s*)\d+', rf'\g<1>{seed}', text, count=1, flags=re.M)
    if n != 1:
        raise ValueError('random_state not found in config.yaml')
    return new


def _output_dir(project_root, config_file):
    from utils import load_config
    out = Path(load_config(str(config_file))['machine']['paths']['output_path'])
    return (out if out.is_absolute() else Path(project_root) / out).resolve()


def create_experiment(project_root, model, name, ref_version='v3', random_state=None,
                      drop_raw_with_imputed=False, skip=(), group_ohe=None, overwrite=False):
    """
    Create config/{model}/{name} = ref_version + one change, and link the ref 04b data.

    Args:
        name: experiment version name, e.g. 'x_seed7' (also the output folder name)
        random_state: new xgboost seed (None = keep ref seed)
        drop_raw_with_imputed / skip / group_ohe: feature rules (see apply_feature_rules)
    Returns:
        output folder path
    """
    project_root = Path(project_root)
    ref = project_root / 'config' / model / ref_version
    dst = project_root / 'config' / model / name
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f'{dst} exists (use overwrite=True)')
        shutil.rmtree(dst)

    for f in VERSION_FILES + ['05b_best_params.yaml']:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref / f, dst / f)

    cfg_file = dst / 'config.yaml'
    text = _update_config_text(cfg_file.read_text(), model, ref_version, name)
    text = re.sub(r' - \S+ full feature list', f' - experiment {name} (from {ref_version})', text, count=1)
    text = _set_stages(text, EXPERIMENT_STAGES)
    if random_state is not None:
        text = _set_random_state(text, random_state)
    cfg_file.write_text(text)

    if drop_raw_with_imputed or skip or group_ohe:
        # Load list base = ref; 04b data is reused, so only the encoding change matters
        apply_feature_rules(project_root, model, name, base_version=ref_version,
                            drop_raw_with_imputed=drop_raw_with_imputed, skip=skip,
                            group_ohe=group_ohe, verbose=False)

    out_ref, out_dst = _output_dir(project_root, ref / 'config.yaml'), _output_dir(project_root, cfg_file)
    for sub in ['data', 'results', 'models', 'config_generated', 'notebooks', 'plots', 'reports']:
        (out_dst / sub).mkdir(parents=True, exist_ok=True)
    for f in REUSED_DATA:
        src, link = out_ref / 'data' / f, out_dst / 'data' / f
        if not src.exists():
            raise FileNotFoundError(f'{src} missing - run {ref_version} through 04b first')
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
    shutil.copy2(dst / '05b_best_params.yaml', out_dst / 'results' / '05b_best_params.yaml')
    print(f'{model}/{name}: from {ref_version}, seed={random_state or "ref"}, raw_drop={drop_raw_with_imputed}, '
          f'skip={list(skip)}, group={group_ohe or {}}')
    return out_dst


def run_experiment(project_root, model, name, kernel_name=None):
    """Run 04c and 05c templates for config/{model}/{name} via papermill (cwd = project root)."""
    import papermill as pm
    from utils import load_config

    project_root = Path(project_root)
    config_path = f'config/{model}/{name}'
    cfg = load_config(f'{project_root}/{config_path}/config.yaml')
    out = _output_dir(project_root, f'{project_root}/{config_path}/config.yaml')
    kernel_name = kernel_name or cfg['machine'].get('conda_env', 'python3')
    for st in EXPERIMENT_STAGES:
        t = STAGE_TEMPLATES[st]
        print(f'  {model}/{name}: stage {st} ...', flush=True)
        pm.execute_notebook(str(project_root / 'templates' / f'{t}.ipynb'),
                            str(out / 'notebooks' / f'{t}.ipynb'),
                            parameters={'config_path': config_path}, kernel_name=kernel_name,
                            cwd=str(project_root), progress_bar=False)
    return out


def _test_predictions(project_root, model, version):
    """05c test predictions (actual, predicted, exposure) + number of model features."""
    out = _output_dir(project_root, Path(project_root) / 'config' / model / version / 'config.yaml')
    d = pd.read_parquet(out / 'results' / '05c_debug_test.parquet',
                        columns=['pp_actual', 'pp_predicted', 'exposure'])
    feats = out / 'results' / '05c_model_features.csv'
    n_feat = len(pd.read_csv(feats)) if feats.exists() else None
    return d, n_feat


def compare_experiments(project_root, model, versions, ref='v3', n_boot=40, seed=0):
    """
    Test gini per version + paired bootstrap of (version - ref) on the same test rows.

    Read: |diff_vs_ref| within the seed runs' spread = noise; boot_lo..boot_hi excluding 0 means
    the difference is not test-sample luck (it can still be seed noise - compare with seed runs).
    Returns DataFrame: version, n_features, test_gini, diff_vs_ref, boot_lo, boot_hi
    """
    from gini_utils import gini

    base, _ = _test_predictions(project_root, model, ref)
    act, w, p_ref = base['pp_actual'].values, base['exposure'].values, base['pp_predicted'].values
    g_ref = gini(act, p_ref, w)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(act), len(act)) for _ in range(n_boot)]
    g_ref_boot = np.array([gini(act[i], p_ref[i], w[i]) for i in idx])

    rows = []
    for v in versions:
        d, n_feat = _test_predictions(project_root, model, v)
        if len(d) != len(base) or not np.allclose(d['pp_actual'].values, act, rtol=1e-5):
            raise ValueError(f'{v}: test rows differ from {ref}')
        p = d['pp_predicted'].values
        diffs = np.array([gini(act[i], p[i], w[i]) for i in idx]) - g_ref_boot
        rows.append({'version': v, 'n_features': n_feat, 'test_gini': round(gini(act, p, w), 5),
                     'diff_vs_ref': round(gini(act, p, w) - g_ref, 5),
                     'boot_lo': round(float(np.percentile(diffs, 2.5)), 5),
                     'boot_hi': round(float(np.percentile(diffs, 97.5)), 5)})
    return pd.DataFrame(rows)

