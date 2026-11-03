"""Build config versions.

build_full_feature_version: every feature enabled in the encoding file gets loaded (v3).
copy_version: copy a version with columns set to skip and an imputation section (v4).
Leakage columns (LEAKAGE_PREFIXES) are never loaded.
"""

import re
import json
import shutil
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pathlib import Path

from model_utils import LEAKAGE_PREFIXES

SKIP_ENCODINGS = ('skip', 'DROP', 'SKIPPED')
STRING_TYPES = ('string', 'large_string')
LOAD_FILE = 'columns_to_load_during_dataassembly.csv'
ENC_FILE = 'config_generated/master_feature_encoding.csv'
# Only files the pipeline reads (00 EDA inputs/outputs and setup files are not copied)
VERSION_FILES = ['config.yaml', LOAD_FILE, 'type_conversions.csv', 'pca_features.csv',
                 'exclusion.csv', 'monotonicity.csv', ENC_FILE]  # + 05b_best_params.yaml
STAGES_ON = ['01', '02', '03', '04a', '04b', '04c', '05a', '05c', '06', '07a', '07b', '08', '08b', '09a', '09b', '09c', '09d']


def _eol(path):
    """Line terminator used by an existing file (keeps CRLF files as CRLF)."""
    return '\r\n' if b'\r\n' in Path(path).read_bytes()[:4096] else '\n'


def _is_numeric_encoding(enc):
    return enc in ('numeric', 'ordinal_0_5', 'remap_2to4_then_ordinal') or str(enc).startswith('binary')


KNOWN_ENCODINGS = ('numeric', 'ordinal_0_5', 'remap_2to4_then_ordinal', 'one_hot', 'one_hot_encode',
                   'group_then_ohe', 'remap_2to4_then_ohe', 'remap_2to4_then_one_hot')


def _is_known_encoding(enc):
    """Encodings apply_master_encoding understands (anything else is silently skipped in 04c)."""
    e = str(enc)
    return e in KNOWN_ENCODINGS or e.startswith('binary:') or e.startswith('group_then_ohe:')


def _numeric_share(master_path, cols, n_rows=200000):
    """Share of non-null, non-'nan' values that parse as numbers, from the first n_rows."""
    if not cols:
        return {}
    batch = next(pq.ParquetFile(master_path).iter_batches(batch_size=n_rows, columns=cols))
    df = batch.to_pandas()
    out = {}
    for c in cols:
        s = df[c].dropna().astype(str)
        s = s[s.str.lower() != 'nan']
        out[c] = float(pd.to_numeric(s, errors='coerce').notna().mean()) if len(s) else 1.0
    return out


def _update_config_text(text, model, src_version, dst_version):
    """Rename experiment, update output paths, enable full run with 05b off."""
    # experiment name: first "name:" line, swap trailing _{src} -> _{dst} (car_coll_v1, car_liab3p_v1)
    text = re.sub(rf'^(\s*name:\s*"?[^"\n#]*?)_{src_version}("?)', rf'\1_{dst_version}\2', text, count=1, flags=re.M)
    text = re.sub(r'(description:\s*"?)([^"\n]*?)("?\s*$)',
                  lambda m: f"{m.group(1)}{re.sub(r' - v[0-9]+ full feature list', '', m.group(2))}"
                            f" - {dst_version} full feature list{m.group(3)}",
                  text, count=1, flags=re.M)
    # output paths ending in /{model}/{src}, quoted or not
    text = re.sub(rf'/{model}/{src_version}(?=["\s]|$)', f'/{model}/{dst_version}', text, flags=re.M)
    for st in STAGES_ON:
        text = re.sub(rf'^(\s*stage_{st}:\s*)\w+', r'\1true', text, flags=re.M)
    text = re.sub(r'^(\s*stage_05b:\s*)\w+', r'\1false', text, flags=re.M)
    # columns_inclusion.csv is not copied (not used by the pipeline)
    text = re.sub(r'^(\s*inclusion_file:\s*).*$', r'\1null  # not used by pipeline', text, count=1, flags=re.M)
    return text


def _find_best_params(project_root, model, src_dir, src_version):
    for p in [src_dir / '05b_best_params.yaml',
              project_root / 'output' / model / src_version / 'results' / '05b_best_params.yaml']:
        if p.exists():
            return p
    return None


def _aux_columns(cfg, src):
    """Aux column names: aux file (if on this machine), else all_columns_aux.csv,
    else car_coll's all_columns_aux.csv (aux files share the same columns across vehicles)."""
    paths = cfg['machine']['paths']
    aux_path = Path(paths['aux_data_path']) / cfg['data']['aux_file']
    if aux_path.exists():
        return set(pq.read_schema(aux_path).names)
    for csv in [src / 'all_columns_aux.csv', src.parents[1] / 'car_coll' / src.name / 'all_columns_aux.csv',
                src.parents[1] / 'car_coll' / 'v1' / 'all_columns_aux.csv']:
        if csv.exists():
            return set(pd.read_csv(csv)['column_name'])
    return None


def build_full_feature_version(project_root, model, src_version='v1', dst_version='v3',
                               aux_columns=None, max_ohe_levels=10, overwrite=False, verbose=True):
    """Create config/{model}/{dst} from {src} with every enabled feature loaded.

    Text columns that hold numbers get a type conversion; text categories become one-hot if they have
    at most max_ohe_levels levels, else skip. Best params are copied from the source.
    """
    from utils import load_config

    project_root = Path(project_root)
    src = project_root / 'config' / model / src_version
    dst = project_root / 'config' / model / dst_version
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists (use overwrite=True)")
        shutil.rmtree(dst)

    cfg = load_config(str(src / 'config.yaml'))
    master_path = Path(cfg['machine']['paths']['master_data_path']) / cfg['data']['master_file']
    master_types = {f.name: str(f.type) for f in pq.read_schema(master_path)}
    aux_columns = set(aux_columns) if aux_columns is not None else _aux_columns(cfg, src)
    if aux_columns is None:
        raise FileNotFoundError(f"No aux schema for {model}; pass aux_columns")

    # 1. Copy only the files the pipeline reads
    for f in VERSION_FILES:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dst / f)

    # 2. config.yaml
    cfg_file = dst / 'config.yaml'
    cfg_file.write_text(_update_config_text(cfg_file.read_text(), model, src_version, dst_version))

    # 3. Encoding: leakage -> skip (read as text so untouched rows are written back unchanged)
    enc = pd.read_csv(dst / ENC_FILE, dtype=str, keep_default_na=False)
    leak = enc['column_name'].str.startswith(LEAKAGE_PREFIXES) & ~enc['encoding'].isin(SKIP_ENCODINGS)
    enc.loc[leak, ['encoding', 'source']] = ['skip', f'{dst_version}_leakage']

    # 4. String columns with numeric encoding -> type conversion, one-hot, or skip
    enabled = ~enc['encoding'].isin(SKIP_ENCODINGS)
    str_num = enc[enabled & enc['encoding'].map(_is_numeric_encoding)
                  & enc['column_name'].map(lambda c: master_types.get(c) in STRING_TYPES)
                  & ~enc['column_name'].isin(aux_columns)]
    share = _numeric_share(master_path, str_num['column_name'].tolist())
    to_convert, to_ohe, to_skip = [], [], []
    for _, r in str_num.iterrows():
        c = r['column_name']
        if share[c] >= 0.99:
            to_convert.append(c)
        elif float(r['unique_count'] or 0) <= max_ohe_levels:
            to_ohe.append(c)
        else:
            to_skip.append(c)
    enc.loc[enc['column_name'].isin(to_ohe), ['encoding', 'source']] = ['one_hot_encode', f'{dst_version}_text']
    enc.loc[enc['column_name'].isin(to_skip), ['encoding', 'source']] = ['skip', f'{dst_version}_text']
    enc.to_csv(dst / ENC_FILE, index=False, lineterminator=_eol(src / ENC_FILE))

    # 5. Load list: source rows unchanged + enabled master columns
    load_eol = _eol(src / LOAD_FILE)
    load_lines = (src / LOAD_FILE).read_bytes().decode().rstrip('\r\n').split(load_eol)
    loaded = set(pd.read_csv(src / LOAD_FILE, comment='#')['column_name'])
    enabled_df = enc[~enc['encoding'].isin(SKIP_ENCODINGS)]
    new_cols = [c for c in enabled_df['column_name']
                if c in master_types and c not in aux_columns and c not in loaded
                and not c.startswith(LEAKAGE_PREFIXES)]
    enc_type = enabled_df.set_index('column_name')['encoding']
    load_lines += [f'{c},{dst_version}_full,N/A,"{dst_version} auto: {enc_type.get(c, "")}"' for c in new_cols]
    (dst / LOAD_FILE).write_bytes((load_eol.join(load_lines) + load_eol).encode())

    # 6. Type conversions
    tc = pd.read_csv(dst / 'type_conversions.csv', dtype=str, keep_default_na=False)
    add_tc = [c for c in to_convert if c not in set(tc['column_name'])]
    if add_tc:
        tc = pd.concat([tc, pd.DataFrame({'column_name': add_tc, 'source_dtype': 'object',
                                          'target_dtype': 'numeric', 'notes': f'{dst_version} auto'})])
        tc.to_csv(dst / 'type_conversions.csv', index=False, lineterminator=_eol(src / 'type_conversions.csv'))

    # 7. Best params copied into dst config folder (copy to output/results before running)
    bp = _find_best_params(project_root, model, src, src_version)
    if bp and bp.parent != dst:
        shutil.copy(bp, dst / '05b_best_params.yaml')

    summary = {'model': model, 'leakage_skipped': int(leak.sum()), 'new_load_cols': len(new_cols),
               'type_conversions_added': add_tc, 'text_to_ohe': to_ohe, 'text_to_skip': to_skip,
               'best_params': str(bp) if bp else None}
    if verbose:
        print(f"{model}: +{len(new_cols)} load cols, {int(leak.sum())} leakage->skip, +{len(add_tc)} type conv, "
              f"ohe={to_ohe}, skip={to_skip}, params={'yes' if bp else 'MISSING'}")
    return summary



IMPUTED_SUFFIXES = ('_imps', '_impa', '_impm', '_imp')


def apply_feature_rules(project_root, model, version, base_version='v1',
                        drop_raw_with_imputed=True, skip=(), group_ohe=None,
                        to_numeric=(), aux_columns=None, verbose=True):
    """Edit encoding choices in config/{model}/{version} and rebuild its load list from base_version.

    drop_raw_with_imputed skips X_raw when X_imps/_impa/_impm/_imp is enabled. group_ohe = {col: min_freq}.
    """
    from utils import load_config

    project_root = Path(project_root)
    dst = project_root / 'config' / model / version
    base = project_root / 'config' / model / base_version
    cfg = load_config(str(dst / 'config.yaml'))
    master_path = Path(cfg['machine']['paths']['master_data_path']) / cfg['data']['master_file']
    master_types = {f.name: str(f.type) for f in pq.read_schema(master_path)}
    aux_columns = set(aux_columns) if aux_columns is not None else _aux_columns(cfg, base)
    tag = f'{version}_rule'

    # 1. Encoding edits (text read/write keeps untouched rows identical)
    enc_eol = _eol(dst / ENC_FILE)
    enc = pd.read_csv(dst / ENC_FILE, dtype=str, keep_default_na=False)
    on = set(enc.loc[~enc['encoding'].isin(SKIP_ENCODINGS), 'column_name'])

    raw_skip = []
    if drop_raw_with_imputed:
        raw_skip = sorted(c for c in on if c.endswith('_raw')
                          and any(c[:-4] + s in on for s in IMPUTED_SUFFIXES))
    skip_all = sorted(set(raw_skip) | set(skip))
    enc.loc[enc['column_name'].isin(skip_all), ['encoding', 'source']] = ['skip', tag]
    for c in to_numeric:
        enc.loc[enc['column_name'] == c, ['encoding', 'source']] = ['numeric', tag]
    for c, freq in (group_ohe or {}).items():
        enc.loc[enc['column_name'] == c, ['encoding', 'source']] = [f'group_then_ohe:{freq}', tag]
    missing = [c for c in list(skip) + list(to_numeric) + list(group_ohe or {}) if c not in set(enc['column_name'])]
    if missing:
        raise ValueError(f"{model}: not in encoding file: {missing}")
    enc.to_csv(dst / ENC_FILE, index=False, lineterminator=enc_eol)

    # 2. Type conversions for text columns turned numeric
    tc_eol = _eol(dst / 'type_conversions.csv')
    tc = pd.read_csv(dst / 'type_conversions.csv', dtype=str, keep_default_na=False)
    add_tc = [c for c in to_numeric if master_types.get(c) in STRING_TYPES and c not in set(tc['column_name'])]
    if add_tc:
        tc = pd.concat([tc, pd.DataFrame({'column_name': add_tc, 'source_dtype': 'object',
                                          'target_dtype': 'numeric', 'notes': tag})])
        tc.to_csv(dst / 'type_conversions.csv', index=False, lineterminator=tc_eol)

    # 3. Rebuild load list: base rows unchanged + enabled master columns (skipped ones dropped)
    load_eol = _eol(base / LOAD_FILE)
    load_lines = (base / LOAD_FILE).read_bytes().decode().rstrip('\r\n').split(load_eol)
    loaded = set(pd.read_csv(base / LOAD_FILE, comment='#')['column_name'])
    enabled_df = enc[~enc['encoding'].isin(SKIP_ENCODINGS)]
    new_cols = [c for c in enabled_df['column_name']
                if c in master_types and c not in aux_columns and c not in loaded
                and not c.startswith(LEAKAGE_PREFIXES)]
    enc_type = enabled_df.set_index('column_name')['encoding']
    load_lines += [f'{c},{version}_full,N/A,"{version} auto: {enc_type.get(c, "")}"' for c in new_cols]
    (dst / LOAD_FILE).write_bytes((load_eol.join(load_lines) + load_eol).encode())

    summary = {'model': model, 'raw_skipped': raw_skip, 'skipped': sorted(skip),
               'to_numeric': list(to_numeric), 'type_conversions_added': add_tc,
               'group_ohe': dict(group_ohe or {}), 'enabled': len(enabled_df),
               'load_rows': len(loaded) + len(new_cols)}
    if verbose:
        print(f"{model}: {len(raw_skip)} raw->skip, skip={sorted(skip)}, numeric={list(to_numeric)}, "
              f"group={dict(group_ohe or {})}, enabled={len(enabled_df)}, load rows={summary['load_rows']}")
    return summary


IMPUTATION_V4 = {
    'defaults_file': 'config/Default_values_AME_MAM.csv',
    'state_column': 'st_raw',
    'columns': {'amc_annual_mileage_estimate_raw': 'AME_median',
                'amc_modeled_annual_mileage_raw': 'MAM_median'},
    'add_missing_flag': False,
}


def _imputation_block(imp):
    """YAML text for the config 'imputation' section, with comments."""
    cols = ''.join(f'    {c}: {s}\n' for c, s in imp['columns'].items())
    return ('\n# Fill missing values from a state x vehicle_type table (stage 02 and 08 scoring).\n'
            '# Only NaN is filled. add_missing_flag adds {column}_missing_ind (1 = was missing).\n'
            'imputation:\n'
            f"  defaults_file: \"{imp['defaults_file']}\"\n"
            f"  state_column: {imp['state_column']}\n"
            f'  columns:\n{cols}'
            f"  add_missing_flag: {str(imp['add_missing_flag']).lower()}\n")


def copy_version(project_root, model, src_version='v3', dst_version='v4', skip=(),
                 imputation=None, overwrite=False, verbose=True):
    """Copy config/{model}/{src} to {dst}, set `skip` columns to skip (source=manual),
    drop them from the load list, and add the 'imputation' config section.

    If imputation has add_missing_flag, a numeric encoding row for each {column}_missing_ind is added.
    """
    project_root = Path(project_root)
    src = project_root / 'config' / model / src_version
    dst = project_root / 'config' / model / dst_version
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists (use overwrite=True)")
        shutil.rmtree(dst)
    for f in VERSION_FILES + ['05b_best_params.yaml']:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dst / f)

    # config.yaml: names/paths -> dst, 05b off, imputation section
    cfg_file = dst / 'config.yaml'
    text = cfg_file.read_text()
    text = re.sub(rf'^(\s*name:\s*"?[^"\n#]*?)_{src_version}("?)', rf'\1_{dst_version}\2', text, count=1, flags=re.M)
    text = re.sub(rf'(description:[^\n]*?)\b{src_version}\b', rf'\1{dst_version}', text, count=1)
    text = re.sub(rf'/{model}/{src_version}(?=["\s]|$)', f'/{model}/{dst_version}', text, flags=re.M)
    text = re.sub(r'^(\s*stage_05b:\s*)\w+', r'\1false', text, flags=re.M)
    if imputation:
        if re.search(r'^imputation:', text, flags=re.M):
            raise ValueError(f'{model}: config already has an imputation section')
        text = text.rstrip('\n') + '\n' + _imputation_block(imputation)
    cfg_file.write_text(text)

    # Encoding: skip rows -> skip/manual, plus optional flag rows
    enc_eol = _eol(dst / ENC_FILE)
    enc = pd.read_csv(dst / ENC_FILE, dtype=str, keep_default_na=False)
    missing = [c for c in skip if c not in set(enc['column_name'])]
    if missing:
        raise ValueError(f"{model}: not in encoding file: {missing}")
    enc.loc[enc['column_name'].isin(skip), ['encoding', 'source']] = ['skip', 'manual']
    flags = []
    if imputation and imputation.get('add_missing_flag'):
        flags = [f'{c}_missing_ind' for c in imputation['columns'] if f'{c}_missing_ind' not in set(enc['column_name'])]
        blank = {k: '' for k in enc.columns}
        enc = pd.concat([enc, pd.DataFrame([{**blank, 'column_name': f, 'encoding': 'numeric',
                                             'source': 'manual'} for f in flags])], ignore_index=True)
    enc.to_csv(dst / ENC_FILE, index=False, lineterminator=enc_eol)

    # Load list: drop skipped columns, every other line unchanged
    load_eol = _eol(dst / LOAD_FILE)
    lines = (dst / LOAD_FILE).read_bytes().decode().rstrip('\r\n').split(load_eol)
    kept = [l for l in lines if l.split(',', 1)[0].strip() not in set(skip)]
    (dst / LOAD_FILE).write_bytes((load_eol.join(kept) + load_eol).encode())

    # Type conversions of skipped columns are not needed
    tc_eol = _eol(dst / 'type_conversions.csv')
    tc = pd.read_csv(dst / 'type_conversions.csv', dtype=str, keep_default_na=False)
    tc[~tc['column_name'].isin(skip)].to_csv(dst / 'type_conversions.csv', index=False, lineterminator=tc_eol)

    summary = {'model': model, 'skipped': len(skip), 'load_rows_removed': len(lines) - len(kept),
               'flags_added': flags}
    if verbose:
        print(f"{model}: {src_version} -> {dst_version}, {len(skip)} skip, "
              f"{len(lines) - len(kept)} load rows removed, flags={flags or 'none'}")
    return summary


def check_copied_version(project_root, model, src_version='v3', dst_version='v4', skip=()):
    """Check a copy_version result against its source. Returns list of problems (empty = OK)."""
    project_root = Path(project_root)
    src = project_root / 'config' / model / src_version
    dst = project_root / 'config' / model / dst_version
    problems = []
    cfg = yaml.safe_load((dst / 'config.yaml').read_text())
    if not str(cfg['experiment']['name']).endswith(f'_{dst_version}'):
        problems.append(f"name={cfg['experiment']['name']}")
    if cfg['execution']['stages'].get('stage_05b'):
        problems.append('stage_05b on')
    for pc, m in cfg['machines'].items():
        op = (m.get('paths') or {}).get('output_path')
        if op and not op.rstrip('/').endswith(f'{model}/{dst_version}'):
            problems.append(f'{pc} output_path={op}')

    load = set(pd.read_csv(dst / LOAD_FILE, comment='#')['column_name'])
    imp = cfg.get('imputation') or {}
    if imp:
        if not (project_root / imp['defaults_file']).exists():
            problems.append(f"defaults_file missing: {imp['defaults_file']}")
        need = [c for c in [imp.get('state_column', 'st_raw'), *imp['columns']] if c not in load]
        if need:
            problems.append(f'imputation columns not loaded: {need}')

    # Only the skip rows (and flag rows) differ from the source encoding
    a = pd.read_csv(src / ENC_FILE, dtype=str, keep_default_na=False).set_index('column_name')
    b = pd.read_csv(dst / ENC_FILE, dtype=str, keep_default_na=False).set_index('column_name')
    changed = sorted(c for c in a.index if not a.loc[c].equals(b.loc[c]))
    if changed != sorted(skip):
        problems.append(f'encoding changes differ from skip list: {sorted(set(changed) ^ set(skip))}')
    rows = b.loc[list(skip)]
    if not ((rows['encoding'] == 'skip') & (rows['source'] == 'manual')).all():
        problems.append('skip rows not skip/manual')
    if load & set(skip):
        problems.append(f'skipped still loaded: {sorted(load & set(skip))}')

    # Load list = source minus skipped
    la_ = [l for l in (src / LOAD_FILE).read_text().splitlines() if l.strip()]
    lb = [l for l in (dst / LOAD_FILE).read_text().splitlines() if l.strip()]
    if lb != [l for l in la_ if l.split(',', 1)[0].strip() not in set(skip)]:
        problems.append('load list is not source minus skip')

    for f in ['pca_features.csv', 'exclusion.csv', 'monotonicity.csv', '05b_best_params.yaml']:
        if (src / f).read_bytes() != (dst / f).read_bytes():
            problems.append(f'{f} differs')
    expected = set(VERSION_FILES) | {'05b_best_params.yaml'}
    actual = {str(p.relative_to(dst)) for p in dst.rglob('*') if p.is_file()}
    if actual != expected:
        problems.append(f'files: extra={sorted(actual - expected)} missing={sorted(expected - actual)}')
    return problems


def check_full_feature_version(project_root, model, src_version='v1', dst_version='v3', aux_columns=None,
                               verbose_excluded=True, base_version='v1'):
    """Validate dst version. Load-list rows are compared with base_version (v1, the curated list).
    Returns list of problems (empty = OK)."""
    project_root = Path(project_root)
    src = project_root / 'config' / model / base_version
    dst = project_root / 'config' / model / dst_version
    problems = []

    cfg = yaml.safe_load((dst / 'config.yaml').read_text())
    if not str(cfg['experiment']['name']).endswith(f'_{dst_version}'):
        problems.append(f"name={cfg['experiment']['name']}")
    if cfg['execution']['stages'].get('stage_05b'):
        problems.append('stage_05b on')
    for pc, m in cfg['machines'].items():
        op = (m.get('paths') or {}).get('output_path')
        if op and not op.rstrip('/').endswith(f'{model}/{dst_version}'):
            problems.append(f'{pc} output_path={op}')

    # Source load rows unchanged, no leakage added
    src_lines = [l for l in (src / LOAD_FILE).read_text().splitlines() if l.strip()]
    dst_lines = [l for l in (dst / LOAD_FILE).read_text().splitlines() if l.strip()]
    if dst_lines[:len(src_lines)] != src_lines:
        problems.append('source load rows changed')
    load = pd.read_csv(dst / LOAD_FILE, comment='#')
    n_src = len(pd.read_csv(src / LOAD_FILE, comment='#'))
    bad = [c for c in load['column_name'].iloc[n_src:] if c.startswith(LEAKAGE_PREFIXES)]
    if bad:
        problems.append(f'leakage added to load: {bad}')
    if load.shape[1] != 4:
        problems.append(f'load file has {load.shape[1]} columns')

    # Encoding: no leakage enabled; every enabled column loaded or in aux
    enc = pd.read_csv(dst / ENC_FILE)
    on = enc[~enc['encoding'].isin(SKIP_ENCODINGS)]
    leak_on = on[on['column_name'].str.startswith(LEAKAGE_PREFIXES)]['column_name'].tolist()
    if leak_on:
        problems.append(f'leakage enabled in encoding: {leak_on}')
    unknown = sorted(set(on.loc[~on['encoding'].map(_is_known_encoding), 'encoding']))
    if unknown:
        problems.append(f'unknown encodings (04c would skip): {unknown}')
    if aux_columns is not None:
        known = set(load['column_name']) | set(aux_columns)
        missing = [c for c in on['column_name'] if c not in known]
        if missing:
            problems.append(f'{len(missing)} enabled not loaded: {missing[:5]}')

    # Target/exposure (or combine parts) loaded; no duplicate load rows
    exp = cfg['experiment']
    comb = exp.get('combine') or {}
    need = (list(comb.get('target', [])) + list(comb.get('exposure', []))) if comb else [exp['target'], exp['exposure']]
    avail = set(load['column_name']) | set(aux_columns or [])
    miss_te = [c for c in need if c not in avail]
    if miss_te:
        problems.append(f'target/exposure not loaded: {miss_te}')
    dup = load['column_name'][load['column_name'].duplicated()].tolist()
    if dup:
        problems.append(f'duplicate load rows: {dup}')

    # Text columns with numeric encoding must have a type conversion
    tc = set(pd.read_csv(dst / 'type_conversions.csv')['column_name'])
    numeric_on = on[on['encoding'].map(_is_numeric_encoding)]
    text_no_tc = [c for c, d in zip(numeric_on['column_name'], numeric_on['dtype'])
                  if str(d) in ('str', 'object') and c in set(load['column_name']) and c not in tc]
    if text_no_tc:
        problems.append(f'text columns without type conversion: {text_no_tc}')

    # Excluded one-hot columns are handled by expand_exclusions in 04c (report only)
    vt = exp.get('vehicle_type')
    ex = pd.read_csv(dst / 'exclusion.csv')
    excluded_ohe = on[on['column_name'].isin(ex.loc[ex[vt] == 1, 'field']) & ~on['encoding'].map(_is_numeric_encoding)]
    if len(excluded_ohe) and verbose_excluded:
        print(f"   {model}: excluded one-hot (dropped in 04c via expand_exclusions): {excluded_ohe['column_name'].tolist()}")

    # Exact file list
    expected = set(VERSION_FILES) | {'05b_best_params.yaml'}
    actual = {str(p.relative_to(dst)) for p in dst.rglob('*') if p.is_file()}
    if actual != expected:
        problems.append(f'files: extra={sorted(actual - expected)} missing={sorted(expected - actual)}')
    return problems


def create_version_runner(project_root, model, src_version='v1', dst_version='v3'):
    """Clone runners/{src}/run_{model}_{src}.ipynb -> runners/{dst}/run_{model}_{dst}.ipynb, outputs cleared."""
    project_root = Path(project_root)
    src_nb = project_root / 'runners' / src_version / f'run_{model}_{src_version}.ipynb'
    dst_nb = project_root / 'runners' / dst_version / f'run_{model}_{dst_version}.ipynb'
    dst_nb.parent.mkdir(parents=True, exist_ok=True)
    nb = json.loads(src_nb.read_text())
    for c in nb['cells']:
        s = ''.join(c['source'])
        s = s.replace(f'config/{model}/{src_version}', f'config/{model}/{dst_version}')
        if c['cell_type'] == 'markdown':
            s = re.sub(rf'\b{src_version}\b', dst_version, s)
        if c['cell_type'] == 'code' and 'Output directories created' in s and 'seed_best_params' not in s:
            s = s.replace('print("Output directories created")',
                          'print("Output directories created")\n\n'
                          '# Use config best params when 05b is off\n'
                          'from utils import seed_best_params\n'
                          'seed_best_params(config_path, output_base)')
        c['source'] = s.splitlines(keepends=True)
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    dst_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    return dst_nb

