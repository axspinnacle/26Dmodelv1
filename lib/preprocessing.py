"""Stage 02 conditioning, shared with 08 scoring."""
import pandas as pd
import numpy as np


def apply_type_conversions(data, type_conversions_path, verbose=True):
    """Stage 02 type conversions: text columns listed as 'numeric' -> pd.to_numeric(errors='coerce')."""
    import os
    if not type_conversions_path or not os.path.exists(type_conversions_path):
        return data
    conversions = pd.read_csv(type_conversions_path)
    done = []
    for _, row in conversions.iterrows():
        col = row['column_name']
        if col in data.columns and row['target_dtype'] == 'numeric':
            data[col] = pd.to_numeric(data[col], errors='coerce')
            done.append(col)
    if verbose:
        print(f"  Type conversions applied: {len(done)} columns")
    return data


def impute_from_defaults(data, cfg, verbose=True):
    """Fill missing values from a state x vehicle_type defaults table (config 'imputation' section).

    Only NaN is filled; real values are kept. States not in the table get the median of the
    vehicle type's values across states. Optional {col}_missing_ind flag (1 = was missing).
    """
    imp = cfg.get('imputation') or {}
    cols = {c: v for c, v in (imp.get('columns') or {}).items() if c in data.columns}
    if not cols:
        return data
    vt = str(cfg['experiment']['vehicle_type']).upper()
    st_col = imp.get('state_column', 'st_raw')
    table = pd.read_csv(imp['defaults_file'])
    table = table[table['vehicle_type'].astype(str).str.upper() == vt].set_index('state')
    if table.empty:
        raise ValueError(f"No rows for vehicle_type={vt} in {imp['defaults_file']}")
    state = data[st_col].astype('string').str.strip().str.upper()
    for col, src in cols.items():
        miss = data[col].isna()
        if imp.get('add_missing_flag', False):
            data[f'{col}_missing_ind'] = miss.astype('int8')
        fill = state[miss].map(table[src]).astype('float64').fillna(float(table[src].median()))
        data.loc[miss, col] = fill.to_numpy()
        if verbose:
            print(f"  Imputed {col}: {int(miss.sum()):,} missing -> {src} by {st_col} ({vt})")
    return data


def condition_data(df, cfg, type_conversions_path=None, copy=True):
    """Stage 02 conditioning, also used by 08 scoring so both match.

    Order: type conversions, defaults imputation, then remaining numeric NaN -> 0.
    type_conversions_path: {config_path}/type_conversions.csv (scoring must pass it).
    """
    data = df.copy() if copy else df

    data = apply_type_conversions(data, type_conversions_path)
    data = impute_from_defaults(data, cfg)

    # Fill missing values in numeric columns with 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(0)
    
    return data


def load_and_condition(file_path, cfg, folds=None):
    """Read a parquet file, optionally keep some folds, and condition it."""
    df = pd.read_parquet(file_path)
    
    if folds and 'fold' in df.columns:
        df = df[df['fold'].isin(folds)]
    
    return condition_data(df, cfg)