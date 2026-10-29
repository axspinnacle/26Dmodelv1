"""
feature_encoder.py
------------------
Master feature encoding system that reads from master_feature_encoding.csv
and applies per-column encoding strategies.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder


def apply_master_encoding(train_df, test_df, config_path, min_frequency=0.01, min_count=50):
    """
    Apply encoding strategies from master_feature_encoding.csv
    
    Returns: train_encoded, test_encoded, encoders, encoding_summary
    """
    
    # Load master encoding CSV
    encoding_path = Path(config_path) / "config_generated" / "master_feature_encoding.csv"
    if not encoding_path.exists():
        raise FileNotFoundError(f"Master encoding file not found: {encoding_path}")
    
    encoding_df = pd.read_csv(encoding_path)
    print(f"[OK] Loaded master encoding: {len(encoding_df)} columns")
    
    # Initialize outputs
    train_parts = []
    test_parts = []
    encoders = {}
    summary_rows = []
    
    # Process each column
    for _, row in encoding_df.iterrows():
        col = row['column_name']
        enc_type = row['encoding']
        
        if col not in train_df.columns:
            continue
        
        if enc_type in ['skip', 'DROP']:
            summary_rows.append({
                'original_column': col, 'encoding_type': enc_type,
                'output_columns': 'SKIPPED', 'n_categories': 0,
                'has_other': False, 'has_missing': False
            })
            continue
        
        elif enc_type == 'numeric':
            train_parts.append(train_df[[col]].astype('float32'))
            test_parts.append(test_df[[col]].astype('float32'))
            summary_rows.append({
                'original_column': col, 'encoding_type': 'numeric',
                'output_columns': col, 'n_categories': 0,
                'has_other': False, 'has_missing': False
            })
        
        elif enc_type == 'ordinal_0_5':
            train_parts.append(train_df[[col]].astype('float32'))
            test_parts.append(test_df[[col]].astype('float32'))
            summary_rows.append({
                'original_column': col, 'encoding_type': 'ordinal_0_5',
                'output_columns': col, 'n_categories': 0,
                'has_other': False, 'has_missing': False
            })
        
        elif enc_type in ['one_hot', 'one_hot_encode']:
            tr_encoded, te_encoded, enc_obj, summary = _encode_one_hot(
                train_df[col], test_df[col], col, group_rare=False
            )
            train_parts.append(tr_encoded)
            test_parts.append(te_encoded)
            encoders[col] = enc_obj
            summary_rows.append(summary)
        
        elif enc_type == 'group_then_ohe' or str(enc_type).startswith('group_then_ohe:'):
            # 'group_then_ohe'      -> keep level if >= min_count rows OR >= min_frequency
            # 'group_then_ohe:0.01' -> keep level only if >= 1% of rows (frequency only)
            if ':' in str(enc_type):
                col_freq, col_count = float(str(enc_type).split(':', 1)[1]), None
            else:
                col_freq, col_count = min_frequency, min_count
            tr_encoded, te_encoded, enc_obj, summary = _encode_one_hot(
                train_df[col], test_df[col], col,
                group_rare=True, min_frequency=col_freq, min_count=col_count
            )
            train_parts.append(tr_encoded)
            test_parts.append(te_encoded)
            encoders[col] = enc_obj
            summary_rows.append(summary)
        
        elif 'binary:' in enc_type:
            tr_encoded, te_encoded, summary = _encode_binary(
                train_df[col], test_df[col], col, enc_type
            )
            train_parts.append(tr_encoded)
            test_parts.append(te_encoded)
            summary_rows.append(summary)
        
        elif enc_type == 'remap_2to4_then_ordinal':
            tr_encoded, te_encoded, summary = _encode_remap_ordinal(
                train_df[col], test_df[col], col
            )
            train_parts.append(tr_encoded)
            test_parts.append(te_encoded)
            summary_rows.append(summary)
        
        elif enc_type in ['remap_2to4_then_ohe', 'remap_2to4_then_one_hot']:
            tr_encoded, te_encoded, enc_obj, summary = _encode_remap_ohe(
                train_df[col], test_df[col], col
            )
            train_parts.append(tr_encoded)
            test_parts.append(te_encoded)
            encoders[col] = enc_obj
            summary_rows.append(summary)
        
        else:
            print(f"[!]  Unknown encoding type '{enc_type}' for column '{col}', skipping")
    
    train_encoded = pd.concat(train_parts, axis=1)
    test_encoded = pd.concat(test_parts, axis=1)
    encoding_summary = pd.DataFrame(summary_rows)
    
    print(f"\n[OK] Encoding complete:")
    print(f"  Train shape: {train_encoded.shape}")
    print(f"  Test shape: {test_encoded.shape}")
    
    return train_encoded, test_encoded, encoders, encoding_summary


def _to_category_str(series):
    """NaN -> '__MISSING__', everything else -> str.
    Fill before astype(str): pandas 3 keeps NaN as float (pandas 2 gave 'nan')."""
    s = series.astype('object')
    return s.where(s.notna(), '__MISSING__').astype(str).replace('nan', '__MISSING__')


def _encode_one_hot(train_series, test_series, col_name, group_rare=False, 
                   min_frequency=0.01, min_count=50):
    """One-hot encode with __OTHER__ and __MISSING__ handling"""
    train_series = train_series.copy()
    test_series = test_series.copy()
    
    # Convert to string, handle NaN as __MISSING__
    # Use object dtype to avoid ArrowStringArray issues
    train_str = _to_category_str(train_series)
    test_str = _to_category_str(test_series)
    
    # Group rare categories if requested
    category_map = None
    if group_rare:
        value_counts = train_str.value_counts()
        total = len(train_str)
        
        # Keep categories that meet threshold (min_count=None -> frequency only)
        keep = value_counts / total >= min_frequency
        if min_count is not None:
            keep = keep | (value_counts >= min_count)
        keep_cats = value_counts[keep].index.tolist()
        
        # Always keep __MISSING__ if present
        if '__MISSING__' in value_counts.index and '__MISSING__' not in keep_cats:
            keep_cats.append('__MISSING__')
        
        # Map rare categories to __OTHER__
        category_map = {}
        for cat in value_counts.index:
            category_map[cat] = cat if cat in keep_cats else '__OTHER__'
        
        train_str = train_str.map(category_map)
        test_str = test_str.map(lambda x: category_map.get(x, '__OTHER__'))
    else:
        # No grouping, but still handle test unknowns
        train_categories = set(train_str.unique())
        test_str = test_str.map(lambda x: x if x in train_categories else '__OTHER__')
    
    # Fit OneHotEncoder on train
    train_categories = sorted(train_str.unique())
    if '__OTHER__' not in train_categories:
        train_categories.append('__OTHER__')
    
    encoder = OneHotEncoder(categories=[train_categories], 
                           sparse_output=False, handle_unknown='ignore')
    
    # Transform - convert to numpy array first to avoid ArrowStringArray issues
    train_encoded = encoder.fit_transform(np.array(train_str).reshape(-1, 1))
    test_encoded = encoder.transform(np.array(test_str).reshape(-1, 1))
    
    # Create DataFrame with proper column names
    feature_names = [f"{col_name}_{cat}" for cat in encoder.categories_[0]]
    train_df = pd.DataFrame(train_encoded, columns=feature_names, 
                           index=train_series.index, dtype='float32')
    test_df = pd.DataFrame(test_encoded, columns=feature_names, 
                          index=test_series.index, dtype='float32')
    
    summary = {
        'original_column': col_name,
        'encoding_type': 'group_then_ohe' if group_rare else 'one_hot',
        'output_columns': ','.join(feature_names),
        'n_categories': len(feature_names),
        'has_other': '__OTHER__' in train_categories,
        'has_missing': '__MISSING__' in train_categories
    }
    
    encoder_obj = {
        'type': 'one_hot',
        'encoder': encoder,
        'category_map': category_map,
        'feature_names': feature_names
    }
    
    return train_df, test_df, encoder_obj, summary


def _apply_binary(series, encoding_str):
    """Map values with a rule like 'binary: {0,1,2}->0 | {3,4,5}->1' (other values -> NaN).
    Shared by training (04c) and scoring (08)."""
    import re
    match = re.search(r'\{([^}]+)\}->0.*\{([^}]+)\}->1', str(encoding_str))
    if not match:
        raise ValueError(f"Cannot parse binary encoding: {encoding_str}")

    low_set = set(int(x.strip()) for x in match.group(1).split(','))
    high_set = set(int(x.strip()) for x in match.group(2).split(','))

    def mapper(v):
        if pd.isna(v):
            return np.nan
        v_int = int(round(v))
        return 0.0 if v_int in low_set else (1.0 if v_int in high_set else np.nan)

    return series.map(mapper).astype('float32')


def _remap_2to4(series, as_float=True):
    """Remap 2 -> 4, keep other values (NaN stays NaN). Shared by training and scoring."""
    def remap(v):
        if pd.isna(v):
            return v
        v_int = int(round(v))
        out = 4 if v_int == 2 else v_int
        return float(out) if as_float else out
    return series.map(remap)


def _encode_binary(train_series, test_series, col_name, encoding_str):
    """Apply binary mapping like 'binary: {0,1,2}->0 | {3,4,5}->1'"""
    train_encoded = _apply_binary(train_series, encoding_str).to_frame()
    test_encoded = _apply_binary(test_series, encoding_str).to_frame()

    summary = {
        'original_column': col_name, 'encoding_type': 'binary',
        'output_columns': col_name, 'n_categories': 2,
        'has_other': False, 'has_missing': False,
        'encoding_rule': encoding_str  # used by apply_saved_encoders in scoring
    }

    return train_encoded, test_encoded, summary


def _encode_remap_ordinal(train_series, test_series, col_name):
    """Remap 2->4, keep as ordinal"""
    train_encoded = _remap_2to4(train_series).astype('float32').to_frame()
    test_encoded = _remap_2to4(test_series).astype('float32').to_frame()
    
    summary = {
        'original_column': col_name, 'encoding_type': 'remap_2to4_then_ordinal',
        'output_columns': col_name, 'n_categories': 0,
        'has_other': False, 'has_missing': False
    }
    
    return train_encoded, test_encoded, summary


def _encode_remap_ohe(train_series, test_series, col_name):
    """Remap 2->4, then one-hot encode"""
    train_remapped = _remap_2to4(train_series, as_float=False)
    test_remapped = _remap_2to4(test_series, as_float=False)

    tr, te, enc_obj, summary = _encode_one_hot(train_remapped, test_remapped, col_name, group_rare=False)
    # Record the real type so scoring applies the same 2->4 remap first
    summary['encoding_type'] = 'remap_2to4_then_ohe'
    return tr, te, enc_obj, summary


def save_encoders(encoders, encoding_summary, output_path):
    """Save encoders and summary to files"""
    output_path = Path(output_path)
    models_dir = output_path / "models"
    results_dir = output_path / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save encoders as pickle
    encoders_file = models_dir / "04c_encoders.pkl"
    with open(encoders_file, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"[OK] Saved encoders: {encoders_file}")
    
    # Save encoding summary as CSV
    summary_file = results_dir / "04c_encoding_summary.csv"
    encoding_summary.to_csv(summary_file, index=False)
    print(f"[OK] Saved encoding summary: {summary_file}")
    
    # Save human-readable mapping as JSON
    mapping_file = models_dir / "04c_encoding_map.json"
    mapping_dict = {}
    for col, enc_obj in encoders.items():
        if enc_obj['type'] == 'one_hot':
            mapping_dict[col] = {
                'type': 'one_hot',
                'categories': list(enc_obj['encoder'].categories_[0]),
                'feature_names': enc_obj['feature_names'],
                'has_grouping': enc_obj['category_map'] is not None
            }
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping_dict, f, indent=2)
    print(f"[OK] Saved encoding map: {mapping_file}")


def load_encoders(output_path):
    """Load saved encoders and summary"""
    output_path = Path(output_path)
    
    encoders_file = output_path / "models" / "04c_encoders.pkl"
    summary_file = output_path / "results" / "04c_encoding_summary.csv"
    
    with open(encoders_file, 'rb') as f:
        encoders = pickle.load(f)
    
    encoding_summary = pd.read_csv(summary_file)
    
    print(f"[OK] Loaded encoders: {len(encoders)} columns")
    print(f"[OK] Loaded summary: {len(encoding_summary)} rows")
    
    return encoders, encoding_summary


def _master_rules(config_path):
    """column -> encoding string from master_feature_encoding.csv (empty if not found)."""
    if config_path is None:
        return {}
    p = Path(config_path) / "config_generated" / "master_feature_encoding.csv"
    if not p.exists():
        print(f"[!]  {p} not found - binary rules only from encoding summary")
        return {}
    enc = pd.read_csv(p)
    return dict(zip(enc['column_name'], enc['encoding'].astype(str)))


def apply_saved_encoders(data_df, encoders, encoding_summary, config_path=None):
    """
    Apply saved encoders to new data (e.g., holdout set), reproducing 04c exactly.

    binary / remap rules: taken from encoding_summary['encoding_rule'] (04c runs after
    this fix) or, for older summaries, from config_path/config_generated/
    master_feature_encoding.csv. Older summaries also record remap_2to4_then_ohe as
    'one_hot'; the master file is used to detect those.
    """
    encoded_parts = []
    rules = _master_rules(config_path)
    has_rule_col = 'encoding_rule' in encoding_summary.columns
    fixed = {'binary': 0, 'remap_2to4_then_ordinal': 0, 'remap_2to4_then_ohe': 0}
    
    for _, row in encoding_summary.iterrows():
        col = row['original_column']
        enc_type = row['encoding_type']
        
        if enc_type in ['skip', 'DROP', 'SKIPPED']:
            continue
        
        if col not in data_df.columns:
            print(f"[!]  Column '{col}' not in data, skipping")
            continue
        
        # Older summaries: one_hot that was really remap_2to4_then_ohe
        if enc_type == 'one_hot' and rules.get(col) in ('remap_2to4_then_ohe', 'remap_2to4_then_one_hot'):
            enc_type = 'remap_2to4_then_ohe'
        
        if enc_type in ['numeric', 'ordinal_0_5']:
            encoded_parts.append(data_df[[col]].astype('float32'))
        
        elif enc_type == 'binary':
            rule = row['encoding_rule'] if has_rule_col and pd.notna(row['encoding_rule']) else rules.get(col)
            if not rule or 'binary' not in str(rule):
                raise ValueError(f"No binary rule for '{col}': pass config_path or rerun 04c")
            encoded_parts.append(_apply_binary(data_df[col], rule).to_frame())
            fixed['binary'] += 1
        
        elif enc_type == 'remap_2to4_then_ordinal':
            encoded_parts.append(_remap_2to4(data_df[col]).astype('float32').to_frame())
            fixed['remap_2to4_then_ordinal'] += 1
        
        elif enc_type in ['one_hot', 'group_then_ohe', 'remap_2to4_then_ohe']:
            if col not in encoders:
                print(f"[!]  No encoder found for '{col}', skipping")
                continue
            
            enc_obj = encoders[col]
            series = data_df[col].copy()
            
            # Remap if needed (same function as 04c)
            if enc_type == 'remap_2to4_then_ohe':
                series = _remap_2to4(series, as_float=False)
                fixed['remap_2to4_then_ohe'] += 1
            
            # Convert to string, handle NaN
            series_str = _to_category_str(series)
            
            # Apply category mapping if exists
            if enc_obj['category_map']:
                series_str = series_str.map(
                    lambda x: enc_obj['category_map'].get(x, '__OTHER__')
                )
            else:
                train_cats = set(enc_obj['encoder'].categories_[0])
                series_str = series_str.map(
                    lambda x: x if x in train_cats else '__OTHER__'
                )
            
            # Transform - convert to numpy array first
            encoded_arr = enc_obj['encoder'].transform(np.array(series_str).reshape(-1, 1))
            encoded_df = pd.DataFrame(
                encoded_arr, 
                columns=enc_obj['feature_names'],
                index=data_df.index,
                dtype='float32'
            )
            encoded_parts.append(encoded_df)
    
    result = pd.concat(encoded_parts, axis=1)
    print(f"[OK] Applied saved encoders: {result.shape} (mapped: {fixed})")
    return result

