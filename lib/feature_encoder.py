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
    print(f"✓ Loaded master encoding: {len(encoding_df)} columns")
    
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
        
        elif enc_type == 'group_then_ohe':
            tr_encoded, te_encoded, enc_obj, summary = _encode_one_hot(
                train_df[col], test_df[col], col, 
                group_rare=True, min_frequency=min_frequency, min_count=min_count
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
            print(f"⚠️  Unknown encoding type '{enc_type}' for column '{col}', skipping")
    
    train_encoded = pd.concat(train_parts, axis=1)
    test_encoded = pd.concat(test_parts, axis=1)
    encoding_summary = pd.DataFrame(summary_rows)
    
    print(f"\n✓ Encoding complete:")
    print(f"  Train shape: {train_encoded.shape}")
    print(f"  Test shape: {test_encoded.shape}")
    
    return train_encoded, test_encoded, encoders, encoding_summary


def _encode_one_hot(train_series, test_series, col_name, group_rare=False, 
                   min_frequency=0.01, min_count=50):
    """One-hot encode with __OTHER__ and __MISSING__ handling"""
    train_series = train_series.copy()
    test_series = test_series.copy()
    
    # Convert to string, handle NaN as __MISSING__
    # Use object dtype to avoid ArrowStringArray issues
    train_str = train_series.astype('object').astype(str).replace('nan', '__MISSING__')
    test_str = test_series.astype('object').astype(str).replace('nan', '__MISSING__')
    
    # Group rare categories if requested
    category_map = None
    if group_rare:
        value_counts = train_str.value_counts()
        total = len(train_str)
        
        # Keep categories that meet threshold
        keep_cats = value_counts[
            (value_counts >= min_count) | (value_counts / total >= min_frequency)
        ].index.tolist()
        
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


def _encode_binary(train_series, test_series, col_name, encoding_str):
    """Apply binary mapping like 'binary: {0,1,2}->0 | {3,4,5}->1'"""
    import re
    match = re.search(r'\{([^}]+)\}->0.*\{([^}]+)\}->1', encoding_str)
    if not match:
        raise ValueError(f"Cannot parse binary encoding: {encoding_str}")
    
    low_set = set(int(x.strip()) for x in match.group(1).split(','))
    high_set = set(int(x.strip()) for x in match.group(2).split(','))
    
    def mapper(v):
        if pd.isna(v):
            return np.nan
        v_int = int(round(v))
        return 0.0 if v_int in low_set else (1.0 if v_int in high_set else np.nan)
    
    train_encoded = train_series.map(mapper).astype('float32').to_frame()
    test_encoded = test_series.map(mapper).astype('float32').to_frame()
    
    summary = {
        'original_column': col_name, 'encoding_type': 'binary',
        'output_columns': col_name, 'n_categories': 2,
        'has_other': False, 'has_missing': False
    }
    
    return train_encoded, test_encoded, summary


def _encode_remap_ordinal(train_series, test_series, col_name):
    """Remap 2->4, keep as ordinal"""
    def remap(v):
        if pd.isna(v):
            return v
        v_int = int(round(v))
        return 4.0 if v_int == 2 else float(v_int)
    
    train_encoded = train_series.map(remap).astype('float32').to_frame()
    test_encoded = test_series.map(remap).astype('float32').to_frame()
    
    summary = {
        'original_column': col_name, 'encoding_type': 'remap_2to4_then_ordinal',
        'output_columns': col_name, 'n_categories': 0,
        'has_other': False, 'has_missing': False
    }
    
    return train_encoded, test_encoded, summary


def _encode_remap_ohe(train_series, test_series, col_name):
    """Remap 2->4, then one-hot encode"""
    def remap(v):
        if pd.isna(v):
            return v
        v_int = int(round(v))
        return 4 if v_int == 2 else v_int
    
    train_remapped = train_series.map(remap)
    test_remapped = test_series.map(remap)
    
    return _encode_one_hot(train_remapped, test_remapped, col_name, group_rare=False)


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
    print(f"✓ Saved encoders: {encoders_file}")
    
    # Save encoding summary as CSV
    summary_file = results_dir / "04c_encoding_summary.csv"
    encoding_summary.to_csv(summary_file, index=False)
    print(f"✓ Saved encoding summary: {summary_file}")
    
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
    print(f"✓ Saved encoding map: {mapping_file}")


def load_encoders(output_path):
    """Load saved encoders and summary"""
    output_path = Path(output_path)
    
    encoders_file = output_path / "models" / "04c_encoders.pkl"
    summary_file = output_path / "results" / "04c_encoding_summary.csv"
    
    with open(encoders_file, 'rb') as f:
        encoders = pickle.load(f)
    
    encoding_summary = pd.read_csv(summary_file)
    
    print(f"✓ Loaded encoders: {len(encoders)} columns")
    print(f"✓ Loaded summary: {len(encoding_summary)} rows")
    
    return encoders, encoding_summary


def apply_saved_encoders(data_df, encoders, encoding_summary):
    """Apply saved encoders to new data (e.g., holdout set)"""
    encoded_parts = []
    
    for _, row in encoding_summary.iterrows():
        col = row['original_column']
        enc_type = row['encoding_type']
        
        if enc_type in ['skip', 'DROP', 'SKIPPED']:
            continue
        
        if col not in data_df.columns:
            print(f"⚠️  Column '{col}' not in data, skipping")
            continue
        
        if enc_type in ['numeric', 'ordinal_0_5', 'binary', 'remap_2to4_then_ordinal']:
            encoded_parts.append(data_df[[col]].astype('float32'))
        
        elif enc_type in ['one_hot', 'group_then_ohe', 'remap_2to4_then_ohe']:
            if col not in encoders:
                print(f"⚠️  No encoder found for '{col}', skipping")
                continue
            
            enc_obj = encoders[col]
            series = data_df[col].copy()
            
            # Remap if needed
            if enc_type == 'remap_2to4_then_ohe':
                def remap(v):
                    if pd.isna(v):
                        return v
                    v_int = int(round(v))
                    return 4 if v_int == 2 else v_int
                series = series.map(remap)
            
            # Convert to string, handle NaN
            series_str = series.astype('object').astype(str).replace('nan', '__MISSING__')
            
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
    print(f"✓ Applied saved encoders: {result.shape}")
    return result

