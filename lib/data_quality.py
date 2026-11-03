"""Data quality helpers (duplicates, exposure floor)."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def analyze_duplicates(
    df: pd.DataFrame,
    key_col: str,
    value_cols: Optional[List[str]] = None
) -> Dict:
    """Analyze duplicate rows based on a key column."""
    total_rows = len(df)
    unique_keys = df[key_col].nunique()
    
    # Count duplicates per key
    dup_counts = df.groupby(key_col).size()
    
    # Distribution of duplicate counts
    dup_distribution = dup_counts.value_counts().sort_index()
    
    # Calculate percentage
    dup_ratio = total_rows / unique_keys if unique_keys > 0 else 0
    
    # Identify keys with duplicates
    duplicated_keys = dup_counts[dup_counts > 1]
    
    result = {
        'total_rows': total_rows,
        'unique_keys': unique_keys,
        'duplicate_ratio': dup_ratio,
        'num_duplicated_keys': len(duplicated_keys),
        'duplicate_distribution': dup_distribution.to_dict(),
        'max_duplicates': dup_counts.max(),
        'avg_duplicates': dup_counts.mean()
    }
    
    # Check value consistency if value columns provided
    if value_cols:
        consistency = check_duplicate_consistency(df, key_col, value_cols)
        result['value_consistency'] = consistency
    
    return result


def check_duplicate_consistency(
    df: pd.DataFrame,
    key_col: str,
    value_cols: List[str]
) -> Dict:
    """Check if duplicated keys have identical values across value columns."""
    # Get only rows with duplicated keys
    duplicated_mask = df.duplicated(subset=key_col, keep=False)
    duplicated_df = df[duplicated_mask]
    
    if len(duplicated_df) == 0:
        return {col: {'consistent': True, 'inconsistent_keys': 0} for col in value_cols}
    
    results = {}
    
    for col in value_cols:
        if col not in df.columns:
            results[col] = {'error': f'Column {col} not found'}
            continue
            
        # For each key, check if all values are the same
        grouped = duplicated_df.groupby(key_col)[col]
        
        # Count unique values per key
        nunique = grouped.nunique()
        
        # Keys where values differ
        inconsistent_keys = nunique[nunique > 1]
        
        results[col] = {
            'consistent': len(inconsistent_keys) == 0,
            'inconsistent_keys': len(inconsistent_keys),
            'total_duplicated_keys': nunique.shape[0],
            'pct_consistent': (1 - len(inconsistent_keys) / nunique.shape[0]) * 100 if nunique.shape[0] > 0 else 100
        }
        
        # Sample inconsistent keys
        if len(inconsistent_keys) > 0:
            sample_key = inconsistent_keys.index[0]
            sample_values = duplicated_df[duplicated_df[key_col] == sample_key][col].values
            results[col]['sample_inconsistent_key'] = sample_key
            results[col]['sample_values'] = sample_values.tolist()
    
    return results


def get_sample_duplicates(
    df: pd.DataFrame,
    key_col: str,
    n_samples: int = 3
) -> pd.DataFrame:
    """Get sample rows showing duplicates."""
    # Find keys that have duplicates
    dup_counts = df.groupby(key_col).size()
    duplicated_keys = dup_counts[dup_counts > 1].index[:n_samples]
    
    # Get all rows for these keys
    sample_df = df[df[key_col].isin(duplicated_keys)].sort_values(key_col)
    
    return sample_df


def summarize_duplicates(
    file_path: str,
    key_col: str,
    value_cols: Optional[List[str]] = None
) -> Dict:
    """Read a parquet file and run analyze_duplicates on it."""
    # Determine columns to load
    cols_to_load = [key_col]
    if value_cols:
        cols_to_load.extend(value_cols)
    
    # Load data
    df = pd.read_parquet(file_path, columns=cols_to_load)
    
    # Analyze
    results = analyze_duplicates(df, key_col, value_cols)
    results['file_path'] = file_path
    results['key_column'] = key_col
    
    return results


def apply_exposure_floor(df, exposure_cols, floor_value, verbose=True):
    """Apply minimum floor to exposure columns."""
    import pandas as pd
    
    df = df.copy()
    
    if verbose:
        print(f"\nApplying exposure floor: {floor_value}")
    
    for col in exposure_cols:
        if col in df.columns:
            n_below = (df[col] < floor_value).sum()
            if n_below > 0:
                df[col] = df[col].clip(lower=floor_value)
                pct = n_below / len(df) * 100
                if verbose:
                    print(f"  {col}: Floored {n_below:,} records ({pct:.2f}%)")
            else:
                if verbose:
                    print(f"  {col}: No records below floor")
        else:
            if verbose:
                print(f"  {col}: Column not found")
    
    return df
