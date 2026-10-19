"""
EDA Helper Functions for Feature Analysis and Encoding Recommendations
"""
import pandas as pd
import numpy as np
import os


def analyze_column(col_name, col_data, skip_patterns=None, target_exposure=None):
    """
    Analyze a single column and recommend encoding strategy.
    
    Parameters
    ----------
    col_name : str
        Column name
    col_data : pd.Series
        Column data
    skip_patterns : list, optional
        Patterns to skip (identifiers)
    target_exposure : set, optional
        Target/exposure column names to skip
    
    Returns
    -------
    dict
        Analysis result with encoding recommendation
    """
    if skip_patterns is None:
        skip_patterns = ['vin', 'policyid', 'reference_num', 'companyid', '__index_level']
    if target_exposure is None:
        target_exposure = set()
    
    dtype_str = str(col_data.dtype)
    
    # Basic stats
    n_total = len(col_data)
    n_null = col_data.isnull().sum()
    null_pct = (n_null / n_total) * 100
    n_unique = col_data.nunique()
    
    # Default values
    category = "unknown"
    encoding = "skip"
    
    # Skip identifiers and targets
    if any(pattern in col_name.lower() for pattern in skip_patterns):
        category = "identifier"
        encoding = "skip"
    elif col_name in target_exposure:
        category = "target_exposure"
        encoding = "skip"
    elif 'timestamp' in dtype_str or 'date' in col_name.lower():
        category = "datetime"
        encoding = "skip"
    elif dtype_str == 'object':
        # String/categorical column
        if n_unique <= 2:
            category = "binary_categorical"
            encoding = "one_hot"
        elif n_unique <= 10:
            category = "low_cardinality_categorical"
            encoding = "one_hot"
        elif n_unique <= 100:
            category = "medium_cardinality_categorical"
            encoding = "one_hot"
        else:
            category = "high_cardinality_categorical"
            encoding = "group_then_ohe"  # Group rare categories
    else:
        # Numeric column - continued in next function
        return _analyze_numeric_column(col_name, col_data, n_unique, dtype_str, null_pct)
    
    return {
        'column_name': col_name,
        'encoding': encoding,
        'category': category,
        'unique_count': n_unique,
        'null_pct': round(null_pct, 2),
        'dtype': dtype_str,
        'source': 'auto'
    }


def _analyze_numeric_column(col_name, col_data, n_unique, dtype_str, null_pct):
    """Helper for numeric column analysis"""
    non_null = col_data.dropna()
    category = "unknown"
    encoding = "numeric"
    
    if len(non_null) > 0:
        try:
            # Ensure numeric comparison (handle mixed types)
            min_val = float(non_null.min())
            max_val = float(non_null.max())
            
            # Check if ordinal 0-5
            if n_unique <= 7 and min_val >= 0 and max_val <= 5:
                category = "ordinal_0_5"
                encoding = "ordinal_0_5"
            elif n_unique <= 2:
                category = "binary_numeric"
                encoding = "numeric"
            elif n_unique <= 20:
                category = "discrete_numeric"
                encoding = "numeric"
            else:
                category = "continuous_numeric"
                encoding = "numeric"
        except (ValueError, TypeError):
            # Can't convert to float - treat as continuous
            category = "continuous_numeric"
            encoding = "numeric"
    
    return {
        'column_name': col_name,
        'encoding': encoding,
        'category': category,
        'unique_count': n_unique,
        'null_pct': round(null_pct, 2),
        'dtype': dtype_str,
        'source': 'auto'
    }


def detect_dtype_conversions(df_sample, success_threshold=0.8):
    """Detect object columns convertible to numeric."""
    dtype_fixes = []
    for col_name in df_sample.columns:
        if df_sample[col_name].dtype == "object":
            test_convert = pd.to_numeric(df_sample[col_name], errors="coerce")
            success_rate = test_convert.notna().sum() / len(test_convert)
            if success_rate > success_threshold:
                dtype_fixes.append({
                    "column_name": col_name,
                    "source_dtype": "object",
                    "target_dtype": "numeric",
                    "success_rate": f"{success_rate:.1%}",
                    "notes": f"Auto: {success_rate:.1%} convertible"
                })
    return pd.DataFrame(dtype_fixes)


def merge_encodings(auto_df, manual_file):
    """Merge auto and manual encodings. Manual wins."""
    if not os.path.exists(manual_file):
        return auto_df.copy()
    manual_df = pd.read_csv(manual_file)
    if "feature" in manual_df.columns:
        manual_df = manual_df.rename(columns={"feature": "column_name"})
    if "liab" in manual_df.columns:
        manual_df = manual_df.rename(columns={"liab": "encoding"})
    master = auto_df.copy()
    for _, manual_row in manual_df.iterrows():
        col = manual_row["column_name"]
        if col in master["column_name"].values:
            idx = master[master["column_name"] == col].index[0]
            master.loc[idx, "encoding"] = manual_row["encoding"]
            master.loc[idx, "source"] = "manual"
    print(f"Merged: {len(master)} total ({(master['source']=='manual').sum()} manual)")
    return master

