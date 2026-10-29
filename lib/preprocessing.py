"""
Data preprocessing functions extracted from Stage 02 for reuse in scoring.
"""
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


def condition_data(df, cfg, type_conversions_path=None):
    """
    Apply Stage 02 conditioning logic to dataframe.

    Args:
        df: Input dataframe
        cfg: Config dict
        type_conversions_path: {config_path}/type_conversions.csv (scoring must pass this
                               so text->numeric columns match training)

    Returns:
        Conditioned dataframe
    """
    data = df.copy()

    # Same order as Stage 02: type conversions, then fill numeric NaN with 0
    data = apply_type_conversions(data, type_conversions_path)

    # Fill missing values in numeric columns with 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(0)
    
    return data


def load_and_condition(file_path, cfg, folds=None):
    """
    Load parquet file and apply conditioning.
    
    Args:
        file_path: Path to parquet file
        cfg: Config dict
        folds: Optional list of folds to filter
        
    Returns:
        Conditioned dataframe
    """
    df = pd.read_parquet(file_path)
    
    if folds and 'fold' in df.columns:
        df = df[df['fold'].isin(folds)]
    
    return condition_data(df, cfg)