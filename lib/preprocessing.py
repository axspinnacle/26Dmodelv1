"""
Data preprocessing functions extracted from Stage 02 for reuse in scoring.
"""
import pandas as pd
import numpy as np


def condition_data(df, cfg):
    """
    Apply Stage 02 conditioning logic to dataframe.
    
    Args:
        df: Input dataframe
        cfg: Config dict
        
    Returns:
        Conditioned dataframe
    """
    data = df.copy()
    
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