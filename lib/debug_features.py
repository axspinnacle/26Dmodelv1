"""
Diagnostic functions for feature correlation analysis with target ratio.
Memory-efficient processing for large datasets.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sample_data(df, frac=0.1, seed=42):
    """Randomly sample fraction of data."""
    print(f"Sampling {frac*100:.0f}% of {len(df):,} rows...")
    sampled = df.sample(frac=frac, random_state=seed)
    print(f"  Sampled: {len(sampled):,} rows")
    return sampled


def check_target_distribution(target_series, weight_series=None, title="Target Ratio"):
    """
    Check distribution of target ratio with histogram and stats.
    
    Returns dict with stats.
    """
    data = target_series.dropna()
    
    if weight_series is not None:
        weights = weight_series[data.index]
        mean_val = np.average(data, weights=weights)
    else:
        mean_val = data.mean()
    
    median_val = data.median()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    
    # Outliers (beyond 3 sigma)
    outlier_threshold = mean_val + 3 * std_val
    outliers = (data > outlier_threshold) | (data < mean_val - 3 * std_val)
    pct_outliers = outliers.sum() / len(data) * 100
    
    stats = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'pct_outliers': pct_outliers,
        'count': len(data)
    }
    
    # Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data.clip(0, data.quantile(0.99)), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title} Distribution (clipped at 99th percentile for display)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return stats, fig


def calculate_correlations_chunked(df, feature_cols, target_col, chunk_size=50):
    """
    Calculate correlations between features and target in chunks to save memory.
    
    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list - feature column names
    target_col : str - target column name
    chunk_size : int - process this many features at a time
    
    Returns
    -------
    pd.DataFrame with columns: feature, correlation
    """
    print(f"\nCalculating correlations for {len(feature_cols)} features...")
    print(f"  Processing in chunks of {chunk_size}")
    
    correlations = []
    
    for i in range(0, len(feature_cols), chunk_size):
        chunk_features = feature_cols[i:i+chunk_size]
        print(f"  Chunk {i//chunk_size + 1}: features {i+1}-{min(i+chunk_size, len(feature_cols))}")
        
        # Calculate correlations for this chunk
        for feat in chunk_features:
            try:
                corr = df[feat].corr(df[target_col])
                correlations.append({'feature': feat, 'correlation': corr})
            except Exception as e:
                print(f"    Warning: Could not calculate correlation for {feat}: {e}")
                correlations.append({'feature': feat, 'correlation': np.nan})
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    
    print(f"\n  Completed: {len(corr_df)} correlations calculated")
    return corr_df


def summarize_feature_signals(corr_df, threshold=0.05):
    """
    Summarize which features have signal above threshold.
    
    Returns dict with summary stats.
    """
    valid_corr = corr_df.dropna()
    abs_corr = valid_corr['correlation'].abs()
    
    strong_features = valid_corr[abs_corr > threshold]
    weak_features = valid_corr[abs_corr <= threshold]
    
    summary = {
        'total_features': len(valid_corr),
        'strong_features': len(strong_features),
        'weak_features': len(weak_features),
        'threshold': threshold,
        'max_corr': abs_corr.max(),
        'top_10': strong_features.head(10)
    }
    
    return summary


def run_full_diagnostic(df, feature_cols, target_col, pred_col, weight_col, 
                        sample_frac=0.1, corr_threshold=0.05, chunk_size=50):
    """
    Run complete feature diagnostic: sampling, distribution, correlations.
    
    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list - feature column names  
    target_col : str - target ratio column name
    pred_col : str - prediction column name (for comparison)
    weight_col : str - weight/exposure column
    sample_frac : float - fraction to sample
    corr_threshold : float - threshold for "strong" correlation
    chunk_size : int - features per chunk
    
    Returns
    -------
    dict with all diagnostic results
    """
    results = {}
    
    # 1. Sample data
    print(f"\n{'='*60}")
    print("STEP 1: Sampling Data")
    print(f"{'='*60}")
    df_sample = sample_data(df, frac=sample_frac)
    results['sample'] = df_sample
    
    # 2. Target distribution
    print(f"\n{'='*60}")
    print("STEP 2: Target Ratio Distribution")
    print(f"{'='*60}")
    stats, fig = check_target_distribution(
        df_sample[target_col], 
        df_sample[weight_col],
        title="Target Ratio (PP/GLM)"
    )
    results['target_stats'] = stats
    results['target_fig'] = fig
    
    print(f"\nTarget Ratio Stats:")
    print(f"  Mean:    {stats['mean']:.4f}")
    print(f"  Median:  {stats['median']:.4f}")
    print(f"  Std:     {stats['std']:.4f}")
    print(f"  Min:     {stats['min']:.4f}")
    print(f"  Max:     {stats['max']:.4f}")
    print(f"  Outliers (>3σ): {stats['pct_outliers']:.2f}%")
    
    plt.show()
    
    # 3. Correlations with target
    print(f"\n{'='*60}")
    print("STEP 3: Feature Correlations with Target Ratio")
    print(f"{'='*60}")
    corr_target = calculate_correlations_chunked(
        df_sample, feature_cols, target_col, chunk_size=chunk_size
    )
    results['corr_target'] = corr_target
    
    # 4. Correlations with pred (for comparison)
    print(f"\n{'='*60}")
    print("STEP 4: Feature Correlations with Original Prediction")
    print(f"{'='*60}")
    corr_pred = calculate_correlations_chunked(
        df_sample, feature_cols, pred_col, chunk_size=chunk_size
    )
    results['corr_pred'] = corr_pred
    
    # 5. Summarize signals
    print(f"\n{'='*60}")
    print("STEP 5: Summary")
    print(f"{'='*60}")
    
    summary_target = summarize_feature_signals(corr_target, threshold=corr_threshold)
    summary_pred = summarize_feature_signals(corr_pred, threshold=corr_threshold)
    
    results['summary_target'] = summary_target
    results['summary_pred'] = summary_pred
    
    print(f"\nFeatures vs Target Ratio:")
    print(f"  Total features:  {summary_target['total_features']}")
    print(f"  Strong (|r|>{corr_threshold}): {summary_target['strong_features']}")
    print(f"  Weak (|r|<={corr_threshold}): {summary_target['weak_features']}")
    print(f"  Max |correlation|: {summary_target['max_corr']:.4f}")
    
    print(f"\nFeatures vs Original Prediction:")
    print(f"  Total features:  {summary_pred['total_features']}")
    print(f"  Strong (|r|>{corr_threshold}): {summary_pred['strong_features']}")
    print(f"  Weak (|r|<={corr_threshold}): {summary_pred['weak_features']}")
    print(f"  Max |correlation|: {summary_pred['max_corr']:.4f}")
    
    print(f"\nTop 10 Features Correlated with Target Ratio:")
    for idx, row in summary_target['top_10'].iterrows():
        print(f"  {row['feature']:40s}: {row['correlation']:7.4f}")
    
    return results


