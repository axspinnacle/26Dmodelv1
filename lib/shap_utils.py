"""
Refactored SHAP utility functions - no global variables!

Clean, reusable functions for SHAP analysis that explicitly
pass data and return results instead of using globals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def compute_shap_aggregate(shap_df, weight_col='weight'):
    """Aggregate SHAP values by feature (weighted). Usage: shagg, shagg_num, shagg2 = compute_shap_aggregate(shap_df)"""
    fc_df2 = shap_df.copy()
    
    # Get all columns except weight and base_value
    cols = [i for i in fc_df2.columns if i not in [weight_col, 'base_value']]
    
    # Weight absolute SHAP values
    for i in cols:
        fc_df2[i] = np.abs(fc_df2[i]) * fc_df2[weight_col]
    
    # Aggregate
    shagg = fc_df2[cols].sum().reset_index()
    shagg.columns = ['field', 'total_shap']
    
    # Sort by importance
    shagg = shagg.sort_values(by='total_shap', ascending=False)
    
    # Filter to non-zero
    shagg2 = shagg.loc[shagg['total_shap'] > 0].copy()
    shagg_num = shagg.loc[shagg['total_shap'] > 0].copy()
    
    # Add cumulative stats
    shagg_num['cum_shap_abs'] = shagg_num['total_shap'].cumsum()
    shagg_num['shap_abs_pct'] = shagg_num['total_shap'] / shagg_num['total_shap'].sum()
    shagg_num['cum_shap_abs_pct'] = shagg_num['cum_shap_abs'] / shagg_num['cum_shap_abs'].max()
    
    return shagg, shagg_num, shagg2


def create_residual_plot(data, feature, weight, round_value=2, print_table=False, dataset_label=""):
    """Plot actual vs predicted by feature. Usage: fig, df = create_residual_plot(data, 'DrvAge_raw', 'weight', dataset_label='Train')"""
    # Aggregate by feature
    agg_dict = {weight: 'sum', 'actual': 'sum', 'pred': 'sum'}
    x = data.groupby([feature]).agg(agg_dict).reset_index()
    x[feature] = round(x[feature], round_value)
    
    # Bin if too many unique values
    if x.shape[0] > 50:
        x = x.groupby(pd.qcut(x[feature], q=20, duplicates='drop')).agg(agg_dict).reset_index()
    
    # Calculate act/pred rates (using weight as denominator)
    x['act_rate'] = x['actual'] / x[weight]
    x['pred_rate'] = x['pred'] / x[weight]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    
    y_max = max(x['act_rate'].max(), x['pred_rate'].max()) * 1.20
    ax2.set_ylim(0, y_max)
    
    x[weight].plot.bar(stacked=False, ax=ax, alpha=0.6, color='lightblue')
    x['act_rate'].plot(kind='line', ax=ax2, marker='o', label='Actual', linewidth=2)
    x['pred_rate'].plot(kind='line', ax=ax2, marker='s', label='Predicted', linewidth=2)
    
    ax.set_xlabel(feature)
    ax.set_ylabel('Weight')
    ax2.set_ylabel('Rate')
    ax2.legend()
    title = f'[{dataset_label}] Residual Plot: {feature}' if dataset_label else f'Residual Plot: {feature}'
    plt.title(title)
    
    if print_table:
        print(f"\n{feature} Summary:")
        print(x[[feature, weight, 'act_rate', 'pred_rate']].to_string(index=False))
    
    return fig, x


def create_shap_range_plot(shap_df, data_df, feature, weight_field,
                           shap_round_level=3, feature_round_to=1,
                           min_ntile=0, max_ntile=0, filter_used_only=False, dataset_label="",
                           max_levels=50):
    """
    SHAP spread plot: for each (rounded) feature value, the exposure-weighted SHAP percentiles
    (0-100, coloured), plus an exposure distribution panel underneath.
    Features with more than max_levels distinct values are grouped into max_levels quantile bins
    (each bin plotted at its median value), so every point has enough rows for percentiles.
    Usage: fig, ntile_df = create_shap_range_plot(shap_df, data_df, 'feature', 'weight', dataset_label='Train')
    Returns: (figure, DataFrame[feature, ntile, SHAP]) - caller saves and closes the figure.
    """
    import seaborn  # noqa: F401  registers the 'vlag' colormap used below

    # Combine feature value, weight and SHAP by position (both frames share row order)
    cf = pd.DataFrame({
        feature: pd.to_numeric(data_df[feature], errors='coerce').to_numpy(dtype='float64'),
        'weight': data_df[weight_field].to_numpy(dtype='float64'),
        'contrib': shap_df[feature].to_numpy(dtype='float64'),
    })
    if feature_round_to:
        cf[feature] = np.round(cf[feature] / feature_round_to) * feature_round_to
    if max_levels and cf[feature].nunique() > max_levels:
        bins = pd.qcut(cf[feature], q=max_levels, duplicates='drop')
        cf[feature] = cf.groupby(bins, observed=True)[feature].transform('median')
    cf['contrib'] = cf['contrib'].round(shap_round_level)
    if filter_used_only:
        cf = cf.loc[cf['contrib'].fillna(0) != 0]

    # Weighted SHAP percentile within each feature value
    g = cf.groupby([feature, 'contrib'], sort=True)['weight'].sum().reset_index()
    g['f_cumsum'] = g.groupby(feature)['weight'].cumsum()
    g['f_weight'] = g.groupby(feature)['weight'].transform('sum')
    g = g.loc[g['f_weight'] > 0]
    g['ntile'] = np.rint(g['f_cumsum'] / g['f_weight'] * 100).astype(int)
    g['sp'] = g['contrib'] * g['weight']
    nt = g.groupby([feature, 'ntile'])[['sp', 'weight']].sum()
    nt['SHAP'] = nt['sp'] / nt['weight']

    # All 0-100 ntiles per value: carry last SHAP upward, fill below first with the lowest SHAP
    wide = nt['SHAP'].unstack('ntile').reindex(columns=range(101))
    wide = wide.ffill(axis=1)
    wide = wide.apply(lambda r: r.fillna(r.min()), axis=1)
    df = wide.stack().rename('SHAP').reset_index()
    df.columns = [feature, 'ntile', 'SHAP']
    if min_ntile == 0 and max_ntile == 0:
        min_ntile, max_ntile = 0, 100
    df = df.loc[(df['ntile'] >= min_ntile) & (df['ntile'] <= max_ntile)].reset_index(drop=True)

    # Exposure distribution: total and where the feature is used (SHAP != 0)
    dist = cf.groupby(feature)['weight'].sum().rename('weight').to_frame()
    dist['used_weight'] = cf.loc[cf['contrib'].fillna(0) != 0].groupby(feature)['weight'].sum()
    dist = dist.fillna(0).reset_index()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})
    sc = ax.scatter(df[feature], df['SHAP'], c=df['ntile'], cmap='vlag', s=12)
    fig.colorbar(sc, ax=[ax, ax2], label='SHAP Percentile', shrink=0.8, pad=0.02)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_ylabel('SHAP')
    title = f'Continuous Feature SHAP Spread Plot; {feature}'
    ax.set_title(f'[{dataset_label}] {title}' if dataset_label else title)
    span = dist[feature].max() - dist[feature].min()
    width = max(span / max(len(dist), 1) * 0.8, 1e-9) if span > 0 else 0.8
    ax2.bar(dist[feature], dist['weight'], width=width, color='grey', alpha=0.4, label='exposure')
    ax2.bar(dist[feature], dist['used_weight'], width=width, color='orange', alpha=0.5, label='SHAP != 0')
    ax2.set_ylabel('Exposure')
    ax2.set_xlabel(feature)
    ax2.legend(loc='upper right', fontsize=8)
    return fig, df


def create_feature_importance_plot(shagg_df, top_n=20):
    """Bar chart of top N features by SHAP. Usage: fig = create_feature_importance_plot(shagg_df, top_n=20)"""
    top_features = shagg_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    ax.barh(range(len(top_features)), top_features['total_shap'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['field'])
    ax.invert_yaxis()
    ax.set_xlabel('Total Weighted Absolute SHAP')
    ax.set_title(f'Top {top_n} Features by SHAP Importance')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def create_shap_beeswarm(shap_df, feature_df, top_n=20, weight_col='weight', max_display=20):
    """
    Create standard SHAP beeswarm plot with individual dots colored by feature value.
    Shows the relationship between feature values (color) and SHAP contributions (x-axis).
    
    Args:
        shap_df: DataFrame with SHAP values (rows=samples, cols=features)
        feature_df: DataFrame with original feature values (same index as shap_df)
        top_n: Number of top features to plot
        weight_col: Weight column name in shap_df
        max_display: Max features to display (same as top_n typically)
    
    Returns:
        fig: matplotlib figure
    
    Note:
        - Each dot is a record
        - X-axis: SHAP value (impact on prediction)
        - Color: Feature value (red=high, blue=low)
        - For residual method, SHAP explains log(PP/GLM)
    """
    import shap
    print(f"  Creating SHAP beeswarm for {len(shap_df):,} samples...")
    
    # Get top features by weighted importance
    shagg, _, shagg2 = compute_shap_aggregate(shap_df, weight_col=weight_col)
    top_features = shagg2.head(top_n)['field'].tolist()
    
    # Filter to top features
    shap_vals = shap_df[top_features].values
    feature_vals = feature_df[top_features].values
    
    # Get base values if available
    base_values = shap_df['base_value'].values if 'base_value' in shap_df.columns else None
    
    # Create SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_values,
        data=feature_vals,
        feature_names=top_features
    )
    
    # Create beeswarm plot
    fig = plt.figure(figsize=(10, max(6, len(top_features) * 0.4)))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    
    print(f"  Beeswarm created with {len(top_features)} features")
    return fig


def create_fast_shap_summary(shap_df, feature_df, top_n=20, weight_col='weight', n_quantiles=20):
    """
    Fast SHAP summary plot using percentile bands instead of scatter.
    Uses 100% of data, plots aggregated percentiles - instant rendering.
    
    LIMITATION: This plot does NOT show the relationship between feature values
    and SHAP contributions. Each feature has one colored band, hiding individual
    record patterns. Use create_shap_beeswarm() for proper interpretation.
    
    Args:
        shap_df: DataFrame with SHAP values (rows=samples, cols=features)
        feature_df: DataFrame with original feature values (same index as shap_df)
        top_n: Number of top features to plot
        weight_col: Weight column name in shap_df
        n_quantiles: Number of quantile bands for feature binning
    
    Returns:
        fig: matplotlib figure
    """
    print(f"  Creating fast SHAP summary for {len(shap_df):,} rows...")
    
    # Get top features by weighted importance
    shagg, _, shagg2 = compute_shap_aggregate(shap_df, weight_col=weight_col)
    top_features = shagg2.head(top_n)['field'].tolist()
    
    # Prepare data for plotting
    results = []
    for feat in top_features:
        if feat not in shap_df.columns or feat not in feature_df.columns:
            continue
            
        shap_vals = shap_df[feat].values
        feat_vals = feature_df[feat].values
        
        # Remove NaN
        mask = ~(np.isnan(shap_vals) | np.isnan(feat_vals))
        shap_vals = shap_vals[mask]
        feat_vals = feat_vals[mask]
        
        if len(shap_vals) == 0:
            continue
        
        # Compute percentiles for this feature
        percentiles = [5, 25, 50, 75, 95]
        shap_pcts = np.percentile(shap_vals, percentiles)
        
        # Normalize feature values to [0, 1] for color
        feat_min, feat_max = np.percentile(feat_vals, [1, 99])
        if feat_max > feat_min:
            feat_norm = np.median((feat_vals - feat_min) / (feat_max - feat_min))
        else:
            feat_norm = 0.5
        
        results.append({
            'feature': feat,
            'p05': shap_pcts[0],
            'p25': shap_pcts[1],
            'p50': shap_pcts[2],
            'p75': shap_pcts[3],
            'p95': shap_pcts[4],
            'feat_norm': np.clip(feat_norm, 0, 1)
        })
    
    if not results:
        print("  Warning: No valid features for summary plot")
        return None
    
    df_plot = pd.DataFrame(results)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.4)))
    
    # Color map (blue to red)
    cmap = plt.cm.RdBu_r
    
    for i, row in df_plot.iterrows():
        y = len(df_plot) - i - 1  # Reverse order
        
        # Plot percentile bands
        # IQR (25-75)
        color = cmap(row['feat_norm'])
        ax.barh(y, row['p75'] - row['p25'], left=row['p25'], height=0.6, 
                color=color, alpha=0.7, edgecolor='none')
        
        # 5-95 whiskers
        ax.plot([row['p05'], row['p95']], [y, y], 'k-', linewidth=1.5, alpha=0.5)
        
        # Median marker
        ax.plot(row['p50'], y, 'ko', markersize=4)
    
    # Set labels
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['feature'][::-1])
    ax.set_xlabel('SHAP Value', fontsize=11)
    ax.set_title(f'SHAP Summary (Percentile Bands, {len(shap_df):,} samples)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Feature Value', rotation=270, labelpad=20)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])
    
    plt.tight_layout()
    print(f"  SHAP summary created")
    return fig


def create_shap_importance_pct(shap_df, top_n=20, weight_col='weight'):
    """
    Create SHAP feature importance as percentage bar chart.
    
    Args:
        shap_df: DataFrame with SHAP values
        top_n: Number of top features to show
        weight_col: Weight column name
    
    Returns:
        fig: matplotlib figure
    """
    print(f"  Creating SHAP importance % chart for {len(shap_df):,} rows...")
    
    # Get weighted importance
    shagg, _, shagg2 = compute_shap_aggregate(shap_df, weight_col=weight_col)
    top_features = shagg2.head(top_n).copy()
    
    # Calculate percentage
    total_shap = top_features['total_shap'].sum()
    top_features['pct'] = (top_features['total_shap'] / total_shap) * 100
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    
    # Horizontal bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_features['pct'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Labels
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['field'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (%)', fontsize=11)
    ax.set_title(f'SHAP Feature Importance (Top {top_n})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['pct'] + 0.5, i, f"{row['pct']:.1f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    print(f"  SHAP importance % chart created")
    return fig
