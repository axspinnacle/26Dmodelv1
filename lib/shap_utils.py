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
                          min_ntile=0, max_ntile=0, filter_used_only=False, dataset_label=""):
    """SHAP range plot across ntiles. Usage: fig, df = create_shap_range_plot(shap_df, data_df, 'feature', 'weight', dataset_label='Train')"""
    import seaborn as sns
    
    # Combine data and SHAP
    cf_df = data_df[[feature, weight_field]].reset_index(drop=True).copy()
    cf_df.rename(columns={weight_field: 'weight'}, inplace=True)
    cf_df['contrib'] = shap_df[feature].reset_index(drop=True)
    
    # Rounding
    if feature_round_to != 0:
        cf_df[feature] = round(cf_df[feature] / feature_round_to, 0) * feature_round_to
    cf_df['contrib'] = round(cf_df['contrib'], shap_round_level)
    
    # Filter
    if filter_used_only:
        cf_df = cf_df.loc[cf_df['contrib'].fillna(0) != 0]
    
    # Aggregate rounded data
    cf_df2 = cf_df.groupby([feature, 'contrib']).agg({'weight': 'sum'}).reset_index()
    
    # Create ntiles
    cf_df2['f_cumsum'] = cf_df2.groupby([feature]).weight.cumsum()
    
    f_weight = cf_df2.groupby([feature]).agg({'weight': 'sum'}).reset_index()
    f_weight.rename(columns={'weight': 'f_weight'}, inplace=True)
    
    cf_df2 = cf_df2.merge(f_weight)
    cf_df2['ntile'] = round(cf_df2['f_cumsum'] / cf_df2['f_weight'], 2) * 100
    cf_df2 = cf_df2.loc[cf_df2['ntile'].notna()]
    cf_df2['ntile'] = cf_df2['ntile'].astype('int')
    
    # Weight SHAP contributions
    cf_df2['sp'] = cf_df2['contrib'] * cf_df2['weight']
    
    # Aggregate by feature and ntile
    cf_df3 = cf_df2.groupby([feature, 'ntile']).agg({'sp': 'sum', 'weight': 'sum'}).reset_index()
    cf_df3['SHAP'] = cf_df3['sp'] / cf_df3['weight']
    del cf_df3['sp'], cf_df3['weight']
    
    # Set up SHAP data for plotting (fill missing ntiles)
    unique_feat_levels = cf_df3[feature].drop_duplicates().tolist()
    ntiles = [i for i in range(101)]
    
    u_df = pd.DataFrame()
    u_df[feature] = unique_feat_levels
    u_df['key'] = 0
    
    n_df = pd.DataFrame()
    n_df['ntile'] = ntiles
    n_df['key'] = 0
    
    df_levels = u_df.merge(n_df)
    del df_levels['key']
    
    cf_df3.sort_values(by=[feature, 'ntile'], inplace=True)
    df_levels.sort_values(by=[feature, 'ntile'], inplace=True)
    
    df = pd.DataFrame()
    for i in unique_feat_levels:
        a = df_levels.loc[df_levels[feature] == i].copy()
        b = cf_df3.loc[cf_df3[feature] == i].copy()
        
        # Fill upwards
        a['ntile'] = a['ntile'].astype('int32')
        b['ntile'] = b['ntile'].astype('int32')
        c = pd.merge_asof(a, b, on='ntile')
        
        # Fill downwards
        lowest_contrib = c['SHAP'].min()
        c['SHAP'].fillna(lowest_contrib, inplace=True)
        
        df = pd.concat([df, c])
    
    df.rename(columns={feature + '_x': feature}, inplace=True)
    del df[feature + '_y']
    
    # Apply ntile filter
    if min_ntile == 0 and max_ntile == 0:
        min_ntile, max_ntile = 0, 100
    df = df.loc[(df['ntile'] >= min_ntile) & (df['ntile'] <= max_ntile)]
    
    # Distribution data
    a = cf_df.groupby([feature]).agg({'weight': 'sum'}).reset_index()
    b = cf_df.loc[cf_df['contrib'].fillna(0) != 0].groupby([feature]).agg({'weight': 'sum'}).reset_index()
    b.rename(columns={'weight': 'used_weight'}, inplace=True)
    
    c = a.merge(b, how='left')
    c[feature] = round(c[feature], 4)
    
    # Create joint plot
    g = sns.jointplot(x=df[feature], y=df['SHAP'], c=df['ntile'], height=12, 
                      joint_kws={"color": None, 'cmap': 'vlag'})
    
    g.fig.colorbar(g.ax_joint.collections[0], ax=[g.ax_joint, g.ax_marg_y, g.ax_marg_x], 
                   use_gridspec=True, orientation='vertical', shrink=.80, anchor=(0, 0), 
                   label='SHAP Percentile', pad=-.15)
    
    g.fig.set_figwidth(12)
    g.fig.set_figheight(8)
    
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()
    
    # Add dataset label to title
    title = f'[{dataset_label}] Continuous Feature SHAP Spread Plot; {feature}' if dataset_label else f'Continuous Feature SHAP Spread Plot; {feature}'
    g.fig.suptitle(title, y=.9)
    
    # Store the main figure
    main_fig = g.fig
    
    # Create weight distribution bar chart
    fig2, ax = plt.subplots(figsize=(12, 2))
    sns.barplot(data=c, x=c[feature], y=c['weight'], color='grey', alpha=.3, ax=ax)
    sns.barplot(data=c, x=c[feature], y=c['used_weight'], color='orange', alpha=.2, ax=ax)
    ax.set(xlabel=None, ylabel=None)
    
    return main_fig, cf_df3


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
    
    return fig
