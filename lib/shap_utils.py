"""
Refactored SHAP utility functions - no global variables!

Clean, reusable functions for SHAP analysis that explicitly
pass data and return results instead of using globals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
