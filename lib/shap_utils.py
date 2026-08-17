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


def create_residual_plot(data, feature, weight, round_value=2, print_table=False):
    """Plot actual vs predicted by feature. Usage: fig, df = create_residual_plot(data, 'DrvAge_raw', 'weight')"""
    # Aggregate by feature
    agg_dict = {weight: 'sum', 'incurred_act': 'sum', 'incurred_pred': 'sum', 'denom': 'sum'}
    x = data.groupby([feature]).agg(agg_dict).reset_index()
    x[feature] = round(x[feature], round_value)
    
    # Bin if too many unique values
    if x.shape[0] > 50:
        x = x.groupby(pd.qcut(x[feature], q=20, duplicates='drop')).agg(agg_dict).reset_index()
    
    # Calculate act/pred rates
    x['act'] = x['incurred_act'] / x['denom']
    x['pred'] = x['incurred_pred'] / x['denom']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    
    y_max = max(x['act'].max(), x['pred'].max()) * 1.20
    ax2.set_ylim(0, y_max)
    
    x[weight].plot.bar(stacked=False, ax=ax, alpha=0.6, color='lightblue')
    x['act'].plot(kind='line', ax=ax2, marker='o', label='Actual', linewidth=2)
    x['pred'].plot(kind='line', ax=ax2, marker='s', label='Predicted', linewidth=2)
    
    ax.set_xlabel(feature)
    ax.set_ylabel('Weight')
    ax2.set_ylabel('Value')
    ax2.legend()
    plt.title(f'Residual Plot: {feature}')
    
    if print_table:
        print(f"\n{feature} Summary:")
        print(x[[feature, weight, 'act', 'pred']].to_string(index=False))
    
    return fig, x


def create_shap_range_plot(shap_df, data_df, feature, weight_field, 
                          shap_round_level=3, feature_round_to=1, 
                          min_ntile=0, max_ntile=0, filter_used_only=False):
    """SHAP range plot across ntiles. Usage: fig, df = create_shap_range_plot(shap_df, data_df, 'feature', 'weight')"""
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
    
    # Set up for plotting
    unique_feat_levels = sorted(cf_df3[feature].drop_duplicates().tolist())
    ntiles = list(range(101))
    
    # Create full ntile grid
    grid_data = []
    for feat_val in unique_feat_levels:
        for ntile in ntiles:
            grid_data.append({feature: feat_val, 'ntile': ntile})
    
    cf_df4 = pd.DataFrame(grid_data)
    cf_df4 = cf_df4.merge(cf_df3, on=[feature, 'ntile'], how='left')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for feat_val in unique_feat_levels:
        subset = cf_df4[cf_df4[feature] == feat_val]
        ax.plot(subset['ntile'], subset['SHAP'], marker='o', label=f'{feature}={feat_val}', alpha=0.7)
    
    # Apply ntile filters if specified
    if min_ntile > 0 or max_ntile > 0:
        min_val = min_ntile if min_ntile > 0 else 0
        max_val = max_ntile if max_ntile > 0 else 100
        ax.set_xlim(min_val, max_val)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Ntile')
    ax.set_ylabel('Average SHAP Contribution')
    ax.set_title(f'SHAP Contribution Range: {feature}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, cf_df3


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
