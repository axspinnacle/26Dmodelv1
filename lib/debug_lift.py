"""
Debug functions for comparing lift charts between GLM, GBM, and combined predictions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_deciles(data, pred_col, weight_col, bins=10):
    """Calculate deciles based on predictions using floor-based method (no bias)."""
    df = data.copy()
    df = df.sort_values(by=pred_col).reset_index(drop=True)
    
    cum_w = df[weight_col].cumsum() / df[weight_col].sum()
    df['decile'] = np.floor(np.round(cum_w, 2) * bins).astype(int)
    df['decile'] = np.where(df['decile'] + 1 > bins, bins, df['decile'] + 1)
    
    return df


def create_lift_summary(data, pred_col, actual_col, weight_col, bins=10):
    """
    Create lift summary table for any prediction column.
    Returns DataFrame with: decile, weight, act, pred, act_rel, pred_rel, error
    """
    # Add deciles based on pred_col
    df = calculate_deciles(data, pred_col, weight_col, bins)
    
    # Aggregate by decile
    agg = df.groupby('decile').agg({
        weight_col: 'sum',
        actual_col: lambda x: (x * df.loc[x.index, weight_col]).sum(),
        pred_col: lambda x: (x * df.loc[x.index, weight_col]).sum()
    }).reset_index()
    
    agg.columns = ['decile', 'weight', 'act_weighted', 'pred_weighted']
    
    # Calculate per-unit values
    agg['act'] = agg['act_weighted'] / agg['weight']
    agg['pred'] = agg['pred_weighted'] / agg['weight']
    
    # Calculate relativities
    overall_act = agg['act_weighted'].sum() / agg['weight'].sum()
    overall_pred = agg['pred_weighted'].sum() / agg['weight'].sum()
    
    agg['act_rel'] = agg['act'] / overall_act
    agg['pred_rel'] = agg['pred'] / overall_pred
    
    # Calculate decile error
    agg['error'] = abs(agg['pred'] / agg['act'] - 1)
    
    # Weight percentage
    agg['weight_pct'] = agg['weight'] / agg['weight'].sum() * 100
    
    return agg[['decile', 'weight', 'weight_pct', 'act', 'pred', 'act_rel', 'pred_rel', 'error']]


def calculate_fit_metrics(lift_df, weight_col='weight'):
    """Calculate fit_quality and model_power from lift summary."""
    total_weight = lift_df[weight_col].sum()
    
    # Fit quality: 1 - weighted average error
    weighted_error = (lift_df['error'] * lift_df[weight_col]).sum() / total_weight
    fit_quality = 1 - weighted_error
    
    # Model power: weighted average of |pred_rel - 1|
    model_power = (abs(lift_df['pred_rel'] - 1) * lift_df[weight_col]).sum() / total_weight
    
    return {
        'fit_quality': fit_quality,
        'model_power': model_power,
        'weighted_avg_error': weighted_error
    }


def compare_lift_charts(data, actual_col, weight_col, 
                        glm_col=None, gbm_col=None, final_col=None,
                        bins=10, title_prefix=""):
    """
    Compare lift charts for GLM, GBM, and/or final predictions.
    Returns dict with lift summaries and metrics for each prediction type.
    """
    results = {}
    
    pred_cols = {
        'GLM': glm_col,
        'GBM': gbm_col,
        'Final': final_col
    }
    
    for name, col in pred_cols.items():
        if col is not None and col in data.columns:
            lift_df = create_lift_summary(data, col, actual_col, weight_col, bins)
            metrics = calculate_fit_metrics(lift_df)
            
            results[name] = {
                'lift': lift_df,
                'metrics': metrics
            }
            
            print(f"\n{'='*50}")
            print(f"{title_prefix}{name} Predictions")
            print(f"{'='*50}")
            print(f"  Fit Quality:  {metrics['fit_quality']:.4f}")
            print(f"  Model Power:  {metrics['model_power']:.4f}")
            print(f"  Avg Error:    {metrics['weighted_avg_error']:.4f}")
            print(f"\nDecile Summary:")
            print(lift_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    return results


def plot_lift_comparison(results, figsize=(14, 5)):
    """Plot lift charts side by side for comparison."""
    n_plots = len(results)
    if n_plots == 0:
        print("No results to plot")
        return None
    
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0], figsize[1]))
    if n_plots == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, results.items()):
        lift_df = data['lift']
        metrics = data['metrics']
        
        # Plot relativities
        ax.plot(lift_df['decile'], lift_df['act_rel'], 'o-', label='Actual', linewidth=2, color='blue')
        ax.plot(lift_df['decile'], lift_df['pred_rel'], 's-', label='Predicted', linewidth=2, color='orange')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Decile')
        ax.set_ylabel('Relativity')
        ax.set_title(f"{name}\nFit: {metrics['fit_quality']:.3f}, Power: {metrics['model_power']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(lift_df['act_rel'].max(), lift_df['pred_rel'].max()) * 1.2)
    
    plt.tight_layout()
    return fig


def debug_glm_vs_gbm(train_data, test_data, 
                     actual_col, weight_col, glm_col,
                     gbm_raw_col=None, final_pp_col=None,
                     bins=10):
    """Full debug comparison of GLM vs GBM vs Final predictions."""
    output = {}
    
    for split_name, data in [('Train', train_data), ('Test', test_data)]:
        if data is None:
            continue
            
        print(f"\n{'#'*60}")
        print(f"# {split_name} Set Analysis")
        print(f"{'#'*60}")
        
        results = compare_lift_charts(
            data=data,
            actual_col=actual_col,
            weight_col=weight_col,
            glm_col=glm_col,
            gbm_col=gbm_raw_col,
            final_col=final_pp_col,
            bins=bins,
            title_prefix=f"{split_name} - "
        )
        
        # Plot
        fig = plot_lift_comparison(results)
        if fig:
            plt.suptitle(f"{split_name} Set Lift Comparison", y=1.02, fontsize=12, fontweight='bold')
            plt.show()
        
        output[split_name] = results
    
    return output
