"""
Fast lift chart implementation - optimized for large datasets
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def create_lift_chart(data, weight_name, bins=10, title="Lift Chart", y_max=4.0):
    """Create lift chart (optimized for large data). Usage: fig, decile_df = create_lift_chart(data, 'weight', bins=10, y_max=4.0)"""
    t0 = time.time()
    
    # Step 1: Create column list
    print(f"  [STEP 1] Creating column list...")
    t1 = time.time()
    cols_needed = ['pred', weight_name, 'incurred_act', 'incurred_pred', 'denom']
    # Optional GBM-only columns (if available)
    if 'incurred_act_gbm' in data.columns:
        cols_needed.extend(['incurred_act_gbm', 'incurred_pred_gbm'])
    print(f"  [STEP 1] Done in {time.time()-t1:.3f}s")
    
    # Step 2: Extract columns
    print(f"  [STEP 2] Extracting {len(cols_needed)} columns from {data.shape[0]:,} rows...")
    t2 = time.time()
    df = data[cols_needed].copy()
    print(f"  [STEP 2] Done in {time.time()-t2:.3f}s")
    
    # Step 3a: Sort values
    print(f"  [STEP 3a] Sorting by prediction (this may take 1-2 min for 2.7M rows)...")
    t3a = time.time()
    df = df.sort_values('pred')
    print(f"  [STEP 3a] sort_values done in {time.time()-t3a:.1f}s")
    
    # Step 3b: Reset index
    print(f"  [STEP 3b] Resetting index...")
    t3b = time.time()
    df = df.reset_index(drop=True)
    print(f"  [STEP 3b] Done in {time.time()-t3b:.3f}s")
    
    # Step 4: Calculate cumulative weight
    print(f"  [STEP 4] Calculating cumulative weight...")
    t4 = time.time()
    w = df[weight_name]
    wsum = w.sum()
    cum_w = w.cumsum() / wsum
    # Floor-based decile assignment (no bias)
    df['decile'] = np.floor(np.round(cum_w, 2) * bins).astype(int)
    df['decile'] = np.where(df['decile'] + 1 > bins, bins, df['decile'] + 1)
    print(f"  [STEP 4] Done in {time.time()-t4:.3f}s")
    
    # Step 5: Aggregate by decile
    print(f"  [STEP 5] Aggregating by decile...")
    t5 = time.time()
    # Build aggregation dict
    agg_dict = {
        weight_name: ['sum', 'count'],
        'incurred_act': 'sum',
        'incurred_pred': 'sum',
        'denom': 'sum'
    }
    # Add GBM columns if present
    has_gbm = 'incurred_act_gbm' in df.columns
    if has_gbm:
        agg_dict['incurred_act_gbm'] = 'sum'
        agg_dict['incurred_pred_gbm'] = 'sum'
    
    x = df.groupby('decile').agg(agg_dict).reset_index()
    
    # Flatten column names (groupby with list creates MultiIndex columns)
    if has_gbm:
        x.columns = ['decile', weight_name, 'n_records', 'incurred_act', 'incurred_pred', 'denom', 'incurred_act_gbm', 'incurred_pred_gbm']
    else:
        x.columns = ['decile', weight_name, 'n_records', 'incurred_act', 'incurred_pred', 'denom']
    print(f"  [STEP 5] Done in {time.time()-t5:.3f}s")
    
    # Calculate act/pred values - both exposure-weighted and simple averages
    # Exposure-weighted average: incurred / exposure
    x['act_ee_weigh_avg'] = x['incurred_act'] / x[weight_name]
    x['pred_ee_weigh_avg'] = x['incurred_pred'] / x[weight_name]
    
    # Simple average: incurred / record count
    x['act_simple_avg'] = x['incurred_act'] / x['denom']
    x['pred_simple_avg'] = x['incurred_pred'] / x['denom']
    
    # Legacy columns for backward compatibility (use exposure-weighted)
    x['act'] = x['act_ee_weigh_avg']
    x['pred'] = x['pred_ee_weigh_avg']
    
    # Relativities (using exposure-weighted averages)
    overall_pred = df['incurred_pred'].sum() / df[weight_name].sum()
    x['act_rel'] = x['act_ee_weigh_avg'] / overall_pred
    x['pred_rel'] = x['pred_ee_weigh_avg'] / overall_pred
    
    # Calculate weight percentage per decile
    x['weight_pct'] = (x[weight_name] / x[weight_name].sum()) * 100
    
    # GBM-only columns (if present)
    if has_gbm:
        # Exposure-weighted average ratios
        x['act_gbm_ee_weigh_avg'] = x['incurred_act_gbm'] / x[weight_name]
        x['pred_gbm_ee_weigh_avg'] = x['incurred_pred_gbm'] / x[weight_name]
        # Simple average ratios
        x['act_gbm_simple_avg'] = x['incurred_act_gbm'] / x['denom']
        x['pred_gbm_simple_avg'] = x['incurred_pred_gbm'] / x['denom']
    
    # Plot
    print(f"  Creating plot...")
    t3 = time.time()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left Y-axis: Relativity
    ax1.plot(x['decile'], x['act_rel'], marker='o', label='Actual Relativity', linewidth=2, color='#1f77b4')
    ax1.plot(x['decile'], x['pred_rel'], marker='s', label='Predicted Relativity', linewidth=2, color='#ff7f0e')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Decile', fontsize=11)
    ax1.set_ylabel('Relativity', fontsize=11)
    ax1.set_ylim(0, y_max)
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right Y-axis: Weight percentage
    ax2 = ax1.twinx()
    ax2.bar(x['decile'], x['weight_pct'], alpha=0.25, color='#2196F3', width=0.6, label='Exposure %')
    ax2.set_ylabel('Weights (%)', fontsize=11, color='#2196F3')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='#2196F3')
    
    # Combined legend (exposure below relativity lines)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.tight_layout()
    print(f"  Plot created in {time.time()-t3:.1f}s")
    
    print(f"  TOTAL TIME: {time.time()-t0:.1f}s")
    
    return fig, x


def generate_scoring_lift_chart(
    predictions,
    target_values,
    exposure_values,
    exposure_col,
    output_dir,
    output_base,
    title_prefix="08 Scoring",
    score_name="holdout",
    suffix="",
    display_chart=True
):
    """
    Generate and save a lift chart for scoring results.
    
    Args:
        predictions: Series of model predictions (PP)
        target_values: Series of actual target values (PP) - capped or uncapped
        exposure_values: Series of exposure values
        exposure_col: Name of exposure column (for chart)
        output_dir: Directory to save chart PNG
        output_base: Base output path for CSV
        title_prefix: Chart title prefix
        score_name: Score name for title
        suffix: Optional suffix for filenames (e.g., "_uncapped")
        display_chart: Whether to display in notebook
        
    Returns:
        DataFrame with decile summary
    """
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    
    print(f'\n* Creating lift chart{suffix}...')
    
    # Prepare dataframe for lift chart
    lift_df = pd.DataFrame({
        'prediction': predictions,
        'pred': predictions,
        'incurred_act': target_values.values * exposure_values.values,
        'incurred_pred': predictions.values * exposure_values.values,
        'denom': 1,
        exposure_col: exposure_values.values
    })
    
    # Remove NaN rows
    lift_df = lift_df.dropna()
    
    print(f'  Creating chart for {len(lift_df):,} rows...')
    
    # Create lift chart
    fig, table = create_lift_chart(
        lift_df,
        exposure_col,
        bins=10,
        title=f'{title_prefix}: {score_name.title()} Lift Chart{suffix.replace("_", " ").title()}'
    )
    
    # Save chart
    chart_file = f'{output_dir}/lift_chart{suffix}.png'
    fig.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {chart_file}')
    
    # Display in notebook
    if display_chart:
        display(Image(chart_file))
        print(f'\nDecile summary:')
        cols_to_show = ['decile', 'n_records', exposure_col, 'incurred_act', 'incurred_pred', 
                        'act_ee_weigh_avg', 'act_simple_avg', 'pred_ee_weigh_avg', 'pred_simple_avg']
        print(table[cols_to_show].to_string(index=False))
    
    # Save table
    table.to_csv(f"{output_base}/results/08_lift{suffix}_table.csv", index=False)
    
    return table


def generate_training_lift_chart(
    data_df,
    predictions,
    target_values,
    exposure_values,
    exposure_col,
    output_base,
    stage="05c",
    dataset="train",
    suffix="",
    display_chart=True
):
    """
    Generate and save a lift chart for training/test data.
    
    Args:
        data_df: Original dataframe (train_orig or test_orig) with 'pred' column
        predictions: Series of model predictions (PP)
        target_values: Series of actual target values (PP) - capped or uncapped
        exposure_values: Series/array of exposure values
        exposure_col: Name of exposure column
        output_base: Base output path
        stage: Stage name (e.g., "05c")
        dataset: "train" or "test"
        suffix: Optional suffix for filenames (e.g., "_uncapped")
        display_chart: Whether to display in notebook
        
    Returns:
        DataFrame with decile summary
    """
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    
    print(f'\n* Creating {dataset} lift chart{suffix}...')
    
    # Prepare dataframe for lift chart
    lift_df = pd.DataFrame({
        'pred': predictions,
        'incurred_act': target_values * exposure_values,
        'incurred_pred': predictions * exposure_values,
        'denom': 1,
        exposure_col: exposure_values
    })
    
    # Remove NaN rows
    lift_df = lift_df.dropna()
    
    print(f'  Creating chart for {len(lift_df):,} rows...')
    
    # Create lift chart
    fig, table = create_lift_chart(
        lift_df,
        exposure_col,
        bins=10,
        title=f'{stage} Production {dataset.title()} Lift Chart{suffix.replace("_", " ").title()}'
    )
    
    # Save chart
    chart_file = f'{output_base}/results/{stage}_lift_{dataset}{suffix}.png'
    fig.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {chart_file}')
    
    # Display in notebook
    if display_chart:
        display(Image(chart_file))
        print(f'\n{dataset.title()} decile summary:')
        cols_to_show = ['decile', 'n_records', exposure_col, 'incurred_act', 'incurred_pred',
                        'act_ee_weigh_avg', 'act_simple_avg', 'pred_ee_weigh_avg', 'pred_simple_avg']
        print(table[cols_to_show].to_string(index=False))
    
    # Save table
    table.to_csv(f"{output_base}/results/{stage}_lift_{dataset}{suffix}_table.csv", index=False)
    
    return table
