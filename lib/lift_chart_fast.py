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


def _to_array(v):
    """Convert Series/array/list to float numpy array."""
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return np.asarray(v.values, dtype=float).ravel()
    return np.asarray(v, dtype=float).ravel()


def generate_ratio_lift_chart(
    gbm_pred,
    ratio_actual,
    exposure_values,
    exposure_col,
    output_base,
    stage="05c",
    dataset="train",
    display_chart=True,
    chart_file=None,
    table_file=None
):
    """GBM ratio lift chart (pp/glm), sorted by GBM ratio, exposure weighted."""
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    print(f'\n* Creating {dataset} GBM ratio lift chart...')

    pred = _to_array(gbm_pred)
    act = _to_array(ratio_actual)
    w = _to_array(exposure_values)

    lift_df = pd.DataFrame({
        'pred': pred,
        'incurred_act': act * w,
        'incurred_pred': pred * w,
        'denom': 1,
        exposure_col: w
    })
    n_before = len(lift_df)
    lift_df = lift_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(lift_df) < n_before:
        print(f'  Dropped {n_before - len(lift_df):,} rows with NaN/inf')

    fig, table = create_lift_chart(
        lift_df,
        exposure_col,
        bins=10,
        title=f'{stage} {dataset.title()} GBM Ratio Lift Chart (PP/GLM, sorted by GBM)'
    )

    chart_file = chart_file or f'{output_base}/results/{stage}_lift_{dataset}_ratio.png'
    table_file = table_file or f"{output_base}/results/{stage}_lift_{dataset}_ratio_table.csv"
    fig.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {chart_file}')

    if display_chart:
        display(Image(chart_file))
        print(f'\n{dataset.title()} ratio summary:')
        cols_to_show = ['decile', 'n_records', exposure_col, 'incurred_act', 'incurred_pred',
                        'act_ee_weigh_avg', 'act_simple_avg', 'pred_ee_weigh_avg', 'pred_simple_avg']
        print(table[cols_to_show].to_string(index=False))

    table.to_csv(table_file, index=False)

    return table


def generate_scoring_ratio_lift_chart(prediction, target_values, glm_pred, exposure_values,
                                      exposure_col, output_dir, output_base, score_name="holdout"):
    """Scoring version of the ratio lift chart: actual = pp/glm, pred = prediction/glm."""
    glm = _to_array(glm_pred)
    return generate_ratio_lift_chart(
        _to_array(prediction) / glm, _to_array(target_values) / glm, exposure_values, exposure_col,
        output_base, stage="08", dataset=score_name,
        chart_file=f"{output_dir}/lift_chart_ratio.png",
        table_file=f"{output_base}/results/08_lift_ratio_table.csv")


def debug_gbm_ratio(
    gbm_pred,
    ratio_actual,
    pp_actual,
    glm_pred,
    exposure_values,
    output_base,
    stage="05c",
    dataset="train",
    exposure_floor=0.0833
):
    """
    Debug GBM average ratios: calibration, GLM scale, distributions, outliers.

    Prints compact report and saves {stage}_gbm_ratio_debug_{dataset}.csv.

    Returns:
        DataFrame with debug metrics (metric, value)
    """
    pred = _to_array(gbm_pred)
    act = _to_array(ratio_actual)
    pp = _to_array(pp_actual)
    glm = _to_array(glm_pred)
    w = _to_array(exposure_values)

    rows = []

    def add(name, val):
        rows.append({'metric': name, 'value': float(val) if val is not None else np.nan})

    # Problem records
    n = len(pred)
    add('n_records', n)
    add('n_glm_le0_or_nan', np.sum(~(glm > 0)))
    add('n_actual_ratio_inf_or_nan', np.sum(~np.isfinite(act)))
    add('n_gbm_pred_inf_or_nan', np.sum(~np.isfinite(pred)))
    add('n_exposure_below_floor', np.sum(w < exposure_floor))

    ok = np.isfinite(act) & np.isfinite(pred) & np.isfinite(pp) & (glm > 0) & np.isfinite(w)
    p, a, y, g, e = pred[ok], act[ok], pp[ok], glm[ok], w[ok]
    wsum = e.sum()

    # Calibration (ratio space)
    add('act_ratio_ee_weigh_avg', (a * e).sum() / wsum)
    add('pred_ratio_ee_weigh_avg', (p * e).sum() / wsum)
    add('act_ratio_simple_avg', a.mean())
    add('pred_ratio_simple_avg', p.mean())
    add('act_over_pred_ratio_ee', (a * e).sum() / (p * e).sum())

    # GLM scale (PP space)
    pp_avg = (y * e).sum() / wsum
    glm_avg = (g * e).sum() / wsum
    add('pp_actual_ee_weigh_avg', pp_avg)
    add('glm_ee_weigh_avg', glm_avg)
    add('pp_over_glm_ee', pp_avg / glm_avg)
    add('final_pred_pp_ee_weigh_avg', (p * g * e).sum() / wsum)

    # Distributions
    for nm, arr in [('gbm_pred', p), ('act_ratio', a), ('glm', g), ('exposure', e)]:
        q = np.percentile(arr, [0, 1, 50, 99, 100])
        for lbl, v in zip(['min', 'p1', 'p50', 'p99', 'max'], q):
            add(f'{nm}_{lbl}', v)

    # Concentration of weighted actual ratio in top 1% records
    contrib = a * e
    tot = contrib.sum()
    k = max(1, int(np.ceil(0.01 * len(contrib))))
    add('top1pct_share_act_ratio_weighted', np.sort(contrib)[::-1][:k].sum() / tot if tot else np.nan)
    add('pct_records_zero_loss', np.mean(a == 0) * 100)

    df = pd.DataFrame(rows)

    print(f'\n* GBM ratio debug ({stage} {dataset}):')
    print(df.to_string(index=False, float_format=lambda x: f'{x:,.4f}'))

    out_file = f'{output_base}/results/{stage}_gbm_ratio_debug_{dataset}.csv'
    df.to_csv(out_file, index=False)
    print(f'  Saved: {out_file}')

    return df

