"""
Fast lift chart implementation - optimized for large datasets
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def create_lift_chart(data, weight_name, bins=10, title="Lift Chart"):
    """Create lift chart (optimized for large data). Usage: fig, decile_df = create_lift_chart(data, 'weight', bins=10)"""
    t0 = time.time()
    
    # Step 1: Create column list
    print(f"  [STEP 1] Creating column list...")
    t1 = time.time()
    cols_needed = ['pred', weight_name, 'incurred_act', 'incurred_pred', 'denom']
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
    df['decile'] = np.ceil(cum_w * bins).astype(int).clip(1, bins)
    print(f"  [STEP 4] Done in {time.time()-t4:.3f}s")
    
    # Step 5: Aggregate by decile
    print(f"  [STEP 5] Aggregating by decile...")
    t5 = time.time()
    x = df.groupby('decile').agg({
        weight_name: 'sum',
        'incurred_act': 'sum',
        'incurred_pred': 'sum',
        'denom': 'sum'
    }).reset_index()
    print(f"  [STEP 5] Done in {time.time()-t5:.3f}s")
    
    # Calculate act/pred values
    x['act'] = x['incurred_act'] / x['denom']
    x['pred'] = x['incurred_pred'] / x['denom']
    
    # Relativities
    overall_pred = df['incurred_pred'].sum() / df['denom'].sum()
    x['act_rel'] = x['act'] / overall_pred
    x['pred_rel'] = x['pred'] / overall_pred
    
    # Plot
    print(f"  Creating plot...")
    t3 = time.time()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x['decile'], x['act_rel'], marker='o', label='Actual Relativity', linewidth=2)
    ax.plot(x['decile'], x['pred_rel'], marker='s', label='Predicted Relativity', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Decile')
    ax.set_ylabel('Relativity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    print(f"  Plot created in {time.time()-t3:.1f}s")
    
    print(f"  TOTAL TIME: {time.time()-t0:.1f}s")
    
    return fig, x
