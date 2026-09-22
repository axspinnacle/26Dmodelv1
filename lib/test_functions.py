def lift_chart_2025update(test_data, weight_name, bins, print_table=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Calculate decile groups
    test_data['decile'] = (round(test_data.sort_values(by='pred')[weight_name].cumsum() / test_data[weight_name].sum(), 2) * bins).apply(np.floor)
    test_data['decile'] = np.where(test_data['decile'] + 1 > bins, bins, test_data['decile'] + 1)
    
    # Group data by deciles
    x = test_data.groupby(['decile'], dropna=False).agg({weight_name: 'sum', 'incurred_act': 'sum', 'incurred_pred': 'sum', 'denom': 'sum'}).reset_index()
    x['act'] = x['incurred_act'] / x['denom']
    x['pred'] = x['incurred_pred'] / x['denom']
    x.drop(columns=['incurred_act', 'incurred_pred', 'denom'], inplace=True)

    # Prepare data for plotting
    dfg = x
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    # Set limits for axes
    y_max = max(dfg['act'].max(), dfg['pred'].max()) * 1.20
    ax.set_ylim(0, y_max)  # Left y-axis (act and pred)
    ax2.set_ylim(0, 100)   # Right y-axis (weights as percentage)

    # Plot bar for weights
    (dfg[weight_name] / dfg[weight_name].sum() * 100).plot.bar(stacked=False, ax=ax2, alpha=0.6)

    # Plot lines for act and pred
    dfg['act'].plot(kind='line', ax=ax, marker='o', linewidth=1, color='blue', label='act')
    dfg['pred'].plot(kind='line', ax=ax, marker='o', linewidth=1, color='orange', label='pred')

    # Add legends and labels
    ax.set_ylabel("Values (act and pred)")
    ax2.set_ylabel("Weights (%)")
    ax.set_xlabel("Decile")
    ax.legend(loc='upper left')
    ax2.legend(["Weights"], loc='upper right')

    plt.show()

    if print_table:
        print(x)