#!/usr/bin/env python3
"""Add remaining cells to EDA notebook"""
import json

nb_path = '/Users/Mach/dev/aps/code/26Dmodelv1/templates/00_eda_for_allfeatures.ipynb'

# Read existing
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Additional cells
new_cells = [
    # Step 1
    {"cell_type": "markdown", "metadata": {}, "source": ["## Step 1: Load Column Metadata"]},
    {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "# Load all_columns_master.csv\n",
            "master_cols_file = f\"{config_path}/all_columns_master.csv\"\n",
            "all_cols_df = pd.read_csv(master_cols_file)\n\n",
            "print(f\"\\n✓ Loaded {len(all_cols_df)} columns from all_columns_master.csv\")\n",
            "print(f\"\\nColumn types:\")\n",
            "print(all_cols_df['dtype'].value_counts())\n\n",
            "all_cols_df.head()"
        ]
    },
    # Step 2
    {"cell_type": "markdown", "metadata": {}, "source": ["## Step 2: Load Sample Data"]},
    {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "# Load sample data\n",
            "data_root = cfg['paths']['master_data_path']\n",
            "master_file = f\"{data_root}/{cfg['data']['master_file']}\"\n\n",
            "print(f\"\\n📂 Loading sample data from: {master_file}\")\n",
            "print(f\"   Sample size: {sample_size:,} rows\")\n\n",
            "# Read parquet with row limit\n",
            "df_sample = pd.read_parquet(master_file).head(sample_size)\n\n",
            "print(f\"\\n✓ Loaded sample: {df_sample.shape}\")\n",
            "print(f\"  Memory: {df_sample.memory_usage(deep=True).sum() / 1e6:.1f} MB\")"
        ]
    },
]

nb['cells'].extend(new_cells)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"Added {len(new_cells)} cells (total: {len(nb['cells'])})")
