#!/usr/bin/env python3
"""
Update Stage 07 template to loop through features and create plots for both train and test
"""

import json
from pathlib import Path

template_path = Path("templates/07_shap_analysis.ipynb")

# Read existing notebook
with open(template_path, 'r') as f:
    notebook = json.load(f)

# Create new cells
new_cells = [
    # Cell 1: Title
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Template 07: SHAP Analysis & Visualization\\n",
            "\\n",
            "**Purpose:** Generate residual plots and SHAP range plots for top N features\\n",
            "\\n",
            "**Inputs:**\\n",
            "- results/06_shap_train.parquet\\n",
            "- results/06_shap_test.parquet\\n",
            "- results/05_predictions.parquet\\n",
            "- data/04_train.parquet\\n",
            "- data/04_test.parquet\\n",
            "\\n",
            "**Outputs:**\\n",
            "- results/07_shap_aggregate.csv\\n",
            "- results/07_plots/{dataset}_{feature}_residual.png (2 plots per feature)\\n",
            "- results/07_plots/{dataset}_{feature}_shap_range.png (2 plots per feature)\\n",
            "\\n",
            "**Configuration:** Controlled by `shap_analysis` section in config.yaml"
        ]
    },
    # Cell 2: Parameters
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["parameters"]},
        "outputs": [],
        "source": [
            "config_path = \\\"config/car_coll/v1\\\""
        ]
    },
    # Cell 3: Imports and Setup
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib\\n",
            "matplotlib.use('Agg')  # Non-interactive backend for papermill\\n",
            "\\n",
            "import pandas as pd\\n",
            "import numpy as np\\n",
            "import yaml\\n",
            "import os\\n",
            "import sys\\n",
            "from pathlib import Path\\n",
            "import matplotlib.pyplot as plt\\n",
            "from datetime import datetime\\n",
            "\\n",
            "sys.path.insert(0, str(Path.cwd() / 'lib'))\\n",
            "from utils import setup_notebook_environment\\n",
            "from shap_utils import compute_shap_aggregate, create_residual_plot, create_shap_range_plot, create_feature_importance_plot\\n",
            "\\n",
            "print(\\\"########################################\\\")\\n",
            "print(\\\"# STAGE 07: SHAP ANALYSIS & VISUALIZATION\\\")\\n",
            "print(\\\"########################################\\\")\\n",
            "\\n",
            "project_root = setup_notebook_environment()"
        ]
    },
    # Cell 4: Load Config
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "config_file = f\\\"{config_path}/config.yaml\\\"\\n",
            "with open(config_file, 'r') as f:\\n",
            "    cfg = yaml.safe_load(f)\\n",
            "\\n",
            "output_base = cfg['paths']['output_base']\\n",
            "target = cfg['experiment']['target']\\n",
            "exposure = cfg['experiment'].get('exposure', None)\\n",
            "\\n",
            "# Load SHAP analysis config\\n",
            "shap_cfg = cfg.get('shap_analysis', {})\\n",
            "top_n = shap_cfg.get('top_n_features', 40)\\n",
            "datasets = shap_cfg.get('datasets', ['train', 'test'])\\n",
            "resid_cfg = shap_cfg.get('residual_plot', {})\\n",
            "shap_range_cfg = shap_cfg.get('shap_range_plot', {})\\n",
            "\\n",
            "print(f\\\"\\\\n* Configuration:\\\")\\n",
            "print(f\\\"  Top N features: {top_n}\\\")\\n",
            "print(f\\\"  Datasets: {datasets}\\\")\\n",
            "print(f\\\"  Target: {target}\\\")\\n",
            "\\n",
            "# Create output directory\\n",
            "os.makedirs(f\\\"{output_base}/results/07_plots\\\", exist_ok=True)"
        ]
    }
]

# Replace first few cells
notebook['cells'] = new_cells

# Save
with open(template_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Updated {template_path}")
print(f"   Created {len(new_cells)} cells so far")
