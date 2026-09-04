#!/usr/bin/env python3
"""
Build the EDA notebook in chunks to avoid size limits
"""
import json

# Base notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.11 (26v1)",
            "language": "python",
            "name": "py311_26v1"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add cells
cells = [
    # Title
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Template 00: EDA for All Features\n\n"
            "**Purpose:** Analyze all columns and generate encoding recommendations\n\n"
            "**Outputs:**\n"
            "- `config_generated/auto_feature_encoding.csv` - Auto-detected encoding recommendations\n"
            "- `config_generated/auto_dtype_fixes.csv` - Auto-detected dtype conversions\n"
            "- `config_generated/master_feature_encoding.csv` - Merged manual + auto (used by pipeline)\n"
            "- `config_generated/master_dtype_fixes.csv` - Merged manual + auto (used by pipeline)\n"
            "- Visualizations (histograms and bar charts)\n\n"
            "**Note:** Run this manually (not part of automated pipeline)"
        ]
    },
    # Parameters
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["parameters"]},
        "outputs": [],
        "source": [
            "# Parameters (can be overridden)\n",
            "config_path = \"config/car_coll/v1\"\n",
            "sample_size = 30000  # Number of rows to sample for analysis"
        ]
    },
    # Setup
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import yaml\n",
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n\n",
            "# Setup\n",
            "sys.path.insert(0, str(Path.cwd() / 'lib'))\n",
            "from utils import setup_notebook_environment, load_config\n\n",
            "print(\"=\"*70)\n",
            "print(\"EDA FOR ALL FEATURES\")\n",
            "print(\"=\"*70)\n\n",
            "project_root = setup_notebook_environment()\n",
            "print(f\"\\nProject root: {project_root}\")\n",
            "print(f\"Config path: {config_path}\")\n",
            "print(f\"Sample size: {sample_size:,} rows\")"
        ]
    },
    # Load config
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load config\n",
            "config_file = f\"{config_path}/config.yaml\"\n",
            "cfg = load_config(config_file)\n\n",
            "# Setup paths\n",
            "config_gen_dir = f\"{config_path}/config_generated\"\n",
            "os.makedirs(config_gen_dir, exist_ok=True)\n\n",
            "print(f\"\\n✓ Config loaded: {cfg['experiment']['name']}\")\n",
            "print(f\"✓ Generated config dir: {config_gen_dir}\")"
        ]
    },
]

notebook['cells'] = cells

# Save
output_path = '/Users/Mach/dev/aps/code/26Dmodelv1/templates/00_eda_for_allfeatures.ipynb'
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Created: {output_path}")
print(f"Cells: {len(cells)}")
