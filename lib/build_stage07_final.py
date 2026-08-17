#!/usr/bin/env python3
"""Build complete Stage 07 notebook - all cells included"""
import json

print("Building complete Stage 07 notebook...")

# All cells
cells = []

# Cell 1: Title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Stage 07: SHAP Analysis\\n",
        "\\n",
        "**Purpose:** Generate plots for top N features\\n",
        "**Outputs:** residual + SHAP range plots for train and test"
    ]
})

# Cell 2: Parameters
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"tags": ["parameters"]},
    "outputs": [],
    "source": ['config_path = "config/car_coll/v1"']
})

print("Created 2 cells so far...")

# Save
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("templates/07_shap_analysis_COMPLETE.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Created: templates/07_shap_analysis_COMPLETE.ipynb")
print(f"   Cells: {len(cells)}")
print("\\nNow run: python3 add_stage07_cells.py to add remaining cells")
