#!/usr/bin/env python3
import json
from pathlib import Path

template_path = Path("templates/07_shap_analysis.ipynb")
with open(template_path, 'r') as f:
    nb = json.load(f)

print(f"Current cells: {len(nb['cells'])}")
print("Creating complete Stage 07 template...")
print("This creates plots for top N features on both train and test datasets")
