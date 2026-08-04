# GBMFirst Templates

This folder contains template notebooks for different modeling approaches.

## Available Templates

### standard_gbm.ipynb
**Default template** - Complete XGBoost pipeline with SHAP analysis

**Features:**
- Load data with machine-specific paths
- Feature selection and clipping
- Monotonicity constraints
- XGBoost training with Tweedie loss
- SHAP analysis and visualizations
- Lift charts and model evaluation
- Self-contained output generation

**Use for:** Standard GBM modeling workflow

## How to Use Templates

Templates are selected via the `template` field in `config.yaml`:

```yaml
experiment:
  name: "CAR COLL v1"
  template: "standard_gbm"  # ← Selects templates/standard_gbm.ipynb
  veh_type: "CAR"
  coverage: "coll"
```

## Creating New Templates

1. Copy an existing template:
   ```bash
   cp templates/standard_gbm.ipynb templates/my_new_template.ipynb
   ```

2. Modify the template for your needs

3. Reference it in config.yaml:
   ```yaml
   experiment:
     template: "my_new_template"
   ```

4. Run via runners/run_one.ipynb

## Template Requirements

All templates must:
- Accept `config_path` parameter (set by papermill)
- Load config using `../lib/utils.py`
- Load functions from `../lib/gbm_functions.ipynb`
- Support SELF_CONTAINED mode for output folders
- Clean up memory at the end

## Path Structure

When running from templates folder:
- Config files: `../config/{model}/{version}/config.yaml`
- Shared libraries: `../lib/`
- Project root: `../`
- Machine ID file: `../.PC`

Created: 2026-07-31
