# GBMFirst Project - 2026 D-Model Version 1

## Overview

Automated machine learning pipeline for insurance pricing models using XGBoost and SHAP analysis.

**Current Version:** Python 3.11+ | XGBoost 3.1.1 | SHAP 0.50.0+

---

## Quick Start

```bash
# 1. Activate environment
conda activate py311_26v1

# 2. Run single model
cd runners
jupyter notebook run_one.ipynb

# 3. Or run all models
jupyter notebook run_all.ipynb
```

See `documentation/QUICKSTART_PY311.md` for detailed setup instructions.

---

## Project Structure

```
26Dmodelv1/
├── config/               # Model configurations (YAML)
│   ├── car_coll/v1/
│   ├── car_comp/v1/
│   └── ...
├── templates/            # Notebook templates (stages 00-07)
├── runners/              # Execution notebooks
├── lib/                  # Python utilities & helper scripts
├── output/               # Model outputs (data, models, results, plots)
├── scripts/              # Shell scripts (setup, verification)
└── documentation/        # All documentation files
```

---

## Pipeline Stages

| Stage | Template | Purpose |
|-------|----------|---------|
| 00 | `00_setup.ipynb` | Environment verification |
| 01 | `01_data_assembly.ipynb` | Load & merge datasets |
| 02 | `02_data_conditioning.ipynb` | Feature engineering |
| 03 | `03_verification.ipynb` | Data quality checks |
| 04 | `04_model_prep.ipynb` | Train/test split |
| 05 | `05_model_training.ipynb` | XGBoost training & metrics |
| 06 | `06_shap_dataframe.ipynb` | SHAP value computation |
| 07 | `07_shap_analysis.ipynb` | Diagnostic plots generation |

---

## Key Features

- ✅ **Automated Pipeline** - Papermill-driven execution
- ✅ **SHAP Explainability** - Full model interpretability
- ✅ **Diagnostic Plots** - Residual & SHAP range plots for top features
- ✅ **Multi-Model Support** - CAR/SUV/TRUCK/VAN × COLL/COMP/LIAB
- ✅ **Reproducible** - Version-controlled configs & code
- ✅ **Debug Modes** - Fast iteration with subset folds

---

## Documentation

| Document | Description |
|----------|-------------|
| `documentation/QUICKSTART_PY311.md` | Setup & first run guide |
| `documentation/PIPELINE_COMPLETE.md` | Full pipeline documentation |
| `documentation/PYTHON_UPGRADE_GUIDE.md` | Python 3.11 upgrade notes |
| `documentation/TemplateCodeExplanation.md` | Code architecture |

---

## Machine Setup

**First-time setup:** Create a `current.pc` file in the project root:
```bash
echo "3" > current.pc  # Use 2 for Linux server, 3 for Mac
```

This file determines which data paths are used (configured in `config.yaml`).

## Configuration

Models are configured via YAML files in `config/{model}/{version}/`:

```yaml
experiment:
  name: "car_bi_v1"
  target: "pp_bi"
  coverage: "coll"

execution:
  stages:
    stage_05: true  # Enable/disable stages
    stage_07: true

shap_analysis:
  top_n_features: 40
  datasets: ["train", "test"]
```

---

## Output Structure

```
output/car_coll/v1/
├── data/
│   ├── 01_assembled.parquet
│   ├── 04_train.parquet
│   └── 04_test.parquet
├── models/
│   └── xgb_model.json
├── results/
│   ├── 05_metrics.yaml
│   ├── 05_predictions.parquet
│   ├── 06_shap_train.parquet
│   └── 07_analysis_summary.yaml
└── plots/
    ├── train_{feature}_residual.png
    └── train_{feature}_shap_range.png
```

---

## Development

### Adding a New Model

1. Create config: `config/{model}/{version}/config.yaml`
2. Add feature list: `config/{model}/{version}/columns_inclusion.csv`
3. Update `runners/run_all.ipynb` to include new model
4. Run pipeline

### Debugging

Set `debug.level` in config:
- `1` = 1 fold (fastest)
- `2` = 2 folds
- `3` = 5 folds
- `0` = production (all folds)

---

## Requirements

- Python 3.11+
- XGBoost 3.1.1
- SHAP 0.50.0+
- See `requirements.txt` for full list

---

## Support

For questions or issues:
1. Check `documentation/` folder
2. Review `.clinerules` for project conventions
3. Check git history for recent changes

---

**Last Updated:** August 2026
