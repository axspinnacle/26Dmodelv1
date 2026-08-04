# GBMFirst - New Dataset Setup Guide

## ✅ Files Copied Successfully

This project has been initialized with the essential GBMFirst template files from `GBMFirstTest_26_v2`.

### Core Files:
- `template.ipynb` - Main ML pipeline (loads data, trains XGBoost, generates SHAP plots)
- `utils.py` - Helper functions (config loader, machine detection)
- `edited_20240314_symbols_gbm_functions.ipynb` - GBM utility functions
- `generate_features.ipynb` - Feature engineering notebook
- `requirements.txt` - Python dependencies
- `.clinerules` - AI assistant project rules
- `.PC` - Machine identifier (currently set to: `3` for Mac)

### Configuration:
- `config/car_coll/v1/` - Example configuration for CAR COLL model
  - `config.yaml` - Experiment settings and paths
  - `feature_selection.csv` - Which features to include/exclude
  - `feature_clipping.csv` - Feature value bounds

### Runners:
- `runners/run_one.ipynb` - Execute a single experiment
- `runners/run_all.ipynb` - Execute multiple experiments in batch

### Output:
- `output/` - Results will be generated here when you run experiments

---

## 🔧 Required Changes for New Dataset

### 1. Update `config/car_coll/v1/config.yaml`

**⚠️ CRITICAL: Update these paths for your new dataset:**

```yaml
machines:
  PC3:  # Mac machine
    conda_env: "py39_26v1"
    paths:
      # ⬇️ CHANGE THIS to your new dataset location
      path_prefix: "/Users/Mach/dev/aps/data/2024_CX_Cmodel/v2/"
      
      # Update if your folder structure is different
      input_data_dir: ""
      ep_data_dir: "ep_data/"
      support_dir: "Other Support/"

files:
  # ⬇️ CHANGE THIS if your input file naming pattern is different
  input_file_pattern: "selectedcolumns_{veh_type}_trainvaliddata_v2.csv"
  
  # ⬇️ CHANGE THIS to match your EP data filename
  ep_file: "data_ep.csv"
  
  feature_selection: "feature_selection.csv"
  feature_clipping: "feature_clipping.csv"
```

**Example for new dataset:**
```yaml
machines:
  PC3:
    conda_env: "py39_26v1"
    paths:
      path_prefix: "/Users/Mach/dev/aps/data/2026_D_model/v1/"  # ← NEW
      input_data_dir: "input_data/"                              # ← NEW
      ep_data_dir: "ep_files/"                                   # ← NEW
      support_dir: "support_files/"                              # ← NEW

files:
  input_file_pattern: "data_{veh_type}_2026.csv"  # ← NEW
  ep_file: "earned_premium_2026.csv"              # ← NEW
```

### 2. Verify `.PC` File

Check that `.PC` contains the correct machine number:
- `2` = Linux server (PC2)
- `3` = Mac local (PC3)

Current value: **3** (Mac)

### 3. Review Feature Selection

Review and update if needed:
- `config/car_coll/v1/feature_selection.csv` - Features to use
- `config/car_coll/v1/feature_clipping.csv` - Value bounds

---

## 🚀 How to Run

### Option 1: Run via Jupyter Notebook (Recommended)

1. **Activate environment:**
   ```bash
   conda activate py39_26v1
   ```

2. **Navigate to runners:**
   ```bash
   cd /Users/Mach/dev/aps/code/26Dmodelv1/runners
   jupyter notebook
   ```

3. **Open and run:**
   - `run_one.ipynb` - Run the CAR COLL v1 experiment
   - Or edit the `config_dir` variable to run a different experiment

### Option 2: Run via Papermill (Command Line)

```bash
conda activate py39_26v1
cd /Users/Mach/dev/aps/code/26Dmodelv1

python -m papermill template.ipynb \
  output/car_coll/v1/car_coll.ipynb \
  -p config_path "config/car_coll/v1/config.yaml" \
  -k py39_26v1
```

---

## 📂 Project Structure

```
26Dmodelv1/
├── template.ipynb                    # Main pipeline
├── utils.py                          # Config utilities
├── edited_20240314_symbols_gbm_functions.ipynb
├── generate_features.ipynb
├── requirements.txt
├── .clinerules
├── .PC                               # Machine ID (2 or 3)
│
├── config/
│   └── car_coll/v1/                  # Example experiment
│       ├── config.yaml               # ⚠️ UPDATE THIS
│       ├── feature_selection.csv
│       └── feature_clipping.csv
│
├── runners/
│   ├── run_one.ipynb                 # Run single experiment
│   └── run_all.ipynb                 # Run multiple experiments
│
└── output/                           # Results go here
    └── car_coll/v1/                  # Self-contained outputs
        ├── car_coll.ipynb            # Executed notebook
        ├── CAR_coll_model.json       # Trained model
        └── ... (supporting files)
```

---

## 🎯 Next Steps

### Immediate Actions:
1. ✅ **Update `config/car_coll/v1/config.yaml`** with new dataset paths
2. ✅ **Verify `.PC` file** contains `3` (or `2` if on Linux)
3. ✅ **Check dataset files exist** at the paths specified in config
4. ✅ **Run a test** with debug mode enabled (see below)

### Test Run (Debug Mode):

Edit `config/car_coll/v1/config.yaml`:
```yaml
model:
  debug: true           # ← Set to true
  debug_rows: 100000    # Small subset for testing
```

Then run `runners/run_one.ipynb` to verify everything works.

### Create Additional Experiments:

To create new experiments (e.g., SUV COMP, reduced features):

1. Copy the config folder:
   ```bash
   cp -r config/car_coll/v1 config/suv_comp/v1
   ```

2. Edit the new `config.yaml`:
   ```yaml
   experiment:
     name: "SUV COMP v1"
     veh_type: "SUV"      # ← Change
     coverage: "comp"     # ← Change
   ```

3. Update feature selection/clipping as needed

4. Run via `runners/run_one.ipynb` (update `config_dir`)

---

## 📝 Important Notes

### What's NOT Included (Reference Only):
- `oldcode/` - Original non-templated notebooks (kept in source project)
- `oldcode_usedforfirsttest/` - Historical code (kept in source project)
- `output/` from old project - Generated results (will be recreated)

### File Sizes:
- `edited_20240314_symbols_gbm_functions.ipynb` (66KB)
- `template.ipynb` (17KB)
- Total project: ~90KB (excluding data/outputs)

### Machine Configuration:
- PC1: Not configured
- PC2: Linux server - `/home/carfax_shared/...`
- PC3: Mac local - `/Users/Mach/dev/aps/data/...`
- PC4: Not configured

---

## ❓ Troubleshooting

### Error: "Config path not found"
- Check that `config_dir` in runner matches actual folder
- Example: `config_dir = "config/car_coll/v1"`

### Error: "Data file not found"
- Verify paths in `config.yaml` → `machines → PC3 → paths`
- Check that `input_file_pattern` matches actual filenames
- Ensure `.PC` file has correct machine number

### Error: "Module not found"
- Activate conda environment: `conda activate py39_26v1`
- Install dependencies: `pip install -r requirements.txt`

### Memory Issues:
- Enable debug mode to test with smaller dataset first
- The template includes memory cleanup at the end

---

## 🔄 Workflow Summary

1. **Setup** (one time):
   - Update config.yaml with new dataset paths
   - Verify .PC file
   - Test with debug mode

2. **Create Experiments**:
   - Copy config folder
   - Edit experiment settings
   - Update feature selection if needed

3. **Run Experiments**:
   - Use `runners/run_one.ipynb` for single runs
   - Use `runners/run_all.ipynb` for batch processing

4. **Review Results**:
   - Check `output/{model}/{version}/` for results
   - Executed notebook includes all plots and metrics
   - Model saved as `.json` file

---

Created: 2026-07-31
Source: GBMFirstTest_26_v2/GBMFirst
