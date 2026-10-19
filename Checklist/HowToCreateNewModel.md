# How To Create New Model

checklist for setting up a new vehicle type + coverage combination.

---

## Prerequisites

- Master data file exists: `master_dataset_{vehicle}.parquet`
- Aux data file exists: `{vehicle}_with_dep_factor_fold.parquet`
- Know the target and exposure column names for your coverage
- Control model output file 

---

## Coverage-Specific Column Names

| Coverage | Target Column | Exposure Column |
|----------|---------------|-----------------|
| **coll** | `pp_coll` | `ee_coll_imps` |
| **comp** | `pp_comp` | `ee_comp_imps` |
| **liab** | `pp_bi` | `ee_bi_imps` |

---

## Step-by-Step Checklist

### 1. Create Config Directory Structure

```bash
# Example: Creating suv_comp model
mkdir -p config/suv_comp/v1/config_generated
```

### 2. Copy Template Configuration Files

Copy from an existing model (e.g., `car_coll/v1/`):

```bash
# From project root
SOURCE="config/car_coll/v1"
TARGET="config/suv_comp/v1"

# Copy all base files
cp $SOURCE/config.yaml $TARGET/
cp $SOURCE/columns_to_load_during_dataassembly.csv $TARGET/
cp $SOURCE/columns_inclusion.csv $TARGET/
cp $SOURCE/exclusion.csv $TARGET/
cp $SOURCE/monotonicity.csv $TARGET/
cp $SOURCE/pca_features.csv $TARGET/
cp $SOURCE/manual_feature_encoding.csv $TARGET/
cp $SOURCE/type_conversions.csv $TARGET/
cp $SOURCE/all_columns_master.csv $TARGET/
cp $SOURCE/all_columns_aux.csv $TARGET/
```

### 3. Update `config.yaml`

**Required changes:**

```yaml
experiment:
  name: "suv_comp_v1"                    # ← Change model name
  description: "SUV comprehensive model" # ← Change description
  target: "pp_comp"                      # ← Change to coverage target
  exposure: "ee_comp_imps"              # ← Change to coverage exposure
  coverage: "comp"                       # ← Change coverage type
  vehicle_type: "SUV"                    # ← Change vehicle type

data:
  master_file: "master_dataset_suv.parquet"          # ← Change vehicle
  aux_file: "suv_with_dep_factor_fold.parquet"       # ← Change vehicle

machines:
  PC3:
    paths:
      output_path: "output/suv_comp/v1"  # ← Change output path
  # Update PC2 paths if needed for server
```

**Check these settings:**
- `run_mode`: Usually `debug2` for initial testing
- `hyperparameter_optimization.param_grid`: Adjust if needed
- Keep other settings unless you have specific requirements

### 4. Update `columns_to_load_during_dataassembly.csv`

**Critical changes (lines 96-98):**

```csv
# OLD (from car_coll):
pp_coll,target,N/A,Pure premium collision (target variable)
ee_coll_imps,exposure,N/A,Earned exposure for collision (weight column)

# NEW (for suv_comp):
pp_comp,target,N/A,Pure premium comprehensive (target variable)
ee_comp_imps,exposure,N/A,Earned exposure for comprehensive (weight column)
```

**⚠️ CRITICAL:** This file must match your coverage type, or Stage 01 will fail with "Missing critical columns" error.

### 5. Update Coverage-Specific Files

#### `exclusion.csv`
- Copy from `config/templates/{coverage}/exclusion.csv`
- Example: For comp model, use `config/templates/comp/exclusion.csv`

```bash
cp config/templates/comp/exclusion.csv config/suv_comp/v1/
```

#### `monotonicity.csv`
- Copy from `config/templates/{coverage}/monotonicity.csv`
- Example: For comp model, use `config/templates/comp/monotonicity.csv`

```bash
cp config/templates/comp/monotonicity.csv config/suv_comp/v1/
```

### 6. Verify Vehicle-Specific Columns

Check that your master data has vehicle-specific columns:
master_dataset_suv.parquet
Should see your target (`pp_comp`) and exposure (`ee_comp_imps`) columns.

### 7. Create Runner Notebook

Copy and update runner:

```bash
cp runners/run_car_coll_v1.ipynb runners/run_suv_comp_v1.ipynb
```

**Edit the runner notebook:**

Open in Jupyter and change **Cell 1**:

```python
# OLD:
config_path = "../config/car_coll/v1/config.yaml"

# NEW:
config_path = "../config/suv_comp/v1/config.yaml"
```

**Verify kernel:** Should be `carfax26v1`

### 8. Validation Checklist

Before running, verify:

- [ ] `config.yaml` has correct target/exposure columns
- [ ] `config.yaml` has correct vehicle_type and coverage
- [ ] `config.yaml` output_path matches folder structure
- [ ] `columns_to_load_during_dataassembly.csv` has correct pp_* and ee_* columns
- [ ] `exclusion.csv` and `monotonicity.csv` are for correct coverage
- [ ] Master data file exists and contains target/exposure columns
- [ ] Aux data file exists
- [ ] Runner notebook points to correct config path

### 9. Test Run

Run  notebook runners/run_suv_comp_v1.ipynb


**Expected behavior:**
- Stage 01 should complete without "Missing critical columns" error
- Check that target and exposure columns are loaded
- Monitor Stage 02-04 for any vehicle-specific issues

---

## Common Mistakes ⚠️

### Mistake 1: Wrong Target/Exposure in columns_to_load file
**Symptom:** `ValueError: Missing critical columns: ['ee_comp_imps']`  
**Fix:** Update lines 97-98 in `columns_to_load_during_dataassembly.csv`

### Mistake 2: Mismatched config.yaml and columns file
**Symptom:** Pipeline loads data but wrong columns are treated as target  
**Fix:** Ensure `config.yaml` experiment.target matches the column in columns_to_load file

### Mistake 3: Wrong exclusion/monotonicity for coverage
**Symptom:** Features excluded/constrained incorrectly, poor model performance  
**Fix:** Use coverage-specific templates from `config/templates/{coverage}/`

### Mistake 4: Wrong vehicle in data filenames
**Symptom:** `FileNotFoundError: master_dataset_suv.parquet not found`  
**Fix:** Update `data.master_file` and `data.aux_file` in config.yaml

### Mistake 5: Wrong output path
**Symptom:** Files saved to wrong location  
**Fix:** Update `machines.PC3.paths.output_path` to match folder structure

---

## Quick Reference: File Locations

```
config/{vehicle}_{coverage}/v1/
├── config.yaml                              # ← Main config, update experiment/data sections
├── columns_to_load_during_dataassembly.csv  # ← Update pp_* and ee_* columns
├── columns_inclusion.csv                    # ← Usually same across models
├── exclusion.csv                            # ← Copy from config/templates/{coverage}/
├── monotonicity.csv                         # ← Copy from config/templates/{coverage}/
├── pca_features.csv                         # ← Usually same across models
├── manual_feature_encoding.csv              # ← Usually same across models
├── type_conversions.csv                     # ← Usually same across models
├── all_columns_master.csv                   # ← Usually same across models
├── all_columns_aux.csv                      # ← Usually same across models
└── config_generated/                        # ← Auto-generated during pipeline run

runners/
└── run_{vehicle}_{coverage}_v1.ipynb        # ← Update config_path in Cell 1

output/{vehicle}_{coverage}/v1/              # ← Created automatically during run
```

---

## Coverage Templates Available

```
config/templates/
├── coll/
│   ├── exclusion.csv
│   └── monotonicity.csv
├── comp/
│   ├── exclusion.csv
│   └── monotonicity.csv
└── liab/
    ├── exclusion.csv
    └── monotonicity.csv
```

---

## Example: Complete Setup for TRUCK_LIAB

```bash
# 1. Create directory
mkdir -p config/truck_liab/v1/config_generated

# 2. Copy base files
cp config/car_coll/v1/*.{yaml,csv} config/truck_liab/v1/

# 3. Copy coverage-specific files
cp config/templates/liab/exclusion.csv config/truck_liab/v1/
cp config/templates/liab/monotonicity.csv config/truck_liab/v1/

# 4. Update config.yaml (use editor)
# - experiment.name: "truck_liab_v1"
# - experiment.target: "pp_bi"
# - experiment.exposure: "ee_bi_imps"
# - experiment.coverage: "liab"
# - experiment.vehicle_type: "TRUCK"
# - data.master_file: "master_dataset_truck.parquet"
# - data.aux_file: "truck_with_dep_factor_fold.parquet"
# - machines.PC3.paths.output_path: "output/truck_liab/v1"

# 5. Update columns_to_load_during_dataassembly.csv (lines 97-98)
# - pp_bi,target,N/A,Pure premium bodily injury (target variable)
# - ee_bi_imps,exposure,N/A,Earned exposure for bodily injury (weight column)

# 6. Create runner
cp runners/run_car_coll_v1.ipynb runners/run_truck_liab_v1.ipynb
# Edit Cell 1: config_path = "../config/truck_liab/v1/config.yaml"

# 7. Test
jupyter notebook runners/run_truck_liab_v1.ipynb
```

---

## Need Help?

- **Data column names:** Check your parquet schema with pyarrow
- **Coverage templates:** See `config/templates/{coverage}/`
- **Working example:** Reference `config/car_coll/v1/` and `config/car_liab/v1/`
- **Pipeline errors:** Check that critical columns exist in data and match config

---

## Notes

- All models use the same template notebooks in `templates/` folder
- The pipeline is parameterized through `config.yaml` only
- Vehicle type affects which master/aux files are loaded
- Coverage type affects target/exposure columns and feature constraints
- PCA features, encoding strategies, and other transformations are coverage-agnostic
