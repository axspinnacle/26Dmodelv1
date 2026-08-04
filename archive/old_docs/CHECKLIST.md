# 🎯 Quick Start Checklist

Use this checklist to get your new dataset running with GBMFirst.

## ✅ Pre-Flight Checklist

### Step 1: Update Configuration (Required)
- [ ] Open `config/car_coll/v1/config.yaml`
- [ ] Update `path_prefix` to your new dataset location
- [ ] Update `input_file_pattern` if different
- [ ] Update `ep_file` filename if different
- [ ] Verify folder paths (`input_data_dir`, `ep_data_dir`, `support_dir`)

**Current paths in config.yaml:**
```yaml
path_prefix: "/Users/Mach/dev/aps/data/2024_CX_Cmodel/v2/"
input_file_pattern: "selectedcolumns_{veh_type}_trainvaliddata_v2.csv"
ep_file: "data_ep.csv"
```

### Step 2: Verify Machine Setup
- [x] `.PC` file exists (contains: `3` for Mac)
- [ ] Conda environment `py39_26v1` is available
- [ ] Dataset files exist at the path specified in config

### Step 3: First Test Run
- [ ] Enable debug mode in config.yaml (`debug: true`)
- [ ] Activate environment: `conda activate py39_26v1`
- [ ] Open `runners/run_one.ipynb` in Jupyter
- [ ] Run all cells
- [ ] Check output folder for results

---

## 📋 What Needs Your Attention

### Files to Update:
1. **config/car_coll/v1/config.yaml** ⚠️ **REQUIRED**
   - Machine paths (PC2 or PC3 section)
   - Data file patterns

2. **config/car_coll/v1/feature_selection.csv** (Optional)
   - Review features for your dataset
   - Mark selected=1 for features to include

3. **config/car_coll/v1/feature_clipping.csv** (Optional)
   - Review min/max bounds for features

### Files Already Configured:
- ✅ `template.ipynb` - Ready to use
- ✅ `utils.py` - Ready to use
- ✅ `runners/run_one.ipynb` - Ready to use
- ✅ `.PC` - Set to `3` (Mac)
- ✅ `.clinerules` - Project rules copied

---

## 🚀 Quick Commands

### Start Jupyter:
```bash
conda activate py39_26v1
cd /Users/Mach/dev/aps/code/26Dmodelv1/runners
jupyter notebook
```

### Run via Command Line:
```bash
conda activate py39_26v1
cd /Users/Mach/dev/aps/code/26Dmodelv1
python -m papermill template.ipynb output/car_coll/v1/car_coll.ipynb \
  -p config_path "config/car_coll/v1/config.yaml" -k py39_26v1
```

### Check Results:
```bash
ls -lh output/car_coll/v1/
```

---

## 🎯 Success Criteria

After your first successful run, you should see:
- [ ] `output/car_coll/v1/car_coll.ipynb` - Executed notebook with results
- [ ] `output/car_coll/v1/CAR_coll_model.json` - Trained XGBoost model
- [ ] SHAP plots in the executed notebook
- [ ] Lift charts showing model performance
- [ ] No errors in execution

---

## ❓ Common Issues

**Problem:** "FileNotFoundError: .PC file not found"
- **Solution:** File exists - verify template.ipynb can find it (should auto-detect)

**Problem:** "Data file not found"
- **Solution:** Check `path_prefix` + `input_data_dir` + `input_file_pattern` in config.yaml

**Problem:** "Module not found: yaml"
- **Solution:** `conda activate py39_26v1` and `pip install pyyaml`

**Problem:** "Kernel not found: py39_26v1"
- **Solution:** `python -m ipykernel install --user --name=py39_26v1`

---

## 📁 File Inventory

Total files copied: **15 files**

```
✅ Template & Core:
   - template.ipynb (main pipeline)
   - utils.py
   - edited_20240314_symbols_gbm_functions.ipynb
   - generate_features.ipynb

✅ Configuration:
   - .clinerules
   - .PC
   - requirements.txt

✅ Experiment Config (car_coll/v1):
   - config.yaml ⚠️ EDIT THIS
   - feature_selection.csv
   - feature_clipping.csv

✅ Runners:
   - runners/run_one.ipynb
   - runners/run_all.ipynb

✅ Documentation:
   - SETUP_GUIDE.md (detailed instructions)
   - CHECKLIST.md (this file)

📁 Directories:
   - config/car_coll/v1/
   - runners/
   - output/ (empty, will be populated)
```

---

## 🔄 Next Actions After First Run

Once your first experiment works:

1. **Create more vehicle/coverage combinations:**
   ```bash
   cp -r config/car_coll/v1 config/suv_comp/v1
   # Edit the new config.yaml
   ```

2. **Run all models:**
   - Open `runners/run_all.ipynb`
   - Add your config paths
   - Execute to run batch processing

3. **Compare results:**
   - Review output folders
   - Compare model performance
   - Iterate on feature selection

---

**Ready to start? Begin with Step 1 above! 🚀**
