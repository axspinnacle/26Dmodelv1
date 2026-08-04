# 🎯 Type 3 Actuarial Encoding - Quick Start Guide

## What This Does

Integrates **domain-informed actuarial encoding** from your 26CF_Dmod_v1 experiments into GBMFirst. Each feature gets encoded based on actuarial knowledge defined in `level_mapping.csv`.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Test with Small Data (fold 1 only)

Edit: `config/car_coll/v1_actuarial/config.yaml`

```yaml
model:
  use_folds: true
  train_folds: [1]
  test_folds: [6]
```

Run from project root:
```bash
jupyter nbconvert --execute templates/actuarial_gbm.ipynb \
  --output-dir output/car_coll/v1_actuarial
```

**Result:** Quick test on ~17% of data (fold 1) in 5-10 minutes

### Step 2: Review Results

Check output folder:
```bash
ls output/car_coll/v1_actuarial/
# Should see: actuarial_gbm.ipynb, model files, metrics
```

### Step 3: Run Full Data

Edit config:
```yaml
model:
  use_folds: false  # Use all data
```

Run again (takes longer):
```bash
jupyter nbconvert --execute templates/actuarial_gbm.ipynb \
  --output-dir output/car_coll/v1_actuarial
```

---

## 📊 Fold Testing Options

| Configuration | Data Used | Purpose | Time |
|--------------|-----------|---------|------|
| `train_folds: [1]` | 17% (1 fold) | Quick test | 5-10 min |
| `train_folds: [1,2]` | 33% (2 folds) | Medium test | 10-20 min |
| `train_folds: [1,2,3]` | 50% (3 folds) | Large test | 20-30 min |
| `use_folds: false` | 100% (all data) | Production run | 40-60 min |

**Note:** Always use `test_folds: [6]` for consistent evaluation.

---

## 📁 File Structure

```
GBMFirst/
├── README_ACTUARIAL.md          ← You are here
│
├── lib/
│   └── encoding_strategies.py    ← Type 3 encoding logic (1024 lines)
│
├── config/car_coll/
│   ├── v1/                       ← Standard approach (unchanged)
│   │   ├── config.yaml
│   │   ├── feature_selection.csv
│   │   └── feature_clipping.csv
│   │
│   └── v1_actuarial/             ← NEW: Actuarial approach
│       ├── config.yaml           ← Configured for Type 3
│       └── level_mapping.csv     ← 138 features with encoding strategies
│
├── templates/
│   ├── standard_gbm.ipynb        ← (if created later)
│   └── actuarial_gbm.ipynb       ← NEW: Uses Type 3 encoding
│
└── output/
    └── car_coll/
        ├── v1/                   ← Standard results
        └── v1_actuarial/         ← Actuarial results
```

---

## 🔍 What's Different?

### Standard (v1) vs Actuarial (v1_actuarial)

| Aspect | Standard | Actuarial |
|--------|----------|-----------|
| **Config File** | feature_selection.csv | level_mapping.csv |
| **Encoding** | Manual clipping/blanking | Per-feature strategies |
| **Strategy** | Select/blank/clip | Domain-informed rules |
| **Flexibility** | Fixed approach | Configurable per feature |
| **Maintenance** | Edit notebook | Edit CSV |

### Encoding Strategies in level_mapping.csv

| Strategy | What It Does | Example Features |
|----------|--------------|------------------|
| `ordered` | Keep 0-5 as-is | Basic ordinal features |
| `binary_low_hi` | {0,1,2}→0, {3,4,5}→1 | Safety features |
| `binary_lo_high` | {0,1,2,3}→0, {4,5}→1 | Premium features |
| `ohe` | One-Hot Encoding | Categorical features |
| `drop` | Exclude from model | Leakage/irrelevant |

---

## ⚙️ Configuration Reference

### Fold-Based Testing

```yaml
model:
  debug: false                    # Old row-based debug (still works)
  debug_rows: 4000000             # If debug=true, use first N rows
  
  # NEW: Fold-based sampling
  use_folds: true                 # Enable fold mode
  train_folds: [1, 2]             # List of training folds
  test_folds: [6]                 # List of test folds  
  fold_column: "fold"             # Column name with fold numbers
```

### When to Use What

**Fold mode (`use_folds: true`):**
- ✅ Faster testing
- ✅ Maintains data distribution
- ✅ Reproducible results
- ✅ Easy to scale up

**Debug mode (`debug: true`):**
- ✅ Very quick iteration
- ⚠️ May not be representative
- ⚠️ Breaks stratification

**Full data (`use_folds: false`):**
- ✅ Production runs
- ✅ Final model training
- ⏱️ Takes longest

---

## 🔄 Comparing Approaches

### Run Both Versions

1. **Standard approach:**
```bash
# Edit config/car_coll/v1/config.yaml (add fold settings)
jupyter nbconvert --execute template.ipynb --output-dir output/car_coll/v1
```

2. **Actuarial approach:**
```bash
jupyter nbconvert --execute templates/actuarial_gbm.ipynb \
  --output-dir output/car_coll/v1_actuarial
```

### Compare Results

```python
import pandas as pd

# Load metrics
v1_metrics = pd.read_csv('output/car_coll/v1/metrics.csv')
v1_act_metrics = pd.read_csv('output/car_coll/v1_actuarial/metrics.csv')

# Compare
comparison = pd.DataFrame({
    'Standard': v1_metrics['value'],
    'Actuarial': v1_act_metrics['value'],
    'Difference': v1_act_metrics['value'] - v1_metrics['value']
}, index=v1_metrics['metric'])

print(comparison)
```

---

## 🐛 Troubleshooting

### Error: "level_mapping.csv not found"

**Solution:** Check config path
```yaml
files:
  level_mapping: "level_mapping.csv"  # Should be in same dir as config.yaml
```

### Error: "encode_type3_actuarial not found"

**Solution:** Encoding strategies not imported
```python
# Should be in notebook imports:
sys.path.insert(0, '../lib')
from encoding_strategies import encode_type3_actuarial
```

### Error: "fold column not found"

**Solution:** Either:
1. Your data doesn't have folds → Set `use_folds: false`
2. Wrong column name → Check `fold_column` in config

### Slow Performance

**Solutions:**
1. Start with fold 1 only: `train_folds: [1]`
2. Reduce XGBoost rounds: `num_round: 50`
3. Enable debug mode: `debug: true, debug_rows: 100000`

---

## 📈 Next Steps

1. ✅ **Test encoding** - Run with fold 1
2. ✅ **Validate results** - Check lift charts look reasonable
3. ✅ **Compare approaches** - Run standard vs actuarial
4. ✅ **Analyze differences** - Which performs better?
5. ✅ **Scale up** - Add more folds or go full data
6. ✅ **Production** - Deploy better-performing approach

---

## 💡 Tips

- **Always test with fold 1 first** - Catches errors quickly
- **Use consistent test fold** - Always test on fold 6 for comparability
- **Check feature counts** - Actuarial may have different # features
- **Compare lift charts** - Visual inspection often reveals issues
- **Save both models** - You can ensemble them later

---

## 📚 More Information

- **Encoding strategies:** See `lib/encoding_strategies.py`
- **Level mapping:** Edit `config/car_coll/v1_actuarial/level_mapping.csv`
- **Original experiment:** `/Users/Mach/dev/aps/code/26CF_Dmod_v1/experiment_levels`

---

## ✅ Checklist

Before your first run:

- [ ] `lib/encoding_strategies.py` exists
- [ ] `config/car_coll/v1_actuarial/level_mapping.csv` exists
- [ ] `config/car_coll/v1_actuarial/config.yaml` has `use_folds: true`
- [ ] Your data has a `fold` column (or `use_folds: false`)
- [ ] You've tested with fold 1 first

**Ready? Run Step 1 above! 🚀**
