# Debug Notebooks

Data quality analysis and debugging tools.

## Notebooks

### analyze_control_duplicates.ipynb

Analyzes duplicate rows in control/auxiliary files.

**Purpose:**
- Identify duplicate join keys in control files
- Check if duplicates have consistent values (safe to drop) or inconsistent (data issue)
- Provide recommendations for handling duplicates

**Usage:**
```bash
# Default (car control file)
jupyter notebook debug/analyze_control_duplicates.ipynb

# Or run with papermill with custom parameters
python -m papermill debug/analyze_control_duplicates.ipynb \
  output_debug.ipynb \
  -p data_path "/path/to/data" \
  -p control_file "your_control_file.parquet" \
  -p join_key "your_key_column" \
  -p value_columns '["col1", "col2"]'
```

**Parameters:**
- `data_path`: Directory containing control files
- `control_file`: Filename of control/auxiliary parquet file
- `join_key`: Column name used for joining (e.g., "vin_date")
- `value_columns`: List of value columns to check for consistency

**Output:**
- Duplicate count statistics
- Distribution of duplicate frequencies
- Value consistency analysis (safe to drop?)
- Sample duplicate rows
- Recommendations

## Utility Functions

See `lib/data_quality.py` for reusable data quality functions:
- `analyze_duplicates()` - Analyze duplicate statistics
- `check_duplicate_consistency()` - Check if duplicates have identical values
- `get_sample_duplicates()` - Get sample duplicate rows
- `summarize_duplicates()` - Convenience wrapper for file analysis

### remove_control_duplicates.ipynb

Removes duplicate rows from control file and saves cleaned version.

**Purpose:**
- Load control file with duplicates
- Verify duplicates are identical (safe to drop)
- Remove duplicates using drop_duplicates()
- Save cleaned file

**Usage:**
```bash
# Run in Jupyter
jupyter notebook debug/remove_control_duplicates.ipynb

# Or use papermill
python -m papermill debug/remove_control_duplicates.ipynb \
  debug_remove_output.ipynb
```

**Parameters:**
- `data_path`: Directory containing control files (default: "/Users/Mach/dev/aps/data/2026_Dmodel_data")
- `input_file`: Input parquet filename (default: "aux_dataset_car_vindate_ctl.parquet")
- `output_file`: Output filename (default: "aux_dataset_car_vindate_ctl_no_duplicate.parquet")
- `key_column`: Column to deduplicate on (default: "vin_date")

**Output:**
- Creates deduplicated parquet file in same directory
- Shows before/after statistics
- Verifies al
### remove_control_duplicates.ipynb

Removes duplicate rows from config
Removes duplicate rows from contrdel
**Purpose:**
- Load control file with duplicates
- Verify ``
