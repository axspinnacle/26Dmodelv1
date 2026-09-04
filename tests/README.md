# EDA Tests

## Test Files

| File | Type | Description |
|------|------|-------------|
| `test_eda_unit.py` | Unit | Tests individual `analyze_column()` function |
| `test_eda_integration.py` | Integration | Tests full workflow with synthetic data |
| `run_tests.sh` | Runner | Runs all tests in sequence |

## Running Tests

### Quick Run (All Tests)
```bash
bash tests/run_tests.sh
```

### Individual Tests
```bash
# Unit tests only
python3 tests/test_eda_unit.py

# Integration tests only
python3 tests/test_eda_integration.py
```

### With Pytest (Optional)
```bash
# If pytest is installed
pytest tests/ -v
```

## Test Coverage

### Unit Tests (`test_eda_unit.py`)
Tests the `analyze_column()` function for:
- ✅ Ordinal 0-5 detection
- ✅ Continuous numeric detection
- ✅ Binary categorical detection
- ✅ High cardinality categorical detection
- ✅ Identifier skip (vin, policyid)
- ✅ Target/exposure skip

### Integration Tests (`test_eda_integration.py`)
Tests full workflow with:
- ✅ 1000 rows of synthetic data
- ✅ 11 columns of various types
- ✅ Null value handling
- ✅ End-to-end column analysis

## Expected Output

```
Running EDA unit tests...
==================================================
✓ Ordinal detection test passed
✓ Continuous numeric test passed
✓ Binary categorical test passed
✓ High cardinality test passed
✓ Identifier skip test passed
✓ Target skip test passed
==================================================
✅ All tests passed!
```

## Test Data

The integration test creates synthetic data with:
- **Ordinal**: safety_feature (0-5)
- **Numeric**: msrp, mileage
- **Binary**: has_sunroof (0/1)
- **Categorical (low)**: state (5 states)
- **Categorical (medium)**: zip_code (50 zips)
- **Categorical (high)**: vehicle_make (150 makes)
- **Identifiers**: vin, policyid
- **Date**: date_column
- **Target**: pp_coll

## Adding New Tests

To add a new test function to `test_eda_unit.py`:

```python
def test_new_feature():
    """Test description"""
    data = pd.Series([...])  # Your test data
    result = analyze_column('test_col', data)
    assert result['encoding'] == 'expected_value'
    print("✓ New test passed")
```

Then add the function call in `if __name__ == '__main__':` block.

## Troubleshooting

**Import errors:**
```bash
# Make sure you're in project root
cd /dev/aps/code/26Dmodelv1
python3 tests/test_eda_unit.py
```

**Missing pandas/numpy:**
```bash
conda activate py311_26v1
```

## Next Steps

After tests pass:
1. Complete `lib/eda_helpers.py` (add remaining functions)
2. Complete EDA notebook
3. Run full EDA workflow
4. Add more tests as needed

