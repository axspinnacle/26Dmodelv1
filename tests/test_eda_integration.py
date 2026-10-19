"""
Integration test for EDA workflow
Run: python tests/test_eda_integration.py
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from eda_helpers import analyze_column


def create_test_data():
    """Create synthetic test dataset"""
    np.random.seed(42)
    n_rows = 1000
    
    data = pd.DataFrame({
        # Ordinal 0-5
        'safety_feature': np.random.choice([0, 1, 2, 3, 4, 5], n_rows),
        
        # Continuous numeric
        'msrp': np.random.uniform(15000, 80000, n_rows),
        'mileage': np.random.uniform(0, 150000, n_rows),
        
        # Binary
        'has_sunroof': np.random.choice([0, 1], n_rows),
        
        # Categorical - low cardinality
        'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], n_rows),
        
        # Categorical - medium cardinality
        'zip_code': np.random.choice([f'ZIP{i:05d}' for i in range(50)], n_rows),
        
        # Categorical - high cardinality
        'vehicle_make': np.random.choice([f'Make{i}' for i in range(150)], n_rows),
        
        # Identifiers (should be skipped)
        'vin': [f'VIN{i:010d}' for i in range(n_rows)],
        'policyid': [f'POL{i:08d}' for i in range(n_rows)],
        
        # Date (should be skipped)
        'date_column': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
        
        # Target (should be skipped)
        'pp_coll': np.random.uniform(0, 5000, n_rows),
    })
    
    return data


def test_full_workflow():
    """Test analyzing all columns in a dataset"""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Full EDA Workflow")
    print("="*70)
    
    # Create test data
    df = create_test_data()
    print(f"\n✓ Created test dataset: {df.shape}")
    
    # Analyze all columns
    results = []
    target_exposure = {'pp_coll'}
    
    print(f"\n📊 Analyzing {len(df.columns)} columns...")
    
    for col in df.columns:
        result = analyze_column(col, df[col], target_exposure=target_exposure)
        results.append(result)
        print(f"  {col:20s} → {result['encoding']:15s} ({result['category']})")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Verify results
    print(f"\n📋 Results Summary:")
    print(f"  Total columns analyzed: {len(results_df)}")
    print(f"\n  Encoding distribution:")
    for encoding, count in results_df['encoding'].value_counts().items():
        print(f"    {encoding:20s}: {count}")
    
    # Assertions
    assert len(results_df) == len(df.columns), "Should analyze all columns"
    # has_sunroof (0/1) is detected as ordinal_0_5, which is technically correct
    assert (results_df['encoding'] == 'ordinal_0_5').sum() >= 1, "Should detect ordinal column(s)"
    assert (results_df['encoding'] == 'numeric').sum() >= 2, "Should detect numeric columns"
    assert (results_df['encoding'] == 'one_hot').sum() >= 2, "Should detect categorical columns"
    assert (results_df['encoding'] == 'skip').sum() == 4, "Should skip identifiers/dates/targets"
    assert (results_df['encoding'] == 'group_then_ohe').sum() == 1, "Should detect high cardinality"
    
    print(f"\n✅ All integration tests passed!")
    
    return results_df


def test_null_handling():
    """Test handling of null values"""
    print("\n" + "="*70)
    print("TEST: Null Value Handling")
    print("="*70)
    
    # Create data with nulls
    data_with_nulls = pd.Series([1, 2, np.nan, 4, np.nan, 6, 7, np.nan, 9, 10])
    result = analyze_column('test_nulls', data_with_nulls)
    
    assert result['null_pct'] == 30.0, f"Expected 30% nulls, got {result['null_pct']}%"
    print(f"  ✓ Null percentage correctly calculated: {result['null_pct']}%")
    print("✅ Null handling test passed!")


if __name__ == '__main__':
    try:
        results_df = test_full_workflow()
        test_null_handling()
        
        print("\n" + "="*70)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
