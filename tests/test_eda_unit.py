"""
Unit tests for EDA helper functions
Run: python tests/test_eda_unit.py
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
from eda_helpers import analyze_column


def test_ordinal_detection():
    """Test ordinal 0-5 detection"""
    data = pd.Series([0, 1, 2, 3, 4, 5, 3, 2, 1, 0])
    result = analyze_column('test_ordinal', data)
    assert result['encoding'] == 'ordinal_0_5', f"Expected ordinal_0_5, got {result['encoding']}"
    assert result['category'] == 'ordinal_0_5'
    print("✓ Ordinal detection test passed")


def test_continuous_numeric():
    """Test continuous numeric detection"""
    # Need >20 unique values for continuous classification
    import numpy as np
    data = pd.Series(np.random.uniform(100, 1000, 50))
    result = analyze_column('test_numeric', data)
    assert result['encoding'] == 'numeric'
    assert result['category'] == 'continuous_numeric'
    print("✓ Continuous numeric test passed")


def test_binary_categorical():
    """Test binary categorical detection"""
    data = pd.Series(['Yes', 'No', 'Yes', 'No', 'Yes'])
    result = analyze_column('test_binary_cat', data)
    assert result['encoding'] == 'one_hot'
    assert result['category'] == 'binary_categorical'
    print("✓ Binary categorical test passed")


def test_high_cardinality():
    """Test high cardinality detection"""
    data = pd.Series([f'cat_{i}' for i in range(150)])
    result = analyze_column('test_high_cat', data)
    assert result['encoding'] == 'group_then_ohe'
    assert result['category'] == 'high_cardinality_categorical'
    print("✓ High cardinality test passed")


def test_identifier_skip():
    """Test identifier skip"""
    data = pd.Series(['VIN12345', 'VIN67890'])
    result = analyze_column('vin_number', data)
    assert result['encoding'] == 'skip'
    assert result['category'] == 'identifier'
    print("✓ Identifier skip test passed")


def test_target_skip():
    """Test target/exposure skip"""
    data = pd.Series([100.5, 200.3, 150.2])
    result = analyze_column('pp_coll', data, target_exposure={'pp_coll'})
    assert result['encoding'] == 'skip'
    assert result['category'] == 'target_exposure'
    print("✓ Target skip test passed")


if __name__ == '__main__':
    print("Running EDA unit tests...")
    print("="*50)
    
    test_ordinal_detection()
    test_continuous_numeric()
    test_binary_categorical()
    test_high_cardinality()
    test_identifier_skip()
    test_target_skip()
    
    print("="*50)
    print("✅ All tests passed!")
