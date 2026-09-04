#!/bin/bash
# Test runner for EDA helpers
# Usage: bash tests/run_tests.sh

echo "=============================================="
echo "Running EDA Helper Tests"
echo "=============================================="

cd "$(dirname "$0")/.." || exit

echo ""
echo "1. Running unit tests..."
echo "----------------------------------------------"
python3 tests/test_eda_unit.py

if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed"
    exit 1
fi

echo ""
echo "2. Running integration tests..."
echo "----------------------------------------------"
python3 tests/test_eda_integration.py

if [ $? -ne 0 ]; then
    echo "❌ Integration tests failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ All tests passed!"
echo "=============================================="
