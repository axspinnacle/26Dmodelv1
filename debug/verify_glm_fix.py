"""
Verification script to check if GLM files have vin_date after re-running 04c
"""
import pandas as pd
import sys

print('Verifying GLM merge fix...')
print('=' * 60)

output_base = 'output/car_liab/v1'

try:
    # Check if GLM files exist
    glm_train = pd.read_parquet(f'{output_base}/data/04c_train_glm.parquet')
    glm_test = pd.read_parquet(f'{output_base}/data/04c_test_glm.parquet')
    
    print(f'\n1. GLM Train file:')
    print(f'   Columns: {glm_train.columns.tolist()}')
    print(f'   Rows: {len(glm_train):,}')
    
    print(f'\n2. GLM Test file:')
    print(f'   Columns: {glm_test.columns.tolist()}')
    print(f'   Rows: {len(glm_test):,}')
    
    # Check for vin_date
    if 'vin_date' in glm_train.columns and 'vin_date' in glm_test.columns:
        print(f'\n✓ SUCCESS: Both files have vin_date column!')
        
        # Check for duplicates
        train_dups = glm_train['vin_date'].duplicated().sum()
        test_dups = glm_test['vin_date'].duplicated().sum()
        
        print(f'\n3. Duplicate check:')
        print(f'   Train duplicates: {train_dups}')
        print(f'   Test duplicates: {test_dups}')
        
        if train_dups == 0 and test_dups == 0:
            print(f'\n✓ SUCCESS: No duplicates in vin_date!')
        else:
            print(f'\n⚠ WARNING: Duplicates found!')
            sys.exit(1)
        
        # Check glm_pred values
        print(f'\n4. GLM predictions:')
        print(f'   Train mean: {glm_train["glm_pred"].mean():.2f}')
        print(f'   Test mean: {glm_test["glm_pred"].mean():.2f}')
        print(f'   Train range: {glm_train["glm_pred"].min():.2f} to {glm_train["glm_pred"].max():.2f}')
        
        print(f'\n' + '=' * 60)
        print('✓ ALL CHECKS PASSED!')
        print('\nYou can now re-run:')
        print('  - 05a_model_initial.ipynb')
        print('  - 05b_hpo.ipynb')
        print('  - 05c_production.ipynb')
        print('  - debug_features.ipynb')
        
    else:
        print(f'\n✗ FAILED: vin_date column not found!')
        print(f'   You need to re-run template 04c_feature_encoding.ipynb')
        sys.exit(1)
        
except FileNotFoundError as e:
    print(f'\n✗ FAILED: GLM parquet files not found!')
    print(f'   Error: {e}')
    print(f'\n   You need to run template 04c_feature_encoding.ipynb first')
    sys.exit(1)
