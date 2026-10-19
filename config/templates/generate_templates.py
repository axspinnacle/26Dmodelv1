#!/usr/bin/env python3
"""
Generate template monotonicity and exclusion files from Sara's master file.
Splits by coverage into template folders.
"""

import pandas as pd
import os

# Paths
sara_file = "/Users/Mach/dev/aps/code/26Dmodelv1/config/car_coll/v1/monotonicity_selections_final_sara.csv"
template_base = "/Users/Mach/dev/aps/code/26Dmodelv1/config/templates"

# Coverage mapping (Sara's names -> template folder names)
coverage_mapping = {
    'liab_3p': 'liab',
    'pip': 'pip',
    'coll': 'coll',
    'comp': 'comp'
}

# Read Sara's file
print(f"Reading {sara_file}...")
df = pd.read_csv(sara_file)
print(f"Loaded {len(df)} rows")

# Process each coverage
for sara_cov, template_folder in coverage_mapping.items():
    print(f"\n=== Processing {sara_cov} -> {template_folder} ===")
    
    # Filter by coverage
    cov_df = df[df['cov'] == sara_cov].copy()
    print(f"  Found {len(cov_df)} rows for {sara_cov}")
    
    if len(cov_df) == 0:
        print(f"  WARNING: No data for {sara_cov}, skipping...")
        continue
    
    # Create monotonicity.csv
    mono_df = cov_df[['field', 'CAR', 'SUV', 'TRUCK', 'VAN']].copy()
    
    # Replace values for monotonicity
    for col in ['CAR', 'SUV', 'TRUCK', 'VAN']:
        mono_df[col] = mono_df[col].replace({
            'x': 0,
            'Exclude': 0  # Exclude is handled in exclusion file
        })
        # Keep -1 and 1 as-is
    
    mono_file = os.path.join(template_base, template_folder, 'monotonicity.csv')
    mono_df.to_csv(mono_file, index=False)
    print(f"  ✓ Created {mono_file}")
    print(f"    Rows: {len(mono_df)}")
    
    # Create exclusion.csv
    excl_df = cov_df[['field', 'CAR', 'SUV', 'TRUCK', 'VAN']].copy()
    
    # Convert to 1=exclude, 0=keep
    for col in ['CAR', 'SUV', 'TRUCK', 'VAN']:
        excl_df[col] = excl_df[col].apply(lambda x: 1 if x == 'Exclude' else 0)
    
    # Only keep rows where at least one vehicle has exclusion
    excl_df = excl_df[(excl_df['CAR'] == 1) | (excl_df['SUV'] == 1) | 
                       (excl_df['TRUCK'] == 1) | (excl_df['VAN'] == 1)]
    
    excl_file = os.path.join(template_base, template_folder, 'exclusion.csv')
    excl_df.to_csv(excl_file, index=False)
    print(f"  ✓ Created {excl_file}")
    print(f"    Rows: {len(excl_df)} (features with at least one exclusion)")

print("\n=== Complete ===")
print(f"Template files created in {template_base}/")
