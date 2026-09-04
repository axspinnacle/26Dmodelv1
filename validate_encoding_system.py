#!/usr/bin/env python
"""
Validation script for feature encoding system
Checks that all components are in place before running pipeline
"""

import os
import sys
from pathlib import Path

def check_file(path, description):
    """Check if file exists and print status"""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def check_import(module_path, description):
    """Check if Python module can be imported"""
    try:
        parts = module_path.split('.')
        if len(parts) == 1:
            __import__(module_path)
        else:
            mod = __import__(parts[0])
            for part in parts[1:]:
                mod = getattr(mod, part)
        print(f"✓ {description}: {module_path}")
        return True
    except Exception as e:
        print(f"✗ {description}: {module_path} - {e}")
        return False

print("=" * 70)
print("FEATURE ENCODING SYSTEM - VALIDATION")
print("=" * 70)

checks = []

print("\n1. Core Library Files")
print("-" * 70)
checks.append(check_file("lib/feature_encoder.py", "Feature encoder module"))

print("\n2. Pipeline Template")
print("-" * 70)
checks.append(check_file("templates/04c_feature_encoding.ipynb", "Stage 04c notebook"))

print("\n3. Configuration")
print("-" * 70)
checks.append(check_file("config/car_coll/v1/config.yaml", "Main config"))

print("\n4. Master Encoding File")
print("-" * 70)
checks.append(check_file("config/car_coll/v1/config_generated/master_feature_encoding.csv", 
                        "Master feature encoding CSV"))

print("\n5. Python Imports")
print("-" * 70)
sys.path.insert(0, str(Path.cwd() / 'lib'))
checks.append(check_import("feature_encoder", "feature_encoder module"))
checks.append(check_import("feature_encoder.apply_master_encoding", "apply_master_encoding"))
checks.append(check_import("feature_encoder.save_encoders", "save_encoders"))
checks.append(check_import("feature_encoder.load_encoders", "load_encoders"))
checks.append(check_import("feature_encoder.apply_saved_encoders", "apply_saved_encoders"))

print("\n6. Modified Pipeline Files")
print("-" * 70)
checks.append(check_file("templates/05_model_training.ipynb", "Updated Stage 05"))
checks.append(check_file("runners/run_car_coll_v1.ipynb", "Updated runner"))

print("\n7. Documentation")
print("-" * 70)
checks.append(check_file("documentation/FEATURE_ENCODING_IMPLEMENTATION.md", "Technical docs"))
checks.append(check_file("ENCODING_SYSTEM_SUMMARY.md", "Summary docs"))

print("\n" + "=" * 70)
passed = sum(checks)
total = len(checks)
print(f"VALIDATION RESULTS: {passed}/{total} checks passed")
print("=" * 70)

if passed == total:
    print("\n✅ ALL CHECKS PASSED - System ready to run!")
    print("\nNext step: Run the pipeline:")
    print("  jupyter notebook runners/run_car_coll_v1.ipynb")
    sys.exit(0)
else:
    print(f"\n❌ {total - passed} checks failed - Please review errors above")
    sys.exit(1)
