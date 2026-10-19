"""
Debug tools for analyzing duplicate vin_dates in datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_check_duplicates(path, key_col='vin_date'):
    """
    Load file and check for duplicates on key column.
    
    Args:
        path: Path to parquet file
        key_col: Column to check for duplicates
        
    Returns:
        DataFrame and duplicate stats dict
    """
    df = pd.read_parquet(path)
    
    total_rows = len(df)
    unique_keys = df[key_col].nunique()
    has_duplicates = total_rows != unique_keys
    num_duplicates = total_rows - unique_keys
    
    stats = {
        'total_rows': total_rows,
        'unique_keys': unique_keys,
        'has_duplicates': has_duplicates,
        'num_duplicate_rows': num_duplicates,
        'duplicate_pct': (num_duplicates / total_rows * 100) if total_rows > 0 else 0
    }
    
    print(f"File: {Path(path).name}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Unique {key_col}: {unique_keys:,}")
    print(f"  Has duplicates: {has_duplicates}")
    if has_duplicates:
        print(f"  Duplicate rows: {num_duplicates:,} ({stats['duplicate_pct']:.1f}%)")
    
    return df, stats


def analyze_duplicate_patterns(df, key_col='vin_date'):
    """
    Analyze how many times each key appears.
    
    Args:
        df: DataFrame to analyze
        key_col: Key column to check
        
    Returns:
        Series of value_counts and summary dict
    """
    counts = df[key_col].value_counts()
    
    # Distribution of occurrences
    occurrence_dist = counts.value_counts().sort_index()
    
    summary = {
        'appears_once': (counts == 1).sum(),
        'appears_2x': (counts == 2).sum(),
        'appears_3x': (counts == 3).sum(),
        'appears_4x': (counts == 4).sum(),
        'appears_5plus': (counts >= 5).sum(),
        'max_occurrences': counts.max()
    }
    
    print(f"\n{key_col} frequency distribution:")
    print(f"  Appears 1x:  {summary['appears_once']:,}")
    print(f"  Appears 2x:  {summary['appears_2x']:,}")
    print(f"  Appears 3x:  {summary['appears_3x']:,}")
    print(f"  Appears 4x:  {summary['appears_4x']:,}")
    print(f"  Appears 5+:  {summary['appears_5plus']:,}")
    print(f"  Max occurrences: {summary['max_occurrences']}")
    
    # Show top duplicates
    top_dups = counts[counts > 1].head(10)
    if len(top_dups) > 0:
        print(f"\nTop 10 most duplicated {key_col}:")
        for key, count in top_dups.items():
            print(f"  {key}: {count} times")
    
    return counts, summary


def compare_duplicate_rows(df, key_col='vin_date', sample_size=5):
    """
    Compare rows that share the same key to see what differs.
    
    Args:
        df: DataFrame to analyze
        key_col: Key column
        sample_size: Number of duplicate groups to sample
        
    Returns:
        Dict with analysis results
    """
    # Find keys with duplicates
    counts = df[key_col].value_counts()
    dup_keys = counts[counts > 1].index[:sample_size]
    
    print(f"\nComparing duplicate rows (sampling {len(dup_keys)} keys):")
    
    results = {}
    for key in dup_keys:
        dup_rows = df[df[key_col] == key]
        print(f"\n{key_col} = {key} ({len(dup_rows)} rows):")
        
        # Check which columns differ
        differing_cols = []
        identical_cols = []
        
        for col in df.columns:
            if col == key_col:
                continue
            try:
                unique_vals = dup_rows[col].nunique()
                if unique_vals > 1:
                    differing_cols.append(col)
                    print(f"  {col}: {unique_vals} different values")
                else:
                    identical_cols.append(col)
            except:
                pass
        
        results[key] = {
            'num_rows': len(dup_rows),
            'differing_cols': differing_cols,
            'identical_cols': identical_cols
        }
        
        print(f"  → {len(differing_cols)} columns differ, {len(identical_cols)} identical")
    
    # Aggregate analysis
    all_differing = set()
    for r in results.values():
        all_differing.update(r['differing_cols'])
    
    print(f"\nSummary: {len(all_differing)} columns differ across duplicate rows:")
    for col in sorted(all_differing):
        print(f"  - {col}")
    
    return results


def suggest_dedup_strategy(df, key_col='vin_date', critical_cols=None):
    """
    Suggest deduplication strategy based on analysis.
    
    Args:
        df: DataFrame to analyze
        key_col: Key column
        critical_cols: List of columns that must not differ (e.g., target, exposure)
        
    Returns:
        Dict with recommendations
    """
    if critical_cols is None:
        critical_cols = []
    
    print(f"\n{'='*60}")
    print("DEDUPLICATION STRATEGY ANALYSIS")
    print(f"{'='*60}")
    
    # Check if there are duplicates
    counts = df[key_col].value_counts()
    dup_keys = counts[counts > 1]
    
    if len(dup_keys) == 0:
        print("✓ No duplicates found - no action needed")
        return {'action': 'none', 'reason': 'no duplicates'}
    
    # Sample duplicate groups
    sample_keys = dup_keys.index[:100]
    
    # Check what columns differ in duplicates
    differing_by_col = {}
    for col in df.columns:
        if col == key_col:
            continue
        try:
            # For each duplicate key, check if this column differs
            differs_count = 0
            for key in sample_keys:
                dup_rows = df[df[key_col] == key]
                if dup_rows[col].nunique() > 1:
                    differs_count += 1
            
            if differs_count > 0:
                differing_by_col[col] = differs_count
        except:
            pass
    
    print(f"\nColumns that differ in duplicate rows (out of {len(sample_keys)} sampled):")
    for col, count in sorted(differing_by_col.items(), key=lambda x: x[1], reverse=True)[:20]:
        pct = count / len(sample_keys) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Check critical columns
    critical_issues = []
    for col in critical_cols:
        if col in differing_by_col:
            critical_issues.append(col)
            print(f"\n⚠️  WARNING: Critical column '{col}' differs in duplicates!")
    
    # Recommend strategy
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    
    if len(critical_issues) > 0:
        print("❌ CANNOT SAFELY DEDUPLICATE")
        print(f"   Critical columns differ: {', '.join(critical_issues)}")
        print("   → Need to investigate why duplicates exist")
        print("   → May need composite key or data cleaning")
        return {
            'action': 'investigate',
            'reason': 'critical columns differ',
            'critical_issues': critical_issues
        }
    
    if len(differing_by_col) == 0:
        print("✓ All columns identical in duplicate rows")
        print("  Strategy: drop_duplicates(subset=[key_col])")
        return {
            'action': 'drop_duplicates',
            'reason': 'all columns identical',
            'method': f"df.drop_duplicates(subset=['{key_col}'], keep='first')"
        }
    
    print("⚠️  Some columns differ in duplicate rows")
    print(f"   {len(differing_by_col)} columns have different values")
    print("   Possible strategies:")
    print("   1. Keep first occurrence: drop_duplicates(keep='first')")
    print("   2. Keep last occurrence: drop_duplicates(keep='last')")
    print("   3. Aggregate: groupby + agg (sum/mean/max)")
    print("   4. Investigate why duplicates exist")
    
    return {
        'action': 'manual_review',
        'reason': 'columns differ',
        'differing_cols': list(differing_by_col.keys()),
        'num_differing': len(differing_by_col)
    }


def create_deduplicated_file(df, key_col='vin_date', strategy='first', output_path=None):
    """
    Create deduplicated version of DataFrame.
    
    Args:
        df: DataFrame to deduplicate
        key_col: Key column
        strategy: 'first', 'last', or dict of {col: agg_func} for groupby
        output_path: Optional path to save result
        
    Returns:
        Deduplicated DataFrame
    """
    print(f"\nDeduplicating by {key_col}...")
    print(f"  Original rows: {len(df):,}")
    
    if strategy in ['first', 'last']:
        df_dedup = df.drop_duplicates(subset=[key_col], keep=strategy)
        print(f"  Strategy: drop_duplicates(keep='{strategy}')")
    else:
        # Groupby aggregation
        print(f"  Strategy: groupby aggregation")
        df_dedup = df.groupby(key_col).agg(strategy).reset_index()
    
    print(f"  Deduplicated rows: {len(df_dedup):,}")
    print(f"  Removed: {len(df) - len(df_dedup):,} rows")
    
    if output_path:
        df_dedup.to_parquet(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    return df_dedup
