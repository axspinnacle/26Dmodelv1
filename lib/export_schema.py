"""
Export parquet file schema to CSV

Usage:
    python lib/export_schema.py <parquet_file> <output_csv>
"""

import pandas as pd
import pyarrow.parquet as pq
import sys
from pathlib import Path

def export_schema(parquet_file, output_csv):
    """Export column names and dtypes from parquet file to CSV"""
    
    print(f"Reading schema from: {parquet_file}")
    
    # Read just the schema (no data) using PyArrow
    parquet_file_obj = pq.ParquetFile(parquet_file)
    schema = parquet_file_obj.schema_arrow
    
    # Convert to pandas for easy CSV export
    columns = schema.names
    dtypes = [str(schema.field(name).type) for name in columns]
    
    # Create schema dataframe
    schema_data = []
    for col, dtype in zip(columns, dtypes):
        schema_data.append({
            'column_name': col,
            'dtype': dtype,
            'notes': ''
        })
    
    schema_df = pd.DataFrame(schema_data)
    
    # Save to CSV
    schema_df.to_csv(output_csv, index=False)
    
    print(f"Exported {len(schema_df)} columns to: {output_csv}")
    print(f"\nColumn types:")
    print(schema_df['dtype'].value_counts())
    
    return schema_df

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python lib/export_schema.py <parquet_file> <output_csv>")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    output_csv = sys.argv[2]
    
    export_schema(parquet_file, output_csv)
