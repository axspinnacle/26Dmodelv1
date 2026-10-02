import json

nb = json.load(open('templates/02_data_conditioning.ipynb'))

print(f'Total cells: {len(nb["cells"])}')

# Find insertion point
for i in range(len(nb['cells'])-1, -1, -1):
    src = ''.join(nb['cells'][i].get('source', []))
    if 'COMPLETE' in src or '###' in src[:10]:
        insert_at = i
        print(f'Inserting at {i}')
        break

# Create cell
new = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# Apply exposure floor\\n',
        'exposure_floor = cfg.get("experiment", {}).get("exposure_floor", None)\\n',
        'exposure_cols = cfg.get("experiment", {}).get("exposure_columns", ["ee_bi_imps"])\\n',
        '\\n',
        'if exposure_floor and exposure_cols:\\n',
        '    print(f"\\\\n* Applying exposure floor ({exposure_floor})...")\\n',
        '    from data_quality import apply_exposure_floor\\n',
        '    df = apply_exposure_floimport json

nb = json.load(open('templates/02_data_conditioning.ipynbpr
nb = jsontal
print(f'Total cells: {len(nb["cells"])}')

# Find insertio   
# Find insertion point
for i in range(lnotfor i in range(len(nb      src = ''.join(nb['cells'][i].get('sourit    if 'COMPLETE' in src or '###' in src[:10]:
   )         insert_at = i
        print(f'Inserti'I        print(f'Inseer        break)
