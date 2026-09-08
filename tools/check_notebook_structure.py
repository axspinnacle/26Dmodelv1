import json
from pathlib import Path

def check_notebook(nb_path):
    """Check for structural issues in notebook"""
    issues = []
    
    with open(nb_path) as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    # Track what variables are set/used
    variables_set = set()
    variables_used = set()
    save_cells = []
    
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue
            
        source = '\n'.join(cell.get('source', []))
        
        # Check for save operations
        if any(x in source for x in ['to_csv', 'to_parquet', 'yaml.dump', 'pickle.dump', 'joblib.dump']):
            save_cells.append(i)
        
        # Check for variable assignments
        for line in source.split('\n'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                var = line.split('=')[0].strip().split()[0] if ' ' in line.split('=')[0] else line.split('=')[0].strip()
                variables_set.add(var)
        
        # Check for specific suspicious patterns
        
        # Orphaned "Build parameters" cells
        if 'Build XGBoost parameters' in source or 'Building XGBoost parameters' in source:
            # Check if params are actually used
            next_cells = cells[i+1:i+3] if i+1 < len(cells) else []
            params_used = any('xgb.XGBRegressor' in '\n'.join(c.get('source', [])) for c in next_cells)
            if not params_used:
                issues.append(f"Cell {i}: Orphaned 'Build parameters' cell (params not used)")
        
        # Configuration cells after main processing
        if 'configuration' in source.lower() and 'cfg[' in source:
            if i > 10:  # Late in notebook
                issues.append(f"Cell {i}: Late configuration cell (should be early)")
        
        # Save cells before data is ready
        if 'to_csv' in source or 'to_parquet' in source or 'yaml.dump' in source:
            # Check if this is saving results or models
            if 'best_params' in source or 'results' in source or 'model' in source:
                # Look back to see if the data was created
                prev_source = '\n'.join('\n'.join(c.get('source', [])) for c in cells[:i])
                if 'best_params' in source and 'best_params =' not in prev_source:
                    issues.append(f"Cell {i}: Save cell before best_params is set")
                if 'results_df' in source and 'results_df =' not in prev_source:
                    issues.append(f"Cell {i}: Save cell before results_df is set")
                if 'model.save' in source or 'joblib.dump(model' in source:
                    if 'model.fit' not in prev_source:
                        issues.append(f"Cell {i}: Save model before training")
        
        # Duplicate imports
        if 'import' in source and i > 3:
            # Check if already imported
            prev_source = '\n'.join('\n'.join(c.get('source', [])) for c in cells[:i])
            imports = [line.strip() for line in source.split('\n') if 'import' in line and not line.strip().startswith('#')]
            for imp in imports:
                if imp in prev_source and 'yaml' not in imp:  # yaml might be reimported for safety
                    issues.append(f"Cell {i}: Duplicate import: {imp[:50]}...")
    
    return issues

# Check all templates
template_dir = Path('templates')
notebooks = sorted(template_dir.glob('*.ipynb'))

print('='*70)
print('NOTEBOOK STRUCTURE CHECKER')
print('='*70)

all_clean = True
for nb_path in notebooks:
    # Skip EDA
    if '00_eda' in nb_path.name:
        continue
    
    print(f'\n{nb_path.name}:')
    issues = check_notebook(nb_path)
    
    if issues:
        all_clean = False
        for issue in issues:
            print(f'  ✗ {issue}')
    else:
        print('  ✓ No structural issues')

print('\n' + '='*70)
if all_clean:
    print('✓ ALL TEMPLATES CLEAN')
else:
    print('✗ ISSUES FOUND - Review above')
print('='*70)
