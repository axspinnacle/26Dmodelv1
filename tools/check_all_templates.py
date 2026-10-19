import json
import ast
from pathlib import Path

def check_notebook(nb_path):
    """Check notebook for common syntax issues"""
    issues = []
    
    with open(nb_path) as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = cell.get('source', [])
        # Handle both string and list formats
        if isinstance(source, list):
            source = '\n'.join(source)
        elif not isinstance(source, str):
            source = str(source)
        if not source.strip():
            continue
        
        # Check 1: Basic Python syntax
        try:
            ast.parse(source)
        except SyntaxError as e:
            issues.append(f"Cell {i}: Syntax error - {e}")
            continue
        
        # Check 2: elif/else without if in same cell
        lines = source.split('\n')
        first_control = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('elif '):
                if first_control is None:
                    issues.append(f"Cell {i}: elif without preceding if in same cell")
                break
            elif stripped.startswith('else:'):
                if first_control is None:
                    issues.append(f"Cell {i}: else without preceding if in same cell")
                break
            elif stripped.startswith('if '):
                first_control = 'if'
        
        # Check 3: Common undefined variables (basic check)
        # Look for variables that should be defined earlier
        undefined_suspects = []
        if 'cfg[' in source and 'import yaml' not in source and i < 5:
            undefined_suspects.append('cfg (may not be defined yet)')
        if 'output_base' in source and 'output_base =' not in source and i < 5:
            undefined_suspects.append('output_base (may not be defined yet)')
        
        if undefined_suspects:
            issues.append(f"Cell {i}: Possible undefined: {', '.join(undefined_suspects)}")
    
    return issues

# Check all template notebooks
template_dir = Path('templates')
notebooks = sorted(template_dir.glob('*.ipynb'))

print('='*60)
print('TEMPLATE SYNTAX CHECKER')
print('='*60)

all_clean = True
for nb_path in notebooks:
    # Skip EDA (not part of main pipeline)
    if '00_eda' in nb_path.name:
        continue
    
    print(f'\n{nb_path.name}:')
    issues = check_notebook(nb_path)
    
    if issues:
        all_clean = False
        for issue in issues:
            print(f'  ✗ {issue}')
    else:
        print('  ✓ No issues found')

print('\n' + '='*60)
if all_clean:
    print('✓ ALL TEMPLATES PASSED')
else:
    print('✗ ISSUES FOUND - Review above')
print('='*60)
