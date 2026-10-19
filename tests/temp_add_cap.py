import json

for nb_file in ['templates/05a_model_initial.ipynb', 'templates/05b_hpo.ipynb', 'templates/05c_production.ipynb']:
    with open(nb_file) as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code' and 'Get actual PP' in ''.join(cell.get('source', [])):
            src = cell['source']
            # Find pp_test_actual line
            for i, line in enumerate(src):
                if 'pp_test_actual = test_orig[target]' in line:
                    # Insert capping after this line
                    src.insert(i+1, '\n')
                    src.insert(i+2, '# Apply target cap if configured\n')
                    src.insert(i+3, 'target_cap = cfg.get("experiment", {}).get("target_cap", None)\n')
                    src.insert(i+4, 'if target_cap:\n')
                    src.insert(i+5, '    n_cap_tr = (pp_train_actual > target_cap).sum()\n')
                    src.insert(i+6, '    n_cap_te = (pp_test_actual > target_cap).sum()\n')
                    src.insert(i+7, '    pp_train_actual = pp_train_actual.clip(upper=target_cap)\n')
                    src.insert(i+8, '    pp_test_actual = pp_test_actual.clip(upper=target_cap)\n')
                    src.insert(i+9, '    if n_cap_tr + n_cap_te > 0:\n')
                    src.insert(i+10, '        print(f"  Capped {n_cap_tr+n_cap_te:,} values at {target_cap:,}")\n')
                    print(f'Updated {nb_file}')
                    break
            break
    
    with open(nb_file, 'w') as f:
        json.dump(nb, f, indent=1)

print('Done')
