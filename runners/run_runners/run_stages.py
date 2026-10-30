#!/usr/bin/env python
"""
run_stages.py - run selected pipeline stages for selected models (no full runner).

WHAT IT DOES
  Runs templates/<stage>_*.ipynb via papermill for each model, e.g. stage 09a-09d
  for all 16 v3 models after the main pipeline has finished. Output path and
  kernel come from each model's config.yaml for this machine (current.pc).
  Executed notebooks go to <output_path>/notebooks/ (same place as the runners).
  The stage_xx true/false switches in config.yaml are ignored (you pick the stages).
  A failed stage is logged and the run continues (unless --stop-on-error).

ENVIRONMENT
  Activate the conda env first (papermill + pyyaml needed):
    conda activate carfax26v1      # PC2 (prod)
    conda activate py311_26v1      # PC3 (Mac)

USAGE (from anywhere; paths are resolved from the project root)
  python runners/run_runners/run_stages.py <version> --stages <stage ...> [options]

ARGUMENTS
  <version>              Config version, e.g. v3 -> config/<model>/v3/config.yaml
  --stages S [S ...]     Stages to run, in this order. Examples: 09a 09b, 07a.
                         A prefix runs all matching stages: 09 -> 09a 09b 09c 09d,
                         07 -> 07a 07b, 04 -> 04a 04b 04c. An exact name runs only
                         that stage (08 -> 08 only; use 08 08b for both).
  --models M [M ...]     Models to run, in this order (default: every
                         config/*/<version>/config.yaml, alphabetical).
  --dry-run              Show machine, models, stages, paths; run nothing.
  --stop-on-error        Stop at the first failed stage.

OUTPUT
  logs/stages_<version>_<YYYYmmdd_HHMMSS>/
    run.log                      start/end/status/duration per model+stage + summary
    <model>_<stage>.log          papermill console output

EXAMPLES
  python runners/run_runners/run_stages.py v3 --stages 09 --dry-run
  python runners/run_runners/run_stages.py v3 --stages 09
  python runners/run_runners/run_stages.py v3 --stages 09a 09b --models car_coll suv_pip
  python runners/run_runners/run_stages.py v3 --stages 07a 07b

  # in background - keeps running if the terminal is closed
  nohup python runners/run_runners/run_stages.py v3 --stages 09 > /dev/null 2>&1 &
  tail -f logs/stages_v3_*/run.log
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


def find_root():
    """Project root = first folder upward (from cwd, then this file) containing current.pc."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for p in (start, *start.parents):
            if (p / 'current.pc').exists():
                return p
    sys.exit('ERROR: current.pc not found (run inside the project)')


def resolve_stages(root, wanted):
    """Stage names / prefixes -> [(stage, template path)], e.g. '09' -> 09a..09d."""
    all_stages = {t.name.split('_')[0]: t for t in sorted((root / 'templates').glob('[0-9][0-9]*.ipynb'))}
    out = []
    for w in wanted:
        hits = [w] if w in all_stages else [s for s in all_stages if s.startswith(w)]
        if not hits:
            sys.exit(f'ERROR: no template for stage "{w}" (available: {" ".join(sorted(all_stages))})')
        out += [(s, all_stages[s]) for s in hits if s not in [x for x, _ in out]]
    return out


def fmt(sec):
    return f'{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m{int(sec % 60):02d}s'


def model_jobs(root, version, models, pc_id):
    """[(model, config dir, output path, kernel)] for this machine + list of models without config."""
    jobs, missing = [], []
    for m in models:
        cfg_file = root / 'config' / m / version / 'config.yaml'
        if not cfg_file.exists():
            missing.append(m)
            continue
        mc = yaml.safe_load(cfg_file.read_text())['machines'].get(pc_id) or {}
        out = Path(mc.get('paths', {}).get('output_path', ''))
        out = out if out.is_absolute() else root / out
        jobs.append((m, cfg_file.parent, out, mc.get('conda_env', 'python3')))
    return jobs, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('version')
    ap.add_argument('--stages', nargs='+', required=True)
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--stop-on-error', action='store_true')
    a = ap.parse_args()

    root = find_root()
    pc_id = f"PC{(root / 'current.pc').read_text().strip()}"
    stages = resolve_stages(root, a.stages)
    models = a.models or sorted(p.parent.parent.name for p in (root / 'config').glob(f'*/{a.version}/config.yaml'))
    jobs, missing = model_jobs(root, a.version, models, pc_id)
    if not jobs:
        sys.exit(f'ERROR: no config/<model>/{a.version}/config.yaml found for {models}')

    print(f'Project root : {root}')
    print(f'Machine      : {pc_id}')
    print(f'Python       : {sys.executable}')
    print(f'Stages       : {" ".join(s for s, _ in stages)}')
    print(f'Models ({len(jobs)})  : {" ".join(m for m, *_ in jobs)}')
    if missing:
        print(f'NOT FOUND    : {" ".join(missing)} (no config/<model>/{a.version}/config.yaml) - skipped')
    if a.dry_run:
        for m, _, out, kernel in jobs:
            note = '' if out.exists() else '  [output path missing]'
            print(f'  {m:12s} kernel={kernel:12s} notebooks -> {out}/notebooks/{note}')
        print('Dry run - nothing executed.')
        return

    import papermill as pm
    log_dir = root / 'logs' / f'stages_{a.version}_{datetime.now():%Y%m%d_%H%M%S}'
    log_dir.mkdir(parents=True)
    run_log = open(log_dir / 'run.log', 'a')

    def log(msg):
        line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
        print(line, flush=True)
        run_log.write(line + '\n')
        run_log.flush()

    log(f'START version={a.version} machine={pc_id} stages={" ".join(s for s, _ in stages)} models={len(jobs)}')
    summary, t_all = [], time.time()
    for m, cfg_dir, out, kernel in jobs:
        for stage, tmpl in stages:
            log(f'{m} {stage}: START')
            t0 = time.time()
            (out / 'notebooks').mkdir(parents=True, exist_ok=True)
            try:
                with open(log_dir / f'{m}_{stage}.log', 'w') as fh:
                    pm.execute_notebook(str(tmpl), str(out / 'notebooks' / tmpl.name),
                                        parameters={'config_path': str(cfg_dir.relative_to(root))},
                                        kernel_name=kernel, cwd=str(root), log_output=True,
                                        stdout_file=fh, stderr_file=fh, progress_bar=False)
                status = 'OK'
            except Exception as e:
                status = 'FAILED'
                msg = str(e).strip().splitlines()
                log(f'{m} {stage}: {type(e).__name__}: {msg[-1][:200] if msg else ""}')
            dur = fmt(time.time() - t0)
            log(f'{m} {stage}: {status} ({dur})')
            summary.append((m, stage, status, dur))
            if status != 'OK' and a.stop_on_error:
                break
        if summary and summary[-1][2] != 'OK' and a.stop_on_error:
            log('Stopping (--stop-on-error)')
            break

    n_bad = sum(s != 'OK' for _, _, s, _ in summary)
    log(f'END total {fmt(time.time() - t_all)} | {len(summary) - n_bad} OK, {n_bad} FAILED | logs: {log_dir}')
    lines = ['', 'SUMMARY', f'{"model":12s} {"stage":6s} {"status":7s} duration']
    lines += [f'{m:12s} {s:6s} {st:7s} {d}' for m, s, st, d in summary]
    print('\n'.join(lines))
    run_log.write('\n'.join(lines) + '\n')
    run_log.close()
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
