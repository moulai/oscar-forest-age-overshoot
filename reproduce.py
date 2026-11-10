#!/usr/bin/env python3
"""Show or explicitly execute the recorded study recipes; never launch on import."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
THREADS = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true')
    action.add_argument('--show', metavar='RECIPE')
    action.add_argument('--check', metavar='RECIPE')
    action.add_argument('--run', metavar='RECIPE')
    p.add_argument('--output-root', type=Path, default=Path('outputs/reproduced'))
    args = p.parse_args()
    recipes = json.loads((ROOT / 'reproduction/commands.json').read_text())['recipes']
    if args.list:
        for name, value in recipes.items():
            print(f'{name:31s} {value["description"]}')
        return 0
    name = args.show or args.check or args.run
    if name not in recipes:
        p.error('Unknown recipe: ' + name)
    recipe = recipes[name]
    values = {'root': str(ROOT), 'python': sys.executable, 'output': str(args.output_root)}
    steps = recipe['steps']
    if args.check:
        required = recipe.get('required_paths', []) + [s['argv'][1] for s in steps if len(s['argv']) > 1 and s['argv'][1].endswith('.py')]
        missing = sorted({x for x in required if not (ROOT / x.format(**values)).exists()})
        print(json.dumps({'recipe': name, 'status': 'missing_inputs' if missing else 'paths_available', 'missing': missing,
                          'scope': 'Existence only; no models, analyses, data hashing or field verification'}, indent=2))
        return 2 if missing else 0
    for step in steps:
        command = [part.format(**values) for part in step['argv']]
        env = dict(os.environ, **THREADS, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
        for key, value in step.get('env', {}).items():
            env[key] = value.format(**values)
        print(shlex.join(['env', 'PYTHONHASHSEED=0', 'PYTHONPATH=' + env['PYTHONPATH'], *command]), flush=True)
        if args.run:
            result = subprocess.run(command, cwd=ROOT, env=env)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
