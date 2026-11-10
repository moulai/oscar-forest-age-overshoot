#!/usr/bin/env python3
"""Rebuild ED5 display tables using the existing birth-source preparation function."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.plot_birth_source_age_response import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.analysis_dir, args.output)
    print('Prepared existing-definition ED5 Source Data:', args.output)


if __name__ == '__main__':
    main()
