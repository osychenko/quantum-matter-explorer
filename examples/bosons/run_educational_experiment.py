"""Patch 32 curated educational contract, independent of any web framework.

python3 -m examples.bosons.run_educational_experiment list
python3 -m examples.bosons.run_educational_experiment run ideal-boson-addition-v1 --beta 2.4
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from qmc.educational_contract import catalog, run_experiment, validate_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List curated MVP experiments')
    run = sub.add_parser('run', help='Produce a versioned experiment result')
    run.add_argument('preset_id')
    run.add_argument('--beta', type=float)
    run.add_argument('--cell-length', type=float)
    run.add_argument('--output', help='Optional JSON output file')
    args = parser.parse_args()
    if args.command == 'list':
        value = dict(schema_version='1.0.0', presets=catalog())
    else:
        params = {}
        if args.beta is not None: params['beta'] = args.beta
        if args.cell_length is not None: params['cell_length'] = args.cell_length
        value = run_experiment(args.preset_id, params)
        validate_result(value)
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(value, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
