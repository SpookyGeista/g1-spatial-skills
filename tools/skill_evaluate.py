#!/usr/bin/env python3
"""Evaluate a finalized actual skill trial; never alter its outcome or trace."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from punktnull.skills.evaluate import evaluate

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--output-name',default='skill_evaluation.json');args=parser.parse_args()
    if Path(args.output_name).name!=args.output_name or not args.output_name.endswith('.json'):
        parser.error('Use one JSON filename')
    output=args.run/args.output_name
    if output.exists():parser.error('Evaluation already exists; preserve it')
    result=evaluate(args.run)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2));return 0 if result['accepted'] else 2

if __name__=='__main__':raise SystemExit(main())
