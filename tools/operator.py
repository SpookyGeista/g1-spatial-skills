#!/usr/bin/env python3
"""Explicit operator controls for one active simulated mission."""
import argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path);p.add_argument('command',choices=['start','takeover','release','abort'])
a=p.parse_args();run=a.run.resolve()
if ROOT/'runs' not in run.parents:p.error('Choose a run inside this workspace')
try:s=json.loads((run/'status.json').read_text())
except (OSError,ValueError):p.error('No readable active mission status')
if time.monotonic()-s.get('monotonic',0)>2:p.error('Mission supervisor is stale; command not sent')
folder=run/'commands'
if not folder.is_dir():p.error('Missing existing mission command directory')
stamp=time.monotonic();temporary=folder/f'{stamp:.9f}.tmp'
temporary.write_text(json.dumps(dict(command=a.command,monotonic=stamp)))
temporary.replace(temporary.with_suffix('.json'))
print('Requested',a.command,'for',run,'; execution must be confirmed by runtime state.')
