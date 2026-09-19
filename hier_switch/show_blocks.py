"""Per-block table for finished tuning runs: .venv/bin/python hier_switch/show_blocks.py v3 [name ...]"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hier_switch_analyses import block_table, print_block_table

tag, names = sys.argv[1], sys.argv[2:]
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'exports', 'hier_switch', f'tune_{tag}')
for f in sorted(glob.glob(os.path.join(root, '*', 'trials.npz'))):
    name = os.path.basename(os.path.dirname(f))
    if names and name not in names:
        continue
    d = dict(np.load(f, allow_pickle=True))
    d['n'] = len(d['correct'])
    for ph in ('no inference learning', 'Learning and inference'):
        rows = block_table(d, ph)
        if rows:
            print_block_table(rows, f'{name} — {ph}')
