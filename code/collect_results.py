#!/usr/bin/env python
"""Aggregate sweep results.

Two metrics per run, from <run>/**/eval.csv:
  avg3  = mean of evaluation/success at 800k, 900k and 1M steps
  final = evaluation/success at 1M steps

    python collect_results.py                       # per-domain table
    python collect_results.py --by task             # per domain x task
    python collect_results.py --csv out.csv         # also dump per-run rows
"""
import argparse
import csv
import glob
import os
import statistics as st

DEFAULT_STEPS = [800_000, 900_000, 1_000_000]


def read_run(run_dir, metric, steps):
    hits = sorted(glob.glob(os.path.join(run_dir, '**', 'eval.csv'), recursive=True))
    if not hits:
        return None
    with open(hits[-1]) as f:
        rows = list(csv.DictReader(f))
    if not rows or metric not in rows[0] or 'step' not in rows[0]:
        return None
    by_step = {int(float(r['step'])): float(r[metric]) for r in rows if r.get(metric) not in (None, '')}
    if not all(s in by_step for s in steps):
        return None
    return dict(avg3=sum(by_step[s] for s in steps) / len(steps), final=by_step[steps[-1]])


def fmt(vals):
    if not vals:
        return '     -    '
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f'{m*100:5.1f}±{s*100:4.1f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='/workspace/rql_results')
    ap.add_argument('--metric', default='evaluation/success')
    ap.add_argument('--steps', default=','.join(str(s) for s in DEFAULT_STEPS))
    ap.add_argument('--by', default='domain', choices=['domain', 'task'])
    ap.add_argument('--csv', default='')
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(',')]

    rows, missing = [], []
    for run_dir in sorted(glob.glob(os.path.join(args.results, 'runs', '*'))):
        rid = os.path.basename(run_dir)
        try:
            domain, task, seed = rid.split('__')
        except ValueError:
            continue
        r = read_run(run_dir, args.metric, steps)
        if r is None:
            missing.append(rid)
            continue
        rows.append(dict(run_id=rid, domain=domain, task=task, seed=seed, **r))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['run_id', 'domain', 'task', 'seed', 'avg3', 'final'])
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {len(rows)} rows to {args.csv}')

    key = (lambda r: r['domain']) if args.by == 'domain' else (lambda r: f"{r['domain']} {r['task']}")
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    width = max([len(k) for k in groups] + [10])
    print(f"\n{args.metric}  (%, mean±std over seeds)")
    print(f"{'group':{width}s} {'n':>3s} {'avg3(800/900/1M)':>17s} {'final(1M)':>12s}")
    print('-' * (width + 36))
    for k in sorted(groups):
        g = groups[k]
        print(f"{k:{width}s} {len(g):3d} {fmt([x['avg3'] for x in g]):>17s} {fmt([x['final'] for x in g]):>12s}")
    if rows:
        print('-' * (width + 36))
        print(f"{'ALL':{width}s} {len(rows):3d} {fmt([x['avg3'] for x in rows]):>17s} "
              f"{fmt([x['final'] for x in rows]):>12s}")
    if missing:
        print(f'\n{len(missing)} run(s) incomplete/missing: ' + ' '.join(missing[:10]) +
              (' ...' if len(missing) > 10 else ''))


if __name__ == '__main__':
    main()
