#!/usr/bin/env python
"""Run the full RQL sweep: 8 domains x 5 tasks x 8 seeds = 320 runs.

Resumable: runs that already finished (DONE marker + eval.csv containing the
final step) are skipped, so re-running this script continues where it stopped.

    python launch_sweep.py                 # run the sweep
    python launch_sweep.py --dry_run       # print the matrix and the commands
    python launch_sweep.py --status        # progress of an existing sweep
    python launch_sweep.py --only humanoidmaze-large-navigate --seeds 0,1
"""
import argparse
from itertools import zip_longest
import glob
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.abspath(__file__))

# Tuned hyperparameters, from hyperparameters.sh (the two 100M-dataset domains
# -- puzzle-4x4-play and cube-quadruple-play -- are deliberately excluded).
DOMAINS = {
    'scene-play':                   dict(alpha=3.0,  expectile=0.7, rho=0.5, h=5, sparse=True),
    'puzzle-3x3-play':              dict(alpha=1.0,  expectile=0.7, rho=0.5, h=5, sparse=True),
    'cube-double-play':             dict(alpha=10.0, expectile=0.9, rho=0.5, h=5),
    'cube-triple-play':             dict(alpha=1.0,  expectile=0.9, rho=0.5, h=5),
    'antmaze-large-navigate':       dict(alpha=0.1,  expectile=0.5, rho=0.5, h=1),
    'antmaze-giant-navigate':       dict(alpha=0.1,  expectile=0.5, rho=0.5, h=1, discount=0.995),
    'humanoidmaze-medium-navigate': dict(alpha=0.3,  expectile=0.5, rho=0.0, h=1, discount=0.995),
    'humanoidmaze-large-navigate':  dict(alpha=0.3,  expectile=0.5, rho=0.0, h=1, discount=0.995),
}
TASKS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

OFFLINE_STEPS = 1_000_000
EVAL_AT = [800_000, 900_000, 1_000_000]
EVAL_EPISODES = 50


def build_matrix(only=None, tasks=None, seeds=None):
    runs = []
    for dom, hp in DOMAINS.items():
        if only and dom not in only:
            continue
        for t in (tasks or TASKS):
            for sd in (seeds or SEEDS):
                runs.append(dict(run_id=f'{dom}__task{t}__sd{sd}', domain=dom, task=t, seed=sd, hp=hp))
    return runs


def run_cmd(run, args):
    hp = run['hp']
    out_dir = os.path.join(args.results, 'runs', run['run_id'])
    cmd = [
        sys.executable, os.path.join(REPO, 'main.py'),
        '--agent', os.path.join(REPO, 'agents/rql.py'),
        '--env_name', f"{run['domain']}-singletask-task{run['task']}-v0",
        '--seed', str(run['seed']),
        '--run_group', f"{run['domain']}_task{run['task']}",
        '--save_dir', out_dir,
        '--offline_steps', str(args.offline_steps),
        '--online_steps', '0',
        '--buffer_size', '1',              # buffer == dataset size; avoids the 100M-row allocation
        '--eval_at', ','.join(str(s) for s in args.eval_at),
        '--eval_episodes', str(args.eval_episodes),
        '--log_interval', '10000',
        '--agent.alpha', str(hp['alpha']),
        '--agent.expectile', str(hp['expectile']),
        '--agent.rho', str(hp['rho']),
        '--agent.h', str(hp['h']),
        '--agent.ensemble_ct', '10',
        '--agent.batch_size', '256',
    ]
    if 'discount' in hp:
        cmd += ['--agent.discount', str(hp['discount'])]
    if hp.get('sparse'):
        cmd += ['--sparse']
    return cmd, out_dir


def child_env(gpu, args):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAX_COMPILATION_CACHE_DIR=args.cache_dir,
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES='0',
        XLA_FLAGS='--xla_gpu_force_compilation_parallelism=1',
        WANDB_MODE=args.wandb,
        OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',
        PYTHONUNBUFFERED='1',
    )
    return env


def eval_csv_of(out_dir):
    hits = glob.glob(os.path.join(out_dir, '**', 'eval.csv'), recursive=True)
    return sorted(hits)[-1] if hits else None


def is_complete(out_dir, final_step):
    """A run counts as finished once its eval.csv holds a row for the final step."""
    path = eval_csv_of(out_dir)
    if path is None:
        return False
    try:
        with open(path) as f:
            header = f.readline().rstrip('\n').split(',')
            if 'step' not in header:
                return False
            si = header.index('step')
            for line in f:
                parts = line.rstrip('\n').split(',')
                if len(parts) > si and parts[si] and int(float(parts[si])) == final_step:
                    return True
    except (OSError, ValueError):
        return False
    return False


class Sweep:
    def __init__(self, args, runs):
        self.args = args
        self.runs = runs
        self.gpus = args.gpus
        self.gpu_load = {g: 0 for g in self.gpus}
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.attempts = {}
        self.done, self.failed, self.skipped = [], [], []
        self.active = {}
        self.stop = threading.Event()
        self.t0 = time.time()

    def pick_gpu(self):
        return min(self.gpu_load, key=lambda g: (self.gpu_load[g], g))

    def worker(self):
        while not self.stop.is_set():
            try:
                run = self.q.get(timeout=1)
            except queue.Empty:
                return
            self.execute(run)
            self.q.task_done()

    def execute(self, run):
        args = self.args
        cmd, out_dir = run_cmd(run, args)
        os.makedirs(out_dir, exist_ok=True)
        with self.lock:
            gpu = self.pick_gpu()
            self.gpu_load[gpu] += 1
            self.active[run['run_id']] = (gpu, time.time())
        log_path = os.path.join(args.results, 'logs', run['run_id'] + '.log')
        try:
            with open(log_path, 'a') as lf:
                lf.write(f"\n=== attempt {self.attempts.get(run['run_id'], 0) + 1} gpu={gpu} "
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                lf.write(' '.join(cmd) + '\n')
                lf.flush()
                p = subprocess.Popen(cmd, cwd=REPO, env=child_env(gpu, args),
                                     stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                with self.lock:
                    self.active[run['run_id']] = (gpu, time.time(), p)
                rc = p.wait()
        finally:
            with self.lock:
                self.gpu_load[gpu] -= 1
                self.active.pop(run['run_id'], None)

        ok = rc == 0 and is_complete(out_dir, args.offline_steps)
        n = self.attempts[run['run_id']] = self.attempts.get(run['run_id'], 0) + 1
        if ok:
            open(os.path.join(out_dir, 'DONE'), 'w').write(str(time.time()))
            with self.lock:
                self.done.append(run['run_id'])
            self.report(f"done   {run['run_id']}")
        elif n <= args.retries and not self.stop.is_set():
            self.report(f"retry  {run['run_id']} (rc={rc}, attempt {n})")
            self.q.put(run)
        else:
            with self.lock:
                self.failed.append(run['run_id'])
            self.report(f"FAILED {run['run_id']} (rc={rc}) -> {log_path}")

    def report(self, msg):
        total = len(self.runs) + len(self.skipped)
        finished = len(self.done) + len(self.skipped)
        el = time.time() - self.t0
        rate = len(self.done) / el if self.done else 0
        eta = (len(self.runs) - len(self.done)) / rate / 3600 if rate else float('nan')
        print(f"[{finished:3d}/{total} | fail {len(self.failed)} | "
              f"{el/3600:5.2f}h elapsed | ETA {eta:5.2f}h] {msg}", flush=True)
        self.write_status()

    def write_status(self):
        with self.lock:
            st = dict(
                started=self.t0, now=time.time(),
                total=len(self.runs) + len(self.skipped),
                done=len(self.done) + len(self.skipped), failed=len(self.failed),
                running=sorted(self.active.keys()),
                failed_ids=self.failed,
                gpu_load=self.gpu_load,
            )
        tmp = os.path.join(self.args.results, 'status.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, os.path.join(self.args.results, 'status.json'))

    def shutdown(self, *_):
        print('\n[sweep] stopping: no new runs will start, killing active ones...', flush=True)
        self.stop.set()
        with self.lock:
            procs = [v[2] for v in self.active.values() if len(v) > 2]
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


def warmup(args, domains):
    """Populate the XLA persistent cache once per domain.

    Without this, dozens of processes JIT-compile at the same time, ptxas fork
    storms, and a large fraction of them die with "Failed to launch ptxas".
    """
    print(f'[sweep] warming compile cache for {len(domains)} domains ...', flush=True)
    procs = []
    for i, dom in enumerate(domains):
        run = dict(run_id=f'warmup__{dom}', domain=dom, task=1, seed=0, hp=DOMAINS[dom])
        wargs = argparse.Namespace(**vars(args))
        wargs.offline_steps, wargs.eval_at, wargs.eval_episodes = 10, [10], 1
        wargs.results = os.path.join(args.results, 'warmup')
        cmd, out_dir = run_cmd(run, wargs)
        os.makedirs(out_dir, exist_ok=True)
        gpu = args.gpus[i % len(args.gpus)]
        lf = open(os.path.join(args.results, 'logs', f'warmup__{dom}.log'), 'w')
        procs.append((dom, subprocess.Popen(cmd, cwd=REPO, env=child_env(gpu, args),
                                            stdout=lf, stderr=subprocess.STDOUT)))
        time.sleep(args.stagger)
    bad = [d for d, p in procs if p.wait() != 0]
    if bad:
        print(f'[sweep] WARNING: warmup failed for {bad} (see {args.results}/logs/warmup__*.log)', flush=True)
    else:
        print('[sweep] compile cache warm.', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='/workspace/rql_results')
    ap.add_argument('--concurrency', type=int, default=48, help='total concurrent runs')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--stagger', type=float, default=2.0, help='seconds between process launches')
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--offline_steps', type=int, default=OFFLINE_STEPS)
    ap.add_argument('--eval_at', default=','.join(str(s) for s in EVAL_AT))
    ap.add_argument('--eval_episodes', type=int, default=EVAL_EPISODES)
    ap.add_argument('--cache_dir', default='/workspace/.jax_cache')
    ap.add_argument('--wandb', default='disabled', choices=['disabled', 'offline', 'online'])
    ap.add_argument('--only', default='', help='comma-separated domains')
    ap.add_argument('--tasks', default='')
    ap.add_argument('--seeds', default='')
    ap.add_argument('--no_warmup', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()

    args.gpus = [int(g) for g in args.gpus.split(',') if g != '']
    args.eval_at = [int(s) for s in args.eval_at.split(',')]
    only = [d for d in args.only.split(',') if d] or None
    tasks = [int(t) for t in args.tasks.split(',') if t] or None
    seeds = [int(s) for s in args.seeds.split(',') if s] or None
    if only:
        unknown = set(only) - set(DOMAINS)
        if unknown:
            sys.exit(f'unknown domain(s): {sorted(unknown)}')

    runs = build_matrix(only, tasks, seeds)
    os.makedirs(os.path.join(args.results, 'logs'), exist_ok=True)

    pending, skipped = [], []
    for r in runs:
        out_dir = os.path.join(args.results, 'runs', r['run_id'])
        (skipped if is_complete(out_dir, args.offline_steps) else pending).append(r)

    # Interleave domains so every concurrency wave carries a mix of light and
    # heavy configs instead of 48 copies of the same one.
    by_dom = {}
    for r in pending:
        by_dom.setdefault(r['domain'], []).append(r)
    pending = [r for grp in zip_longest(*by_dom.values()) for r in grp if r is not None]

    if args.status:
        print(f'{len(skipped)}/{len(runs)} complete, {len(pending)} remaining')
        sp = os.path.join(args.results, 'status.json')
        if os.path.exists(sp):
            print(json.dumps(json.load(open(sp)), indent=1)[:2000])
        return

    print(f'[sweep] {len(runs)} runs total | {len(skipped)} already complete | {len(pending)} to run')
    print(f'[sweep] concurrency={args.concurrency} gpus={args.gpus} results={args.results}')
    if args.dry_run:
        for r in pending[:3] + (['...'] if len(pending) > 3 else []):
            print(' '.join(run_cmd(r, args)[0]) if r != '...' else '...')
        print(f'[dry run] {len(pending)} runs would be launched')
        return
    if not pending:
        print('[sweep] nothing to do'); return

    if not args.no_warmup:
        warmup(args, sorted({r['domain'] for r in pending}))

    sweep = Sweep(args, pending)
    sweep.skipped = [r['run_id'] for r in skipped]
    signal.signal(signal.SIGINT, sweep.shutdown)
    signal.signal(signal.SIGTERM, sweep.shutdown)
    for r in pending:
        sweep.q.put(r)

    threads = []
    for _ in range(min(args.concurrency, len(pending))):
        t = threading.Thread(target=sweep.worker, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(args.stagger)
    for t in threads:
        t.join()

    sweep.write_status()
    print(f'\n[sweep] finished: {len(sweep.done)} ok, {len(sweep.failed)} failed, '
          f'{(time.time() - sweep.t0)/3600:.2f}h elapsed')
    if sweep.failed:
        print('failed runs:', ' '.join(sweep.failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
