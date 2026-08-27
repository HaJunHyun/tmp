# RQL sweep — 8 domains × 5 tasks × 8 seeds

Results and run tooling for a reproduction sweep of **Reversal Q-Learning**
([arXiv:2606.17551](https://arxiv.org/abs/2606.17551), code: [aoberai/rql](https://github.com/aoberai/rql)),
on the 8 OGBench domains that do **not** need the 100M-sized datasets.

Live progress: [PROGRESS.md](PROGRESS.md).

## Sweep

| | |
|---|---|
| domains | scene-play, puzzle-3x3-play, cube-double-play, cube-triple-play, antmaze-large-navigate, antmaze-giant-navigate, humanoidmaze-medium-navigate, humanoidmaze-large-navigate |
| tasks | `task1`–`task5` per domain |
| seeds | 0–7 |
| runs | 8 × 5 × 8 = **320** |
| steps | 1M offline gradient steps, 0 online |
| eval | 50 episodes at 800k / 900k / 1M steps |
| hardware | 8× RTX 4090, 48 concurrent runs |

Excluded: `puzzle-4x4-play` and `cube-quadruple-play`, whose tuned configs in the
upstream `hyperparameters.sh` require the 100M-transition datasets.

Hyperparameters are the tuned per-domain values from upstream `hyperparameters.sh`
(`--sparse` for scene-play / puzzle-3x3-play; `discount=0.995` for antmaze-giant and
both humanoidmaze domains); see the `DOMAINS` table in `code/launch_sweep.py`.

## Metrics

Two numbers per run, from `evaluation/success` in each run's `eval.csv`:

- **avg3** — mean of the evaluations at 800k, 900k and 1M steps
- **final** — the 1M-step evaluation

## Layout

```
results/runs/<domain>__task<t>__sd<seed>/    eval.csv, train.csv, flags.json
results/per_run.csv                          both metrics, one row per run
results/summary_by_domain.txt                mean±std over seeds, per domain
results/summary_by_task.txt                  mean±std over seeds, per domain×task
results/status.json                          launcher progress snapshot
code/launch_sweep.py                         320-run orchestrator (resumable)
code/collect_results.py                      aggregator
code/download_ogbench_data.py                dataset fetcher (HF mirror)
code/patches/rql-compat.patch                changes applied to upstream rql
code/SETUP_NOTES.md                          environment setup notes
code/requirements.lock.txt                   pinned environment
sync.sh                                      refresh this tree from the live sweep
```

## Reproducing

```bash
git clone https://github.com/aoberai/rql && cd rql
git checkout 229c956efb4494c2b9bb0bbddbd67b761c93f1cc
git apply /path/to/code/patches/rql-compat.patch
pip install -r /path/to/code/requirements.lock.txt
python /path/to/code/download_ogbench_data.py scene-play-v0 ...    # datasets
python /path/to/code/launch_sweep.py --concurrency=48
python /path/to/code/collect_results.py
```

The patch covers four things needed to run upstream as-is on a current stack:
numpy 2.x dropped `np.reshape(..., newshape=)`; `setup_wandb` now honours
`WANDB_MODE`; and `main.py` gains `--eval_at` so evaluation happens only at the
three steps this protocol reports. See `code/SETUP_NOTES.md` for the full list,
including the wandb version pin and the dataset mirror (the official host
`rail.eecs.berkeley.edu` was unreachable).
