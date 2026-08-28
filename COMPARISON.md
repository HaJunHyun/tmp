# Reproduction vs. paper

Our numbers are `avg3` — the mean of `evaluation/success` at 800k, 900k and 1M steps,
averaged over 5 tasks x 8 seeds per domain. Paper numbers are the RQL column of Table 1
in [arXiv:2606.17551](https://arxiv.org/abs/2606.17551) (4 seeds, 95% CI in brackets).

## Per domain

| domain | ours (5 tasks x 8 seeds) | paper (4 seeds) | diff |
|---|---|---|---|
| antmaze-large | 83.2 ± 7.2 | 83 [82, 84] | +0.2 |
| antmaze-giant | 37.0 ± 20.8 | 37 [33, 41] | -0.0 |
| humanoidmaze-medium | 96.5 ± 7.4 | 93 [84, 98] | +3.5 |
| humanoidmaze-large | 37.3 ± 22.5 | 39 [38, 41] | -1.7 |
| scene-sparse | 88.8 ± 12.4 | 89 [89, 90] | -0.2 |
| puzzle-3x3-sparse | 100.0 ± 0.0 | 100 [100, 100] | +0.0 |
| cube-double | 20.1 ± 15.5 | 23 [22, 23] | -2.9 |
| cube-triple | 3.2 ± 4.8 | 4 [3, 5] | -0.8 |
| **mean (8 domains)** | **58.3** | **58.5** | **-0.2** |

## Per task (ours / paper)

| domain | task1 | task2 | task3 | task4 | task5 |
|---|---|---|---|---|---|
| antmaze-large | 82.1 / 84 | 79.0 / 80 | 94.8 / 95 | 83.1 / 81 | 77.2 / 76 |
| antmaze-giant | 8.8 / 15 | 45.3 / 44 | 32.5 / 21 | 36.2 / 35 | 62.1 / 69 |
| humanoidmaze-medium | 97.8 / 96 | 99.2 / 99 | 99.1 / 99 | 87.1 / 72 | 99.6 / 99 |
| humanoidmaze-large | 70.5 / 76 | 4.2 / 4 | 31.8 / 36 | 41.2 / 42 | 38.7 / 37 |
| scene-sparse | 99.4 / 100 | 73.6 / 72 | 96.2 / 96 | 99.4 / 100 | 75.3 / 79 |
| puzzle-3x3-sparse | 100.0 / 100 | 100.0 / 100 | 100.0 / 100 | 100.0 / 100 | 100.0 / 100 |
| cube-double | 45.3 / 51 | 28.3 / 25 | 11.9 / 19 | 5.2 / 6 | 9.6 / 12 |
| cube-triple | 11.9 / 11 | 0.6 / 1 | 1.0 / 1 | 0.2 / 0 | 2.3 / 5 |

## Not run — the two 100M-dataset domains

Excluded from this sweep because their tuned configs need the 100M-transition datasets,
which the unreachable official host serves and the HF mirror does not carry.
Paper RQL scores, for reference:

| domain | paper agg | task1 | task2 | task3 | task4 | task5 |
|---|---|---|---|---|---|---|
| puzzle-4x4-100M-sparse | 37 [34, 39] | 64 | 26 | 32 | 40 | 21 |
| cube-quadruple-100M | 51 [47, 55] | 87 | 81 | 62 | 25 | 0 |

Paper mean over all 10 domains: **55.6**.

## Caveats

- **Seeds.** Paper uses 4 seeds and reports 95% confidence intervals; this sweep uses 8
  seeds and reports standard deviation, so the spreads are not directly comparable.
- **Training length.** Table 2 of the paper lists "Gradient steps 2M", but the authors'
  own `hyperparameters.sh` — what this sweep runs — uses `--offline_steps=1000000`, and
  the paper's BDPO footnote states BDPO "uses twice as many offline steps as RQL",
  implying 1M for RQL. This sweep is 1M. The agreement below supports that reading.
- Hyperparameters are the per-domain values from upstream `hyperparameters.sh`, unchanged.
