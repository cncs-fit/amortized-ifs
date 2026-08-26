# Amortized Set Prediction for Inverse IFS Reconstruction from Density Maps

Code and checkpoints to reproduce the results of

> *Amortized Set Prediction for Inverse IFS Reconstruction from Density Maps*
> ([arXiv:2608.24175](https://arxiv.org/abs/2608.24175))

A feed-forward estimator predicts the affine-map set of an Iterated Function
System (IFS) from a visit-frequency density map in a single forward pass,
replacing per-image optimization. Training pairs are self-generated from the
fully known forward model; an optional test-time refinement polishes the
one-shot prediction.

## Layout

```
data/, models/, losses/   core library (sampler, renderers, estimator, losses)
scripts/                  training, evaluation, and analysis entry points
configs/                  the four training configurations used in the paper
checkpoints/              released model checkpoints (fetched from GitHub Releases; see below)
tests/                    unit tests
third_party/              instructions for fetching Tu et al.'s public code
```

All commands below are run from this directory. Scripts write to `outputs/`
(created on demand) and training caches to `cache/`.

## Setup

```bash
conda env create -f environment.yml
conda activate ifs-amortized
python -m unittest discover -s tests   # smoke test (CPU, ~1 min)
```

A single NVIDIA RTX 3090 (24 GB) was used for all experiments.

Reproducibility: evaluation protocols are fully seeded, but GPU training is not
bitwise deterministic, so the released checkpoints are the canonical artifacts
behind the paper's numbers. Zero-step (one-shot) metrics reproduce exactly at
the paper's displayed precision; metrics after gradient-based refinement
reproduce to about three significant digits.

## Released checkpoints

The checkpoints (~120 MB) are distributed as a GitHub Releases asset rather
than tracked in git. Fetch and extract them at the repository root:

```bash
curl -L -O https://github.com/cncs-fit/amortized-ifs/releases/download/v1.0/checkpoints.tar.gz
tar xzf checkpoints.tar.gz    # creates checkpoints/
# sha256: 97a4a949c8db3883c904ded5a69c1d24b325183a70f7d92cebbc6144867ced92
```

| Directory | Contents | Used for |
|---|---|---|
| `checkpoints/paper_base50k` | n=4 estimator, 50k steps, best-val (`best_model.pt`) | all main experiments |
| `checkpoints/paper50k_matchonly3k` | +3k match-only continuation, final (`model.pt`) | auxiliary-loss ablation |
| `checkpoints/paper50k_md4_3k` | +3k with reconstruction auxiliary, final (`model.pt`) | auxiliary-loss ablation |
| `checkpoints/paper_base50k_seed3026`, `..._seed4026` | same recipe, other training seeds | seed-robustness paragraph |
| `checkpoints/explore_n10_base100k` | n=10 estimator, 100k steps, best-val | MNIST / Fashion-MNIST |

Each entry is a minimal training-run directory (`result.json` + checkpoint)
and can be passed directly as `--run-dir`.

## Reproducing the paper's results

Overview of which command reproduces which paper result (details in the
sections below; approximate GPU runtimes on an RTX 3090 in parentheses):

| Paper result | Entry point |
|---|---|
| Table 2, Figures 2--3, A.1 (quality--speed Pareto) | `run_paper50k_e1_hero.sh` |
| Table 3, Figure 4 (long-horizon convergence) | `run_advice027_long_horizon.sh` |
| Table 4 (reconstruction-auxiliary ablation) | `run_recon_aux_from_checkpoints.sh` |
| Seed-robustness paragraph (Section 6.3) | `run_advice028_seed_robustness.sh` |
| Table 5, Figure 5 (within-family OOD) | `run_ood_fractal_eval.sh` |
| Table 1, Figure 1 (identifiability) | `analyze_identifiability.py` |
| Tables 6--7, Figures 6--7, B.1--B.4 (MNIST / Fashion-MNIST vs Tu) | see the MNIST section |

### Central result: quality-speed Pareto (hero table and figures)

```bash
bash scripts/run_paper50k_e1_hero.sh          # (~1-2 h)
```

Evaluates the base estimator at 0/10/20/30 refinement steps and the
random-initialized baseline (4 restarts) at 0/10/20/30/60 steps on the fixed
`test256` set. Aggregated outputs under
`outputs/oracle/paper50k_e1_hero_pareto_test256_chamfer010_p512_robust2048/`:

- `summary.csv` -- the table rows
- the quality-speed figures
- `per_sample_wins.csv` -- per-sample win counts

The `(W,b)` diagnostic column is produced from the same runs:

```bash
python scripts/summarize_oracle_param_error.py \
  --run base50k_best-0 outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512 \
  --run base50k_best-10 outputs/oracle/paper_base50k_fresh_best_s10_test256_chamfer010_p512 \
  --run base50k_best-20 outputs/oracle/paper_base50k_fresh_best_s20_test256_chamfer010_p512 \
  --run base50k_best-30 outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512 \
  --run random_r4-0 outputs/oracle/paper50k_e1_random_r4_s0_test256_chamfer010_p512 \
  --run random_r4-10 outputs/oracle/paper50k_e1_random_r4_s10_test256_chamfer010_p512 \
  --run random_r4-20 outputs/oracle/paper50k_e1_random_r4_s20_test256_chamfer010_p512 \
  --run random_r4-30 outputs/oracle/paper50k_e1_random_r4_s30_test256_chamfer010_p512 \
  --run random_r4-60 outputs/oracle/paper50k_e1_random_r4_s60_test256_chamfer010_p512 \
  --output-dir outputs/oracle/param_error_hero --output-prefix hero_pareto_param_error
```

The qualitative example figures (Figures 3 and A.1):

```bash
python scripts/create_hero_qualitative_test256.py   # see --help for the four run dirs
```

### Convergence under long optimization

```bash
bash scripts/run_advice027_long_horizon.sh    # (~3-4 h)
```

Extends refinement to 1000 steps for amortized and random (r1 / r4)
initializations and produces the convergence table, success-rate curve, and
pairwise scatter under
`outputs/oracle/advice027_long_horizon_test256_chamfer010_p512_bs256/analysis/`.
Note: this experiment re-measures metrics under its own batching protocol, so
small differences from the hero table (e.g., mean Chamfer 0.0262 vs 0.0259)
are expected; the paper notes this explicitly.

### Reconstruction-auxiliary ablation

From the released checkpoints, without retraining:

```bash
bash scripts/run_recon_aux_from_checkpoints.sh   # (~1 h)
```

Produces the ablation table rows, per-sample win counts, the paired-bootstrap
confidence intervals, and the `(W,b)` diagnostic. The GT-theta validation
losses quoted in the paper are read directly from
`checkpoints/*/result.json` (`best_val_eval` / final `val` entries).

To instead retrain everything from scratch (base 50k, then both 3k
continuations, then all evaluations):

```bash
bash scripts/run_paper50k_pipeline.sh         # (~6-8 h)
```

### Seed robustness

```bash
# to skip retraining, pre-seed the expected run layout with the released checkpoints:
mkdir -p outputs/paper_base50k_advice028_seed_robustness/seed3026 \
         outputs/paper_base50k_advice028_seed_robustness/seed4026
cp -r checkpoints/paper_base50k_seed3026 outputs/paper_base50k_advice028_seed_robustness/seed3026/20260625_130724
cp -r checkpoints/paper_base50k_seed4026 outputs/paper_base50k_advice028_seed_robustness/seed4026/20260625_175407

bash scripts/run_advice028_seed_robustness.sh  # (~1 h with pre-seeded checkpoints; ~7 h if retraining)
```

This evaluation reuses the random-r4 baseline from the hero runs, so run that
section first. Outputs under `outputs/oracle/advice028_seed_robustness/analysis/`.

### Within-family OOD generalization

```bash
bash scripts/run_ood_fractal_smoke.sh          # quick 32-sample smoke (~10 min)
bash scripts/run_ood_fractal_eval.sh           # full 256-sample S1/T1 evaluation (~2-3 h)
```

### Identifiability study

```bash
python scripts/analyze_identifiability.py \
  --run-dir checkpoints/paper_base50k --split test --num-samples 256
python scripts/plot_identifiability_epsilon_curve.py \
  --pair-csv outputs/identifiability/*/pair_distances.csv \
  --output-dir outputs/identifiability/epsilon_curve
```

The pairwise statistics (correlations, near-image/far-parameter pairs, the
distance table) are written to `summary.json`. The run directory supplies only
the dataset configuration (test seed 4100), so any released checkpoint
directory works.

### MNIST / Fashion-MNIST and the comparison with Tu et al.

First fetch the third-party code (see `third_party/README.md`):

```bash
git clone https://github.com/andytu28/LearningFractals refs/LearningFractals
```

Then, per dataset (`--dataset mnist` or `--dataset fashion-mnist`):

```bash
# ours: one-shot + density-aware refinement (n=10 estimator)
python scripts/run_mnist_p0.py \
  --run-dir checkpoints/explore_n10_base100k --dataset mnist \
  --num-samples 50 --selection balanced --steps 0 10 20 30 100 \
  --output-dir outputs/mnist_p0/n10_balanced50                 # (~30 min)

# occupancy-GD ablation from our initialization
python scripts/refine_mnist_tu32_from_ours.py \
  --ours-output outputs/mnist_p0/n10_balanced50/mnist_p0_outputs.pt \
  --dataset mnist --steps 30 100                               # (~10 min)

# Tu et al. baseline with their public code (n=4 and n=10, 1000 iterations)
python scripts/run_tu_mnist_balanced10.py \
  --samples-per-digit 5 --target mnist                         # (~100-160 s per image)

# common-condition evaluation: density-aware and occupancy metrics + galleries
python scripts/evaluate_mnist_common_rendering.py \
  --ours-output outputs/mnist_tu32_refine/.../mnist_p0_outputs_with_tu32gd.pt \
  --dataset mnist --output-dir outputs/mnist_common/n10_balanced50_vs_tu
```

For Fashion-MNIST the paper compares against Tu at n=10 only. Wall-clock speed
numbers quoted in the paper come from the per-sample timings reported by
`run_mnist_p0.py` and the Tu runs.

### Training from scratch

```bash
python scripts/train_phase0.py --config configs/paper_base50k.yaml        # n=4, 50k steps (~3 h)
python scripts/train_phase0.py --config configs/explore_n10_base100k.yaml # n=10, 100k steps (~10 h)
```

The two 3k continuations are trained by `run_paper50k_pipeline.sh` (or run
`train_phase0.py` with `configs/paper_matchonly3k.yaml` /
`configs/paper_md4_3k.yaml` and `--init-model-path`). Training data is
self-generated; no dataset download is needed for the synthetic experiments.

## Notes

- Rolling training caches under `cache/` can grow large (tens of GB) when
  `train_cache_save` is enabled; the paper runs used `--no-train-cache-save`.
- Figures are written as both PNG and PDF; PDF fonts are embedded as TrueType
  (`scripts/matplotlib_config.py`).
- `checkpoints/` is not tracked in git; it is distributed as a GitHub Releases
  asset (see "Released checkpoints" above).

## License

MIT (see `LICENSE`). The third-party LearningFractals code fetched into
`refs/` retains its own BSD-3-Clause license.

## Citation

```bibtex
@misc{yamaguti2026amortized,
  title         = {Amortized Set Prediction for Inverse {IFS} Reconstruction from Density Maps},
  author        = {Yamaguti, Yutaka},
  year          = {2026},
  eprint        = {2608.24175},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2608.24175}
}
```
