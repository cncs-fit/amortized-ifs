# Amortized Set Prediction for Inverse IFS Reconstruction from Density Maps

Code and checkpoints to reproduce the results of

> *Amortized Set Prediction for Inverse IFS Reconstruction from Density Maps*
> ([arXiv:2608.24175](https://arxiv.org/abs/2608.24175))

A feed-forward estimator predicts the affine-map set of an Iterated Function
System (IFS) from a visit-frequency density map in a single forward pass,
replacing per-image optimization. Training pairs are self-generated from the
fully known forward model; an optional test-time refinement polishes the
one-shot prediction.

**Versions.** `v1.0` reproduces the originally submitted version
(arXiv:2608.24175v1). `v2.0` adds the code, configurations, and checkpoints for
the experiments of the revised manuscript (`scripts/rev1/`, `configs/rev1/`,
`checkpoints/rev1/`); the core library is unchanged, so every v1.0 result is
reproduced as before. Table, figure, and section numbers below refer to the revised
manuscript.

## Layout

```
data/, models/, losses/   core library (sampler, renderers, estimator, losses)
scripts/                  training, evaluation, and analysis entry points
scripts/rev1/             experiments added in the revision (training variants, robustness,
                          external benchmark, comparison with Djeacoumar et al.)
configs/                  the four training configurations of the original version
configs/rev1/             the eight training variants of the revision
checkpoints/              released model checkpoints (fetched from GitHub Releases; see below)
tests/                    unit tests
third_party/              fetch scripts and instructions for third-party code and data
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

The checkpoints (~120 MB for v1.0, ~165 MB for the revision) are distributed as
GitHub Releases assets rather than tracked in git. Fetch and extract them at the repository root:

```bash
curl -L -O https://github.com/cncs-fit/amortized-ifs/releases/download/v1.0/checkpoints.tar.gz
tar xzf checkpoints.tar.gz    # creates checkpoints/
# sha256: 97a4a949c8db3883c904ded5a69c1d24b325183a70f7d92cebbc6144867ced92

# revision experiments (v2.0, ~165 MB)
curl -L -O https://github.com/cncs-fit/amortized-ifs/releases/download/v2.0/checkpoints_rev1.tar.gz
tar xzf checkpoints_rev1.tar.gz    # creates checkpoints/rev1/
# sha256: 2afb052474807902d3c96cab3d842f9de8fc89787969a15238846d287a33f53e
```

| Directory | Contents | Used for |
|---|---|---|
| `checkpoints/paper_base50k` | n=4 estimator, 50k steps, best-val (`best_model.pt`) | all main experiments |
| `checkpoints/paper50k_matchonly3k` | +3k match-only continuation, final (`model.pt`) | auxiliary-loss ablation |
| `checkpoints/paper50k_md4_3k` | +3k with reconstruction auxiliary, final (`model.pt`) | auxiliary-loss ablation |
| `checkpoints/paper_base50k_seed3026`, `..._seed4026` | same recipe, other training seeds | seed-robustness paragraph |
| `checkpoints/explore_n10_base100k` | n=10 estimator, 100k steps, best-val | MNIST / Fashion-MNIST, external benchmark |
| `checkpoints/rev1/I1_hung_nofp` | Hungarian matching without the fixed-point term | Table 4 |
| `checkpoints/rev1/I1_ordered` | regression in fixed-point order (no set matching) | Table 4 |
| `checkpoints/rev1/I2_recon_only` | reconstruction-only training (no ground truth) | Table 4 |
| `checkpoints/rev1/I2_multihyp4` | multi-hypothesis head, K=4, relaxed winner-takes-all | Table 4, Section 6.5 |
| `checkpoints/rev1/I2_collage_c070` | collage-only training, output s_max <= 0.70, final (`model.pt`) | Table 4 |
| `checkpoints/rev1/I2_collage` | collage-only training with penalties only, final (`model.pt`) | Appendix D |
| `checkpoints/rev1/I3_spectral` | spectrally constrained output, s_max <= 0.95 | Section 4.1, Appendices A, D |
| `checkpoints/rev1/I4b_reflect` | trained on a prior with reflections | Section 6.6, Table E.2 |

Each entry is a minimal training-run directory (`result.json` + checkpoint)
and can be passed directly as `--run-dir`.

All revision variants share the encoder, data, and 50k-step budget of
`paper_base50k`; their `result.json` keeps the full training history. Apart from the
training signal, the multi-hypothesis head outputs four candidate sets, and the
reconstruction-only and collage variants start from maps spread over distinct fixed
points and are selected by validation reconstruction. The two collage variants are
used at their final checkpoint, because their validation reconstruction is lowest at
the untrained initialization.

## Reproducing the paper's results

Overview of which command reproduces which paper result (details in the
sections below; approximate GPU runtimes on an RTX 3090 in parentheses):

| Paper result | Entry point |
|---|---|
| Table 2, Figures 2--3, A.1 (quality--speed Pareto) | `run_paper50k_e1_hero.sh` |
| Table 3, Figure 4 (long-horizon convergence) | `run_advice027_long_horizon.sh` |
| Table 4, `+aux` row (reconstruction-auxiliary continuation) | `run_recon_aux_from_checkpoints.sh` |
| Seed-robustness paragraph (Section 6.3) | `run_advice028_seed_robustness.sh` |
| Table 5 (S1/T1 rows), Figure 6 (within-family OOD) | `run_ood_fractal_eval.sh` |
| Table 1, Figure 1 (identifiability) | `analyze_identifiability.py` |
| Tables 8--9, Figures 7--8, B.1--B.4 (MNIST / Fashion-MNIST vs Tu) | see the MNIST section |
| Table 4, Appendix D (choice of training signal); Section 4.1 (spectral output) | `rev1/train_variants.sh` (optional), `rev1/run_training_signal.sh` |
| Table 3, `+SA` rows (hybrid search of Djeacoumar et al.) | `rev1/run_hybrid.sh` |
| Figure 5, noise floor (threshold-free convergence); Section 5 (input ambiguity); Table A.1 (contractivity) | `rev1/run_analyses.sh` |
| Table 5 (R/P rows), Table 6, Tables E.1--E.2 (observation mismatch, reflections, probabilities) | `rev1/run_shift_evals.sh` |
| Table 7, Tables F.1--F.2, Figure F.1 (external benchmark, hand-specified IFS) | `rev1/run_external.sh` |
| Table F.3 (official code of Djeacoumar et al. on `test256`) | `rev1/run_i7a2.sh` |
| Times in Table 7 and of the multi-hypothesis selection | `rev1/run_timing.sh` |

The revision entry points are described in the section "Experiments added in the
revision" below.

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

## Experiments added in the revision

The scripts in `scripts/rev1/` reuse the core library and the v1.0 runs. Run the
following v1.0 entry points first, since the revision compares against their
outputs: `run_paper50k_e1_hero.sh` (base at 0-30 steps),
`run_advice027_long_horizon.sh` (1000 steps), and
`run_recon_aux_from_checkpoints.sh` (the `+aux` continuation). All drivers skip
runs whose outputs already exist, so they can be resumed. Approximate times are for
one RTX 3090.

### Choice of training signal and spectral output (Table 4, Appendix D, Section 4.1)

```bash
bash scripts/rev1/train_variants.sh          # optional: retrain the eight variants (~3-5 h each)
bash scripts/rev1/run_training_signal.sh     # evaluate all variants (~9 h)
```

`train_variants.sh` writes `outputs/rev1/<variant>/<timestamp>/`; to evaluate
retrained runs instead of the released ones, point `RUNS` at a directory with one run
directory per variant (named as in `checkpoints/rev1/`). `RUNS` is also passed to the
multi-hypothesis analysis, the collage diagnosis, and `run_timing.sh`. Outputs that
already exist are not recomputed, so use a fresh `outputs/` when switching runs. `run_training_signal.sh`
evaluates every variant on `test256` with the protocol of the main comparison
(`run_variant_eval.sh`) and writes the table rows to
`outputs/rev1/paper/training_signal_{main,appendix}_rows.tex`, the per-hypothesis
analysis of the multi-hypothesis head to `outputs/rev1/I2_multihyp4/analysis/`, and
the collage-objective diagnosis (the collage error of the predictions and of the
generating parameters) to `outputs/rev1/I2_collage_c070/analysis/`, and the
measurements of where refinement leads from the reconstruction-only and base
predictions (gradient norm of the refinement objective, distance moved, refinement
from perturbed predictions) to `outputs/rev1/I2_stationarity/`.

### Hybrid search (Table 3, `+SA` rows)

```bash
bash scripts/rev1/run_hybrid.sh              # (~3-4 h)
```

Gradient descent with the simulated-annealing candidate search of Djeacoumar et
al., re-implemented in our objective (`optimize_hybrid.py`), from random (four
restarts) and amortized initialization at 100/300/1000 steps.

### Threshold-free convergence, input ambiguity, contractivity

```bash
bash scripts/rev1/run_analyses.sh            # (~10 min)
```

Success rate as a function of the threshold and the noise floor of the ground truth
re-rendered (Figure 5, `outputs/rev1/I8_tau/`), the ambiguity of the inputs on a
reference set of 131,072 systems (Section 5, `outputs/rev1/I2_ambiguity/`), and
the largest singular values of every predicted and refined map (Table A.1,
`outputs/rev1/I3_contractivity/`). The contractivity table reads the runs of all
other sections; rows whose runs are missing are skipped and listed in the output.

### Observation mismatch and prior shifts (Tables 5-6, E.1-E.2)

```bash
bash scripts/rev1/run_shift_evals.sh         # (~10-14 h)
```

Varies one property of the observation renderer at a time (trajectories, smoothing
width, resolution, splatting, burn-in), with refinement in the training renderer or
in the observation's renderer and with test-time pre-blurring; then the reflection
shift R, the probability shift P, and the estimator trained with reflections. Each
setting compares the base estimator with random-r4 at equal budgets
(`eval_shift.py`); the table rows go to `outputs/rev1/paper/robustness_*_rows.tex`.

### External benchmark and hand-specified IFS (Table 7, Tables F.1-F.2, Figure F.1)

```bash
bash third_party/fetch_chaotic_fractals.sh   # their code and benchmark data -> refs/
bash third_party/setup_chaotic_env.sh        # env "chaotic-fractals" for their code
bash scripts/rev1/run_external.sh            # (~4 h)
```

Exports the codes published with the benchmark (their method and three baselines)
in their own env, brings every code into our frame (`prepare_external_ifs.py`
follows their bounding-box normalization and their probability rule
`|det W| + 0.01`), evaluates our n=10 model on all 101 systems and our n=4 model on the
43 systems with at most four maps and on six hand-specified IFS, and runs the
best-fit alignment check (`check_external_alignment.py`).

### Official code of Djeacoumar et al. on `test256` (Table F.3)

```bash
bash scripts/rev1/run_i7a2.sh                # (~50 min per system with their code; 16 + 16 + 6 systems)
```

Writes the first 16 `test256` systems in their code format, draws each target with
their ground-truth generator, runs their optimizer with its default configuration
(10 maps, 15,000 iterations, seed 5), and scores all methods against the same
ground-truth point clouds (`i7a2_score.py`); `i7a2_check.py` repeats the scoring
with their probability rule and with best-fit alignment. The protocol checks of
Appendix F follow: their code with the true number of maps (4), and on the six most
strongly contracting systems of their own benchmark, scored against the codes they
publish (`i7a2_score_benchmark.py`; run `run_external.sh` first); then Figure F.2
(`plot_i7a2_examples.py`). Needs the 100-step base run from `run_training_signal.sh`.

### Timing

```bash
bash scripts/rev1/run_timing.sh              # on an otherwise idle GPU (~3 h)
```

The cost of the multi-hypothesis selection at batch sizes 1 and 32
(`outputs/rev1/timing/multihyp.json`), the per-image times of Table 7 in batches, and
the single-image times (batch 1) of Table 7 and Section 6.8.

### Reproducibility of the revision results

Zero-step metrics of all released checkpoints reproduce exactly. Refined metrics
reproduce to about three significant digits, as for v1.0. Per-system win counts over
16 systems (Table F.3) can change by one when the 1000-step v1.0 runs are recomputed.
Script docstrings refer to the revision items by the identifiers below.

| Identifier | Experiment | Manuscript |
|---|---|---|
| I1 | ordered regression vs. Hungarian matching, fixed-point term | Section 6.5, Table 4 |
| I2 | reconstruction-only, multi-hypothesis, and collage training; input ambiguity | Sections 5, 6.5 |
| I3 | contractivity statistics, spectrally constrained output | Section 4.1, Appendix A |
| I4 | reflections, selection probabilities, reflection-trained estimator | Section 6.6, Appendix E |
| I5 | observation-renderer mismatch | Section 6.6, Table 6, Appendix E |
| I6 | external benchmark, hand-specified IFS | Section 6.7, Appendix F |
| I7 | Djeacoumar et al.: hybrid search (a1), official code (a2) | Sections 6.4, 6.7, Appendix F |
| I8 | threshold-free convergence comparison | Section 6.4, Figure 5 |

## Notes

- Rolling training caches under `cache/` can grow large (tens of GB) when
  `train_cache_save` is enabled; the paper runs used `--no-train-cache-save`.
- Figures are written as both PNG and PDF; PDF fonts are embedded as TrueType
  (`scripts/matplotlib_config.py`).
- `checkpoints/` is not tracked in git; it is distributed as a GitHub Releases
  asset (see "Released checkpoints" above).

## License

MIT (see `LICENSE`). Third-party code and data fetched into `refs/` keep their own
licenses: LearningFractals (BSD-3-Clause), and the code of Djeacoumar et al. (MIT)
with its bundled Gaussian rasterizer (Gaussian-Splatting License, non-commercial
research use only). See `third_party/README.md`.

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
