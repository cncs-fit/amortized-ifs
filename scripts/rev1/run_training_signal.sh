#!/usr/bin/env bash
# Choice of training signal (Section 6.5, Table 4, Appendix D) and the spectrally constrained output
# (Section 4.1): evaluate every variant on test256 at the budgets reported in the paper, then build
# the table rows, the multi-hypothesis analysis, and the collage-objective diagnosis.
# Needs the base hero runs (bash scripts/run_paper50k_e1_hero.sh), the long-horizon runs
# (bash scripts/run_advice027_long_horizon.sh), and the reconstruction-auxiliary runs
# (bash scripts/run_recon_aux_from_checkpoints.sh) for the base / +aux entries at 0-30 and 1000 steps.
# Env: RUNS (default checkpoints/rev1) holds one run directory per variant.  About 9 h on an RTX 3090,
# most of it in the 1000-step refinements.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
RUNS="${RUNS:-checkpoints/rev1}"
EV=scripts/rev1/run_variant_eval.sh
# base and +aux at the budgets not covered by the v1.0 runs (same protocol as the variants)
BASE_STEPS="" bash $EV base checkpoints/paper_base50k 100 1000
CKPT=final BASE_STEPS="" bash $EV aux_md4_3k checkpoints/paper50k_md4_3k 100
for v in I1_hung_nofp I1_ordered I2_recon_only I2_multihyp4; do bash $EV "$v" "$RUNS/$v" 100 1000; done
bash $EV I3_spectral "$RUNS/I3_spectral" 100
# collage variants: final checkpoint (validation reconstruction is lowest at the untrained initialization)
for v in I2_collage I2_collage_c070; do CKPT=final bash $EV "${v}_final" "$RUNS/$v" 100; done
"$PY" scripts/rev1/summarize_variants.py
"$PY" scripts/rev1/make_training_signal_table.py       # -> outputs/rev1/paper/training_signal_{main,appendix}_rows.tex
"$PY" scripts/rev1/analyze_multihyp.py --run-dir "$RUNS/I2_multihyp4"       # -> outputs/rev1/I2_multihyp4/analysis/
"$PY" scripts/rev1/analyze_collage.py --config-run "$RUNS/I2_collage_c070"  # -> outputs/rev1/I2_collage_c070/analysis/
# where refinement leads from the reconstruction-only and base predictions (Section 6.5): gradient norm of
# the refinement objective and distance moved, then refinement from perturbed predictions
"$PY" scripts/rev1/analyze_stationarity.py                 # -> outputs/rev1/I2_stationarity/
FID=fidelity_res128_traj16_steps1024_sigma2_robust2048
for pair in "recon_only:$RUNS/I2_recon_only" "base:checkpoints/paper_base50k"; do
  label=${pair%%:*}; run=${pair#*:}
  for noise in 0.05 0.1; do for steps in 0 100; do
    out=outputs/rev1/I2_stationarity/perturb/${label}_noise${noise}_s${steps}
    [[ -f "$out/summary.json" ]] || "$PY" scripts/rev1/optimize_rev1.py --run-dir "$run" --checkpoint best --split test \
      --num-samples 256 --seed 4100 --batch-size 32 --init-mode model --init-noise $noise --steps $steps --lr 0.005 \
      --reconstruction-resolution 128 --reconstruction-num-trajectories 16 --reconstruction-num-steps 1024 \
      --reconstruction-burn-in 128 --reconstruction-smoothing-sigma 2.0 --reconstruction-seed 12345 \
      --eval-reconstruction-seed 22318 --reconstruction-map-probability-mode determinant \
      --reconstruction-match-render-config --point-chamfer-loss-weight 0.10 --point-chamfer-num-pred-points 512 \
      --point-chamfer-num-target-points 512 --point-chamfer-seed 24680 --log-interval 0 --device cuda \
      --output-dir "$out" > /dev/null 2>&1
    [[ -f "$out/$FID.json" ]] || "$PY" scripts/evaluate_oracle_fidelity.py --oracle-dir "$out" --resolution 128 \
      --num-trajectories 16 --num-steps 1024 --burn-in 128 --smoothing-sigma 2.0 --seed 53100 --chamfer-max-points 2048 \
      --plot-samples 4 --device cuda --output-name "$FID" > /dev/null 2>&1
  done; done
done
