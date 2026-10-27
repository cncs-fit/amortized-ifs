#!/usr/bin/env bash
# Analyses of the revision that reuse saved runs:
#   * threshold-free convergence comparison and the noise floor (Section 6.4, Figure 5): needs
#     run_advice027_long_horizon.sh; the floor is the ground truth re-rendered and scored;
#   * ambiguity of the inputs on a reference set of 131,072 systems (Section 5);
#   * contractivity of every predicted and refined map (Section 4.1, Table A.1): rows whose runs are
#     missing are skipped and listed in the output.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
FID=fidelity_res128_traj16_steps1024_sigma2_robust2048
GT=outputs/rev1/I8_tau/at_gt_s0_test256
if [[ ! -f "$GT/$FID.json" ]]; then
  "$PY" scripts/optimize_oracle.py --run-dir checkpoints/paper_base50k --checkpoint best --split test --num-samples 256 \
    --seed 4100 --batch-size 32 --init-mode target --steps 0 --lr 0.005 --reconstruction-resolution 128 \
    --reconstruction-num-trajectories 16 --reconstruction-num-steps 1024 --reconstruction-burn-in 128 \
    --reconstruction-smoothing-sigma 2.0 --reconstruction-seed 12345 --eval-reconstruction-seed 22318 \
    --reconstruction-map-probability-mode determinant --reconstruction-match-render-config \
    --point-chamfer-loss-weight 0.10 --point-chamfer-num-pred-points 512 --point-chamfer-num-target-points 512 \
    --point-chamfer-seed 24680 --log-interval 0 --device cuda --output-dir "$GT" > /dev/null 2>&1
  "$PY" scripts/evaluate_oracle_fidelity.py --oracle-dir "$GT" --resolution 128 --num-trajectories 16 --num-steps 1024 \
    --burn-in 128 --smoothing-sigma 2.0 --seed 53100 --chamfer-max-points 2048 --plot-samples 4 --device cuda \
    --output-name "$FID" > /dev/null 2>&1
fi
"$PY" scripts/rev1/analyze_tau_threshold.py         # -> outputs/rev1/I8_tau/analysis/
"$PY" scripts/rev1/plot_convergence_tau_figures.py  # -> outputs/rev1/I8_tau/paper/
"$PY" scripts/rev1/analyze_ambiguity.py             # -> outputs/rev1/I2_ambiguity/
"$PY" scripts/rev1/analyze_contractivity.py         # -> outputs/rev1/I3_contractivity/
