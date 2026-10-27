#!/usr/bin/env bash
# Evaluate one estimator with the protocol of the paper's main comparison: test256, refinement for
# 0/10/20/30 steps (plus optional extra step counts), matched renderer, fidelity scoring.
# Usage: bash scripts/rev1/run_variant_eval.sh <label> <run_dir> [extra steps ...]
# Env:   CKPT=best|final (default best); BASE_STEPS="0 10 20 30" (default; "" runs only the extra steps);
#        PYTHON (default python).  Output: outputs/rev1/variant_eval/<label>_s<steps>_test256/
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
LABEL="$1"; RUN="$2"; shift 2
CKPT="${CKPT:-best}"
read -r -a BASE <<< "${BASE_STEPS-0 10 20 30}"
STEPS=("${BASE[@]}" "$@")
FID=fidelity_res128_traj16_steps1024_sigma2_robust2048
for s in "${STEPS[@]}"; do
  out="outputs/rev1/variant_eval/${LABEL}_s${s}_test256"
  if [[ ! -f "$out/summary.json" ]]; then
    "$PY" scripts/rev1/optimize_rev1.py --run-dir "$RUN" --checkpoint "$CKPT" --split test --num-samples 256 --seed 4100 \
      --batch-size 32 --init-mode model --steps "$s" --lr 0.005 --reconstruction-resolution 128 \
      --reconstruction-num-trajectories 16 --reconstruction-num-steps 1024 --reconstruction-burn-in 128 \
      --reconstruction-smoothing-sigma 2.0 --reconstruction-seed 12345 --eval-reconstruction-seed 22318 \
      --reconstruction-map-probability-mode determinant --reconstruction-match-render-config \
      --point-chamfer-loss-weight 0.10 --point-chamfer-num-pred-points 512 --point-chamfer-num-target-points 512 \
      --point-chamfer-seed 24680 --plot-samples 4 --point-cloud-max-points 3000 --log-interval 0 --device cuda \
      --output-dir "$out" > /dev/null 2>&1 || { echo "optimize failed: $out"; continue; }
  fi
  if [[ ! -f "$out/$FID.json" ]]; then
    "$PY" scripts/evaluate_oracle_fidelity.py --oracle-dir "$out" --resolution 128 --num-trajectories 16 --num-steps 1024 \
      --burn-in 128 --smoothing-sigma 2.0 --seed 53100 --chamfer-max-points 2048 --plot-samples 4 --device cuda \
      --output-name "$FID" > /dev/null 2>&1 || echo "fidelity failed: $out"
  fi
  echo "done $LABEL s$s"
done
