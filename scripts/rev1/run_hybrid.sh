#!/usr/bin/env bash
# Hybrid search of Djeacoumar et al. re-implemented in our objective (Section 6.4, Table 3): gradient
# descent with simulated-annealing candidate search, from random (4 restarts) and amortized
# initialization, at 100/300/1000 steps on test256.  About 3-4 h on an RTX 3090.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
FID=fidelity_res128_traj16_steps1024_sigma2_robust2048
for steps in 100 300 1000; do
  for mode in random model; do
    out=outputs/rev1/I7_hybrid/${mode}_s${steps}
    if [[ ! -f "$out/summary.json" ]]; then
      echo "== $(date -Is) $out"
      "$PY" scripts/rev1/optimize_hybrid.py --output-dir "$out" --init-mode "$mode" --steps "$steps" --batch-size 128 2>&1 | grep -v -i warn | tail -3
    fi
    if [[ ! -f "$out/$FID.json" ]]; then
      "$PY" scripts/evaluate_oracle_fidelity.py --oracle-dir "$out" --resolution 128 --num-trajectories 16 --num-steps 1024 \
        --burn-in 128 --smoothing-sigma 2.0 --seed 53100 --chamfer-max-points 2048 --plot-samples 4 --device cuda \
        --output-name "$FID" > /dev/null 2>&1 || echo "fidelity failed: $out"
    fi
  done
done
