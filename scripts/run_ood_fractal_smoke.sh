#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-checkpoints/paper_base50k}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
FIDELITY_NAME="fidelity_res128_traj16_steps1024_sigma2_robust2048"
SUMMARY_ROOT="outputs/oracle/ood_fractal_smoke_n${NUM_SAMPLES}_chamfer010_p512_robust2048"

run_oracle() {
  local scenario="$1"
  local scale_low="$2"
  local scale_high="$3"
  local fixed_point_range="$4"
  local label="$5"
  local init_mode="$6"
  local restarts="$7"
  local steps="$8"
  local seed="$9"
  local out_dir="outputs/oracle/ood_${scenario}_${label}_s${steps}_n${NUM_SAMPLES}_chamfer010_p512"

  echo "== OOD ${scenario}: ${label} init=${init_mode} r=${restarts} steps=${steps} =="
  if [[ ! -f "$out_dir/summary.json" ]]; then
    "$PYTHON_BIN" scripts/optimize_oracle.py \
      --run-dir "$BASE_RUN" \
      --checkpoint best \
      --split test \
      --num-samples "$NUM_SAMPLES" \
      --seed "$seed" \
      --sample-label "$scenario" \
      --sample-scale-range "$scale_low" "$scale_high" \
      --sample-fixed-point-range "$fixed_point_range" \
      --batch-size "$BATCH_SIZE" \
      --init-mode "$init_mode" \
      --restarts "$restarts" \
      --steps "$steps" \
      --lr 0.005 \
      --reconstruction-resolution 128 \
      --reconstruction-num-trajectories 16 \
      --reconstruction-num-steps 1024 \
      --reconstruction-burn-in 128 \
      --reconstruction-smoothing-sigma 2.0 \
      --reconstruction-seed 12345 \
      --eval-reconstruction-seed 22318 \
      --reconstruction-map-probability-mode determinant \
      --reconstruction-match-render-config \
      --point-chamfer-loss-weight 0.10 \
      --point-chamfer-num-pred-points 512 \
      --point-chamfer-num-target-points 512 \
      --point-chamfer-seed 24680 \
      --plot-samples 4 \
      --point-cloud-max-points 3000 \
      --device cuda \
      --output-dir "$out_dir"
  else
    echo "skip optimize: $out_dir/summary.json exists"
  fi

  if [[ ! -f "$out_dir/${FIDELITY_NAME}.json" ]]; then
    "$PYTHON_BIN" scripts/evaluate_oracle_fidelity.py \
      --oracle-dir "$out_dir" \
      --resolution 128 \
      --num-trajectories 16 \
      --num-steps 1024 \
      --burn-in 128 \
      --smoothing-sigma 2.0 \
      --seed 53100 \
      --chamfer-max-points 2048 \
      --plot-samples 4 \
      --device cuda \
      --output-name "$FIDELITY_NAME"
  else
    echo "skip fidelity: $out_dir/${FIDELITY_NAME}.json exists"
  fi
}

summarize_scenario() {
  local scenario="$1"
  local out_dir="${SUMMARY_ROOT}/${scenario}"
  "$PYTHON_BIN" scripts/summarize_oracle_quality_speed.py \
    --run model-0 "outputs/oracle/ood_${scenario}_model_s0_n${NUM_SAMPLES}_chamfer010_p512" \
    --run model-30 "outputs/oracle/ood_${scenario}_model_s30_n${NUM_SAMPLES}_chamfer010_p512" \
    --run random_r4-0 "outputs/oracle/ood_${scenario}_random_r4_s0_n${NUM_SAMPLES}_chamfer010_p512" \
    --run random_r4-30 "outputs/oracle/ood_${scenario}_random_r4_s30_n${NUM_SAMPLES}_chamfer010_p512" \
    --fidelity-name "$FIDELITY_NAME" \
    --output-dir "$out_dir"
}

run_scenario() {
  local scenario="$1"
  local scale_low="$2"
  local scale_high="$3"
  local fixed_point_range="$4"
  local seed="$5"

  run_oracle "$scenario" "$scale_low" "$scale_high" "$fixed_point_range" model model 1 0 "$seed"
  run_oracle "$scenario" "$scale_low" "$scale_high" "$fixed_point_range" model model 1 30 "$seed"
  run_oracle "$scenario" "$scale_low" "$scale_high" "$fixed_point_range" random_r4 random 4 0 "$seed"
  run_oracle "$scenario" "$scale_low" "$scale_high" "$fixed_point_range" random_r4 random 4 30 "$seed"
  summarize_scenario "$scenario"
}

echo "OOD fractal smoke"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "BASE_RUN=$BASE_RUN"
echo "NUM_SAMPLES=$NUM_SAMPLES"
echo "SUMMARY_ROOT=$SUMMARY_ROOT"

run_scenario s1_scale_high 0.70 0.85 0.75 7100
run_scenario s2_scale_low 0.08 0.20 0.75 7200
run_scenario t1_wide_fixed_point 0.20 0.70 1.20 7300

echo "wrote $SUMMARY_ROOT"
