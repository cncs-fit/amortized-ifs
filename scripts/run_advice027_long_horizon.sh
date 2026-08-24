#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-checkpoints/paper_base50k}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUT_ROOT="${OUT_ROOT:-outputs/oracle/advice027_long_horizon_test256_chamfer010_p512_bs256}"
FIDELITY_NAME="fidelity_res128_traj16_steps1024_sigma2_robust2048"
STEPS=(0 30 100 300 1000)

run_oracle() {
  local label="$1"
  local init_mode="$2"
  local restarts="$3"
  local steps="$4"
  local out_dir="${OUT_ROOT}/${label}_s${steps}"

  echo "== ${label} init=${init_mode} r=${restarts} steps=${steps} =="
  if [[ ! -f "${out_dir}/summary.json" ]]; then
    "$PYTHON_BIN" scripts/optimize_oracle.py \
      --run-dir "$BASE_RUN" \
      --checkpoint best \
      --split test \
      --num-samples "$NUM_SAMPLES" \
      --seed 4100 \
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
      --log-interval 100 \
      --device cuda \
      --output-dir "$out_dir"
  else
    echo "skip optimize: ${out_dir}/summary.json exists"
  fi

  if [[ ! -f "${out_dir}/${FIDELITY_NAME}.json" ]]; then
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
    echo "skip fidelity: ${out_dir}/${FIDELITY_NAME}.json exists"
  fi
}

echo "advice027 long-horizon refinement"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "BASE_RUN=$BASE_RUN"
echo "NUM_SAMPLES=$NUM_SAMPLES"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "OUT_ROOT=$OUT_ROOT"
df -h .

if [[ ! -f "$BASE_RUN/result.json" || ! -f "$BASE_RUN/best_model.pt" ]]; then
  echo "missing BASE_RUN result/checkpoint: $BASE_RUN" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

for steps in "${STEPS[@]}"; do
  run_oracle model_r1 model 1 "$steps"
done
for steps in "${STEPS[@]}"; do
  run_oracle random_r1 random 1 "$steps"
done
for steps in "${STEPS[@]}"; do
  run_oracle random_r4 random 4 "$steps"
done

echo "== summarize quality-speed =="
"$PYTHON_BIN" scripts/summarize_oracle_quality_speed.py \
  --run model_r1-0 "${OUT_ROOT}/model_r1_s0" \
  --run model_r1-30 "${OUT_ROOT}/model_r1_s30" \
  --run model_r1-100 "${OUT_ROOT}/model_r1_s100" \
  --run model_r1-300 "${OUT_ROOT}/model_r1_s300" \
  --run model_r1-1000 "${OUT_ROOT}/model_r1_s1000" \
  --run random_r1-0 "${OUT_ROOT}/random_r1_s0" \
  --run random_r1-30 "${OUT_ROOT}/random_r1_s30" \
  --run random_r1-100 "${OUT_ROOT}/random_r1_s100" \
  --run random_r1-300 "${OUT_ROOT}/random_r1_s300" \
  --run random_r1-1000 "${OUT_ROOT}/random_r1_s1000" \
  --run random_r4-0 "${OUT_ROOT}/random_r4_s0" \
  --run random_r4-30 "${OUT_ROOT}/random_r4_s30" \
  --run random_r4-100 "${OUT_ROOT}/random_r4_s100" \
  --run random_r4-300 "${OUT_ROOT}/random_r4_s300" \
  --run random_r4-1000 "${OUT_ROOT}/random_r4_s1000" \
  --fidelity-name "$FIDELITY_NAME" \
  --output-dir "${OUT_ROOT}/summary"

echo "== summarize param error =="
"$PYTHON_BIN" scripts/summarize_oracle_param_error.py \
  --output-dir "${OUT_ROOT}/param_error" \
  --output-prefix advice027_param_error \
  --run model_r1-0 "${OUT_ROOT}/model_r1_s0" \
  --run model_r1-30 "${OUT_ROOT}/model_r1_s30" \
  --run model_r1-100 "${OUT_ROOT}/model_r1_s100" \
  --run model_r1-300 "${OUT_ROOT}/model_r1_s300" \
  --run model_r1-1000 "${OUT_ROOT}/model_r1_s1000" \
  --run random_r1-0 "${OUT_ROOT}/random_r1_s0" \
  --run random_r1-30 "${OUT_ROOT}/random_r1_s30" \
  --run random_r1-100 "${OUT_ROOT}/random_r1_s100" \
  --run random_r1-300 "${OUT_ROOT}/random_r1_s300" \
  --run random_r1-1000 "${OUT_ROOT}/random_r1_s1000" \
  --run random_r4-0 "${OUT_ROOT}/random_r4_s0" \
  --run random_r4-30 "${OUT_ROOT}/random_r4_s30" \
  --run random_r4-100 "${OUT_ROOT}/random_r4_s100" \
  --run random_r4-300 "${OUT_ROOT}/random_r4_s300" \
  --run random_r4-1000 "${OUT_ROOT}/random_r4_s1000"

echo "== analyze distributions =="
"$PYTHON_BIN" scripts/analyze_long_horizon_refinement.py \
  --run model_r1-0 "${OUT_ROOT}/model_r1_s0" \
  --run model_r1-30 "${OUT_ROOT}/model_r1_s30" \
  --run model_r1-100 "${OUT_ROOT}/model_r1_s100" \
  --run model_r1-300 "${OUT_ROOT}/model_r1_s300" \
  --run model_r1-1000 "${OUT_ROOT}/model_r1_s1000" \
  --run random_r1-0 "${OUT_ROOT}/random_r1_s0" \
  --run random_r1-30 "${OUT_ROOT}/random_r1_s30" \
  --run random_r1-100 "${OUT_ROOT}/random_r1_s100" \
  --run random_r1-300 "${OUT_ROOT}/random_r1_s300" \
  --run random_r1-1000 "${OUT_ROOT}/random_r1_s1000" \
  --run random_r4-0 "${OUT_ROOT}/random_r4_s0" \
  --run random_r4-30 "${OUT_ROOT}/random_r4_s30" \
  --run random_r4-100 "${OUT_ROOT}/random_r4_s100" \
  --run random_r4-300 "${OUT_ROOT}/random_r4_s300" \
  --run random_r4-1000 "${OUT_ROOT}/random_r4_s1000" \
  --fidelity-name "$FIDELITY_NAME" \
  --param-error-csv "${OUT_ROOT}/param_error/advice027_param_error_per_sample.csv" \
  --output-dir "${OUT_ROOT}/analysis"

df -h .
echo "advice027 long-horizon refinement complete"
