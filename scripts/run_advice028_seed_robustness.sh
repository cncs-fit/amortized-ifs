#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/paper_base50k.yaml}"
TRAIN_OUT_ROOT="${TRAIN_OUT_ROOT:-outputs/paper_base50k_advice028_seed_robustness}"
ORACLE_ROOT="${ORACLE_ROOT:-outputs/oracle/advice028_seed_robustness}"
SUMMARY_DIR="${SUMMARY_DIR:-${ORACLE_ROOT}/summary}"
SEEDS=(${SEEDS:-3026 4026})
BATCH_SIZE="${BATCH_SIZE:-32}"
FIDELITY_NAME="fidelity_res128_traj16_steps1024_sigma2_robust2048"
STEPS=(0 10 20 30)

SEED1_RUN="${SEED1_RUN:-checkpoints/paper_base50k}"
SEED1_ORACLE_PREFIX="${SEED1_ORACLE_PREFIX:-outputs/oracle/paper_base50k_fresh_best}"
RANDOM_ORACLE_PREFIX="${RANDOM_ORACLE_PREFIX:-outputs/oracle/paper50k_e1_random_r4}"

latest_complete_run() {
  local out_root="$1"
  if [[ ! -d "$out_root" ]]; then
    return 0
  fi
  find "$out_root" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | while read -r candidate; do
    if [[ -f "${candidate}/result.json" && -f "${candidate}/best_model.pt" ]]; then
      echo "$candidate"
    fi
  done | tail -n 1
}

train_seed() {
  local seed="$1"
  local out_root="${TRAIN_OUT_ROOT}/seed${seed}"
  local env_var="BASE_RUN_SEED${seed}"
  local override="${!env_var:-}"

  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi

  local existing
  existing="$(latest_complete_run "$out_root")"
  if [[ -n "$existing" ]]; then
    echo "skip train seed=${seed}: using existing ${existing}" >&2
    echo "$existing"
    return
  fi

  echo "== train base50k seed=${seed} ==" >&2
  "$PYTHON_BIN" scripts/train_phase0.py \
    --config "$CONFIG" \
    --seed "$seed" \
    --output-dir "$out_root" >&2

  local run_dir
  run_dir="$(latest_complete_run "$out_root")"
  if [[ -z "$run_dir" ]]; then
    echo "could not locate completed run for seed ${seed} under ${out_root}" >&2
    exit 1
  fi
  echo "$run_dir"
}

run_oracle() {
  local label="$1"
  local run_dir="$2"
  local steps="$3"
  local out_dir="${ORACLE_ROOT}/${label}_s${steps}"

  echo "== oracle ${label} steps=${steps} =="
  if [[ ! -f "${out_dir}/summary.json" ]]; then
    "$PYTHON_BIN" scripts/optimize_oracle.py \
      --run-dir "$run_dir" \
      --checkpoint best \
      --split test \
      --num-samples 256 \
      --seed 4100 \
      --batch-size "$BATCH_SIZE" \
      --init-mode model \
      --restarts 1 \
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

echo "advice028 seed robustness"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CONFIG=$CONFIG"
echo "TRAIN_OUT_ROOT=$TRAIN_OUT_ROOT"
echo "ORACLE_ROOT=$ORACLE_ROOT"
echo "SEEDS=${SEEDS[*]}"
echo "BATCH_SIZE=$BATCH_SIZE"
df -h .

if [[ ! -f "$SEED1_RUN/result.json" || ! -f "$SEED1_RUN/best_model.pt" ]]; then
  echo "missing seed1 run: ${SEED1_RUN}" >&2
  exit 1
fi

mkdir -p "$ORACLE_ROOT" "$SUMMARY_DIR"
BASE_RUNS_TSV="${ORACLE_ROOT}/base_runs.tsv"
{
  printf "2026\t%s\n" "$SEED1_RUN"
  for seed in "${SEEDS[@]}"; do
    run_dir="$(train_seed "$seed")"
    printf "%s\t%s\n" "$seed" "$run_dir"
  done
} > "$BASE_RUNS_TSV"

echo "== base runs =="
cat "$BASE_RUNS_TSV"

while IFS=$'\t' read -r seed run_dir; do
  if [[ "$seed" == "2026" ]]; then
    continue
  fi
  for steps in "${STEPS[@]}"; do
    run_oracle "seed${seed}_base" "$run_dir" "$steps"
  done
done < "$BASE_RUNS_TSV"

summary_args=()
param_args=()
wins_run_args=()
wins_comp_args=()

while IFS=$'\t' read -r seed run_dir; do
  for steps in "${STEPS[@]}"; do
    label="seed${seed}_base-${steps}"
    if [[ "$seed" == "2026" ]]; then
      oracle_dir="${SEED1_ORACLE_PREFIX}_s${steps}_test256_chamfer010_p512"
    else
      oracle_dir="${ORACLE_ROOT}/seed${seed}_base_s${steps}"
    fi
    summary_args+=(--run "$label" "$oracle_dir")
    param_args+=(--run "$label" "$oracle_dir")
    if [[ "$steps" == "30" ]]; then
      wins_run_args+=(--run "$label" "$oracle_dir")
      wins_comp_args+=(--comparison "seed${seed}_base30_vs_random30" "$label" "random_r4-30")
      wins_comp_args+=(--comparison "seed${seed}_base30_vs_random60" "$label" "random_r4-60")
    fi
  done
done < "$BASE_RUNS_TSV"

for steps in 30 60; do
  random_label="random_r4-${steps}"
  random_dir="${RANDOM_ORACLE_PREFIX}_s${steps}_test256_chamfer010_p512"
  summary_args+=(--run "$random_label" "$random_dir")
  param_args+=(--run "$random_label" "$random_dir")
  wins_run_args+=(--run "$random_label" "$random_dir")
done

echo "== summarize quality-speed =="
"$PYTHON_BIN" scripts/summarize_oracle_quality_speed.py \
  "${summary_args[@]}" \
  --fidelity-name "$FIDELITY_NAME" \
  --output-dir "$SUMMARY_DIR"

echo "== summarize param error =="
"$PYTHON_BIN" scripts/summarize_oracle_param_error.py \
  "${param_args[@]}" \
  --output-dir "${ORACLE_ROOT}/param_error" \
  --output-prefix advice028_param_error

echo "== per-sample wins =="
"$PYTHON_BIN" scripts/compare_oracle_per_sample_wins.py \
  "${wins_run_args[@]}" \
  "${wins_comp_args[@]}" \
  --fidelity-name "$FIDELITY_NAME" \
  --output-csv "${SUMMARY_DIR}/per_sample_wins.csv"

echo "== aggregate seed robustness =="
"$PYTHON_BIN" scripts/analyze_seed_robustness.py \
  --summary-csv "${SUMMARY_DIR}/summary.csv" \
  --param-summary-csv "${ORACLE_ROOT}/param_error/advice028_param_error_summary.csv" \
  --per-sample-wins-csv "${SUMMARY_DIR}/per_sample_wins.csv" \
  --output-dir "${ORACLE_ROOT}/analysis"

df -h .
echo "advice028 seed robustness complete"
