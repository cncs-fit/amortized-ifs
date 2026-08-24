#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-checkpoints/paper_base50k}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FIDELITY_NAME="fidelity_res128_traj16_steps1024_sigma2_robust2048"
SUMMARY_DIR="outputs/oracle/paper50k_e1_hero_pareto_test256_chamfer010_p512_robust2048"

run_oracle() {
  local label="$1"
  local init_mode="$2"
  local restarts="$3"
  local steps="$4"
  local out_dir="outputs/oracle/${label}_s${steps}_test256_chamfer010_p512"

  echo "== ${label} init=${init_mode} r=${restarts} steps=${steps} =="
  if [[ ! -f "$out_dir/summary.json" ]]; then
    "$PYTHON_BIN" scripts/optimize_oracle.py \
      --run-dir "$BASE_RUN" \
      --checkpoint best \
      --split test \
      --num-samples 256 \
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

echo "paper50k E1 hero Pareto"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "BASE_RUN=$BASE_RUN"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "SUMMARY_DIR=$SUMMARY_DIR"
df -h .

if [[ ! -f "$BASE_RUN/result.json" || ! -f "$BASE_RUN/best_model.pt" ]]; then
  echo "missing BASE_RUN result/checkpoint: $BASE_RUN" >&2
  exit 1
fi

for steps in 0 10 20 30; do
  run_oracle paper_base50k_fresh_best model 1 "$steps"
done

for steps in 0 10 20 30 60; do
  run_oracle paper50k_e1_random_r4 random 4 "$steps"
done

echo "== summarize E1 quality-speed curves =="
"$PYTHON_BIN" scripts/summarize_oracle_quality_speed.py \
  --run base50k_best-0 outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512 \
  --run base50k_best-10 outputs/oracle/paper_base50k_fresh_best_s10_test256_chamfer010_p512 \
  --run base50k_best-20 outputs/oracle/paper_base50k_fresh_best_s20_test256_chamfer010_p512 \
  --run base50k_best-30 outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512 \
  --run random_r4-0 outputs/oracle/paper50k_e1_random_r4_s0_test256_chamfer010_p512 \
  --run random_r4-10 outputs/oracle/paper50k_e1_random_r4_s10_test256_chamfer010_p512 \
  --run random_r4-20 outputs/oracle/paper50k_e1_random_r4_s20_test256_chamfer010_p512 \
  --run random_r4-30 outputs/oracle/paper50k_e1_random_r4_s30_test256_chamfer010_p512 \
  --run random_r4-60 outputs/oracle/paper50k_e1_random_r4_s60_test256_chamfer010_p512 \
  --fidelity-name "$FIDELITY_NAME" \
  --output-dir "$SUMMARY_DIR"

echo "== E1 Pareto dominance =="
"$PYTHON_BIN" scripts/analyze_oracle_pareto.py \
  --summary-csv "$SUMMARY_DIR/summary.csv" \
  --output-dir "$SUMMARY_DIR/pareto"

echo "== E1 per-sample wins =="
"$PYTHON_BIN" scripts/compare_oracle_per_sample_wins.py \
  --run base50k_best-0 outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512 \
  --run base50k_best-10 outputs/oracle/paper_base50k_fresh_best_s10_test256_chamfer010_p512 \
  --run base50k_best-20 outputs/oracle/paper_base50k_fresh_best_s20_test256_chamfer010_p512 \
  --run base50k_best-30 outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512 \
  --run random_r4-0 outputs/oracle/paper50k_e1_random_r4_s0_test256_chamfer010_p512 \
  --run random_r4-10 outputs/oracle/paper50k_e1_random_r4_s10_test256_chamfer010_p512 \
  --run random_r4-20 outputs/oracle/paper50k_e1_random_r4_s20_test256_chamfer010_p512 \
  --run random_r4-30 outputs/oracle/paper50k_e1_random_r4_s30_test256_chamfer010_p512 \
  --run random_r4-60 outputs/oracle/paper50k_e1_random_r4_s60_test256_chamfer010_p512 \
  --comparison base0_vs_random0 base50k_best-0 random_r4-0 \
  --comparison base10_vs_random10 base50k_best-10 random_r4-10 \
  --comparison base20_vs_random20 base50k_best-20 random_r4-20 \
  --comparison base30_vs_random30 base50k_best-30 random_r4-30 \
  --comparison base30_vs_random60 base50k_best-30 random_r4-60 \
  --fidelity-name "$FIDELITY_NAME" \
  --output-csv "$SUMMARY_DIR/per_sample_wins.csv"

df -h .
echo "paper50k E1 hero Pareto complete"
