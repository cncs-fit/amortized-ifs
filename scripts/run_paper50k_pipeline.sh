#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
FIDELITY_NAME="fidelity_res128_traj16_steps1024_sigma2_robust2048"
SUMMARY_DIR="outputs/oracle/paper50k_base_matchonly_md4_refinement_test256_chamfer010_p512_robust2048"

latest_completed_run() {
  local output_root="$1"
  find "$output_root" -mindepth 2 -maxdepth 2 -name result.json -printf '%T@ %h\n' 2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

run_oracle() {
  local label="$1"
  local run_dir="$2"
  local checkpoint="$3"
  local steps="$4"
  local out_dir="outputs/oracle/${label}_s${steps}_test256_chamfer010_p512"

  echo "== ${label} steps=${steps} checkpoint=${checkpoint} =="
  if [[ ! -f "$out_dir/summary.json" ]]; then
    "$PYTHON_BIN" scripts/optimize_oracle.py \
      --run-dir "$run_dir" \
      --checkpoint "$checkpoint" \
      --split test \
      --num-samples 256 \
      --seed 4100 \
      --batch-size 32 \
      --init-mode model \
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

run_curve() {
  local label="$1"
  local run_dir="$2"
  local checkpoint="$3"
  for steps in 0 10 20 30; do
    run_oracle "$label" "$run_dir" "$checkpoint" "$steps"
  done
}

echo "paper50k pipeline"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SUMMARY_DIR=$SUMMARY_DIR"
echo "disk before:"
df -h .

echo "== train fresh paper_base50k =="
"$PYTHON_BIN" scripts/train_phase0.py --config configs/paper_base50k.yaml --no-train-cache-save
BASE_RUN="$(latest_completed_run outputs/paper_base50k)"
if [[ -z "$BASE_RUN" || ! -f "$BASE_RUN/best_model.pt" ]]; then
  echo "failed to locate completed paper_base50k run" >&2
  exit 1
fi
echo "BASE_RUN=$BASE_RUN"

echo "== base 0-step gate =="
run_oracle paper_base50k_fresh_best "$BASE_RUN" best 0

echo "== train paper50k match-only 3k =="
"$PYTHON_BIN" scripts/train_phase0.py \
  --config configs/paper_matchonly3k.yaml \
  --init-model-path "$BASE_RUN/best_model.pt" \
  --output-dir outputs/paper50k_matchonly3k \
  --no-train-cache-save
MATCH_RUN="$(latest_completed_run outputs/paper50k_matchonly3k)"
if [[ -z "$MATCH_RUN" || ! -f "$MATCH_RUN/result.json" ]]; then
  echo "failed to locate completed paper50k_matchonly3k run" >&2
  exit 1
fi
echo "MATCH_RUN=$MATCH_RUN"

echo "== train paper50k md4 3k =="
"$PYTHON_BIN" scripts/train_phase0.py \
  --config configs/paper_md4_3k.yaml \
  --init-model-path "$BASE_RUN/best_model.pt" \
  --output-dir outputs/paper50k_md4_3k \
  --no-train-cache-save
MD4_RUN="$(latest_completed_run outputs/paper50k_md4_3k)"
if [[ -z "$MD4_RUN" || ! -f "$MD4_RUN/result.json" ]]; then
  echo "failed to locate completed paper50k_md4_3k run" >&2
  exit 1
fi
echo "MD4_RUN=$MD4_RUN"

echo "== refinement curves =="
run_curve paper_base50k_fresh_best "$BASE_RUN" best
run_curve paper50k_matchonly3k_final "$MATCH_RUN" final
run_curve paper50k_md4_3k_final "$MD4_RUN" final

echo "== auxiliary 0-step checkpoints =="
run_oracle paper_base50k_fresh_final "$BASE_RUN" final 0
run_oracle paper50k_matchonly3k_best "$MATCH_RUN" best 0
run_oracle paper50k_md4_3k_best "$MD4_RUN" best 0

echo "== summarize quality-speed curves =="
"$PYTHON_BIN" scripts/summarize_oracle_quality_speed.py \
  --run paper_base50k_fresh_best-0 outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512 \
  --run paper_base50k_fresh_best-10 outputs/oracle/paper_base50k_fresh_best_s10_test256_chamfer010_p512 \
  --run paper_base50k_fresh_best-20 outputs/oracle/paper_base50k_fresh_best_s20_test256_chamfer010_p512 \
  --run paper_base50k_fresh_best-30 outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512 \
  --run paper50k_matchonly3k_final-0 outputs/oracle/paper50k_matchonly3k_final_s0_test256_chamfer010_p512 \
  --run paper50k_matchonly3k_final-10 outputs/oracle/paper50k_matchonly3k_final_s10_test256_chamfer010_p512 \
  --run paper50k_matchonly3k_final-20 outputs/oracle/paper50k_matchonly3k_final_s20_test256_chamfer010_p512 \
  --run paper50k_matchonly3k_final-30 outputs/oracle/paper50k_matchonly3k_final_s30_test256_chamfer010_p512 \
  --run paper50k_md4_3k_final-0 outputs/oracle/paper50k_md4_3k_final_s0_test256_chamfer010_p512 \
  --run paper50k_md4_3k_final-10 outputs/oracle/paper50k_md4_3k_final_s10_test256_chamfer010_p512 \
  --run paper50k_md4_3k_final-20 outputs/oracle/paper50k_md4_3k_final_s20_test256_chamfer010_p512 \
  --run paper50k_md4_3k_final-30 outputs/oracle/paper50k_md4_3k_final_s30_test256_chamfer010_p512 \
  --run paper_base50k_fresh_final-0 outputs/oracle/paper_base50k_fresh_final_s0_test256_chamfer010_p512 \
  --run paper50k_matchonly3k_best-0 outputs/oracle/paper50k_matchonly3k_best_s0_test256_chamfer010_p512 \
  --run paper50k_md4_3k_best-0 outputs/oracle/paper50k_md4_3k_best_s0_test256_chamfer010_p512 \
  --fidelity-name "$FIDELITY_NAME" \
  --output-dir "$SUMMARY_DIR"

echo "== Pareto dominance =="
"$PYTHON_BIN" scripts/analyze_oracle_pareto.py \
  --summary-csv "$SUMMARY_DIR/summary.csv" \
  --output-dir "$SUMMARY_DIR/pareto"

echo "disk after:"
df -h .
du -sh cache 2>/dev/null || true
echo "paper50k pipeline complete"
