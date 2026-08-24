#!/usr/bin/env bash
# Reproduce the reconstruction-auxiliary ablation table (base / match-only / +aux
# at 0 and 30 refinement steps) from the released checkpoints, without retraining.
# Mirrors the evaluation stage of run_paper50k_pipeline.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN="${BASE_RUN:-checkpoints/paper_base50k}"
MATCH_RUN="${MATCH_RUN:-checkpoints/paper50k_matchonly3k}"
MD4_RUN="${MD4_RUN:-checkpoints/paper50k_md4_3k}"
FIDELITY_NAME="fidelity_res128_traj16_steps1024_sigma2_robust2048"
SUMMARY_DIR="outputs/oracle/paper50k_base_matchonly_md4_refinement_test256_chamfer010_p512_robust2048"

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

for steps in 0 10 20 30; do
  run_oracle paper_base50k_fresh_best "$BASE_RUN" best "$steps"
  run_oracle paper50k_matchonly3k_final "$MATCH_RUN" final "$steps"
  run_oracle paper50k_md4_3k_final "$MD4_RUN" final "$steps"
done

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
  --fidelity-name "$FIDELITY_NAME" \
  --output-dir "$SUMMARY_DIR"

echo "== per-sample wins (base vs +aux / match-only) =="
"$PYTHON_BIN" scripts/compare_oracle_per_sample_wins.py \
  --run base-0 outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512 \
  --run base-10 outputs/oracle/paper_base50k_fresh_best_s10_test256_chamfer010_p512 \
  --run base-20 outputs/oracle/paper_base50k_fresh_best_s20_test256_chamfer010_p512 \
  --run base-30 outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512 \
  --run matchonly-0 outputs/oracle/paper50k_matchonly3k_final_s0_test256_chamfer010_p512 \
  --run matchonly-10 outputs/oracle/paper50k_matchonly3k_final_s10_test256_chamfer010_p512 \
  --run matchonly-20 outputs/oracle/paper50k_matchonly3k_final_s20_test256_chamfer010_p512 \
  --run matchonly-30 outputs/oracle/paper50k_matchonly3k_final_s30_test256_chamfer010_p512 \
  --run md4-0 outputs/oracle/paper50k_md4_3k_final_s0_test256_chamfer010_p512 \
  --run md4-10 outputs/oracle/paper50k_md4_3k_final_s10_test256_chamfer010_p512 \
  --run md4-20 outputs/oracle/paper50k_md4_3k_final_s20_test256_chamfer010_p512 \
  --run md4-30 outputs/oracle/paper50k_md4_3k_final_s30_test256_chamfer010_p512 \
  --comparison base_vs_md4_0 base-0 md4-0 \
  --comparison base_vs_md4_10 base-10 md4-10 \
  --comparison base_vs_md4_20 base-20 md4-20 \
  --comparison base_vs_md4_30 base-30 md4-30 \
  --comparison base_vs_matchonly_0 base-0 matchonly-0 \
  --comparison base_vs_matchonly_10 base-10 matchonly-10 \
  --comparison base_vs_matchonly_20 base-20 matchonly-20 \
  --comparison base_vs_matchonly_30 base-30 matchonly-30 \
  --comparison md4_vs_matchonly_0 md4-0 matchonly-0 \
  --comparison md4_vs_matchonly_10 md4-10 matchonly-10 \
  --comparison md4_vs_matchonly_20 md4-20 matchonly-20 \
  --comparison md4_vs_matchonly_30 md4-30 matchonly-30 \
  --output-csv "$SUMMARY_DIR/p0_per_sample_wins.csv"

echo "== paired bootstrap CI (base+30 vs +aux+30) =="
"$PYTHON_BIN" scripts/bootstrap_paired_ci.py \
  --left-dir outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512 \
  --right-dir outputs/oracle/paper50k_md4_3k_final_s30_test256_chamfer010_p512 \
  --comparison-label base_vs_md4_30 \
  --output-json "$SUMMARY_DIR/p0_base_vs_md4_30_bootstrap.json"

echo "== (W,b) parameter-error diagnostic =="
"$PYTHON_BIN" scripts/summarize_oracle_param_error.py \
  --run base50k_best-0 outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512 \
  --run base50k_best-30 outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512 \
  --run matchonly3k_final-0 outputs/oracle/paper50k_matchonly3k_final_s0_test256_chamfer010_p512 \
  --run matchonly3k_final-30 outputs/oracle/paper50k_matchonly3k_final_s30_test256_chamfer010_p512 \
  --run md4_aux3k_final-0 outputs/oracle/paper50k_md4_3k_final_s0_test256_chamfer010_p512 \
  --run md4_aux3k_final-30 outputs/oracle/paper50k_md4_3k_final_s30_test256_chamfer010_p512 \
  --output-dir "$SUMMARY_DIR/param_error" \
  --output-prefix recon_aux_param_error

echo "recon-aux evaluation complete: $SUMMARY_DIR"
