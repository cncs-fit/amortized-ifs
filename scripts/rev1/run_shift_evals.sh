#!/usr/bin/env bash
# Generalization under observation mismatch and prior shift (Section 6.6, Tables 5-6, Appendix E):
# renderer sweep, reflections (R), independent selection probabilities (P), and the estimator trained
# with reflections (checkpoints/rev1/I4b_reflect).  Each setting evaluates the base estimator and
# random-r4 at equal budgets with eval_shift.py; the table rows are written at the end.
# About 10-14 h on an RTX 3090.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
RUNS="${RUNS:-checkpoints/rev1}"
run() {  # run <out_dir> <args...>
  local out="$1"; shift
  if [[ -f "$out/summary.json" ]]; then echo "skip $out"; return; fi
  echo "== $(date -Is) $out $*"
  "$PY" scripts/rev1/eval_shift.py --output-dir "$out" "$@" 2>&1 | grep -v -i warn | tail -6
}
R=outputs/rev1/I5_renderer
run $R/default --steps 30   # the training renderer itself (first row of Tables 6 and E.1)
for t in 2 4 8 64 256; do run $R/traj$t --obs-num-trajectories $t --steps 30; done
for s in 0 1 3 4; do run $R/sigma$s --obs-sigma $s --steps 30; done
for r in 64 96 256; do run $R/res$r --obs-resolution $r --steps 30; done
run $R/splat_bilinear --obs-splat bilinear --steps 30
run $R/burnin0 --obs-burn-in 0 --steps 30
run $R/dense_t256_s4096 --obs-num-trajectories 256 --obs-num-steps 4096 --steps 30
# renderer known at test time: refine with the observation's renderer settings
for t in 2 4; do run $R/traj${t}_refobs --obs-num-trajectories $t --refine-renderer obs --steps 30; done
for s in 0 4; do run $R/sigma${s}_refobs --obs-sigma $s --refine-renderer obs --steps 30; done
# test-time pre-blur of sharper observations to the training smoothing
run $R/sigma0_preblur2 --obs-sigma 0 --input-preblur 2 --steps 30
run $R/sigma1_preblur2 --obs-sigma 1 --input-preblur 2 --steps 30
run $R/sigma1_preblur1.73 --obs-sigma 1 --input-preblur 1.732 --steps 30
R4=outputs/rev1/I4_prior
run $R4/reflect_detpen0.1 --gt-source reflect --steps 30 60
run $R4/reflect_detpen0 --gt-source reflect --det-negative-penalty-weight 0 --steps 30 60
for a in 0.25 0.5 0.75 1.0; do run $R4/probmix_a$a --prob-mode mix --prob-alpha $a --steps 30 60; done
# estimator trained with reflections, on R and on test256 (variant protocol)
for pen in 0 0.1; do
  run $R4/I4b_reflect_detpen$pen --run-dir "$RUNS/I4b_reflect" --gt-source reflect --det-negative-penalty-weight $pen --steps 30 60
done
bash scripts/rev1/run_variant_eval.sh I4b_reflect "$RUNS/I4b_reflect"
"$PY" scripts/rev1/make_robustness_tables.py           # -> outputs/rev1/paper/robustness_*_rows.tex
