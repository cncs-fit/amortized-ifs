#!/usr/bin/env bash
# External IFS (Section 6.7, Table 7, Appendix F): the 101-system benchmark of Djeacoumar et al. (2025)
# with the codes they publish for their method and three baselines, and six hand-specified IFS.
# Requires the benchmark data and their code (bash third_party/fetch_chaotic_fractals.sh) and the
# separate env for their code (bash third_party/setup_chaotic_env.sh), which is needed only to export
# their published results (pickled modules) to JSON.  About 4 h on an RTX 3090.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
CHAOS_ENV="${CHAOS_ENV:-chaotic-fractals}"
E=outputs/rev1/I6_external
N4=checkpoints/paper_base50k
N10=checkpoints/explore_n10_base100k
[[ -d "$E/djeacoumar_optimized_json" ]] || conda run -n "$CHAOS_ENV" python scripts/rev1/export_chaotic_codes.py
[[ -f "$E/djeacoumar_all.pt" ]] || "$PY" scripts/rev1/prepare_external_ifs.py   # -> djeacoumar_{all,le4}.pt, classical.pt
run() {
  local out="$1"; shift
  if [[ -f "$out/summary.json" ]]; then echo "skip $out"; return; fi
  echo "== $(date -Is) $out $*"
  "$PY" scripts/rev1/eval_shift.py --output-dir "$out" "$@" 2>&1 | grep -v -i warn | tail -12
}
run $E/djeacoumar_all_n10 --run-dir $N10 --gt-source file --gt-file $E/djeacoumar_all.pt --prob-mode file --external-preds --num-samples 101 --steps 30 100 1000 --batch-size 16
run $E/djeacoumar_le4_n4 --run-dir $N4 --gt-source file --gt-file $E/djeacoumar_le4.pt --prob-mode file --external-preds --num-samples 43 --steps 30 100 1000
run $E/classical_n4 --run-dir $N4 --gt-source file --gt-file $E/classical.pt --prob-mode file --num-samples 6 --steps 30 100 1000
"$PY" scripts/rev1/check_external_alignment.py      # best-fit similarity check (Appendix F)
"$PY" scripts/rev1/plot_external_examples.py        # Figure F.1
