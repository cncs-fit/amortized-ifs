#!/usr/bin/env bash
# Official code of Djeacoumar et al. on the first 16 test256 systems (Section 6.7, Table F.3): the
# systems are written in their code format, each target is drawn by their ground-truth generator and
# optimized with their default configuration (10 maps, 15,000 iterations; about 50 min per system on
# an RTX 3090), and all methods are scored with our metrics.  Needs their code and env
# (third_party/fetch_chaotic_fractals.sh, third_party/setup_chaotic_env.sh) and our test256 runs at
# 0/30 steps (run_paper50k_e1_hero.sh), 100 steps (run_training_signal.sh), and 1000 steps
# (run_advice027_long_horizon.sh).  Resumable: finished systems are skipped.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
CHAOS_ENV="${CHAOS_ENV:-chaotic-fractals}"
"$PY" scripts/rev1/i7a2_make_inputs.py --num 16
conda run --no-capture-output -n "$CHAOS_ENV" python scripts/rev1/i7a2_run_official.py --names $(seq -f "t256_%03g" 0 15)
"$PY" scripts/rev1/i7a2_score.py    # -> outputs/rev1/I7_official/{per_sample.csv,summary.json}
"$PY" scripts/rev1/i7a2_check.py    # probability rule, best-fit alignment, floor -> outputs/rev1/I7_official/check/
# protocol checks (Appendix F): their code with the true number of maps, and on the six most strongly
# contracting systems of their own benchmark (compared with the codes they publish; run_external.sh first)
conda run --no-capture-output -n "$CHAOS_ENV" python scripts/rev1/i7a2_run_official.py --num-maps 4 --suffix _n4 \
  --tag rev1_i7b_n4 --names $(seq -f "t256_%03g" 0 15)
"$PY" scripts/rev1/i7a2_score.py --suffix _n4
conda run --no-capture-output -n "$CHAOS_ENV" python scripts/rev1/i7a2_run_official.py --benchmark --suffix _bench \
  --tag rev1_i7b_bench --names fdb_104 fdb_96 fdb_62 fdb_23 fdb_19 fdb_79
"$PY" scripts/rev1/i7a2_score_benchmark.py   # -> outputs/rev1/I7_official/benchmark_repro/
"$PY" scripts/rev1/plot_i7a2_examples.py     # Figure F.2
