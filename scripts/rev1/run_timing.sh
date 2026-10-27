#!/usr/bin/env bash
# Wall-clock times quoted in the revision, to be run on an otherwise idle GPU: the cost of the
# multi-hypothesis selection (Section 6.5) and the per-image times of Table 7 (the external benchmark
# re-run with the arguments of run_external.sh into a separate folder).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
RUNS="${RUNS:-checkpoints/rev1}"
E=outputs/rev1/I6_external
"$PY" scripts/rev1/time_multihyp.py --mh-run "$RUNS/I2_multihyp4"   # -> outputs/rev1/timing/multihyp.json (needs the base 100-step run)
"$PY" scripts/rev1/eval_shift.py --output-dir outputs/rev1/timing/djeacoumar_all_n10_idle \
  --run-dir checkpoints/explore_n10_base100k --gt-source file --gt-file $E/djeacoumar_all.pt --prob-mode file \
  --external-preds --num-samples 101 --steps 30 100 1000 --batch-size 16 2>&1 | grep -v -i warn | tail -12
# single-image times (batch 1) for Table 7 and Section 6.8: four systems each
"$PY" scripts/rev1/eval_shift.py --output-dir outputs/rev1/timing/djeacoumar_n10_batch1 \
  --run-dir checkpoints/explore_n10_base100k --gt-source file --gt-file $E/djeacoumar_all.pt --prob-mode file \
  --num-samples 4 --steps 30 100 1000 --batch-size 1 2>&1 | grep -v -i warn | tail -3
"$PY" scripts/run_mnist_p0.py --run-dir checkpoints/explore_n10_base100k --dataset mnist --num-samples 4 \
  --selection balanced --steps 0 30 100 --batch-size 1 --output-dir outputs/rev1/timing/mnist_batch1
