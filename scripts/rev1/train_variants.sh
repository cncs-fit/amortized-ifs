#!/usr/bin/env bash
# Train the estimator variants of the revision (Section 6.5, Table 4 and Appendix D; spectral output,
# Section 4.1; reflection prior, Section 6.6).  Each run uses the encoder, head, data, and 50k-step
# recipe of the base estimator and writes outputs/rev1/<variant>/<timestamp>/ (about 3-5 h each on an
# RTX 3090).  The released checkpoints (checkpoints/rev1/) make this step optional.
# Usage: bash scripts/rev1/train_variants.sh [config-name ...]   (default: all eight)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
NAMES=("$@")
[[ ${#NAMES[@]} -gt 0 ]] || NAMES=(i1_hung_nofp i1_ordered i2_recon_only i2_multihyp4 i2_collage i2_collage_c070 i3_spectral i4b_reflect)
for name in "${NAMES[@]}"; do
  cfg="configs/rev1/${name}.yaml"
  echo "== train $cfg $(date -Is)"
  if [[ "$name" == i1_hung_nofp ]]; then
    "$PY" scripts/train_phase0.py --config "$cfg"   # base recipe with the fixed-point term switched off
  else
    "$PY" scripts/rev1/train_rev1.py --config "$cfg"
  fi
done
