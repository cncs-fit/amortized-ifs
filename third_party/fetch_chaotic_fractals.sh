#!/usr/bin/env bash
# Fetch the official code and benchmark data of Djeacoumar et al. (2025), used by the external-benchmark
# and official-code experiments of the revision (scripts/rev1/run_external.sh, run_i7a2.sh).
# Nothing of theirs is redistributed here; this script downloads it into refs/ (git-ignored):
#   * code: github.com/sshadr/chaotic-fractals at a pinned commit, with submodules.  The bundled
#     rasterizer needs one added line (#include <cstdint>) to build with GCC >= 13.
#   * data: their benchmark targets (full_dataset) and published optimized codes, from their project page.
# Existing directories are left untouched.  Then build the env with setup_chaotic_env.sh.
# DEST overrides the target directory (default: refs/ at the repository root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$ROOT/refs}"
COMMIT=0e4751320e1860889e6c65603198dbab31b724c3
DATA_URL=https://chaotic-fractals.mpi-inf.mpg.de/data
mkdir -p "$DEST"

REPO="$DEST/chaotic-fractals"
if [ ! -d "$REPO" ]; then
  git clone https://github.com/sshadr/chaotic-fractals "$REPO"
  git -C "$REPO" checkout -q "$COMMIT"
  git -C "$REPO" submodule update --init --recursive
  HDR="$REPO/lib/diff-isogaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
  grep -q '#include <cstdint>' "$HDR" || sed -i 's|^#pragma once$|#pragma once\n#include <cstdint>  // added for GCC>=13|' "$HDR"
  grep -q '#include <cstdint>' "$HDR" || { echo "build fix not applied: $HDR" >&2; exit 1; }
else
  echo "skip: $REPO exists"
fi

DATA="$DEST/chaotic_data"
if [ ! -d "$DATA" ]; then
  mkdir -p "$DATA"
  # the zips have no top-level folder; layout used by scripts/rev1: full_dataset/, optimized/
  for pair in full_dataset:full_dataset optimized_fractal_codes:optimized; do
    name=${pair%%:*}; sub=${pair##*:}
    curl -fsSL -o "$DATA/$name.zip" "$DATA_URL/$name.zip"
    unzip -q "$DATA/$name.zip" -d "$DATA/$sub"
  done
else
  echo "skip: $DATA exists"
fi
