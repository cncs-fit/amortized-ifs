"""I7-a2: write a subset of test256 as fractal codes for the official Djeacoumar et al. code.

The first ``--num`` systems of test256 (an i.i.d. sample, so the first ones are as good as any)
are written in the benchmark's column-vector convention (x' = M x + t, M = W^T, t = b).  Their
renderer draws every code with p ~ |det| + 0.01 and normalizes it by its bounding box, so ifs_p
is informative only.  Run in the main env; the images and the optimization are made by
``i7a2_run_official.py`` in the chaotic-fractals env.
Output: ``outputs/rev1/I7_official/codes/t256_XXX.json`` and ``subset.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import params_to_affine

SOURCE = ROOT / "outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512/oracle_outputs.pt"
OUT = ROOT / "outputs/rev1/I7_official"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=16)
    args = parser.parse_args()
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    w, b = params_to_affine(payload["target_params"].float())
    (OUT / "codes").mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(args.num):
        m = w[i].transpose(-1, -2)
        det = torch.linalg.det(w[i]).abs()
        name = f"t256_{i:03d}"
        code = {"ifs_m": m.tolist(), "ifs_t": b[i].tolist(), "ifs_p": (det / det.sum()).tolist()}
        (OUT / "codes" / f"{name}.json").write_text(json.dumps(code))
        names.append(name)
    (OUT / "subset.json").write_text(json.dumps({"source": str(SOURCE.relative_to(ROOT)), "indices": list(range(args.num)),
                                                 "names": names}, indent=2))
    print(f"wrote {len(names)} codes to {OUT / 'codes'}")


if __name__ == "__main__":
    main()
