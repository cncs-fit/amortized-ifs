"""I6/I7: export the published optimized codes of the Djeacoumar et al. (2025) benchmark to JSON.

Run in the separate ``chaotic-fractals`` env (it needs the authors' ``system.blocks`` classes to
unpickle their checkpoints, which store whole modules):

    conda run -n chaotic-fractals python scripts/rev1/export_chaotic_codes.py

Writes outputs/rev1/I6_external/djeacoumar_optimized_json/<method>/<name>.json with the
benchmark's column-vector convention (ifs_m, ifs_t) plus per-map opacity for the authors'
method.  Probabilities are not stored: the benchmark renderer always uses |det| + 0.01.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "refs/chaotic-fractals"
DATA = ROOT / "refs/chaotic_data/optimized"
OUT = ROOT / "outputs/rev1/I6_external/djeacoumar_optimized_json"
sys.path.insert(0, str(REPO / "base"))

import torch  # noqa: E402


def export_ours() -> None:
    target = OUT / "chaotic_fractals"
    target.mkdir(parents=True, exist_ok=True)
    opacities = []
    for path in sorted((DATA / "ours").glob("*.pth")):
        code = torch.load(path, map_location="cpu", weights_only=False)  # official release files
        ms, ts, ops = [], [], []
        for fn in code.contractive_functions:
            ll = fn.ll
            with torch.no_grad():
                w = ll.weights().detach().cpu().reshape(2, 2)
                b = ll.bias.detach().cpu().reshape(2) if torch.is_tensor(ll.bias) else ll.bias().detach().cpu().reshape(2)
                op = float(torch.clamp(ll.opacity.detach().cpu(), 0.0, 1.0))
            ms.append(w.tolist())
            ts.append(b.tolist())
            ops.append(op)
        opacities.extend(ops)
        (target / f"{path.stem}.json").write_text(json.dumps({"ifs_m": ms, "ifs_t": ts, "opacity": ops}))
    low = sum(o < 0.5 for o in opacities)
    print(f"chaotic_fractals: {len(list(target.glob('*.json')))} codes, maps={len(opacities)}, opacity<0.5: {low}")


def copy_json_baselines() -> None:
    for method in ("neural_regression", "cuckoo", "evol_pcov"):
        target = OUT / method
        target.mkdir(parents=True, exist_ok=True)
        for path in (DATA / method).glob("fdb_*.json"):
            shutil.copy(path, target / path.name)
        print(f"{method}: {len(list(target.glob('*.json')))} codes")


if __name__ == "__main__":
    export_ours()
    copy_json_baselines()
