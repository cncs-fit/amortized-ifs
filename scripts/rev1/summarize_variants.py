"""Summarize revision-1 variants against the paper's base on test256 (quality only).

Reads fidelity JSONs/per-sample CSVs written by run_variant_eval.sh and the paper's base
runs, and reports mean density SSE / Chamfer / HD95 / coverage@2px / (W,b) distance,
plus per-sample Chamfer wins of base over each variant at the same step.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.analyze_identifiability import affine_set_distance
from data.sampler import affine_matrices_to_vector, params_to_affine

FID = "fidelity_res128_traj16_steps1024_sigma2_robust2048"
STEPS = (0, 10, 20, 30, 100, 1000)
BASE = {s: ROOT / f"outputs/oracle/paper_base50k_fresh_best_s{s}_test256_chamfer010_p512" for s in (0, 10, 20, 30)}
BASE[1000] = ROOT / "outputs/oracle/advice027_long_horizon_test256_chamfer010_p512_bs256/model_r1_s1000"
# base re-run with run_variant_eval.sh (same protocol as the variants) takes precedence where it exists
for _s in STEPS:
    _same = ROOT / f"outputs/rev1/variant_eval/base_s{_s}_test256"
    if (_same / f"{FID}.json").exists():
        BASE[_s] = _same


def load(dir_: Path) -> dict | None:
    if not (dir_ / f"{FID}.json").exists():
        return None
    rows = [r for r in csv.DictReader((dir_ / f"{FID}_per_sample.csv").open()) if r["variant"] == "oracle"]
    payload = torch.load(dir_ / "oracle_outputs.pt", map_location="cpu", weights_only=False)
    w, b = params_to_affine(payload["target_params"].float())
    target = affine_matrices_to_vector(w, b)
    pred = payload["oracle_params"].float()
    param = [affine_set_distance(pred[i], target[i])["param_distance"] for i in range(pred.shape[0])]
    col = lambda key: [float(r[key]) for r in rows]
    return {
        "density_sse": col("density_sse_to_input"),
        "chamfer": col("chamfer_to_target_points"),
        "hd95": col("hausdorff_p95_to_target_points"),
        "coverage_2px": col("coverage_symmetric_2px_to_target_points"),
        "param_distance": param,
    }


def mean(values):
    return sum(values) / len(values)


def main() -> None:
    labels = sys.argv[1:] or sorted({p.name.rsplit("_s", 1)[0] for p in (ROOT / "outputs/rev1/variant_eval").glob("*_test256")} - {"base"})
    table = []
    for step in STEPS:
        base = load(BASE[step]) if step in BASE else None
        entries = [("base", base)] + [(lab, load(ROOT / f"outputs/rev1/variant_eval/{lab}_s{step}_test256")) for lab in labels]
        for label, data in entries:
            if data is None:
                continue
            row = {"step": step, "label": label}
            row.update({k: mean(v) for k, v in data.items()})
            if label != "base" and base is not None:
                row["base_wins_chamfer"] = sum(a < b for a, b in zip(base["chamfer"], data["chamfer"]))
            table.append(row)
    out = ROOT / "outputs/rev1/variant_eval/summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in table for k in r}, key=lambda k: ["step", "label", "density_sse", "chamfer", "hd95", "coverage_2px", "param_distance", "base_wins_chamfer"].index(k)))
        writer.writeheader()
        writer.writerows(table)
    for r in table:
        print(f"s{r['step']:<5d} {r['label']:22s} SSE={r['density_sse']:.3e} CD={r['chamfer']:.4f} HD95={r['hd95']:.4f} "
              f"cov={r['coverage_2px']:.4f} WB={r['param_distance']:.3f}" + (f" base_wins={r['base_wins_chamfer']}" if "base_wins_chamfer" in r else ""))


if __name__ == "__main__":
    main()
