"""I7-a2 protocol check: does our run of the official code reproduce the results Djeacoumar et al. publish?

For benchmark systems optimized by ``i7a2_run_official.py --benchmark --suffix _bench`` (their own target
images, default configuration), scores our run and the code they publish for the same system
(``export_chaotic_codes.py``) against the same ground truth in the same way: every code is drawn with
their rule p ~ |det| + 0.01 and mapped to the frame of the ground truth through the bounding box of its
attractor (as in ``i7a2_score.py``), and Chamfer / HD95 / coverage@2px are computed on point clouds.
Output: ``outputs/rev1/I7_official/benchmark_repro/{per_sample.csv,summary.json}``.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.rev1.eval_shift import eval_point_clouds
from scripts.rev1.i7a2_check import their_probs
from scripts.rev1.i7a2_score import FIXED_RANGE, OUT, PIXEL, align, bbox, code_affine, score

CODES = ROOT / "refs/chaotic_data/full_dataset/ifs_codes"
PUBLISHED = ROOT / "outputs/rev1/I6_external/djeacoumar_optimized_json/chaotic_fractals"
RUNS = OUT / "optimized_bench"
DEST = OUT / "benchmark_repro"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEST.mkdir(parents=True, exist_ok=True)
    timing = {}
    if (OUT / "timing_bench.jsonl").exists():
        for line in (OUT / "timing_bench.jsonl").read_text().splitlines():
            rec = json.loads(line)
            timing[rec["name"]] = rec["seconds"]
    rows = []
    for path in sorted(RUNS.glob("fdb_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        name = path.stem
        gt = code_affine(json.loads((CODES / f"{name}.json").read_text()))
        gt_p = their_probs(gt)
        # score in our domain: the ground truth is first mapped to the box of prepare_external_ifs.py
        # (centered, half-extent 1.125), then every code is aligned to it through its own bounding box
        _, half = bbox(gt, gt_p)
        scale = 1.125 / half
        target = align(gt, gt_p, torch.zeros(2), 1.125)
        cloud = eval_point_clouds(target[None], gt_p[None], seed=53_100, fixed_range=FIXED_RANGE, device=device,
                                  max_points=2048)[0]
        ref_center, ref_half = torch.zeros(2), 1.125
        row = {"name": name, "num_maps": int(gt.shape[0]), "seconds": timing.get(name), "scale": scale,
               "smax": float(torch.linalg.svdvals(gt[:, :4].reshape(-1, 2, 2)).max())}
        for label, src in (("ours_run", path), ("published", PUBLISHED / f"{name}.json")):
            code = code_affine(json.loads(src.read_text()))
            p = their_probs(code)
            for key, value in score(align(code, p, ref_center, ref_half), p, cloud, device).items():
                row[f"{label}_{key}"] = value
        rows.append(row)
        print(name, {k: round(v, 4) for k, v in row.items() if isinstance(v, float) and k.endswith("chamfer")}, flush=True)
    if not rows:
        print("no runs yet")
        return
    with (DEST / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"num_systems": len(rows), "names": [r["name"] for r in rows]}
    for label in ("ours_run", "published"):
        cds = [r[f"{label}_chamfer"] for r in rows]
        summary[label] = {"chamfer_mean": statistics.fmean(cds), "chamfer_median": statistics.median(cds),
                          "above_0.03": sum(c > 0.03 for c in cds), "success_1px": sum(c <= PIXEL for c in cds) / len(cds)}
    summary["ours_run_better"] = sum(r["ours_run_chamfer"] < r["published_chamfer"] for r in rows)
    secs = [r["seconds"] for r in rows if r["seconds"]]
    if secs:
        summary["seconds_mean"] = statistics.fmean(secs)
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
