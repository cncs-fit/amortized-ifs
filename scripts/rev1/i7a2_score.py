"""I7-a2: score the official Djeacoumar et al. results on the test256 subset against our method.

The authors' renderer normalizes every attractor by its bounding box (center, longer side), so
their optimized code is defined up to a translation and a uniform scale.  We map it back to our
frame by the similarity that takes the bounding box of its attractor to that of the ground truth
(the conjugation keeps the linear parts), then score every method with one function: Chamfer,
HD95 and coverage@2px between attractor point clouds (standard chaos game 16 x 1024, burn-in
128, 2048 points), ground truth with p ~ |det W| as in test256, the authors' codes with their
rule p ~ |det| + 0.01.  Our predictions are taken from the stored refinement outputs for the
same systems (0/30 steps: main comparison; 100: variant evaluation; 1000: long-horizon run).
Output: ``outputs/rev1/I7_official/{per_sample.csv,summary.json}``; ``--suffix _n4`` scores the run with
the true number of maps (``optimized_n4/``) into ``per_sample_n4.csv`` and ``summary_n4.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import affine_vector_probabilities
from scripts.evaluate_oracle_fidelity import point_metrics
from scripts.rev1.eval_shift import chaos_points, eval_point_clouds

OUT = ROOT / "outputs/rev1/I7_official"
FIXED_RANGE = (-1.5, 1.5)
PIXEL = 3.0 / 128.0
OURS = {  # label -> oracle_outputs.pt with oracle_params for all of test256
    "ours-0": "outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512",
    "ours-30": "outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512",
    "ours-100": "outputs/rev1/variant_eval/base_s100_test256",
    "ours-1000": "outputs/oracle/advice027_long_horizon_test256_chamfer010_p512_bs256/model_r1_s1000",
    "random-r4-1000": "outputs/oracle/advice027_long_horizon_test256_chamfer010_p512_bs256/random_r4_s1000",
}


def code_affine(code: dict) -> torch.Tensor:
    w = torch.tensor(code["ifs_m"], dtype=torch.float32).transpose(-1, -2)  # column -> row convention
    b = torch.tensor(code["ifs_t"], dtype=torch.float32)
    return torch.cat((w.reshape(-1, 4), b), dim=-1)


def bbox(affine: torch.Tensor, probs: torch.Tensor) -> tuple[torch.Tensor, float]:
    pts = chaos_points(affine[None], probs[None], num_trajectories=16, num_steps=4096, burn_in=128,
                       generator=torch.Generator().manual_seed(0))[0].cpu()
    lo, hi = pts.min(0).values, pts.max(0).values
    return 0.5 * (lo + hi), float((hi - lo).max() / 2)


def align(affine: torch.Tensor, probs: torch.Tensor, ref_center: torch.Tensor, ref_half: float) -> torch.Tensor:
    """Conjugate by T(x) = c_ref + s (x - c) so that the attractor's bounding box matches the reference."""
    center, half = bbox(affine, probs)
    s = ref_half / half
    w = affine[:, :4].reshape(-1, 2, 2)
    b = affine[:, 4:6]
    # f'(y) = T(f(T^-1 y)), T^-1 y = c + (y - c_ref)/s  =>  W' = W,  b' = s (b + c W - c) + c_ref - c_ref W
    b_new = (s * (b + torch.einsum("c,ncd->nd", center, w) - center) + ref_center
             - torch.einsum("c,ncd->nd", ref_center, w))
    return torch.cat((affine[:, :4], b_new), dim=-1)


def score(affine: torch.Tensor, probs: torch.Tensor, target_cloud: torch.Tensor, device) -> dict:
    cloud = eval_point_clouds(affine[None], probs[None], seed=53_107, fixed_range=FIXED_RANGE,
                              device=device, max_points=2048)[0]
    m = point_metrics(cloud, target_cloud, device=device, coverage_thresholds={"2px": 2 * PIXEL})
    return {"chamfer": m["chamfer"], "hd95": m["hausdorff_p95"], "cov2px": m["coverage_symmetric_2px"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="score optimized<suffix>/ (e.g. _n4); outputs get the suffix")
    sfx = parser.parse_args().suffix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = json.loads((OUT / "subset.json").read_text())
    done = [(i, n) for i, n in zip(subset["indices"], subset["names"]) if (OUT / f"optimized{sfx}" / f"{n}.json").exists()]
    if not done:
        print("no optimized results yet")
        return
    ours = {k: torch.load(ROOT / v / "oracle_outputs.pt", map_location="cpu", weights_only=False)["oracle_params"].float()
            for k, v in OURS.items() if (ROOT / v / "oracle_outputs.pt").exists()}
    timing = {}
    if (OUT / f"timing{sfx}.jsonl").exists():
        for line in (OUT / f"timing{sfx}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            timing[rec["name"]] = rec["seconds"]
    rows = []
    for index, name in done:
        gt = code_affine(json.loads((OUT / "codes" / f"{name}.json").read_text()))
        gt_p = affine_vector_probabilities(gt[None])[0]
        target_cloud = eval_point_clouds(gt[None], gt_p[None], seed=53_100, fixed_range=FIXED_RANGE,
                                         device=device, max_points=2048)[0]
        ref_center, ref_half = bbox(gt, gt_p)
        theirs = code_affine(json.loads((OUT / f"optimized{sfx}" / f"{name}.json").read_text()))
        their_p = torch.linalg.det(theirs[:, :4].reshape(-1, 2, 2)).abs() + 0.01
        their_p = their_p / their_p.sum()
        theirs = align(theirs, their_p, ref_center, ref_half)
        row = {"index": index, "name": name, "official_seconds": timing.get(name)}
        for key, value in score(theirs, their_p, target_cloud, device).items():
            row[f"djeacoumar_{key}"] = value
        row["djeacoumar_smax"] = float(torch.linalg.svdvals(theirs[:, :4].reshape(-1, 2, 2)).max())
        for label, params in ours.items():
            pred = params[index]
            for key, value in score(pred, affine_vector_probabilities(pred[None])[0], target_cloud, device).items():
                row[f"{label}_{key}"] = value
        rows.append(row)
        print(name, {k: round(v, 4) for k, v in row.items() if k.endswith("chamfer")}, flush=True)
    with (OUT / f"per_sample{sfx}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = ["djeacoumar"] + list(ours)
    summary = {"num_systems": len(rows), "names": [r["name"] for r in rows]}
    for label in labels:
        cds = [r[f"{label}_chamfer"] for r in rows]
        summary[label] = {
            "chamfer_mean": statistics.fmean(cds), "chamfer_median": statistics.median(cds),
            "hd95_mean": statistics.fmean(r[f"{label}_hd95"] for r in rows),
            "cov2px_mean": statistics.fmean(r[f"{label}_cov2px"] for r in rows),
            "success_1px": sum(c <= PIXEL for c in cds) / len(cds),
        }
        if label != "djeacoumar":
            summary[label]["better_than_djeacoumar"] = sum(r[f"{label}_chamfer"] < r["djeacoumar_chamfer"] for r in rows)
    secs = [r["official_seconds"] for r in rows if r["official_seconds"]]
    if secs:
        summary["djeacoumar_seconds_mean"] = statistics.fmean(secs)
    (OUT / f"summary{sfx}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
