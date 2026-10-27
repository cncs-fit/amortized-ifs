"""I7-a2 checks: is the comparison with the official Djeacoumar et al. code fair to them?

``i7a2_score.py`` scores every method against a ground-truth cloud drawn with p ~ |det W| (the
test256 convention), whereas the authors' optimizer saw a target image drawn by their generator with
p ~ |det| + 0.01, and their code is mapped back to our frame through the bounding box of its
attractor.  Here
  * floor: the ground truth itself, drawn with fresh samples (p ~ |det|) and with the authors'
    probabilities and the same bounding-box alignment as their code (p ~ |det| + 0.01);
  * their code is also scored against a ground-truth cloud drawn with p ~ |det| + 0.01;
  * every method gets the best-fit similarity of ``check_external_alignment.py`` (uniform scale
    and translation minimizing the Chamfer distance to the truth), then fresh samples are scored.
Output: ``outputs/rev1/I7_official/check/{per_sample.csv,summary.json}``.
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

from data.sampler import affine_vector_probabilities
from scripts.rev1.check_external_alignment import best_similarity, conjugate
from scripts.rev1.eval_shift import eval_point_clouds
from scripts.rev1.i7a2_score import FIXED_RANGE, OURS, OUT, PIXEL, align, bbox, code_affine, score

CHECK = OUT / "check"


def their_probs(affine: torch.Tensor) -> torch.Tensor:
    p = torch.linalg.det(affine[:, :4].reshape(-1, 2, 2)).abs() + 0.01
    return p / p.sum()


def cloud(affine: torch.Tensor, probs: torch.Tensor, seed: int, device) -> torch.Tensor:
    return eval_point_clouds(affine[None], probs[None], seed=seed, fixed_range=FIXED_RANGE, device=device,
                             max_points=2048)[0]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECK.mkdir(parents=True, exist_ok=True)
    subset = json.loads((OUT / "subset.json").read_text())
    ours = {k: torch.load(ROOT / OURS[k] / "oracle_outputs.pt", map_location="cpu", weights_only=False)["oracle_params"].float()
            for k in ("ours-30", "ours-1000")}
    rows = []
    for index, name in zip(subset["indices"], subset["names"]):
        if not (OUT / "optimized" / f"{name}.json").exists():
            continue
        gt = code_affine(json.loads((OUT / "codes" / f"{name}.json").read_text()))
        gt_p, gt_q = affine_vector_probabilities(gt[None])[0], their_probs(gt)
        target = cloud(gt, gt_p, 53_100, device)  # as in i7a2_score.py
        target_q = cloud(gt, gt_q, 53_200, device)
        ref_center, ref_half = bbox(gt, gt_p)
        theirs = code_affine(json.loads((OUT / "optimized" / f"{name}.json").read_text()))
        theirs = align(theirs, their_probs(theirs), ref_center, ref_half)
        fresh = score(gt, gt_p, target, device)
        row = {"index": index, "name": name,
               "floor_fresh": fresh["chamfer"], "floor_fresh_hd95": fresh["hd95"], "floor_fresh_cov2px": fresh["cov2px"],
               "floor_their_rule": score(align(gt, gt_q, ref_center, ref_half), gt_q, target, device)["chamfer"],
               "djeacoumar_vs_their_rule": score(theirs, their_probs(theirs), target_q, device)["chamfer"]}
        fit_truth = cloud(gt, gt_p, 71_000 + index, device)
        eval_truth = cloud(gt, gt_p, 72_000 + index, device)
        for label, params, probs in (("djeacoumar", theirs, their_probs(theirs)),
                                     *((k, v[index], affine_vector_probabilities(v[index][None])[0]) for k, v in ours.items())):
            s, t = best_similarity(cloud(params, probs, 73_000 + index, device).to(device), fit_truth.to(device))
            row[f"{label}_as_scored"] = score(params, probs, eval_truth, device)["chamfer"]
            row[f"{label}_best_fit"] = score(conjugate(params, s, t), probs, eval_truth, device)["chamfer"]
            row[f"{label}_scale"] = s
        rows.append(row)
        print(name, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)
    with (CHECK / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = [k for k in rows[0] if k not in ("index", "name") and not k.endswith("_scale")]
    summary = {"num_systems": len(rows)}
    for key in keys:
        v = [r[key] for r in rows]
        summary[key] = {"mean": statistics.fmean(v), "median": statistics.median(v),
                        "success_1px": sum(x <= PIXEL for x in v) / len(v)}
    for label in ("ours-30", "ours-1000"):
        for kind in ("as_scored", "best_fit"):
            summary[f"{label}_better_than_djeacoumar_{kind}"] = sum(
                r[f"{label}_{kind}"] < r[f"djeacoumar_{kind}"] for r in rows)
    (CHECK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
