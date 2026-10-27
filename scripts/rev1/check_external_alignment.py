"""I6/I7 check: how much does the bounding-box alignment of external codes cost them?

In the external benchmark, the codes published by Djeacoumar et al. are defined up to a similarity
(their renderer normalizes every attractor by its bounding box), so we normalized them by the
bounding box of a chaos-game sample, while our predictions were made on the normalized ground truth
and needed no alignment.  Errors of the box estimate therefore count against them only.  Here every
method gets the same best-fit similarity: a uniform scale and a translation of its attractor that
minimize the Chamfer distance to the ground-truth cloud (Adam on point clouds, 300 steps); the
conjugated code is then re-scored with fresh, independent point samples.
Output: ``outputs/rev1/I6_external/alignment_check/{per_sample.csv,summary.json}``.
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
from scripts.evaluate_oracle_fidelity import point_metrics
from scripts.rev1.eval_shift import eval_point_clouds

RUN = ROOT / "outputs/rev1/I6_external/djeacoumar_all_n10"
GT_FILE = ROOT / "outputs/rev1/I6_external/djeacoumar_all.pt"
OUT = ROOT / "outputs/rev1/I6_external/alignment_check"
FIXED_RANGE = (-1.5, 1.5)
PIXEL = 3.0 / 128.0
METHODS = ("ext-chaotic_fractals", "model-100", "model-1000")


def chamfer_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(a, b)
    return 0.5 * (d.min(dim=1).values.square().mean() + d.min(dim=0).values.square().mean()).sqrt()


def best_similarity(pred: torch.Tensor, truth: torch.Tensor, steps: int = 300) -> tuple[float, torch.Tensor]:
    log_s = torch.zeros((), device=pred.device, requires_grad=True)
    t = torch.zeros(2, device=pred.device, requires_grad=True)
    opt = torch.optim.Adam([log_s, t], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = chamfer_torch(pred * log_s.exp() + t, truth)
        loss.backward()
        opt.step()
    return float(log_s.exp()), t.detach().cpu()


def conjugate(affine: torch.Tensor, s: float, t: torch.Tensor) -> torch.Tensor:
    """Code of T o f o T^-1 for T(x) = s x + t (row vectors): W' = W, b' = s b + t - t W."""
    w = affine[:, :4].reshape(-1, 2, 2)
    b = affine[:, 4:6]
    return torch.cat((affine[:, :4], s * b + t - torch.einsum("c,ncd->nd", t, w)), dim=-1)


def score(affine, probs, truth_cloud, seed, device) -> float:
    cloud = eval_point_clouds(affine[None], probs[None], seed=seed, fixed_range=FIXED_RANGE, device=device, max_points=2048)[0]
    return point_metrics(cloud, truth_cloud, device=device)["chamfer"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = torch.load(RUN / "outputs.pt", map_location="cpu", weights_only=False)
    gt = torch.load(GT_FILE, map_location="cpu", weights_only=False)
    truth, truth_p = out["variants"]["truth"].float(), out["probs"].float()
    ext_p = {f"ext-{k}": v["probs"].float() for k, v in gt["external_predictions"].items()}
    n = truth.shape[0]
    fit_truth = eval_point_clouds(truth, truth_p, seed=71_000, fixed_range=FIXED_RANGE, device=device, max_points=2048)
    eval_truth = eval_point_clouds(truth, truth_p, seed=72_000, fixed_range=FIXED_RANGE, device=device, max_points=2048)
    rows = [{"index": i, "name": out["names"][i]} for i in range(n)]
    for m_index, method in enumerate(METHODS):
        params = out["variants"][method].float()
        probs = ext_p.get(method, affine_vector_probabilities(params))
        fit_pred = eval_point_clouds(params, probs, seed=73_000 + m_index, fixed_range=FIXED_RANGE, device=device, max_points=2048)
        for i in range(n):
            s, t = best_similarity(fit_pred[i].to(device), fit_truth[i].to(device))
            aligned = conjugate(params[i], s, t)
            rows[i][f"{method}_as_scored"] = score(params[i], probs[i], eval_truth[i], 74_000 + i, device)
            rows[i][f"{method}_best_fit"] = score(aligned, probs[i], eval_truth[i], 74_000 + i, device)
            rows[i][f"{method}_scale"] = s
            rows[i][f"{method}_shift"] = float(t.norm())
        print(method, "done", flush=True)
    with (OUT / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in METHODS:
        for kind in ("as_scored", "best_fit"):
            v = [r[f"{method}_{kind}"] for r in rows]
            summary[f"{method}_{kind}"] = {"mean": statistics.fmean(v), "median": statistics.median(v),
                                          "above_0.03": sum(x > 0.03 for x in v), "success_1px": sum(x <= PIXEL for x in v) / n}
        summary[f"{method}_scale_dev_median"] = statistics.median(abs(r[f"{method}_scale"] - 1) for r in rows)
        summary[f"{method}_shift_median"] = statistics.median(r[f"{method}_shift"] for r in rows)
    for kind in ("as_scored", "best_fit"):
        summary[f"model-1000_better_than_djeacoumar_{kind}"] = sum(
            r[f"model-1000_{kind}"] < r[f"ext-chaotic_fractals_{kind}"] for r in rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
