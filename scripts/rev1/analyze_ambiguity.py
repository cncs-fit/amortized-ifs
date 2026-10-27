"""I2(a): how target ambiguity relates to the parameter-matching surrogate (R2-W1, R1-2a).

1. Reference bank: M self-generated (theta, y) pairs from the training generator
   (seeds disjoint from training/validation/test).
2. Input ambiguity of each test256 sample y_j: among bank images with density L2
   distance <= eps (near-identical inputs), the median Hungarian parameter-set distance
   to theta_j.  Large values mean the same image is produced by distant parameter sets.
3. Disagreement: the base estimator's parameter error vs. its reconstruction errors
   (0 / 30 / 1000 refinement steps), stratified by ambiguity.
4. Irreducible part of the matching loss: for a squared-error target, the Bayes risk is
   E[Var(theta | y)] = 1/2 E||theta - theta'||^2 for theta, theta' drawn from the same
   posterior.  Using bank neighbors within eps as draws from p(theta | y ~ y_j) gives an
   estimate (an upper bound in practice, since neighbors are only near-identical) that we
   compare with the trained estimator's matching loss.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.dataset import generate_train_cache_batched_from_iterable_config, make_fixed_dataset_from_iterable_config
from data.sampler import affine_matrices_to_vector, params_to_affine
from scripts.evaluate_checkpoint import _config_from_payload
from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()
import matplotlib.pyplot as plt  # noqa: E402

LONG = ROOT / "outputs/oracle/advice027_long_horizon_test256_chamfer010_p512_bs256/analysis/per_sample_metrics.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="checkpoints/paper_base50k")
    p.add_argument("--bank-size", type=int, default=131072)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--bank-seed", type=int, default=9_000_000)
    p.add_argument("--eps", type=float, nargs="+", default=[0.013, 0.0168, 0.02, 0.025])
    p.add_argument("--main-eps", type=float, default=0.0168)
    p.add_argument("--knn", type=int, default=32)
    p.add_argument("--output-dir", default=str(ROOT / "outputs/rev1/I2_ambiguity"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def affine_of(params: torch.Tensor) -> torch.Tensor:
    w, b = params_to_affine(params)
    return affine_matrices_to_vector(w, b)


_PERMS = None


def matched_sq_cost(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean matched squared (W,b) cost between sets a [P,n,6] and b [P,n,6] (n small)."""
    global _PERMS
    n = a.shape[1]
    if _PERMS is None or _PERMS.shape[1] != n:
        _PERMS = torch.tensor(list(itertools.permutations(range(n))))
    cost = (a[:, :, None, :] - b[:, None, :, :]).square().sum(-1)  # [P,n,n]
    rows = torch.arange(n)
    return cost[:, rows[None, :], _PERMS].sum(-1).min(dim=1).values / n


def load_per_sample() -> dict[tuple[str, int], dict[int, dict]]:
    out: dict[tuple[str, int], dict[int, dict]] = {}
    with LONG.open() as handle:
        for row in csv.DictReader(handle):
            out.setdefault((row["family"], int(row["step"])), {})[int(row["sample_index"])] = row
    return out


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads((ROOT / args.run_dir / "result.json").read_text())
    config = _config_from_payload(result["train_config"])

    test = make_fixed_dataset_from_iterable_config(config, num_samples=256, seed=4100)
    y_test = test.tensors[0].reshape(256, -1).float()
    y_test = y_test / y_test.sum(1, keepdim=True)
    a_test = affine_of(test.tensors[1].float())

    bank_path = out / f"bank_{args.bank_size}_seed{args.bank_seed}.pt"
    if bank_path.exists():
        bank = torch.load(bank_path, weights_only=False)
        y_bank, a_bank = bank["images"], bank["affine"]
    else:
        ys, aa = [], []
        for i in range(args.bank_size // args.chunk):
            cfg = replace(config, seed=args.bank_seed + 1_000_003 * i)
            ds = generate_train_cache_batched_from_iterable_config(cfg, num_samples=args.chunk, batch_size=512, device=args.device)
            imgs = ds.tensors[0].reshape(args.chunk, -1).float().cpu()
            ys.append((imgs / imgs.sum(1, keepdim=True)).half())
            aa.append(affine_of(ds.tensors[1].float().cpu()))
            print(f"bank chunk {i + 1}/{args.bank_size // args.chunk}", flush=True)
        y_bank, a_bank = torch.cat(ys), torch.cat(aa)
        torch.save({"images": y_bank, "affine": a_bank}, bank_path)

    # Density L2 distances test x bank (fp32 accumulation, chunked).
    test_sq = y_test.square().sum(1)
    dists = torch.empty(256, y_bank.shape[0])
    for start in range(0, y_bank.shape[0], args.chunk):
        yb = y_bank[start : start + args.chunk].float()
        d2 = test_sq[:, None] + yb.square().sum(1)[None, :] - 2.0 * y_test @ yb.T
        dists[:, start : start + args.chunk] = d2.clamp_min(0).sqrt()

    per = load_per_sample()
    records = []
    floor_by_eps = {}
    for eps in args.eps:
        floor_terms = []
        for j in range(256):
            idx = torch.nonzero(dists[j] <= eps).flatten()
            rec = next((r for r in records if r["sample_index"] == j), None)
            if rec is None:
                rec = {"sample_index": j}
                records.append(rec)
            rec[f"n_neighbors@{eps}"] = int(idx.numel())
            if idx.numel() >= 3:
                sq = matched_sq_cost(a_test[j].expand(idx.numel(), -1, -1), a_bank[idx])
                rec[f"ambiguity@{eps}"] = float(sq.sqrt().median())
                rec[f"frac_far@{eps}"] = float((sq.sqrt() >= 1.0).float().mean())
                floor_terms.append(0.5 * float(sq.mean()))
            else:
                rec[f"ambiguity@{eps}"] = float("nan")
                rec[f"frac_far@{eps}"] = float("nan")
        floor_by_eps[eps] = {
            "bayes_floor_estimate": float(np.mean(floor_terms)) if floor_terms else float("nan"),
            "num_test_with_neighbors": len(floor_terms),
        }
    knn_d, knn_i = dists.topk(args.knn, dim=1, largest=False)
    for j in range(256):
        sq = matched_sq_cost(a_test[j].expand(args.knn, -1, -1), a_bank[knn_i[j]])
        records[j]["knn_density_l2_median"] = float(knn_d[j].median())
        records[j][f"knn{args.knn}_param_distance_median"] = float(sq.sqrt().median())
        for family, step in (("model_r1", 0), ("model_r1", 30), ("model_r1", 1000)):
            row = per[(family, step)][j]
            records[j][f"param_distance_s{step}"] = float(row["param_distance"])
            records[j][f"chamfer_s{step}"] = float(row["chamfer_to_target_points"])
            records[j][f"density_sse_s{step}"] = float(row["density_sse_to_input"])

    with (out / "per_sample_ambiguity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    key = f"ambiguity@{args.main_eps}"
    amb = np.array([r[key] for r in records])
    valid = np.isfinite(amb)
    quart = np.nanquantile(amb, [0.25, 0.5, 0.75])
    bins = np.digitize(amb, quart)
    bin_rows = []
    for b in range(4):
        sel = valid & (bins == b)
        row = {"bin": b, "count": int(sel.sum()), "ambiguity_mean": float(amb[sel].mean())}
        for step in (0, 30, 1000):
            pe = np.array([r[f"param_distance_s{step}"] for r in records])[sel]
            cd = np.array([r[f"chamfer_s{step}"] for r in records])[sel]
            ds = np.array([r[f"density_sse_s{step}"] for r in records])[sel]
            row[f"param_distance_s{step}"] = float(pe.mean())
            row[f"chamfer_s{step}"] = float(cd.mean())
            row[f"density_sse_s{step}"] = float(ds.mean())
            row[f"spearman_param_vs_chamfer_s{step}"] = spearman(pe, cd)
        bin_rows.append(row)
    corr = {}
    for step in (0, 30, 1000):
        pe = np.array([r[f"param_distance_s{step}"] for r in records])
        cd = np.array([r[f"chamfer_s{step}"] for r in records])
        ds = np.array([r[f"density_sse_s{step}"] for r in records])
        corr[f"s{step}"] = {
            "spearman_ambiguity_vs_param_error": spearman(amb, pe),
            "spearman_ambiguity_vs_chamfer": spearman(amb, cd),
            "spearman_ambiguity_vs_density_sse": spearman(amb, ds),
            "spearman_param_error_vs_chamfer": spearman(pe, cd),
        }
    summary = {
        "bank_size": int(y_bank.shape[0]),
        "main_eps": args.main_eps,
        "ambiguity_quartiles": quart.tolist(),
        "valid_test_samples": int(valid.sum()),
        "bins": bin_rows,
        "correlations": corr,
        "bayes_floor_by_eps": floor_by_eps,
        "reference_val_matching_loss_base": 0.16944918781518936,
    }
    (out / "ambiguity_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    x = [r["ambiguity_mean"] for r in bin_rows]
    ax = axes[0]
    ax2 = ax.twinx()
    for step, style in ((0, "o-"), (30, "s--")):
        ax.plot(x, [r[f"param_distance_s{step}"] for r in bin_rows], style, color="#7c3aed", label=f"(W,b) error, {step} steps")
        ax2.plot(x, [r[f"chamfer_s{step}"] for r in bin_rows], style, color="#1f5fa8", label=f"Chamfer, {step} steps")
    ax.set_xlabel(rf"input ambiguity (median param. distance of neighbors, $\varepsilon={args.main_eps}$)")
    ax.set_ylabel("(W,b) error", color="#7c3aed")
    ax2.set_ylabel("Chamfer", color="#1f5fa8")
    ax.grid(alpha=0.3)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, fontsize=7, loc="upper left")
    ax = axes[1]
    eps_list = sorted(floor_by_eps)
    ax.plot(eps_list, [floor_by_eps[e]["bayes_floor_estimate"] for e in eps_list], "o-", color="0.2", label="ambiguity floor estimate")
    ax.axhline(0.1694, color="#c2410c", ls="--", label="trained estimator (val. matching loss)")
    ax.set_xlabel(r"neighbor radius $\varepsilon$ (density $L_2$)")
    ax.set_ylabel("matching loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ambiguity_analysis.png", dpi=360)
    fig.savefig(out / "ambiguity_analysis.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
