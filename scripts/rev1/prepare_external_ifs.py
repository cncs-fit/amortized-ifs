"""I6: convert external IFS codes to the evaluation format of ``eval_shift.py``.

Sources
  * Djeacoumar et al. (2025) benchmark (101 fractals generated with the Anderson & Farrell
    2022 sampler): ground-truth JSON codes and, optionally, every method's optimized codes
    exported to JSON (``export_chaotic_codes.py`` in the separate ``chaotic-fractals`` env).
  * Classical named fractals (hand-coded below).

Conventions
  * External codes use column vectors (x' = M x + t); we use row vectors (x' = x W + b),
    so W = M^T, b = t.
  * Their renderer normalizes every attractor by its own bounding box before drawing
    (``apply_mv_transform``: center at the box center, longer side = 75% of the view), both
    for the ground-truth images and during optimization, so codes are defined up to a
    similarity.  We apply the same normalization to every code (ground truth and each
    method's result): longer side = 75% of our domain [-1.5, 1.5]^2 (half-extent 1.125).
    Conjugation by a similarity leaves the linear parts (and singular values) unchanged.
  * Variable map counts are padded to n_max with zero maps of probability 0, which the
    chaos game never selects.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

HALF_EXTENT = 1.125  # 75% of the [-1.5, 1.5] domain, as in the benchmark renderer


def code_to_affine(code: dict, *, scale: float = 1.0, benchmark_rule: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    m = torch.tensor(code["ifs_m"], dtype=torch.float32)  # [n, 2, 2], column-vector convention
    t = torch.tensor(code["ifs_t"], dtype=torch.float32)  # [n, 2]
    w = m.transpose(-1, -2)
    n = w.shape[0]
    if benchmark_rule or not code.get("ifs_p"):
        # The benchmark's renderer (ModelInfer.optimized_probs) draws every code -- ground truth
        # included -- with p proportional to |det| + 0.01, ignoring any stored ifs_p.
        p = torch.linalg.det(w).abs() + 0.01
    else:
        p = torch.tensor(code["ifs_p"], dtype=torch.float32)
    p = p / p.sum()
    affine = torch.cat((w.reshape(n, 4), scale * t), dim=-1)
    return affine, p


def pad(affines: list[torch.Tensor], probs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    n_max = max(a.shape[0] for a in affines)
    out_a = torch.zeros(len(affines), n_max, 6)
    out_p = torch.zeros(len(affines), n_max)
    for i, (a, p) in enumerate(zip(affines, probs)):
        out_a[i, : a.shape[0]] = a
        out_p[i, : p.shape[0]] = p
    return out_a, out_p


def classical() -> dict[str, dict]:
    """Classical IFS (column-vector form, as in the literature), probabilities as published."""
    c, s = math.cos, math.sin
    codes = {
        "sierpinski": {"ifs_m": [[[0.5, 0], [0, 0.5]]] * 3, "ifs_t": [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.366]], "ifs_p": [1 / 3] * 3},
        "barnsley_fern": {
            "ifs_m": [[[0, 0], [0, 0.16]], [[0.85, 0.04], [-0.04, 0.85]], [[0.2, -0.26], [0.23, 0.22]], [[-0.15, 0.28], [0.26, 0.24]]],
            "ifs_t": [[0, 0], [0, 1.6], [0, 1.6], [0, 0.44]],
            "ifs_p": [0.01, 0.85, 0.07, 0.07],
        },
        "heighway_dragon": {
            "ifs_m": [[[0.5, -0.5], [0.5, 0.5]], [[-0.5, -0.5], [0.5, -0.5]]],
            "ifs_t": [[0, 0], [1, 0]],
            "ifs_p": [0.5, 0.5],
        },
        "levy_c": {
            "ifs_m": [[[0.5, -0.5], [0.5, 0.5]], [[0.5, 0.5], [-0.5, 0.5]]],
            "ifs_t": [[0, 0], [0.5, 0.5]],
            "ifs_p": [0.5, 0.5],
        },
        "sierpinski_carpet4": {  # 4-map square variant (corners), a clean n=4 case
            "ifs_m": [[[0.45, 0], [0, 0.45]]] * 4,
            "ifs_t": [[-0.55, -0.55], [0.55, -0.55], [-0.55, 0.55], [0.55, 0.55]],
            "ifs_p": [0.25] * 4,
        },
        "spiral_tree": {
            "ifs_m": [[[0.6 * c(0.6), -0.6 * s(0.6)], [0.6 * s(0.6), 0.6 * c(0.6)]], [[0.35, 0], [0, 0.35]], [[0.3, 0.1], [-0.1, 0.3]]],
            "ifs_t": [[0.2, 0.1], [-0.6, -0.4], [0.5, -0.5]],
        },
    }
    return codes


def fit_to_window(affine: torch.Tensor, probs: torch.Tensor, *, half_extent: float = HALF_EXTENT) -> torch.Tensor:
    """Similarity-conjugate an IFS so that its attractor's bounding box fits [-half_extent, half_extent]^2."""
    from scripts.rev1.eval_shift import chaos_points

    pts = chaos_points(affine[None], probs[None], num_trajectories=16, num_steps=4096, burn_in=128,
                       generator=torch.Generator().manual_seed(0))[0]
    lo, hi = pts.min(0).values, pts.max(0).values
    center = 0.5 * (lo + hi)
    scale = half_extent / float((hi - lo).max() / 2)
    w = affine[:, :4].reshape(-1, 2, 2)
    b = affine[:, 4:6]
    # T(x) = scale * (x - center); f'(y) = T(f(T^-1 y)) => W' = W, b' = scale*(b + center W - center)
    b_new = scale * (b + torch.einsum("c,ncd->nd", center, w) - center)
    return torch.cat((affine[:, :4], b_new), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-dir", default=str(ROOT / "refs/chaotic_data/full_dataset/ifs_codes"))
    parser.add_argument("--optimized-dir", default=str(ROOT / "outputs/rev1/I6_external/djeacoumar_optimized_json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rev1/I6_external"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{args.codes_dir}/*.json"), key=lambda f: int(Path(f).stem.split("_")[-1]))
    names = [Path(f).stem for f in files]
    gt = [code_to_affine(json.load(open(f)), benchmark_rule=True) for f in files]
    gt = [(fit_to_window(a, p), p) for a, p in gt]
    n_maps = [a.shape[0] for a, _ in gt]
    for label, keep in (("djeacoumar_le4", [n <= 4 for n in n_maps]), ("djeacoumar_all", [True] * len(n_maps))):
        idx = [i for i, k in enumerate(keep) if k]
        affine, probs = pad([gt[i][0] for i in idx], [gt[i][1] for i in idx])
        payload = {"affine": affine, "probs": probs, "names": [names[i] for i in idx], "n_maps": [n_maps[i] for i in idx],
                   "convention": "row-vector W=M^T; each code similarity-normalized by its own bbox to half-extent 1.125; zero-prob padding"}
        # Optimized codes of the benchmark methods, if exported.
        preds = {}
        opt_root = Path(args.optimized_dir)
        if opt_root.exists():
            for method_dir in sorted(p for p in opt_root.iterdir() if p.is_dir()):
                aff_list, p_list, ok = [], [], True
                for i in idx:
                    f = method_dir / f"{names[i]}.json"
                    if not f.exists():
                        ok = False
                        break
                    a, p = code_to_affine(json.load(open(f)), benchmark_rule=True)
                    a = fit_to_window(a, p)
                    aff_list.append(a)
                    p_list.append(p)
                if ok:
                    preds[method_dir.name] = dict(zip(("affine", "probs"), pad(aff_list, p_list)))
        payload["external_predictions"] = preds
        torch.save(payload, out / f"{label}.pt")
        print(label, affine.shape, "methods:", list(preds))

    cl = classical()
    aff_list, p_list, cl_names = [], [], []
    for name, code in cl.items():
        a, p = code_to_affine(code, scale=1.0)
        a = fit_to_window(a, p)
        aff_list.append(a)
        p_list.append(p)
        cl_names.append(name)
    affine, probs = pad(aff_list, p_list)
    torch.save({"affine": affine, "probs": probs, "names": cl_names, "n_maps": [a.shape[0] for a in aff_list],
                "convention": "row-vector, similarity-normalized by bbox to half-extent 1.125; published probabilities"}, out / "classical.pt")
    print("classical", affine.shape, cl_names)


if __name__ == "__main__":
    main()
