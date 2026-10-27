"""I7-a2: qualitative figure of the official Djeacoumar et al. code on the 16 test256 systems.

For every system: the target, their optimized code (default configuration, 10 maps; and, when
available, the run with the true number of maps, ``optimized_n4``), and our estimate after 30 and
1000 refinement steps, all drawn as density maps on our standard renderer in our frame (their codes
mapped back through the bounding box of the attractor and drawn with their rule p ~ |det| + 0.01, as in
``i7a2_score.py``).  The 16 systems are shown in two blocks of eight, in index order.
Output: ``outputs/rev1/I7_official/paper/official_examples.{png,pdf}``.
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

from data.sampler import affine_vector_probabilities
from scripts.matplotlib_config import configure_matplotlib_pdf_fonts
from scripts.rev1.eval_shift import standard_density
from scripts.rev1.i7a2_check import their_probs
from scripts.rev1.i7a2_score import FIXED_RANGE, OURS, OUT, align, bbox, code_affine, score
from scripts.rev1.eval_shift import eval_point_clouds

configure_matplotlib_pdf_fonts()
import matplotlib.pyplot as plt  # noqa: E402

PAPER = OUT / "paper"


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = json.loads((OUT / "subset.json").read_text())
    names = [n for n in subset["names"] if (OUT / "optimized" / f"{n}.json").exists()]
    indices = [subset["indices"][subset["names"].index(n)] for n in names]
    chamfer = {}
    with (OUT / "per_sample.csv").open() as handle:
        for row in csv.DictReader(handle):
            for key in row:
                if key.endswith("_chamfer"):
                    chamfer[(key[: -len("_chamfer")], row["name"])] = float(row[key])
    columns = [("truth", "target"), ("djeacoumar", "Djeacoumar et al.")]
    with_n4 = all((OUT / "optimized_n4" / f"{n}.json").exists() for n in names)
    if with_n4:
        columns.append(("djeacoumar_n4", "Djeacoumar, 4 maps"))
    columns += [("ours-30", "ours-30"), ("ours-1000", "ours-1000")]
    ours = {k: torch.load(ROOT / OURS[k] / "oracle_outputs.pt", map_location="cpu", weights_only=False)["oracle_params"].float()
            for k in ("ours-30", "ours-1000")}
    images = {}
    for name, index in zip(names, indices):
        gt = code_affine(json.loads((OUT / "codes" / f"{name}.json").read_text()))
        gt_p = affine_vector_probabilities(gt[None])[0]
        center, half = bbox(gt, gt_p)
        target_cloud = eval_point_clouds(gt[None], gt_p[None], seed=53_100, fixed_range=FIXED_RANGE,
                                         device=device, max_points=2048)[0]
        sets = {"truth": (gt, gt_p)}
        for key, sub in (("djeacoumar", "optimized"), ("djeacoumar_n4", "optimized_n4")):
            path = OUT / sub / f"{name}.json"
            if path.exists():
                code = code_affine(json.loads(path.read_text()))
                p = their_probs(code)
                sets[key] = (align(code, p, center, half), p)
        for key, params in ours.items():
            sets[key] = (params[index], affine_vector_probabilities(params[index][None])[0])
        for key, (a, p) in sets.items():
            images[(key, name)] = standard_density(a[None], p[None], seed=777, fixed_range=FIXED_RANGE, device=device)[0, 0]
            if key == "djeacoumar_n4":
                chamfer[(key, name)] = score(a, p, target_cloud, device)["chamfer"]
    blocks = [names[:8], names[8:]]
    ncol = len(columns)
    fig, axes = plt.subplots(8, 2 * ncol, figsize=(1.05 * 2 * ncol, 1.05 * 8), squeeze=False)
    for b, block in enumerate(blocks):
        for row in range(8):
            for col, (key, title) in enumerate(columns):
                ax = axes[row, b * ncol + col]
                ax.axis("off")
                if row >= len(block):
                    continue
                name = block[row]
                vmax = float(torch.quantile(torch.stack([images[(k, name)] for k, _ in columns]).flatten(), 0.995))
                ax.imshow(images[(key, name)].T.numpy(), cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
                if row == 0:
                    ax.set_title(title, fontsize=6.5)
                text = name.replace("t256_", "#") if key == "truth" else f"C={chamfer[(key, name)]:.3f}"
                ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color="white",
                        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2, "edgecolor": "none"})
    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    fig.savefig(PAPER / "official_examples.png", dpi=360)
    fig.savefig(PAPER / "official_examples.pdf", dpi=360)
    plt.close(fig)
    print("saved", PAPER / "official_examples.pdf", "with_n4" if with_n4 else "")


if __name__ == "__main__":
    main()
