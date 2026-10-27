"""I6: qualitative figure for the hand-specified IFS (classical fractals) of the external evaluation.

Reads ``outputs/rev1/I6_external/classical_n4/{outputs.pt,per_sample.csv}`` written by
``scripts/rev1/eval_shift.py`` and draws, for every IFS, the target and the predictions
(ours-0, ours-100, random-r4-100, random-r4-1000) as density maps on the standard renderer
(16 trajectories x 1024 steps, burn-in 128, sigma 2, 128^2). Targets use their published
probabilities, predictions p ~ |det W| (as in refinement). Maps are shown with x to the right
and y up, so the classical fractals appear in their usual orientation.
Output: ``outputs/rev1/I6_external/paper/classical_examples.{png,pdf}``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import affine_vector_probabilities
from scripts.matplotlib_config import configure_matplotlib_pdf_fonts
from scripts.rev1.eval_shift import standard_density

configure_matplotlib_pdf_fonts()
import matplotlib.pyplot as plt  # noqa: E402

RUN = ROOT / "outputs/rev1/I6_external/classical_n4"
OUT = ROOT / "outputs/rev1/I6_external/paper"
COLUMNS = (("truth", "target"), ("model-0", "ours-0"), ("model-100", "ours-100"),
           ("random-r4-100", "random-r4-100"), ("random-r4-1000", "random-r4-1000"))
LABELS = {"sierpinski": "Sierpiński triangle", "barnsley_fern": "Barnsley fern",
          "heighway_dragon": "Heighway dragon", "levy_c": "Lévy C curve",
          "sierpinski_carpet4": "four-corner Cantor dust", "spiral_tree": "three-map spiral"}
FIXED_RANGE = (-1.5, 1.5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = torch.load(RUN / "outputs.pt", map_location="cpu", weights_only=False)
    names, variants = payload["names"], payload["variants"]
    chamfer = {}
    with (RUN / "per_sample.csv").open() as handle:
        for row in csv.DictReader(handle):
            chamfer[(row["variant"], row["name"])] = float(row["chamfer"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rendered = {}
    for key, _ in COLUMNS:
        params = variants[key]
        probs = payload["probs"] if key == "truth" else affine_vector_probabilities(params)
        rendered[key] = standard_density(params, probs, seed=777, fixed_range=FIXED_RANGE, device=device)
    fig, axes = plt.subplots(len(names), len(COLUMNS), figsize=(1.75 * len(COLUMNS), 1.75 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        vmax = float(torch.quantile(torch.stack([rendered[k][row] for k, _ in COLUMNS]).flatten(), 0.995))
        for col, (key, title) in enumerate(COLUMNS):
            ax = axes[row, col]
            ax.imshow(rendered[key][row, 0].T.numpy(), cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            if row == 0:
                ax.set_title(title, fontsize=9)
            text = LABELS.get(name, name) if key == "truth" else f"C={chamfer[(key, name)]:.3f}"
            ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"})
            ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "classical_examples.png", dpi=360)
    fig.savefig(OUT / "classical_examples.pdf", dpi=360)
    plt.close(fig)
    print("saved", OUT / "classical_examples.pdf")


if __name__ == "__main__":
    main()
