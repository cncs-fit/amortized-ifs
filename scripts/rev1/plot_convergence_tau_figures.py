"""I8: paper figures for the threshold-independent convergence comparison.

* convergence_success_rate_1px: success rate vs refinement steps at the model-independent
  threshold of one pixel width (3/128), same style as the original Fig. 4 (left).
* convergence_success_vs_tau: success rate as a function of the threshold at 1000 steps
  (and base at 30 steps), with the one-pixel width and the 95th percentile of the measured
  Chamfer noise floor marked.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()
import matplotlib.pyplot as plt  # noqa: E402

LONG = ROOT / "outputs/oracle/advice027_long_horizon_test256_chamfer010_p512_bs256/analysis/per_sample_metrics.csv"
SUMMARY = ROOT / "outputs/rev1/I8_tau/analysis/tau_success_summary.json"
OUT = ROOT / "outputs/rev1/I8_tau/paper"
PIXEL = 3.0 / 128.0
COLOR = {"model_r1": "#2563eb", "random_r1": "#dc2626", "random_r4": "#7c3aed"}
LABEL = {"model_r1": "amortized init. (base)", "random_r1": "random r1", "random_r4": "random r4 (best of 4)"}


def load() -> dict[tuple[str, int], np.ndarray]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    with LONG.open() as handle:
        for row in csv.DictReader(handle):
            grouped[(row["family"], int(row["step"]))].append(float(row["chamfer_to_target_points"]))
    return {key: np.array(values) for key, values in grouped.items()}


def save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=360)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cd = load()
    floor_p95 = json.loads(SUMMARY.read_text())["floor_chamfer"]["p95"]
    steps = [0, 30, 100, 300, 1000]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    records = {}
    for family in ("model_r1", "random_r1", "random_r4"):
        rates = [float((cd[(family, s)] <= PIXEL).mean()) for s in steps]
        records[family] = dict(zip(steps, rates))
        ax.plot(steps, rates, marker="o", color=COLOR[family], label=LABEL[family])
    ax.set_xlabel("refinement steps", fontsize=11)
    ax.set_ylabel(r"success rate $\Pr[\mathrm{CD}\leq 1\,\mathrm{px}]$", fontsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=10)
    save(fig, "convergence_success_rate_1px")

    grid = np.linspace(0.005, 0.08, 301)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for family, step, style in (("model_r1", 1000, "-"), ("model_r1", 30, "--"), ("random_r4", 1000, "-"), ("random_r1", 1000, "-")):
        values = cd[(family, step)]
        ax.plot(grid, (values[None, :] <= grid[:, None]).mean(axis=1), style, color=COLOR[family], lw=1.7,
                label=f"{LABEL[family]}, {step} steps")
    for x, text in ((floor_p95, "noise floor p95"), (PIXEL, "1 px")):
        ax.axvline(x, color="#475569", lw=0.9, ls=":")
        ax.text(x + 0.0007, 0.97, text, rotation=90, fontsize=8.5, color="#334155", va="top")
    ax.set_xlabel(r"Chamfer threshold $\tau$", fontsize=11)
    ax.set_ylabel(r"$\Pr[\mathrm{CD}\leq\tau]$", fontsize=11)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "convergence_success_vs_tau")
    (OUT / "success_rate_1px_by_step.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
