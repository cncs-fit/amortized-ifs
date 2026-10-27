"""I8: threshold-free view of the long-horizon convergence comparison (Tab. 3).

Reviewer R1 noted that tau = 0.0262 equals the mean Chamfer of base+30.  This
script (i) sweeps tau over a range and reports Pr[CD <= tau] for every method and
step (an empirical CDF), (ii) evaluates model-independent anchor thresholds
(pixel width, and the Chamfer noise floor measured by re-rendering the true
parameters with the same protocol), and (iii) a per-sample floor-relative
criterion CD_i <= k * floor_i that uses no global threshold at all.

Inputs are existing per-sample CSVs; no optimization is rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()
import matplotlib.pyplot as plt  # noqa: E402

PIXEL = 3.0 / 128.0
LONG_HORIZON = ROOT / "outputs/oracle/advice027_long_horizon_test256_chamfer010_p512_bs256/analysis/per_sample_metrics.csv"
FLOOR = ROOT / "outputs/rev1/I8_tau/at_gt_s0_test256/fidelity_res128_traj16_steps1024_sigma2_robust2048_per_sample.csv"
FAMILY_LABEL = {"model_r1": "base (amortized init.)", "random_r1": "random-r1", "random_r4": "random-r4 (best of 4)"}
FAMILY_COLOR = {"model_r1": "#1f5fa8", "random_r1": "#c2410c", "random_r4": "#7c3aed"}


def load_long_horizon(path: Path) -> dict[tuple[str, int], np.ndarray]:
    data: dict[tuple[str, int], dict[int, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["family"], int(row["step"]))
            data.setdefault(key, {})[int(row["sample_index"])] = float(row["chamfer_to_target_points"])
    return {key: np.array([values[i] for i in sorted(values)]) for key, values in data.items()}


def load_floor(path: Path, variant: str = "oracle") -> np.ndarray:
    values: dict[int, float] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == variant:
                values[int(row["sample_index"])] = float(row["chamfer_to_target_points"])
    return np.array([values[i] for i in sorted(values)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rev1/I8_tau/analysis"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cd = load_long_horizon(LONG_HORIZON)
    floor = load_floor(FLOOR)
    floor_stats = {
        "mean": float(floor.mean()),
        "median": float(np.median(floor)),
        "p95": float(np.quantile(floor, 0.95)),
        "max": float(floor.max()),
    }
    anchors = {
        "tau_paper_base30_mean": 0.0262,
        "one_pixel": PIXEL,
        "1.5_pixel": 1.5 * PIXEL,
        "floor_p95_x2": 2.0 * floor_stats["p95"],
        "floor_mean_x3": 3.0 * floor_stats["mean"],
        "floor_max": floor_stats["max"],
    }
    grid = np.linspace(0.005, 0.08, 301)

    rows = []
    curves = {}
    for (family, step), values in sorted(cd.items()):
        curve = (values[None, :] <= grid[:, None]).mean(axis=1)
        curves[(family, step)] = curve
        record = {
            "family": family,
            "step": step,
            "mean_cd": float(values.mean()),
            "auc_success_0.005_0.08": float(np.trapezoid(curve, grid) / (grid[-1] - grid[0])),
        }
        for name, tau in anchors.items():
            record[f"success@{name}"] = float((values <= tau).mean())
        for k in (1.5, 2.0, 3.0, 4.0):
            record[f"floor_rel_success@{k:g}x"] = float((values <= k * floor).mean())
        rows.append(record)

    # Dominance of base over each random family at matched step, across the whole tau grid.
    dominance = []
    for step in sorted({s for _, s in cd}):
        for rival in ("random_r1", "random_r4"):
            if ("model_r1", step) not in curves or (rival, step) not in curves:
                continue
            diff = curves[("model_r1", step)] - curves[(rival, step)]
            dominance.append({
                "step": step,
                "rival": rival,
                "min_diff_over_grid": float(diff.min()),
                "max_diff_over_grid": float(diff.max()),
                "base_never_below": bool((diff >= -1e-12).all()),
            })
    # Also compare base at 30 steps with random at 1000 steps (33x fewer steps).
    for rival in ("random_r1", "random_r4"):
        diff = curves[("model_r1", 30)] - curves[(rival, 1000)]
        dominance.append({
            "step": "base30_vs_rival1000",
            "rival": rival,
            "min_diff_over_grid": float(diff.min()),
            "max_diff_over_grid": float(diff.max()),
            "base_never_below": bool((diff >= -1e-12).all()),
            "tau_where_base_below": [float(t) for t in grid[diff < -1e-12][:5]],
        })

    keys = list(rows[0].keys())
    with (out / "tau_success_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / "tau_success_summary.json").write_text(
        json.dumps({"floor_chamfer": floor_stats, "anchors": anchors, "rows": rows, "dominance": dominance}, indent=2)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    for family, step, style in (
        ("model_r1", 30, "--"),
        ("model_r1", 1000, "-"),
        ("random_r1", 1000, "-"),
        ("random_r4", 1000, "-"),
    ):
        ax.plot(grid, curves[(family, step)], style, color=FAMILY_COLOR[family], lw=1.6,
                label=f"{FAMILY_LABEL[family]}, {step} steps")
    for name, tau, ls in (("1 px", PIXEL, ":"), (r"$\tau{=}0.0262$", 0.0262, "-."), ("floor p95", floor_stats["p95"], ":")):
        ax.axvline(tau, color="0.45", lw=0.9, ls=ls)
        ax.text(tau, 0.02, f" {name}", rotation=90, fontsize=7, color="0.3", va="bottom")
    ax.set_xlabel(r"Chamfer threshold $\tau$")
    ax.set_ylabel(r"success rate $\Pr[\mathrm{CD}\leq\tau]$")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    ks = np.linspace(1.0, 8.0, 141)
    for family, step, style in (("model_r1", 30, "--"), ("model_r1", 1000, "-"), ("random_r1", 1000, "-"), ("random_r4", 1000, "-")):
        values = cd[(family, step)]
        ax.plot(ks, [(values <= k * floor).mean() for k in ks], style, color=FAMILY_COLOR[family], lw=1.6,
                label=f"{FAMILY_LABEL[family]}, {step} steps")
    ax.set_xlabel(r"multiple $k$ of the per-sample noise floor")
    ax.set_ylabel(r"$\Pr[\mathrm{CD}_i \leq k\,\mathrm{floor}_i]$")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tau_success_curves.png", dpi=360)
    fig.savefig(out / "tau_success_curves.pdf")
    plt.close(fig)
    print(json.dumps({"floor_chamfer": floor_stats, "anchors": anchors}, indent=2))
    for row in rows:
        if row["step"] in (30, 1000):
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    for record in dominance:
        print(record)


if __name__ == "__main__":
    main()
