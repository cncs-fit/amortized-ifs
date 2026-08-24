"""Plot epsilon-dependent image/parameter ambiguity curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10),
    )
    return parser.parse_args()


def _read_pairs(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "density_l2": float(row["density_l2"]),
                    "density_mse": float(row["density_mse"]),
                    "param_distance": float(row["param_distance"]),
                    "w_fro": float(row["w_fro"]),
                    "b_l2": float(row["b_l2"]),
                    "fixed_point_l2": float(row["fixed_point_l2"]),
                }
            )
    return rows


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": math.nan, "p90": math.nan, "p95": math.nan, "max": math.nan}
    sorted_values = sorted(values)
    return {
        "median": _quantile(sorted_values, 0.50),
        "p90": _quantile(sorted_values, 0.90),
        "p95": _quantile(sorted_values, 0.95),
        "max": sorted_values[-1],
    }


def build_curve(rows: list[dict[str, float]], quantiles: list[float]) -> list[dict[str, float]]:
    density_values = sorted(row["density_l2"] for row in rows)
    curve = []
    total = len(rows)
    for q in quantiles:
        epsilon = _quantile(density_values, q)
        selected = [row for row in rows if row["density_l2"] <= epsilon]
        record = {
            "density_l2_quantile": float(q),
            "epsilon_density_l2": epsilon,
            "epsilon_density_sse_equivalent": epsilon * epsilon,
            "pair_count": len(selected),
            "pair_fraction": len(selected) / total if total else math.nan,
        }
        for key in ("param_distance", "w_fro", "b_l2", "fixed_point_l2"):
            summary = _summarize([row[key] for row in selected])
            for stat, value in summary.items():
                record[f"{key}_{stat}"] = value
        curve.append(record)
    return curve


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(curve: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x = [row["epsilon_density_l2"] for row in curve]
    panels = [
        ("param_distance", "IFS set distance"),
        ("w_fro", "W Frobenius"),
        ("b_l2", "b L2"),
        ("fixed_point_l2", "fixed point L2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    axes = axes.reshape(-1)
    for ax, (key, ylabel) in zip(axes, panels):
        for stat, color in (("median", "#2563eb"), ("p90", "#b45309"), ("max", "#dc2626")):
            ax.plot(
                x,
                [row[f"{key}_{stat}"] for row in curve],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=color,
                label=stat,
            )
        ax.set_xscale("log")
        ax.set_xlabel("epsilon: density L2")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "epsilon_identifiability_curve.png", dpi=360)
    fig.savefig(output_dir / "epsilon_identifiability_curve.pdf")
    plt.close(fig)


def save_single_plot(curve: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, NullLocator
    except ImportError:
        return

    x = [row["epsilon_density_l2"] for row in curve]
    tick_count = min(5, len(x))
    tick_indices = sorted({round(idx * (len(x) - 1) / max(tick_count - 1, 1)) for idx in range(tick_count)})
    x_ticks = [x[idx] for idx in tick_indices]
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.5))
    for stat, color in (("median", "#2563eb"), ("p90", "#b45309"), ("max", "#dc2626")):
        ax.plot(
            x,
            [row[f"param_distance_{stat}"] for row in curve],
            marker="o",
            linewidth=1.5,
            markersize=3.8,
            color=color,
            label=stat,
        )
    ax.set_xscale("log")
    ax.set_xlim(min(x) * 0.96, max(x) * 1.04)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{value:.3f}" for value in x_ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("epsilon: density L2")
    ax.set_ylabel("IFS set distance")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "epsilon_identifiability_curve_single.png", dpi=360)
    fig.savefig(output_dir / "epsilon_identifiability_curve_single.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_pairs(Path(args.pair_csv))
    curve = build_curve(rows, list(args.quantiles))
    write_csv(curve, output_dir / "epsilon_curve.csv")
    (output_dir / "epsilon_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    save_plot(curve, output_dir)
    save_single_plot(curve, output_dir)
    print(json.dumps(curve, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
