"""Summarize oracle/refinement runs on shared quality-speed axes."""

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
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "ORACLE_DIR"),
        required=True,
        help="Run label and oracle output directory. Can be repeated.",
    )
    parser.add_argument("--fidelity-name", default="fidelity_res128_traj16_steps1024_sigma2_robust")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_metric(fidelity: dict, variant: str, metric: str) -> float:
    return float(fidelity["summaries"][variant][metric]["mean"])


def _safe_summary(summary: dict, path: tuple[str, ...], default: float = math.nan) -> float:
    value = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return float(value)


def read_run(label: str, oracle_dir: Path, fidelity_name: str) -> dict:
    summary = _load_json(oracle_dir / "summary.json")
    fidelity_payload = _load_json(oracle_dir / f"{fidelity_name}.json")
    fidelity = fidelity_payload[0] if isinstance(fidelity_payload, list) else fidelity_payload
    row = {
        "label": label,
        "oracle_dir": str(oracle_dir),
        "init_mode": summary.get("init_mode"),
        "restarts": summary.get("restarts"),
        "steps": summary.get("steps"),
        "seconds_per_sample": float(summary.get("seconds_per_sample", math.nan)),
        "objective_weight": _safe_summary(summary, ("objective_config", "point_chamfer_loss_weight")),
        "objective_pred_points": _safe_summary(summary, ("objective_config", "point_chamfer_num_pred_points")),
        "objective_target_points": _safe_summary(summary, ("objective_config", "point_chamfer_num_target_points")),
        "density_sse": _mean_metric(fidelity, "oracle", "density_sse_to_input"),
        "chamfer": _mean_metric(fidelity, "oracle", "chamfer_to_target_points"),
        "hd95": _mean_metric(fidelity, "oracle", "hausdorff_p95_to_target_points"),
        "coverage_1px": _mean_metric(fidelity, "oracle", "coverage_symmetric_1px_to_target_points"),
        "coverage_2px": _mean_metric(fidelity, "oracle", "coverage_symmetric_2px_to_target_points"),
        "coverage_4px": _mean_metric(fidelity, "oracle", "coverage_symmetric_4px_to_target_points"),
        "raw_hausdorff": _mean_metric(fidelity, "oracle", "hausdorff_to_target_points"),
        "modified_hausdorff_mean": _mean_metric(
            fidelity,
            "oracle",
            "modified_hausdorff_mean_to_target_points",
        ),
        "w_fro": _safe_summary(summary, ("oracle", "w_fro")),
        "b_l2": _safe_summary(summary, ("oracle", "b_l2")),
    }
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    marker_by_mode = {"model": "o", "random": "s", "target": "^", "constant": "D", "mixed": "P"}
    color_by_mode = {"model": "#2563eb", "random": "#dc2626", "target": "#059669"}
    panels = [
        ("chamfer", "Chamfer", "quality_speed_chamfer"),
        ("hd95", "HD95", "quality_speed_hd95"),
        ("density_sse", "density SSE", "quality_speed_density"),
        ("coverage_2px", "coverage@2px", "quality_speed_coverage2px"),
    ]
    for metric, ylabel, stem in panels:
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        for row in rows:
            mode = str(row.get("init_mode"))
            ax.scatter(
                row["seconds_per_sample"],
                row[metric],
                marker=marker_by_mode.get(mode, "o"),
                s=54,
                color=color_by_mode.get(mode, "#475569"),
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            ax.annotate(
                row["label"],
                (row["seconds_per_sample"], row[metric]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("seconds / sample")
        ax.set_ylabel(ylabel)
        if metric != "coverage_2px":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}.png", dpi=360)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [read_run(label, Path(path), args.fidelity_name) for label, path in args.run]
    write_csv(rows, output_dir / "summary.csv")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_plots(rows, output_dir)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
