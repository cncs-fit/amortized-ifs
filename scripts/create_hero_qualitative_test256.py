"""Create central-result qualitative figures for the test256 Pareto experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.renderer import render_density_from_affine_vector
from data.sampler import affine_matrices_to_vector, params_to_affine

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


FIDELITY_NAME = "fidelity_res128_traj16_steps1024_sigma2_robust2048"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ours0-dir",
        default="outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512",
    )
    parser.add_argument(
        "--ours30-dir",
        default="outputs/oracle/paper_base50k_fresh_best_s30_test256_chamfer010_p512",
    )
    parser.add_argument(
        "--random30-dir",
        default="outputs/oracle/paper50k_e1_random_r4_s30_test256_chamfer010_p512",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_figures/advice024_central_qualitative",
    )
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit fixed test-set indices. If omitted, the first "
            "--num-examples samples are used, independent of all metrics."
        ),
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num-trajectories", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=1024)
    parser.add_argument("--burn-in", type=int, default=128)
    parser.add_argument("--smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--fixed-range", type=float, nargs=2, default=(-1.5, 1.5))
    parser.add_argument("--seed", type=int, default=53100)
    parser.add_argument("--max-display-points", type=int, default=2400)
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--point-alpha", type=float, default=0.58)
    parser.add_argument("--column-font-size", type=float, default=11.0)
    parser.add_argument("--cell-label-font-size", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=360)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_oracle_metrics(oracle_dir: Path) -> dict[int, dict[str, float]]:
    path = oracle_dir / f"{FIDELITY_NAME}_per_sample.csv"
    records: dict[int, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "oracle":
                continue
            index = int(row["sample_index"])
            records[index] = {
                "chamfer": float(row["chamfer_to_target_points"]),
                "density_sse": float(row["density_sse_to_input"]),
                "hd95": float(row["hausdorff_p95_to_target_points"]),
                "coverage2px": float(row["coverage_symmetric_2px_to_target_points"]),
            }
    return records


def select_examples(
    ours0_metrics: dict[int, dict[str, float]],
    ours30_metrics: dict[int, dict[str, float]],
    random30_metrics: dict[int, dict[str, float]],
    *,
    num_examples: int,
    sample_indices: list[int] | None,
) -> list[dict[str, float]]:
    shared = sorted(set(ours0_metrics) & set(ours30_metrics) & set(random30_metrics))
    if sample_indices:
        selected_indices = list(sample_indices)
    else:
        selected_indices = shared[:num_examples]
    missing = [index for index in selected_indices if index not in shared]
    if missing:
        raise ValueError(f"sample indices are not available in all runs: {missing}")

    selected = []
    for index in selected_indices[:num_examples]:
        ours0 = ours0_metrics[index]["chamfer"]
        ours30 = ours30_metrics[index]["chamfer"]
        random30 = random30_metrics[index]["chamfer"]
        selected.append(
            {
                "sample_index": index,
                "ours0_chamfer": ours0,
                "ours30_chamfer": ours30,
                "random30_chamfer": random30,
                "ours0_to_ours30_improvement": ours0 - ours30,
                "random30_minus_ours30": random30 - ours30,
                "ours30_density_sse": ours30_metrics[index]["density_sse"],
                "ours30_hd95": ours30_metrics[index]["hd95"],
                "ours30_coverage2px": ours30_metrics[index]["coverage2px"],
            }
        )
    return selected


def _target_affine_from_params(target_params: torch.Tensor) -> torch.Tensor:
    target_w, target_b = params_to_affine(target_params)
    return affine_matrices_to_vector(target_w, target_b)


def _render(
    params: torch.Tensor,
    *,
    args: argparse.Namespace,
    sample_index: int,
    variant_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(
        int(args.seed) + 10_003 * int(sample_index) + int(variant_offset)
    )
    points, density = render_density_from_affine_vector(
        params.to(device=device, dtype=torch.float32),
        resolution=args.resolution,
        fixed_range=tuple(args.fixed_range),
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        burn_in=args.burn_in,
        density_smoothing_sigma=args.smoothing_sigma,
        generator=generator,
    )
    return points.detach().cpu().float(), density.detach().cpu().float()


def _filter_and_subsample(
    points: torch.Tensor,
    *,
    fixed_range: tuple[float, float],
    max_points: int,
    seed: int,
) -> torch.Tensor:
    low, high = fixed_range
    valid = (
        torch.isfinite(points).all(dim=-1)
        & (points[:, 0] >= low)
        & (points[:, 0] <= high)
        & (points[:, 1] >= low)
        & (points[:, 1] <= high)
    )
    points = points[valid].detach().cpu().float()
    if max_points > 0 and points.shape[0] > max_points:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(points.shape[0], generator=generator)[:max_points]
        points = points[order]
    return points


def _write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _overlay(ax, text: str, *, color: str = "white", font_size: float = 9.0) -> None:
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=font_size,
        color=color,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2.0, "edgecolor": "none"},
    )


def save_density_figure(
    rendered: dict[int, dict[str, torch.Tensor]],
    selected: list[dict[str, float]],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    columns = ("target", "ours-0", "ours-30", "random-r4-30")
    titles = {
        "target": "target",
        "ours-0": "ours-0",
        "ours-30": "ours-30",
        "random-r4-30": "random-r4-30",
    }
    all_density = torch.stack(
        [rendered[int(record["sample_index"])][column] for record in selected for column in columns]
    )
    vmax = float(torch.quantile(all_density.flatten(), 0.995).item())
    fig, axes = plt.subplots(
        len(selected),
        len(columns),
        figsize=(2.05 * len(columns), 2.05 * len(selected)),
        squeeze=False,
    )
    for row, record in enumerate(selected):
        index = int(record["sample_index"])
        for col, column in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(rendered[index][column][0], cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            if row == 0:
                ax.set_title(titles[column], fontsize=args.column_font_size)
            if column == "target":
                _overlay(ax, f"idx {index}", font_size=args.cell_label_font_size)
            else:
                metric_key = {
                    "ours-0": "ours0_chamfer",
                    "ours-30": "ours30_chamfer",
                    "random-r4-30": "random30_chamfer",
                }[column]
                _overlay(ax, f"C={record[metric_key]:.3f}", font_size=args.cell_label_font_size)
            ax.axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / "hero_qualitative_test256_density.png", dpi=args.dpi)
    fig.savefig(output_dir / "hero_qualitative_test256_density.pdf", dpi=args.dpi)
    plt.close(fig)


def save_point_figure(
    rendered_points: dict[int, dict[str, torch.Tensor]],
    selected: list[dict[str, float]],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    columns = ("target", "ours-0", "ours-30", "random-r4-30")
    colors = {
        "target": "#111827",
        "ours-0": "#64748b",
        "ours-30": "#2563eb",
        "random-r4-30": "#b45309",
    }
    low, high = tuple(args.fixed_range)
    fig, axes = plt.subplots(
        len(selected),
        len(columns),
        figsize=(2.05 * len(columns), 2.05 * len(selected)),
        squeeze=False,
    )
    for row, record in enumerate(selected):
        index = int(record["sample_index"])
        for col, column in enumerate(columns):
            ax = axes[row, col]
            points = rendered_points[index][column]
            if points.numel() > 0:
                ax.scatter(
                    points[:, 0].numpy(),
                    points[:, 1].numpy(),
                    s=args.point_size,
                    c=colors[column],
                    alpha=args.point_alpha,
                    linewidths=0,
                    rasterized=True,
                )
            if row == 0:
                ax.set_title(column, fontsize=args.column_font_size)
            if column == "target":
                _overlay(ax, f"idx {index}", color="white", font_size=args.cell_label_font_size)
            else:
                metric_key = {
                    "ours-0": "ours0_chamfer",
                    "ours-30": "ours30_chamfer",
                    "random-r4-30": "random30_chamfer",
                }[column]
                _overlay(ax, f"C={record[metric_key]:.3f}", color="white", font_size=args.cell_label_font_size)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / "hero_qualitative_test256_points.png", dpi=args.dpi)
    fig.savefig(output_dir / "hero_qualitative_test256_points.pdf", dpi=args.dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    ours0_dir = _resolve(args.ours0_dir)
    ours30_dir = _resolve(args.ours30_dir)
    random30_dir = _resolve(args.random30_dir)
    ours0_metrics = _read_oracle_metrics(ours0_dir)
    ours30_metrics = _read_oracle_metrics(ours30_dir)
    random30_metrics = _read_oracle_metrics(random30_dir)
    selected = select_examples(
        ours0_metrics,
        ours30_metrics,
        random30_metrics,
        num_examples=args.num_examples,
        sample_indices=args.sample_indices,
    )
    if not selected:
        raise RuntimeError("no samples were available for qualitative figure selection")

    ours0_payload = torch.load(ours0_dir / "oracle_outputs.pt", map_location="cpu")
    ours30_payload = torch.load(ours30_dir / "oracle_outputs.pt", map_location="cpu")
    random30_payload = torch.load(random30_dir / "oracle_outputs.pt", map_location="cpu")
    images = ours0_payload["images"].float()
    target_affine = _target_affine_from_params(ours0_payload["target_params"].float())
    variant_params = {
        "target": target_affine,
        "ours-0": ours0_payload["oracle_params"].float(),
        "ours-30": ours30_payload["oracle_params"].float(),
        "random-r4-30": random30_payload["oracle_params"].float(),
    }
    rendered_density: dict[int, dict[str, torch.Tensor]] = {}
    rendered_points: dict[int, dict[str, torch.Tensor]] = {}
    variant_offsets = {"target": 0, "ours-0": 17_000, "ours-30": 34_000, "random-r4-30": 51_000}
    for record in selected:
        index = int(record["sample_index"])
        rendered_density[index] = {"target": images[index].detach().cpu().float()}
        rendered_points[index] = {}
        for variant, params in variant_params.items():
            points, density = _render(
                params[index],
                args=args,
                sample_index=index,
                variant_offset=variant_offsets[variant],
                device=device,
            )
            if variant != "target":
                rendered_density[index][variant] = density
            rendered_points[index][variant] = _filter_and_subsample(
                points,
                fixed_range=tuple(args.fixed_range),
                max_points=args.max_display_points,
                seed=args.seed + 91_000 + variant_offsets[variant] + index,
            )

    save_density_figure(rendered_density, selected, output_dir=output_dir, args=args)
    save_point_figure(rendered_points, selected, output_dir=output_dir, args=args)

    metadata = {
        "description": "Fixed-order test256 qualitative examples for the central Pareto result.",
        "selection_rule": (
            "No metric-based selection is used. Unless --sample-indices is provided, "
            "the figure uses the first --num-examples indices from the fixed test256 "
            "evaluation order."
        ),
        "sample_indices_source": "explicit --sample-indices" if args.sample_indices else "first fixed test256 indices",
        "input_dirs": {
            "ours0": str(ours0_dir),
            "ours30": str(ours30_dir),
            "random30": str(random30_dir),
        },
        "render_config": {
            "resolution": args.resolution,
            "num_trajectories": args.num_trajectories,
            "num_steps": args.num_steps,
            "burn_in": args.burn_in,
            "smoothing_sigma": args.smoothing_sigma,
            "fixed_range": list(args.fixed_range),
            "seed": args.seed,
            "max_display_points": args.max_display_points,
            "point_size": args.point_size,
            "point_alpha": args.point_alpha,
            "column_font_size": args.column_font_size,
            "cell_label_font_size": args.cell_label_font_size,
        },
        "selected_samples": selected,
        "outputs": [
            "hero_qualitative_test256_density.png",
            "hero_qualitative_test256_density.pdf",
            "hero_qualitative_test256_points.png",
            "hero_qualitative_test256_points.pdf",
        ],
    }
    (output_dir / "hero_qualitative_test256_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _write_csv(selected, output_dir / "hero_qualitative_test256_selected_samples.csv")
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
