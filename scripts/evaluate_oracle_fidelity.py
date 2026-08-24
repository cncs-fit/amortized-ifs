"""High-fidelity reconstruction metrics for per-instance oracle outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from data.renderer import render_density_from_affine_vector
from data.sampler import affine_matrices_to_vector, params_to_affine

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", action="append", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num-trajectories", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=1024)
    parser.add_argument("--burn-in", type=int, default=128)
    parser.add_argument("--smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--fixed-range", type=float, nargs=2, default=(-1.5, 1.5))
    parser.add_argument("--seed", type=int, default=53100)
    parser.add_argument("--chamfer-max-points", type=int, default=2048)
    parser.add_argument("--coverage-pixel-thresholds", type=float, nargs="+", default=(1.0, 2.0, 4.0))
    parser.add_argument("--plot-samples", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return {
        "mean": _to_float(tensor.mean()),
        "median": _to_float(tensor.median()),
        "p90": _to_float(torch.quantile(tensor, 0.90)),
        "p95": _to_float(torch.quantile(tensor, 0.95)),
        "min": _to_float(tensor.min()),
        "max": _to_float(tensor.max()),
    }


def _pixel_threshold_label(pixel_threshold: float) -> str:
    if float(pixel_threshold).is_integer():
        return f"{int(pixel_threshold)}px"
    return f"{pixel_threshold:g}px".replace(".", "p")


def _coverage_thresholds(
    *,
    fixed_range: tuple[float, float],
    resolution: int,
    pixel_thresholds: Iterable[float],
) -> dict[str, float]:
    low, high = fixed_range
    pixel_width = (high - low) / float(resolution)
    return {
        _pixel_threshold_label(pixel_threshold): float(pixel_threshold) * pixel_width
        for pixel_threshold in pixel_thresholds
    }


def _point_metric_keys(coverage_thresholds: dict[str, float] | None = None) -> list[str]:
    keys = [
        "chamfer",
        "hausdorff",
        "hausdorff_p90",
        "hausdorff_p95",
        "modified_hausdorff_mean",
        "modified_hausdorff_rms",
    ]
    for label in coverage_thresholds or {}:
        keys.extend(
            [
                f"coverage_pred_to_target_{label}",
                f"coverage_target_to_pred_{label}",
                f"coverage_symmetric_{label}",
            ]
        )
    return keys


def _empty_point_metrics(coverage_thresholds: dict[str, float] | None = None) -> dict[str, float]:
    return {key: math.nan for key in _point_metric_keys(coverage_thresholds)}


def _identity_point_metrics(coverage_thresholds: dict[str, float] | None = None) -> dict[str, float]:
    metrics = {key: 0.0 for key in _point_metric_keys(coverage_thresholds)}
    for label in coverage_thresholds or {}:
        metrics[f"coverage_pred_to_target_{label}"] = 1.0
        metrics[f"coverage_target_to_pred_{label}"] = 1.0
        metrics[f"coverage_symmetric_{label}"] = 1.0
    return metrics


def _write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _target_density(images: torch.Tensor, *, resolution: int, device: torch.device) -> torch.Tensor:
    target = images.to(device=device, dtype=torch.float32)
    if target.shape[-2:] != (resolution, resolution):
        target = F.interpolate(target, size=(resolution, resolution), mode="area")
    return target / target.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)


def _filter_and_subsample(
    points: torch.Tensor,
    *,
    fixed_range: tuple[float, float],
    max_points: int,
    generator: torch.Generator,
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
        order = torch.randperm(points.shape[0], generator=generator)[:max_points]
        points = points[order]
    return points


def point_metrics(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    *,
    device: torch.device,
    coverage_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    if points_a.numel() == 0 or points_b.numel() == 0:
        return _empty_point_metrics(coverage_thresholds)
    a = points_a.to(device=device, dtype=torch.float32)
    b = points_b.to(device=device, dtype=torch.float32)
    distances = torch.cdist(a, b, p=2)
    a_to_b = distances.min(dim=1).values
    b_to_a = distances.min(dim=0).values
    metrics = {
        "chamfer": _to_float(0.5 * (a_to_b.square().mean() + b_to_a.square().mean()).sqrt()),
        "hausdorff": _to_float(torch.maximum(a_to_b.max(), b_to_a.max())),
        "hausdorff_p90": _to_float(torch.maximum(torch.quantile(a_to_b, 0.90), torch.quantile(b_to_a, 0.90))),
        "hausdorff_p95": _to_float(torch.maximum(torch.quantile(a_to_b, 0.95), torch.quantile(b_to_a, 0.95))),
        "modified_hausdorff_mean": _to_float(torch.maximum(a_to_b.mean(), b_to_a.mean())),
        "modified_hausdorff_rms": _to_float(torch.maximum(a_to_b.square().mean().sqrt(), b_to_a.square().mean().sqrt())),
    }
    for label, threshold in (coverage_thresholds or {}).items():
        pred_to_target = (a_to_b <= threshold).float().mean()
        target_to_pred = (b_to_a <= threshold).float().mean()
        metrics[f"coverage_pred_to_target_{label}"] = _to_float(pred_to_target)
        metrics[f"coverage_target_to_pred_{label}"] = _to_float(target_to_pred)
        metrics[f"coverage_symmetric_{label}"] = _to_float(0.5 * (pred_to_target + target_to_pred))
    return metrics


def target_affine_from_params(target_params: torch.Tensor) -> torch.Tensor:
    target_w, target_b = params_to_affine(target_params)
    return affine_matrices_to_vector(target_w, target_b)


def render_one(
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
    return points, density


def evaluate_oracle_dir(oracle_dir: Path, *, args: argparse.Namespace, device: torch.device) -> dict:
    payload = torch.load(oracle_dir / "oracle_outputs.pt", map_location="cpu")
    images = payload["images"].float()
    target_params = payload["target_params"].float()
    variants = {
        "target": target_affine_from_params(target_params),
        "init": payload["init_params"].float(),
        "oracle": payload["oracle_params"].float(),
    }
    target_images = _target_density(images, resolution=args.resolution, device=device)
    coverage_thresholds = _coverage_thresholds(
        fixed_range=tuple(args.fixed_range),
        resolution=args.resolution,
        pixel_thresholds=args.coverage_pixel_thresholds,
    )
    point_metric_keys = _point_metric_keys(coverage_thresholds)

    records: list[dict] = []
    rendered_for_plot: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    target_rendered_points: dict[int, torch.Tensor] = {}
    for sample_index in range(images.shape[0]):
        target_points, target_density = render_one(
            variants["target"][sample_index],
            args=args,
            sample_index=sample_index,
            variant_offset=0,
            device=device,
        )
        downsample_generator = torch.Generator(device="cpu").manual_seed(args.seed + 71_000 + sample_index)
        target_sample_points = _filter_and_subsample(
            target_points,
            fixed_range=tuple(args.fixed_range),
            max_points=args.chamfer_max_points,
            generator=downsample_generator,
        )
        target_rendered_points[sample_index] = target_sample_points

        for variant_index, (variant_name, params) in enumerate(variants.items()):
            if variant_name == "target":
                points = target_points
                density = target_density
                sampled_points = target_sample_points
            else:
                points, density = render_one(
                    params[sample_index],
                    args=args,
                    sample_index=sample_index,
                    variant_offset=17_000 * (variant_index + 1),
                    device=device,
                )
                downsample_generator = torch.Generator(device="cpu").manual_seed(
                    args.seed + 91_000 + 1009 * variant_index + sample_index
                )
                sampled_points = _filter_and_subsample(
                    points,
                    fixed_range=tuple(args.fixed_range),
                    max_points=args.chamfer_max_points,
                    generator=downsample_generator,
                )

            target_image = target_images[sample_index]
            density = density.to(device=device, dtype=torch.float32)
            diff = density - target_image
            pmetrics = (
                _identity_point_metrics(coverage_thresholds)
                if variant_name == "target"
                else point_metrics(
                    sampled_points,
                    target_rendered_points[sample_index],
                    device=device,
                    coverage_thresholds=coverage_thresholds,
                )
            )
            record = {
                "sample_index": int(sample_index),
                "variant": variant_name,
                "density_sse_to_input": _to_float(diff.square().sum()),
                "density_mse_to_input": _to_float(diff.square().mean()),
                "density_l1_to_input": _to_float(diff.abs().sum()),
                "num_points": int(points.shape[0]),
                "num_points_used": int(sampled_points.shape[0]),
            }
            for metric_key in point_metric_keys:
                record[f"{metric_key}_to_target_points"] = pmetrics[metric_key]
            records.append(record)
            if sample_index < args.plot_samples:
                rendered_for_plot[variant_name].append(density.detach().cpu())

    summaries = {}
    for variant_name in variants:
        selected = [record for record in records if record["variant"] == variant_name]
        summaries[variant_name] = {
            "density_sse_to_input": _summarize([record["density_sse_to_input"] for record in selected]),
            "density_mse_to_input": _summarize([record["density_mse_to_input"] for record in selected]),
            "density_l1_to_input": _summarize([record["density_l1_to_input"] for record in selected]),
        }
        for metric_key in point_metric_keys:
            record_key = f"{metric_key}_to_target_points"
            summaries[variant_name][record_key] = _summarize([record[record_key] for record in selected])

    output_name = args.output_name
    if output_name is None:
        output_name = (
            f"fidelity_res{args.resolution}_traj{args.num_trajectories}_"
            f"steps{args.num_steps}_sigma{args.smoothing_sigma:g}"
        )
    summary = {
        "oracle_dir": str(oracle_dir),
        "num_samples": int(images.shape[0]),
        "eval_config": {
            "resolution": args.resolution,
            "num_trajectories": args.num_trajectories,
            "num_steps": args.num_steps,
            "burn_in": args.burn_in,
            "smoothing_sigma": args.smoothing_sigma,
            "fixed_range": list(args.fixed_range),
            "seed": args.seed,
            "chamfer_max_points": args.chamfer_max_points,
            "coverage_pixel_thresholds": list(args.coverage_pixel_thresholds),
            "coverage_thresholds": coverage_thresholds,
        },
        "summaries": summaries,
    }
    (oracle_dir / f"{output_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(records, oracle_dir / f"{output_name}_per_sample.csv")
    save_density_figure(
        target_images.detach().cpu(),
        rendered_for_plot,
        output_dir=oracle_dir,
        output_stem=output_name,
    )
    return summary


def save_density_figure(
    target_images: torch.Tensor,
    rendered: dict[str, list[torch.Tensor]],
    *,
    output_dir: Path,
    output_stem: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rendered or not next(iter(rendered.values())):
        return
    variants = list(rendered.keys())
    num_cols = len(next(iter(rendered.values())))
    if num_cols <= 0:
        return
    fig, axes = plt.subplots(len(variants) + 1, num_cols, figsize=(2.1 * num_cols, 2.0 * (len(variants) + 1)), squeeze=False)
    for col in range(num_cols):
        axes[0, col].imshow(target_images[col, 0], cmap="magma", origin="lower")
        axes[0, col].set_title(f"input\nidx={col}", fontsize=7)
        axes[0, col].axis("off")
    for row, variant in enumerate(variants, start=1):
        for col, density in enumerate(rendered[variant]):
            axes[row, col].imshow(density[0], cmap="magma", origin="lower")
            axes[row, col].set_title(variant, fontsize=7)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_stem}_density_examples.png", dpi=360)
    fig.savefig(output_dir / f"{output_stem}_density_examples.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.burn_in >= args.num_steps:
        raise ValueError("burn-in must be smaller than num-steps")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    outputs = []
    for oracle_dir_value in args.oracle_dir:
        oracle_dir = Path(oracle_dir_value)
        print(f"evaluating {oracle_dir}", flush=True)
        outputs.append(evaluate_oracle_dir(oracle_dir, args=args, device=device))
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
