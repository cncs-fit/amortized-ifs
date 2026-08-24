"""Analyze per-sample errors for saved Phase 0/1 checkpoints."""

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
from torch.utils.data import DataLoader

from data.dataset import make_fixed_dataset_from_iterable_config
from data.renderer import render_density_from_affine_vector, render_density_from_params
from data.sampler import affine_matrices_to_vector, affine_vector_to_matrices, params_to_affine
from losses.hungarian import (
    fixed_points_from_matrices,
    hungarian_indices,
    pairwise_direct_affine_distance,
)
from scripts.evaluate_checkpoint import _build_model, _checkpoint_paths, _config_from_payload

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--point-cloud-max-points", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": _to_float(tensor.mean()),
        "median": _to_float(tensor.median()),
        "p90": _to_float(torch.quantile(tensor, 0.90)),
        "p95": _to_float(torch.quantile(tensor, 0.95)),
        "max": _to_float(tensor.max()),
    }


def _pearson(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x = torch.tensor(x_values, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denom.item() <= 1e-12:
        return None
    return _to_float((x * y).sum() / denom)


def _target_affine_from_params(target_params: torch.Tensor) -> torch.Tensor:
    target_w, target_b = params_to_affine(target_params)
    return affine_matrices_to_vector(target_w, target_b)


def _target_min_pair_distance(target_affine: torch.Tensor) -> float:
    if target_affine.shape[0] < 2:
        return math.nan
    cost = pairwise_direct_affine_distance(target_affine, target_affine)
    diagonal = torch.eye(cost.shape[0], device=cost.device, dtype=torch.bool)
    return _to_float(cost.masked_fill(diagonal, float("inf")).min().sqrt())


def _analyze_one(pred_affine: torch.Tensor, target_params: torch.Tensor, index: int) -> dict:
    target_w, target_b = params_to_affine(target_params)
    target_affine = affine_matrices_to_vector(target_w, target_b)
    cost = pairwise_direct_affine_distance(pred_affine, target_affine)
    row_ind, col_ind = hungarian_indices(cost)
    row_ind = row_ind.to(device=cost.device)
    col_ind = col_ind.to(device=cost.device)

    pred_w, pred_b = affine_vector_to_matrices(pred_affine[row_ind])
    matched_target_w = target_w[col_ind]
    matched_target_b = target_b[col_ind]
    w_errors = (pred_w - matched_target_w).square().sum(dim=(-1, -2)).sqrt()
    b_errors = (pred_b - matched_target_b).square().sum(dim=-1).sqrt()
    pred_fp = fixed_points_from_matrices(pred_w, pred_b)
    target_fp = fixed_points_from_matrices(matched_target_w, matched_target_b)
    fp_errors = (pred_fp - target_fp).square().sum(dim=-1).sqrt()

    pred_singular = torch.linalg.svdvals(pred_w)
    target_singular = torch.linalg.svdvals(target_w)
    matched_target_singular = target_singular[col_ind]
    pred_det = torch.linalg.det(pred_w)
    target_det = torch.linalg.det(target_w)
    matched_target_det = target_det[col_ind]

    per_map = []
    for matched_idx in range(row_ind.numel()):
        target_smax = matched_target_singular[matched_idx].amax()
        pred_smax = pred_singular[matched_idx].amax()
        target_smin = matched_target_singular[matched_idx].amin()
        pred_smin = pred_singular[matched_idx].amin()
        ratio = pred_smax / target_smax.clamp_min(1e-8)
        per_map.append(
            {
                "sample_index": int(index),
                "matched_index": int(matched_idx),
                "pred_index": int(row_ind[matched_idx].detach().cpu().item()),
                "target_index": int(col_ind[matched_idx].detach().cpu().item()),
                "w_fro": _to_float(w_errors[matched_idx]),
                "b_l2": _to_float(b_errors[matched_idx]),
                "fixed_point_l2": _to_float(fp_errors[matched_idx]),
                "target_spectral_max": _to_float(target_smax),
                "pred_spectral_max": _to_float(pred_smax),
                "spectral_max_error": _to_float(pred_smax - target_smax),
                "spectral_max_ratio": _to_float(ratio),
                "target_spectral_min": _to_float(target_smin),
                "pred_spectral_min": _to_float(pred_smin),
                "target_det": _to_float(matched_target_det[matched_idx]),
                "pred_det": _to_float(pred_det[matched_idx]),
                "target_det_abs": _to_float(matched_target_det[matched_idx].abs()),
                "pred_det_abs": _to_float(pred_det[matched_idx].abs()),
                "det_error": _to_float(pred_det[matched_idx] - matched_target_det[matched_idx]),
            }
        )

    return {
        "index": int(index),
        "loss": _to_float(cost[row_ind, col_ind].mean()),
        "w_fro_mean": _to_float(w_errors.mean()),
        "w_fro_max": _to_float(w_errors.max()),
        "b_l2_mean": _to_float(b_errors.mean()),
        "b_l2_max": _to_float(b_errors.max()),
        "fixed_point_l2_mean": _to_float(fp_errors.mean()),
        "fixed_point_l2_max": _to_float(fp_errors.max()),
        "target_min_pair_distance": _target_min_pair_distance(target_affine),
        "target_spectral_max_mean": _to_float(target_singular.amax(dim=-1).mean()),
        "target_det_abs_mean": _to_float(target_det.abs().mean()),
        "pred_spectral_max_mean": _to_float(pred_singular.amax(dim=-1).mean()),
        "pred_det_abs_mean": _to_float(pred_det.abs().mean()),
        "pred_det_negative_count": int((pred_det < 0).sum().item()),
        "assignment_pred_rows": [int(value) for value in row_ind.cpu().tolist()],
        "assignment_target_cols": [int(value) for value in col_ind.cpu().tolist()],
        "per_map_w_fro": [_to_float(value) for value in w_errors],
        "per_map_b_l2": [_to_float(value) for value in b_errors],
        "per_map_fixed_point_l2": [_to_float(value) for value in fp_errors],
        "per_map": per_map,
    }


def predict_all(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            preds.append(model(images.to(device)).detach().cpu())
    return torch.cat(preds, dim=0)


def _filter_points(
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
    points = points[valid].detach().cpu()
    if max_points > 0 and points.shape[0] > max_points:
        points = points[torch.randperm(points.shape[0], generator=generator)[:max_points]]
    return points


def save_error_summary_figure(records: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    w_errors = [record["w_fro_mean"] for record in records]
    b_errors = [record["b_l2_mean"] for record in records]
    separations = [record["target_min_pair_distance"] for record in records]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].hist(w_errors, bins=24, color="#2563eb", alpha=0.85)
    axes[0].set_xlabel("sample W Frobenius")
    axes[0].set_ylabel("count")
    axes[1].hist(b_errors, bins=24, color="#b45309", alpha=0.85)
    axes[1].set_xlabel("sample b L2")
    axes[2].scatter(separations, w_errors, s=12, color="#334155", alpha=0.75)
    axes[2].set_xlabel("target map separation")
    axes[2].set_ylabel("sample W Frobenius")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "error_summary.png", dpi=360)
    fig.savefig(output_dir / "error_summary.pdf")
    plt.close(fig)


def _bin_map_records(map_records: list[dict]) -> list[dict]:
    bins = [
        (0.0, 0.35),
        (0.35, 0.45),
        (0.45, 0.55),
        (0.55, 0.65),
        (0.65, float("inf")),
    ]
    summaries = []
    for low, high in bins:
        selected = [
            record
            for record in map_records
            if record["target_spectral_max"] >= low and record["target_spectral_max"] < high
        ]
        if not selected:
            continue
        summaries.append(
            {
                "target_spectral_max_range": [
                    low,
                    None if math.isinf(high) else high,
                ],
                "count": len(selected),
                "w_fro_mean": sum(record["w_fro"] for record in selected) / len(selected),
                "b_l2_mean": sum(record["b_l2"] for record in selected) / len(selected),
                "target_spectral_max_mean": (
                    sum(record["target_spectral_max"] for record in selected) / len(selected)
                ),
                "pred_spectral_max_mean": (
                    sum(record["pred_spectral_max"] for record in selected) / len(selected)
                ),
                "spectral_max_error_mean": (
                    sum(record["spectral_max_error"] for record in selected) / len(selected)
                ),
                "spectral_max_ratio_mean": (
                    sum(record["spectral_max_ratio"] for record in selected) / len(selected)
                ),
            }
        )
    return summaries


def save_spectral_bias_figure(map_records: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    target_smax = [record["target_spectral_max"] for record in map_records]
    pred_smax = [record["pred_spectral_max"] for record in map_records]
    w_errors = [record["w_fro"] for record in map_records]
    smax_errors = [record["spectral_max_error"] for record in map_records]
    binned = _bin_map_records(map_records)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.1))
    axes[0].scatter(target_smax, pred_smax, s=10, color="#2563eb", alpha=0.65)
    max_value = max(max(target_smax), max(pred_smax))
    axes[0].plot([0.0, max_value], [0.0, max_value], color="#475569", linewidth=1)
    axes[0].set_xlabel("target spectral max")
    axes[0].set_ylabel("pred spectral max")

    axes[1].scatter(target_smax, w_errors, s=10, color="#334155", alpha=0.65)
    axes[1].set_xlabel("target spectral max")
    axes[1].set_ylabel("map W Frobenius")

    axes[2].scatter(target_smax, smax_errors, s=10, color="#b45309", alpha=0.65)
    axes[2].axhline(0.0, color="#475569", linewidth=1)
    axes[2].set_xlabel("target spectral max")
    axes[2].set_ylabel("pred-target spectral max")

    labels = []
    values = []
    for record in binned:
        low, high = record["target_spectral_max_range"]
        labels.append(f"{low:.2f}+" if high is None else f"{low:.2f}-{high:.2f}")
        values.append(record["spectral_max_error_mean"])
    axes[3].bar(labels, values, color="#7c3aed", alpha=0.8)
    axes[3].axhline(0.0, color="#475569", linewidth=1)
    axes[3].set_xlabel("target spectral max bin")
    axes[3].set_ylabel("mean spectral bias")
    axes[3].tick_params(axis="x", labelrotation=35)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "spectral_bias.png", dpi=360)
    fig.savefig(output_dir / "spectral_bias.pdf")
    plt.close(fig)


def save_worst_density_figure(
    records: list[dict],
    predictions: torch.Tensor,
    dataset,
    *,
    config,
    output_dir: Path,
    max_samples: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    selected = records[:max_samples]
    fig, axes = plt.subplots(3, len(selected), figsize=(2.1 * len(selected), 6.2), squeeze=False)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 51_000)
    for col, record in enumerate(selected):
        idx = int(record["index"])
        target_image = dataset.tensors[0][idx, 0]
        _, pred_density = render_density_from_affine_vector(
            predictions[idx],
            resolution=config.resolution,
            fixed_range=config.fixed_range,
            num_trajectories=config.num_trajectories,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            density_smoothing_sigma=config.density_smoothing_sigma,
            generator=generator,
        )
        pred_image = pred_density[0]
        diff = (target_image - pred_image).abs()
        panels = ((target_image, "input"), (pred_image, "pred"), (diff, "abs diff"))
        for row, (image, title) in enumerate(panels):
            axes[row, col].imshow(image, cmap="magma", origin="lower")
            axes[row, col].set_title(f"{title}\nidx={idx} W={record['w_fro_mean']:.3f}", fontsize=7)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "worst_density.png", dpi=360)
    fig.savefig(output_dir / "worst_density.pdf")
    plt.close(fig)


def save_worst_point_cloud_figure(
    records: list[dict],
    predictions: torch.Tensor,
    dataset,
    *,
    config,
    output_dir: Path,
    max_samples: int,
    max_points: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    selected = records[:max_samples]
    fig, axes = plt.subplots(2, len(selected), figsize=(2.2 * len(selected), 4.4), squeeze=False)
    low, high = config.fixed_range
    for col, record in enumerate(selected):
        idx = int(record["index"])
        target_generator = torch.Generator(device="cpu").manual_seed(config.seed + 61_000 + idx)
        pred_generator = torch.Generator(device="cpu").manual_seed(config.seed + 71_000 + idx)
        downsample_generator = torch.Generator(device="cpu").manual_seed(config.seed + 81_000 + idx)
        target_points, _ = render_density_from_params(
            dataset.tensors[1][idx],
            resolution=config.resolution,
            fixed_range=config.fixed_range,
            num_trajectories=config.num_trajectories,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            density_smoothing_sigma=0.0,
            generator=target_generator,
        )
        pred_points, _ = render_density_from_affine_vector(
            predictions[idx],
            resolution=config.resolution,
            fixed_range=config.fixed_range,
            num_trajectories=config.num_trajectories,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            density_smoothing_sigma=0.0,
            generator=pred_generator,
        )
        clouds = (
            (_filter_points(
                target_points,
                fixed_range=config.fixed_range,
                max_points=max_points,
                generator=downsample_generator,
            ), "#1f2937", "target"),
            (_filter_points(
                pred_points,
                fixed_range=config.fixed_range,
                max_points=max_points,
                generator=downsample_generator,
            ), "#b45309", "prediction"),
        )
        for row, (points, color, title) in enumerate(clouds):
            ax = axes[row, col]
            if points.numel() > 0:
                ax.scatter(
                    points[:, 0].numpy(),
                    points[:, 1].numpy(),
                    s=0.12,
                    c=color,
                    alpha=0.55,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{title}\nidx={idx} W={record['w_fro_mean']:.3f}", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "worst_point_clouds.png", dpi=360)
    fig.savefig(output_dir / "worst_point_clouds.pdf")
    plt.close(fig)


def write_csv(records: Iterable[dict], path: Path) -> None:
    records = list(records)
    fieldnames = [
        "index",
        "loss",
        "w_fro_mean",
        "w_fro_max",
        "b_l2_mean",
        "b_l2_max",
        "fixed_point_l2_mean",
        "fixed_point_l2_max",
        "target_min_pair_distance",
        "target_spectral_max_mean",
        "target_det_abs_mean",
        "pred_spectral_max_mean",
        "pred_det_abs_mean",
        "pred_det_negative_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def write_map_csv(records: Iterable[dict], path: Path) -> None:
    records = list(records)
    fieldnames = [
        "sample_index",
        "matched_index",
        "pred_index",
        "target_index",
        "w_fro",
        "b_l2",
        "fixed_point_l2",
        "target_spectral_max",
        "pred_spectral_max",
        "spectral_max_error",
        "spectral_max_ratio",
        "target_spectral_min",
        "pred_spectral_min",
        "target_det",
        "pred_det",
        "target_det_abs",
        "pred_det_abs",
        "det_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    args_payload = result["args"]
    if args_payload["target_representation"] != "affine":
        raise ValueError("per-sample analysis currently supports direct affine outputs only")

    config = _config_from_payload(result["train_config"])
    split_seed_key = "val_seed" if args.split == "val" else "test_seed"
    seed = int(args_payload[split_seed_key] if args.seed is None else args.seed)
    batch_size = int(args_payload["eval_batch_size"] if args.batch_size is None else args.batch_size)
    device = torch.device(args.device)

    dataset = make_fixed_dataset_from_iterable_config(config, num_samples=args.num_samples, seed=seed)
    model = _build_model(args_payload, device=device)
    checkpoint_paths = _checkpoint_paths(result, run_dir=run_dir, checkpoint=args.checkpoint)
    checkpoint_path = checkpoint_paths[args.checkpoint]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictions = predict_all(model, dataset, batch_size=batch_size, device=device)

    records = [
        _analyze_one(predictions[idx], dataset.tensors[1][idx], idx)
        for idx in range(args.num_samples)
    ]
    map_records = [map_record for record in records for map_record in record["per_map"]]
    sorted_records = sorted(records, key=lambda record: record["w_fro_mean"], reverse=True)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": seed,
        "batch_size": batch_size,
        "summaries": {
            "loss": _summarize([record["loss"] for record in records]),
            "w_fro_mean": _summarize([record["w_fro_mean"] for record in records]),
            "w_fro_max": _summarize([record["w_fro_max"] for record in records]),
            "b_l2_mean": _summarize([record["b_l2_mean"] for record in records]),
            "fixed_point_l2_mean": _summarize(
                [record["fixed_point_l2_mean"] for record in records]
            ),
            "map_w_fro": _summarize([record["w_fro"] for record in map_records]),
            "map_spectral_max_error": _summarize(
                [record["spectral_max_error"] for record in map_records]
            ),
            "map_spectral_max_ratio": _summarize(
                [record["spectral_max_ratio"] for record in map_records]
            ),
        },
        "correlations": {
            "target_min_pair_distance_vs_w_fro_mean": _pearson(
                [record["target_min_pair_distance"] for record in records],
                [record["w_fro_mean"] for record in records],
            ),
            "target_spectral_max_mean_vs_w_fro_mean": _pearson(
                [record["target_spectral_max_mean"] for record in records],
                [record["w_fro_mean"] for record in records],
            ),
            "target_det_abs_mean_vs_w_fro_mean": _pearson(
                [record["target_det_abs_mean"] for record in records],
                [record["w_fro_mean"] for record in records],
            ),
            "map_target_spectral_max_vs_w_fro": _pearson(
                [record["target_spectral_max"] for record in map_records],
                [record["w_fro"] for record in map_records],
            ),
            "map_target_spectral_max_vs_spectral_max_error": _pearson(
                [record["target_spectral_max"] for record in map_records],
                [record["spectral_max_error"] for record in map_records],
            ),
            "map_target_det_abs_vs_w_fro": _pearson(
                [record["target_det_abs"] for record in map_records],
                [record["w_fro"] for record in map_records],
            ),
        },
        "map_spectral_bins": _bin_map_records(map_records),
        "worst_by_w_fro_mean": sorted_records[: args.top_k],
    }

    if args.output_dir is None:
        output_dir = run_dir / f"error_analysis_{args.split}_{args.checkpoint}_n{args.num_samples}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "per_sample_errors.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_map_errors.json").write_text(
        json.dumps(map_records, indent=2),
        encoding="utf-8",
    )
    write_csv(records, output_dir / "per_sample_errors.csv")
    write_map_csv(map_records, output_dir / "per_map_errors.csv")
    save_error_summary_figure(records, output_dir)
    save_spectral_bias_figure(map_records, output_dir)
    save_worst_density_figure(
        sorted_records,
        predictions,
        dataset,
        config=config,
        output_dir=output_dir,
        max_samples=min(args.top_k, args.num_samples),
    )
    save_worst_point_cloud_figure(
        sorted_records,
        predictions,
        dataset,
        config=config,
        output_dir=output_dir,
        max_samples=min(args.top_k, args.num_samples),
        max_points=args.point_cloud_max_points,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
