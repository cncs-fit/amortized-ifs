"""Analyze identifiability of synthetic IFS evaluation sets.

This script compares pairs of target samples, not model predictions.  It asks
whether two density maps can be close while their underlying IFS parameter sets
are far apart after optimal set matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.dataset import make_fixed_dataset_from_iterable_config
from data.renderer import points_to_density_map, render_density_from_params
from data.sampler import affine_matrices_to_vector, affine_vector_to_matrices, params_to_affine
from data.sampler import iterate_affine_vector_points
from losses.hungarian import (
    fixed_points_from_matrices,
    hungarian_indices,
    pairwise_direct_affine_distance,
)
from scripts.evaluate_checkpoint import _config_from_payload

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--near-quantile", type=float, default=0.01)
    parser.add_argument("--far-quantile", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chamfer-top-k", type=int, default=24)
    parser.add_argument("--chamfer-max-points", type=int, default=1024)
    parser.add_argument("--point-cloud-max-points", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-pairs-csv", type=int, default=200000)
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
        "p01": _to_float(torch.quantile(tensor, 0.01)),
        "p05": _to_float(torch.quantile(tensor, 0.05)),
        "p10": _to_float(torch.quantile(tensor, 0.10)),
        "p90": _to_float(torch.quantile(tensor, 0.90)),
        "p95": _to_float(torch.quantile(tensor, 0.95)),
        "p99": _to_float(torch.quantile(tensor, 0.99)),
        "min": _to_float(tensor.min()),
        "max": _to_float(tensor.max()),
    }


def _pearson(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2:
        return None
    x = torch.tensor(x_values, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denom.item() <= 1e-12:
        return None
    return _to_float((x * y).sum() / denom)


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def _spearman(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2:
        return None
    return _pearson(
        _rankdata(torch.tensor(x_values, dtype=torch.float32)).tolist(),
        _rankdata(torch.tensor(y_values, dtype=torch.float32)).tolist(),
    )


@lru_cache(maxsize=16)
def _permutation_indices(num_items: int) -> torch.Tensor:
    import itertools

    return torch.tensor(list(itertools.permutations(range(num_items))), dtype=torch.long)


def _target_affine(params: torch.Tensor) -> torch.Tensor:
    w, b = params_to_affine(params)
    return affine_matrices_to_vector(w, b)


def affine_set_distance(source_affine: torch.Tensor, target_affine: torch.Tensor) -> dict[str, float]:
    """Return matching-based distances between two direct-affine IFS sets."""
    cost = pairwise_direct_affine_distance(source_affine, target_affine)
    num_source, num_target = cost.shape
    if num_source == num_target and num_source <= 6:
        perms = _permutation_indices(num_source).to(cost.device)
        rows = torch.arange(num_source, device=cost.device)
        values = cost[rows[None, :], perms].mean(dim=1)
        best_idx = int(torch.argmin(values).item())
        row_ind = rows
        col_ind = perms[best_idx]
    else:
        row_ind, col_ind = hungarian_indices(cost)
        row_ind = row_ind.to(cost.device)
        col_ind = col_ind.to(cost.device)

    source_w, source_b = affine_vector_to_matrices(source_affine[row_ind])
    target_w, target_b = affine_vector_to_matrices(target_affine[col_ind])

    w_errors = (source_w - target_w).square().sum(dim=(-1, -2)).sqrt()
    b_errors = (source_b - target_b).square().sum(dim=-1).sqrt()
    source_fp = fixed_points_from_matrices(source_w, source_b)
    target_fp = fixed_points_from_matrices(target_w, target_b)
    fp_errors = (source_fp - target_fp).square().sum(dim=-1).sqrt()
    selected_loss = cost[row_ind, col_ind].mean()
    return {
        "param_loss": _to_float(selected_loss),
        "param_distance": _to_float(selected_loss.sqrt()),
        "w_fro": _to_float(w_errors.mean()),
        "b_l2": _to_float(b_errors.mean()),
        "fixed_point_l2": _to_float(fp_errors.mean()),
    }


def _sample_spectral_summary(params: torch.Tensor) -> dict[str, float]:
    w, _ = params_to_affine(params)
    singular = torch.linalg.svdvals(w)
    return {
        "spectral_max_mean": _to_float(singular.amax(dim=-1).mean()),
        "spectral_max_max": _to_float(singular.amax(dim=-1).max()),
        "det_abs_mean": _to_float(torch.linalg.det(w).abs().mean()),
    }


def build_pair_records(images: torch.Tensor, params: torch.Tensor) -> list[dict]:
    images = images.float().contiguous()
    params = params.float().contiguous()
    flat = images.reshape(images.shape[0], -1)
    density_l2 = torch.cdist(flat, flat, p=2)
    density_l1 = torch.cdist(flat, flat, p=1)
    density_mse = density_l2.square() / float(flat.shape[1])
    affines = [_target_affine(params[idx]) for idx in range(params.shape[0])]
    spectral = [_sample_spectral_summary(params[idx]) for idx in range(params.shape[0])]

    records = []
    for i in range(params.shape[0]):
        for j in range(i + 1, params.shape[0]):
            distances = affine_set_distance(affines[i], affines[j])
            record = {
                "i": int(i),
                "j": int(j),
                "density_l2": _to_float(density_l2[i, j]),
                "density_mse": _to_float(density_mse[i, j]),
                "density_l1": _to_float(density_l1[i, j]),
                "spectral_max_mean_i": spectral[i]["spectral_max_mean"],
                "spectral_max_mean_j": spectral[j]["spectral_max_mean"],
                "spectral_max_max_i": spectral[i]["spectral_max_max"],
                "spectral_max_max_j": spectral[j]["spectral_max_max"],
                "det_abs_mean_i": spectral[i]["det_abs_mean"],
                "det_abs_mean_j": spectral[j]["det_abs_mean"],
            }
            record.update(distances)
            record["far_near_score"] = record["param_distance"] / (record["density_l2"] + 1e-8)
            records.append(record)
    return records


def nearest_image_records(pair_records: list[dict], num_samples: int) -> list[dict]:
    nearest: list[dict | None] = [None for _ in range(num_samples)]
    for record in pair_records:
        i = record["i"]
        j = record["j"]
        if nearest[i] is None or record["density_l2"] < nearest[i]["density_l2"]:
            nearest[i] = {**record, "source_index": i, "neighbor_index": j}
        if nearest[j] is None or record["density_l2"] < nearest[j]["density_l2"]:
            reversed_record = {**record, "source_index": j, "neighbor_index": i}
            nearest[j] = reversed_record
    return [record for record in nearest if record is not None]


def _quantile(values: list[float], q: float) -> float:
    return _to_float(torch.quantile(torch.tensor(values, dtype=torch.float32), float(q)))


def select_ambiguous_pairs(
    pair_records: list[dict],
    *,
    near_quantile: float,
    far_quantile: float,
    fallback_k: int,
) -> tuple[list[dict], dict[str, float]]:
    density_values = [record["density_l2"] for record in pair_records]
    param_values = [record["param_distance"] for record in pair_records]
    thresholds = {
        "density_near_quantile": near_quantile,
        "density_near_threshold": _quantile(density_values, near_quantile),
        "param_far_quantile": far_quantile,
        "param_far_threshold": _quantile(param_values, far_quantile),
    }
    selected = [
        record
        for record in pair_records
        if record["density_l2"] <= thresholds["density_near_threshold"]
        and record["param_distance"] >= thresholds["param_far_threshold"]
    ]
    if not selected:
        selected = sorted(
            pair_records,
            key=lambda record: (
                record["param_distance"] / (record["density_l2"] + 1e-8),
                -record["density_l2"],
            ),
            reverse=True,
        )[:fallback_k]
    else:
        selected = sorted(
            selected,
            key=lambda record: (record["param_distance"], record["far_near_score"]),
            reverse=True,
        )
    return selected, thresholds


def write_csv(records: Iterable[dict], path: Path, *, limit: int | None = None) -> None:
    records = list(records)
    if limit is not None and limit > 0:
        records = records[:limit]
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


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
        order = torch.randperm(points.shape[0], generator=generator)[:max_points]
        points = points[order]
    return points


def generate_target_points(
    params: torch.Tensor,
    index: int,
    *,
    config,
    max_points: int,
    seed_offset: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(config.seed + seed_offset + index)
    downsample_generator = torch.Generator(device="cpu").manual_seed(
        config.seed + seed_offset + 100_003 + index
    )
    points, _ = render_density_from_params(
        params[index],
        resolution=config.resolution,
        fixed_range=config.fixed_range,
        num_trajectories=config.num_trajectories,
        num_steps=config.num_steps,
        burn_in=config.burn_in,
        density_smoothing_sigma=0.0,
        generator=generator,
    )
    return _filter_points(
        points,
        fixed_range=config.fixed_range,
        max_points=max_points,
        generator=downsample_generator,
    )


def approximate_chamfer(points_a: torch.Tensor, points_b: torch.Tensor, *, device: torch.device) -> float:
    if points_a.numel() == 0 or points_b.numel() == 0:
        return math.nan
    a = points_a.to(device=device, dtype=torch.float32)
    b = points_b.to(device=device, dtype=torch.float32)
    distances = torch.cdist(a, b, p=2).square()
    chamfer_sq = 0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
    return _to_float(chamfer_sq.sqrt())


def add_chamfer_to_pairs(
    pair_records: list[dict],
    params: torch.Tensor,
    *,
    config,
    max_pairs: int,
    max_points: int,
    device: torch.device,
) -> list[dict]:
    selected = pair_records[: max(0, max_pairs)]
    if not selected:
        return selected
    unique_indices = sorted({record["i"] for record in selected} | {record["j"] for record in selected})
    points_by_index = {
        index: generate_target_points(
            params,
            index,
            config=config,
            max_points=max_points,
            seed_offset=91_000,
        )
        for index in unique_indices
    }
    enriched = []
    for record in selected:
        chamfer = approximate_chamfer(
            points_by_index[record["i"]],
            points_by_index[record["j"]],
            device=device,
        )
        enriched.append({**record, "approx_chamfer": chamfer})
    return enriched


def save_scatter_figure(
    pair_records: list[dict],
    ambiguous_records: list[dict],
    output_dir: Path,
    thresholds: dict[str, float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    density_l2 = [record["density_l2"] for record in pair_records]
    param_distance = [record["param_distance"] for record in pair_records]
    w_fro = [record["w_fro"] for record in pair_records]
    b_l2 = [record["b_l2"] for record in pair_records]
    fp_l2 = [record["fixed_point_l2"] for record in pair_records]
    nearest_param_by_density = [
        record["param_distance"] for record in sorted(pair_records, key=lambda r: r["density_l2"])[:512]
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0))
    axes = axes.reshape(-1)
    panels = [
        (param_distance, "IFS set distance"),
        (w_fro, "W Frobenius"),
        (b_l2, "b L2"),
        (fp_l2, "fixed point L2"),
    ]
    ambiguous_pairs = {(record["i"], record["j"]) for record in ambiguous_records}
    amb_x = [record["density_l2"] for record in pair_records if (record["i"], record["j"]) in ambiguous_pairs]
    for ax, (y_values, ylabel) in zip(axes, panels):
        ax.scatter(density_l2, y_values, s=5, color="#475569", alpha=0.20, linewidths=0)
        if amb_x:
            amb_y = [
                record[
                    {
                        "IFS set distance": "param_distance",
                        "W Frobenius": "w_fro",
                        "b L2": "b_l2",
                        "fixed point L2": "fixed_point_l2",
                    }[ylabel]
                ]
                for record in pair_records
                if (record["i"], record["j"]) in ambiguous_pairs
            ]
            ax.scatter(amb_x, amb_y, s=12, color="#dc2626", alpha=0.75, linewidths=0)
        ax.axvline(thresholds["density_near_threshold"], color="#2563eb", linewidth=1)
        if ylabel == "IFS set distance":
            ax.axhline(thresholds["param_far_threshold"], color="#b45309", linewidth=1)
        ax.set_xlabel("density L2")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"nearest 512 median param={torch.tensor(nearest_param_by_density).median():.3f}")
    fig.tight_layout()
    fig.savefig(output_dir / "identifiability_scatter.png", dpi=360)
    fig.savefig(output_dir / "identifiability_scatter.pdf")
    plt.close(fig)


def save_scatter_single_figure(
    pair_records: list[dict],
    ambiguous_records: list[dict],
    output_dir: Path,
    thresholds: dict[str, float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    density_l2 = [record["density_l2"] for record in pair_records]
    param_distance = [record["param_distance"] for record in pair_records]
    ambiguous_pairs = {(record["i"], record["j"]) for record in ambiguous_records}
    amb_x = [record["density_l2"] for record in pair_records if (record["i"], record["j"]) in ambiguous_pairs]
    amb_y = [record["param_distance"] for record in pair_records if (record["i"], record["j"]) in ambiguous_pairs]

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.7))
    ax.scatter(density_l2, param_distance, s=6, color="#475569", alpha=0.20, linewidths=0)
    if amb_x:
        ax.scatter(amb_x, amb_y, s=16, color="#dc2626", alpha=0.80, linewidths=0)
    ax.axvline(
        thresholds["density_near_threshold"],
        color="#2563eb",
        linewidth=1.2,
        label=f"density <= {thresholds['density_near_threshold']:.4f}",
    )
    ax.axhline(
        thresholds["param_far_threshold"],
        color="#b45309",
        linewidth=1.2,
        label=f"IFS dist >= {thresholds['param_far_threshold']:.3f}",
    )
    ax.set_xlabel("density L2")
    ax.set_ylabel("IFS set distance")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "identifiability_scatter_single.png", dpi=360)
    fig.savefig(output_dir / "identifiability_scatter_single.pdf")
    plt.close(fig)


def save_ambiguous_density_figure(
    ambiguous_records: list[dict],
    images: torch.Tensor,
    output_dir: Path,
    *,
    max_pairs: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    selected = ambiguous_records[:max_pairs]
    if not selected:
        return
    fig, axes = plt.subplots(3, len(selected), figsize=(2.1 * len(selected), 6.2), squeeze=False)
    for col, record in enumerate(selected):
        image_i = images[record["i"], 0]
        image_j = images[record["j"], 0]
        diff = (image_i - image_j).abs()
        panels = (
            (image_i, f"idx={record['i']}"),
            (image_j, f"idx={record['j']}"),
            (diff, f"dL2={record['density_l2']:.3f}\nparam={record['param_distance']:.3f}"),
        )
        for row, (image, title) in enumerate(panels):
            axes[row, col].imshow(image, cmap="magma", origin="lower")
            axes[row, col].set_title(title, fontsize=7)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "ambiguous_density_pairs.png", dpi=360)
    fig.savefig(output_dir / "ambiguous_density_pairs.pdf")
    plt.close(fig)


def save_ambiguous_point_cloud_figure(
    ambiguous_records: list[dict],
    params: torch.Tensor,
    output_dir: Path,
    *,
    config,
    max_pairs: int,
    max_points: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    selected = ambiguous_records[:max_pairs]
    if not selected:
        return
    unique_indices = sorted({record["i"] for record in selected} | {record["j"] for record in selected})
    points_by_index = {
        index: generate_target_points(
            params,
            index,
            config=config,
            max_points=max_points,
            seed_offset=121_000,
        )
        for index in unique_indices
    }
    fig, axes = plt.subplots(2, len(selected), figsize=(2.1 * len(selected), 4.4), squeeze=False)
    low, high = config.fixed_range
    for col, record in enumerate(selected):
        pair = ((record["i"], "#1f2937"), (record["j"], "#b45309"))
        for row, (index, color) in enumerate(pair):
            ax = axes[row, col]
            points = points_by_index[index]
            if points.numel() > 0:
                ax.scatter(
                    points[:, 0].numpy(),
                    points[:, 1].numpy(),
                    s=0.35,
                    c=color,
                    alpha=0.65,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"idx={index}", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "ambiguous_point_cloud_pairs.png", dpi=360)
    fig.savefig(output_dir / "ambiguous_point_cloud_pairs.pdf")
    plt.close(fig)


def sierpinski_affine_vectors() -> tuple[torch.Tensor, torch.Tensor]:
    sqrt3 = math.sqrt(3.0)
    vertices = torch.tensor(
        [
            [-0.5, -sqrt3 / 6.0],
            [0.5, -sqrt3 / 6.0],
            [0.0, sqrt3 / 3.0],
        ],
        dtype=torch.float32,
    )
    eye = torch.eye(2, dtype=torch.float32)
    w3 = eye.mul(0.5).repeat(3, 1, 1)
    b3 = vertices.mul(0.5)
    standard = affine_matrices_to_vector(w3, b3)

    angle = 2.0 * math.pi / 3.0
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)
    row_rotation = torch.tensor(
        [
            [cos_t, sin_t],
            [-sin_t, cos_t],
        ],
        dtype=torch.float32,
    )
    rotated_w = row_rotation.unsqueeze(0) @ w3
    rotated = affine_matrices_to_vector(rotated_w, b3)
    return standard, rotated


def save_sierpinski_nonidentifiability_figure(output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    maps3, maps_rotated = sierpinski_affine_vectors()
    generator3 = torch.Generator(device="cpu").manual_seed(23_003)
    generator_rotated = torch.Generator(device="cpu").manual_seed(23_103)
    points3 = iterate_affine_vector_points(
        maps3,
        num_trajectories=32,
        num_steps=8192,
        burn_in=64,
        generator=generator3,
        prob_floor=0.0,
    )
    points_rotated = iterate_affine_vector_points(
        maps_rotated,
        num_trajectories=32,
        num_steps=8192,
        burn_in=64,
        generator=generator_rotated,
        prob_floor=0.0,
    )
    fixed_range = (-0.62, 0.62)
    density3 = points_to_density_map(
        points3,
        resolution=192,
        fixed_range=fixed_range,
        smoothing_sigma=0.8,
    )[0]
    density_rotated = points_to_density_map(
        points_rotated,
        resolution=192,
        fixed_range=fixed_range,
        smoothing_sigma=0.8,
    )[0]
    vmax = float(torch.quantile(torch.cat((density3.flatten(), density_rotated.flatten())), 0.995).item())

    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.8), squeeze=False)
    panels = (
        (density3, "standard 3-map\nW = 0.5 I", vmax),
        (density_rotated, "rotated 3-map\nW = 0.5 Rot(120)", vmax),
    )
    for ax, (image, title, image_vmax) in zip(axes[0], panels):
        ax.imshow(image.cpu(), cmap="magma", origin="lower", vmin=0.0, vmax=image_vmax)
        ax.set_title(title, fontsize=9, pad=3)
        ax.axis("off")
    fig.tight_layout(pad=0.35, rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_dir / "sierpinski_nonidentifiability.png", dpi=360)
    fig.savefig(output_dir / "sierpinski_nonidentifiability.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    run_dir = Path(args.run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    args_payload = result["args"]
    config = _config_from_payload(result["train_config"])
    seed_key = "val_seed" if args.split == "val" else "test_seed"
    seed = int(args_payload[seed_key] if args.seed is None else args.seed)

    if args.output_dir is None:
        output_dir = (
            ROOT
            / "outputs"
            / "identifiability"
            / f"{run_dir.parent.name}_{run_dir.name}_{args.split}_n{args.num_samples}"
        )
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    dataset = make_fixed_dataset_from_iterable_config(config, num_samples=args.num_samples, seed=seed)
    images, params = dataset.tensors
    dataset_elapsed = time.perf_counter() - started

    pair_started = time.perf_counter()
    pair_records = build_pair_records(images, params)
    pair_elapsed = time.perf_counter() - pair_started
    ambiguous_records, thresholds = select_ambiguous_pairs(
        pair_records,
        near_quantile=args.near_quantile,
        far_quantile=args.far_quantile,
        fallback_k=max(args.top_k, args.chamfer_top_k),
    )
    nearest_records = nearest_image_records(pair_records, args.num_samples)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    chamfer_records = add_chamfer_to_pairs(
        ambiguous_records,
        params,
        config=config,
        max_pairs=args.chamfer_top_k,
        max_points=args.chamfer_max_points,
        device=device,
    )

    density_l2 = [record["density_l2"] for record in pair_records]
    density_mse = [record["density_mse"] for record in pair_records]
    param_distance = [record["param_distance"] for record in pair_records]
    w_fro = [record["w_fro"] for record in pair_records]
    b_l2 = [record["b_l2"] for record in pair_records]
    fp_l2 = [record["fixed_point_l2"] for record in pair_records]
    summary = {
        "run_dir": str(run_dir),
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": seed,
        "num_pairs": len(pair_records),
        "dataset_elapsed_sec": dataset_elapsed,
        "pair_elapsed_sec": pair_elapsed,
        "thresholds": thresholds,
        "ambiguous_pair_count": len(
            [
                record
                for record in pair_records
                if record["density_l2"] <= thresholds["density_near_threshold"]
                and record["param_distance"] >= thresholds["param_far_threshold"]
            ]
        ),
        "summaries": {
            "density_l2": _summarize(density_l2),
            "density_mse": _summarize(density_mse),
            "param_distance": _summarize(param_distance),
            "w_fro": _summarize(w_fro),
            "b_l2": _summarize(b_l2),
            "fixed_point_l2": _summarize(fp_l2),
            "nearest_image_param_distance": _summarize(
                [record["param_distance"] for record in nearest_records]
            ),
            "nearest_image_w_fro": _summarize([record["w_fro"] for record in nearest_records]),
        },
        "correlations": {
            "density_l2_vs_param_distance_pearson": _pearson(density_l2, param_distance),
            "density_l2_vs_param_distance_spearman": _spearman(density_l2, param_distance),
            "density_l2_vs_w_fro_pearson": _pearson(density_l2, w_fro),
            "density_l2_vs_w_fro_spearman": _spearman(density_l2, w_fro),
            "density_l2_vs_b_l2_pearson": _pearson(density_l2, b_l2),
            "density_l2_vs_fixed_point_l2_pearson": _pearson(density_l2, fp_l2),
        },
        "top_ambiguous_pairs": ambiguous_records[: args.top_k],
        "top_ambiguous_pairs_with_chamfer": chamfer_records[: args.top_k],
        "nearest_image_pairs_top_param": sorted(
            nearest_records,
            key=lambda record: record["param_distance"],
            reverse=True,
        )[: args.top_k],
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(pair_records, output_dir / "pair_distances.csv", limit=args.max_pairs_csv)
    write_csv(ambiguous_records, output_dir / "near_image_far_param_pairs.csv")
    write_csv(chamfer_records, output_dir / "near_image_far_param_pairs_chamfer.csv")
    write_csv(nearest_records, output_dir / "nearest_image_pairs.csv")
    save_scatter_figure(pair_records, ambiguous_records[: args.top_k], output_dir, thresholds)
    save_scatter_single_figure(pair_records, ambiguous_records, output_dir, thresholds)
    save_ambiguous_density_figure(ambiguous_records, images, output_dir, max_pairs=args.top_k)
    save_ambiguous_point_cloud_figure(
        ambiguous_records,
        params,
        output_dir,
        config=config,
        max_pairs=args.top_k,
        max_points=args.point_cloud_max_points,
    )
    save_sierpinski_nonidentifiability_figure(output_dir)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
