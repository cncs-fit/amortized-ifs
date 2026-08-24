"""Common-condition MNIST/Fashion-MNIST visual and metric comparison.

This script compares the amortized+refined IFS outputs and Tu et al. style
per-image optimized IFS checkpoints under two shared renderers:

1. The paper high-fidelity density renderer at 128x128.
2. Tu's 32x32 grayscale RBF renderer with min-over-random-sequences MSE.

Tu checkpoints are optional.  Missing Tu variants are skipped so the same
script can be used before and after running the public Tu optimization code.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from data.renderer import points_to_density_map
from data.sampler import affine_vector_probabilities, iterate_affine_vector_points
from losses.reconstruction import density_images_to_point_samples, target_density_for_reconstruction
from scripts.evaluate_oracle_fidelity import _coverage_thresholds, _filter_and_subsample, point_metrics

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-output", default="outputs/mnist_p0/base50k_best_n4_balanced10/mnist_p0_outputs.pt")
    parser.add_argument("--dataset", choices=("auto", "mnist", "fashion-mnist"), default="auto")
    parser.add_argument("--tu-root", default="refs/LearningFractals")
    parser.add_argument("--data-root", default="refs/LearningFractals/data")
    parser.add_argument("--output-dir", default="outputs/mnist_common/base50k_best_n4_balanced10_vs_tu")
    parser.add_argument("--tu-target", choices=("auto", "mnist", "fmnist", "kmnist"), default="auto")
    parser.add_argument("--ours-variants", nargs="+", default=("model-0", "model-30"))
    parser.add_argument("--tu-num-transforms", type=int, nargs="+", default=(4, 10))
    parser.add_argument("--tu-iteration", type=int, default=1000)
    parser.add_argument("--tu-lr", type=float, default=0.05)
    parser.add_argument("--tu-std", type=float, default=1.0)
    parser.add_argument("--tu-noise", type=float, default=0.1)
    parser.add_argument("--tu-init-seed", type=int, default=100)
    parser.add_argument("--tu-image-size", type=int, default=32)
    parser.add_argument("--tu-num-coords", type=int, default=300)
    parser.add_argument("--tu-gen-batch-size", type=int, default=50)
    parser.add_argument("--tu-tar-batch-size", type=int, default=1)
    parser.add_argument("--tu-num-rand-seq", type=int, default=100)
    parser.add_argument("--tu-sigma", type=float, default=1.0)
    parser.add_argument("--density-resolution", type=int, default=128)
    parser.add_argument("--density-num-trajectories", type=int, default=16)
    parser.add_argument("--density-num-steps", type=int, default=1024)
    parser.add_argument("--density-burn-in", type=int, default=128)
    parser.add_argument("--density-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--fixed-range", type=float, nargs=2, default=(-1.5, 1.5))
    parser.add_argument("--tu-native-range", type=float, nargs=2, default=(-5.0, 5.0))
    parser.add_argument("--chamfer-max-points", type=int, default=2048)
    parser.add_argument("--coverage-pixel-thresholds", type=float, nargs="+", default=(1.0, 2.0, 4.0))
    parser.add_argument("--grid-preview-rows", type=int, default=10)
    parser.add_argument("--density-all50-two-column-output", default=None)
    parser.add_argument("--seed", type=int, default=53100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def dataset_class(name: str):
    if name == "mnist":
        return datasets.MNIST
    if name == "fashion-mnist":
        return datasets.FashionMNIST
    raise ValueError(f"unsupported dataset: {name}")


def resolve_dataset_name(args: argparse.Namespace, payload: dict | None = None) -> str:
    if args.dataset != "auto":
        return str(args.dataset)
    if payload is not None:
        return str(payload.get("dataset", "mnist"))
    return "mnist"


def dataset_to_tu_target(dataset_name: str) -> str:
    if dataset_name == "mnist":
        return "mnist"
    if dataset_name == "fashion-mnist":
        return "fmnist"
    raise ValueError(f"unsupported dataset for Tu target: {dataset_name}")


def resolve_tu_target(args: argparse.Namespace, dataset_name: str) -> str:
    if args.tu_target != "auto":
        return str(args.tu_target)
    return dataset_to_tu_target(dataset_name)


def summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_grayscale_images(
    *,
    dataset_name: str,
    data_root: str,
    mnist_indices: list[int],
    image_size: int,
) -> torch.Tensor:
    dataset = dataset_class(dataset_name)(
        data_root,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        ),
    )
    images = [dataset[index][0].float() for index in mnist_indices]
    return torch.stack(images, dim=0).contiguous()


def tu_inner_dir(args: argparse.Namespace, num_transforms: int) -> str:
    return (
        f"img{args.tu_image_size}"
        f"_nc{args.tu_num_coords}"
        f"_tbs{args.tu_tar_batch_size}"
        f"_gbs{args.tu_gen_batch_size}"
        f"_nt{num_transforms}"
        f"_lr{args.tu_lr}"
        f"_std{args.tu_std}"
        f"_n{args.tu_noise}"
        f"_initseed{args.tu_init_seed}"
    )


def tu_checkpoint_path(
    args: argparse.Namespace,
    *,
    tu_target: str,
    digit: int,
    sample_idx: int,
    num_transforms: int,
) -> Path:
    return (
        Path(args.tu_root)
        / f"IMAGEMATCH_{tu_target.upper()}"
        / tu_inner_dir(args, num_transforms)
        / f"IDX{digit}-Sample{sample_idx}"
        / f"iter{args.tu_iteration}_opti_ifs_code.pth"
    )


def rotation_matrices(theta: torch.Tensor) -> torch.Tensor:
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    row0 = torch.stack((cos_t, -sin_t), dim=-1)
    row1 = torch.stack((sin_t, cos_t), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def tu_svd_weight_to_affine(raw_w: torch.Tensor, raw_b: torch.Tensor) -> torch.Tensor:
    """Convert Tu public-code SVD weights to row-vector affine parameters."""
    theta1 = raw_w[:, 0]
    theta2 = raw_w[:, 1]
    sigma1 = torch.sigmoid(raw_w[:, 2])
    sigma2 = torch.sigmoid(raw_w[:, 3])
    d1 = torch.sign(raw_w[:, 4]).clamp(min=-1.0, max=1.0)
    d2 = torch.sign(raw_w[:, 5]).clamp(min=-1.0, max=1.0)
    d1 = torch.where(d1 == 0, torch.ones_like(d1), d1)
    d2 = torch.where(d2 == 0, torch.ones_like(d2), d2)
    zeros = torch.zeros_like(sigma1)
    sig = torch.stack(
        (
            torch.stack((sigma1, zeros), dim=-1),
            torch.stack((zeros, sigma2), dim=-1),
        ),
        dim=-2,
    )
    diag_sign = torch.stack(
        (
            torch.stack((d1, zeros), dim=-1),
            torch.stack((zeros, d2), dim=-1),
        ),
        dim=-2,
    )
    column_w = rotation_matrices(theta1) @ sig @ rotation_matrices(theta2) @ diag_sign
    row_w = column_w.transpose(-1, -2)
    return torch.cat((row_w.reshape(raw_w.shape[0], 4), raw_b.reshape(raw_b.shape[0], 2)), dim=-1).float()


def digit_sample_indices(labels: list[int]) -> list[int]:
    seen: dict[int, int] = {}
    sample_indices = []
    for label in labels:
        sample_index = seen.get(int(label), 0)
        sample_indices.append(sample_index)
        seen[int(label)] = sample_index + 1
    return sample_indices


def load_tu_params(
    args: argparse.Namespace,
    *,
    tu_target: str,
    labels: list[int],
    num_transforms: int,
) -> tuple[torch.Tensor | None, list[str]]:
    params = []
    missing = []
    sample_indices = digit_sample_indices(labels)
    for digit, sample_idx in zip(labels, sample_indices):
        path = tu_checkpoint_path(
            args,
            tu_target=tu_target,
            digit=int(digit),
            sample_idx=int(sample_idx),
            num_transforms=num_transforms,
        )
        if not path.exists():
            missing.append(str(path))
            continue
        payload = torch.load(path, map_location="cpu")
        params.append(tu_svd_weight_to_affine(payload["w"].float(), payload["b"].float()))
    if missing:
        return None, missing
    return torch.stack(params, dim=0), []


def load_variants(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], dict[str, torch.Tensor], dict[str, tuple[float, float]], list[str]]:
    payload = torch.load(args.ours_output, map_location="cpu", weights_only=False)
    dataset_name = resolve_dataset_name(args, payload)
    tu_target = resolve_tu_target(args, dataset_name)
    images = payload["images"].float()
    labels = [int(value) for value in payload["labels"]]
    mnist_indices = [int(value) for value in payload["mnist_indices"]]
    variants = {}
    native_ranges = {}
    for name in args.ours_variants:
        if name not in payload["params_by_label"]:
            raise KeyError(f"missing ours variant {name} in {args.ours_output}")
        variant_name = f"ours-{name.removeprefix('model-')}"
        variants[variant_name] = payload["params_by_label"][name].float()
        native_ranges[variant_name] = tuple(float(value) for value in args.fixed_range)

    missing_paths: list[str] = []
    for num_transforms in args.tu_num_transforms:
        tu_params, missing = load_tu_params(
            args,
            tu_target=tu_target,
            labels=labels,
            num_transforms=int(num_transforms),
        )
        if tu_params is None:
            missing_paths.extend(missing)
            continue
        variant_name = f"tu-n{num_transforms}"
        variants[variant_name] = tu_params.float()
        native_ranges[variant_name] = tuple(float(value) for value in args.tu_native_range)

    tu32_targets = load_grayscale_images(
        dataset_name=dataset_name,
        data_root=args.data_root,
        mnist_indices=mnist_indices,
        image_size=args.tu_image_size,
    )
    return images, tu32_targets, labels, mnist_indices, variants, native_ranges, missing_paths


def rescale_points(
    points: torch.Tensor,
    *,
    source_range: tuple[float, float],
    target_range: tuple[float, float],
) -> torch.Tensor:
    source_low, source_high = source_range
    target_low, target_high = target_range
    scale = float(target_high - target_low) / float(source_high - source_low)
    return (points - float(source_low)) * scale + float(target_low)


def fixed_point_first_map(params: torch.Tensor) -> torch.Tensor:
    w = params[0, 0:4].reshape(2, 2)
    b = params[0, 4:6]
    eye = torch.eye(2, dtype=params.dtype, device=params.device)
    try:
        return torch.linalg.solve(eye - w.T, b)
    except RuntimeError:
        return torch.zeros(2, dtype=params.dtype, device=params.device)


def sample_affine_points_tu_protocol(
    params: torch.Tensor,
    *,
    num_sequences: int,
    num_coords: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    params = params.to(device=device, dtype=torch.float32)
    w = params[:, 0:4].reshape(-1, 2, 2)
    b = params[:, 4:6]
    probs = affine_vector_probabilities(params, prob_floor=0.0)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        probs = torch.full_like(probs, 1.0 / float(probs.numel()))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    seqs = torch.multinomial(
        probs,
        num_sequences * num_coords,
        replacement=True,
        generator=generator,
    ).reshape(num_sequences, num_coords)
    current = fixed_point_first_map(params).view(1, 2).repeat(num_sequences, 1)
    coords = []
    for step in range(num_coords):
        ids = seqs[:, step]
        current = torch.bmm(current.unsqueeze(1), w[ids]).squeeze(1) + b[ids]
        coords.append(current)
    return torch.stack(coords, dim=1)


def render_tu32_images(
    params: torch.Tensor,
    *,
    args: argparse.Namespace,
    native_range: tuple[float, float],
    sample_index: int,
    variant_offset: int,
    device: torch.device,
) -> torch.Tensor:
    coords = sample_affine_points_tu_protocol(
        params,
        num_sequences=args.tu_num_rand_seq,
        num_coords=args.tu_num_coords,
        seed=args.seed + 101_000 + 10_003 * sample_index + int(variant_offset),
        device=device,
    )
    # Tu's public evaluator fixes the render boundary to [-5, 5]
    # (refs/LearningFractals/evaluate_mse.py:105); adaptive bbox is only used
    # during training.
    low, high = tuple(float(value) for value in args.tu_native_range)
    coords = rescale_points(coords, source_range=native_range, target_range=(low, high))
    size = int(args.tu_image_size)
    norm = ((coords - low) / (high - low)).clamp(0.0, 1.0) * float(size - 1)
    grid = torch.arange(size, device=device, dtype=torch.float32)
    x_coords = grid.view(1, 1, size, 1)
    y_coords = grid.view(1, 1, 1, size)
    dist = (norm[:, :, 0].unsqueeze(-1).unsqueeze(-1) - x_coords).square()
    dist = dist + (norm[:, :, 1].unsqueeze(-1).unsqueeze(-1) - y_coords).square()
    images = torch.exp(-dist / float(args.tu_sigma)).sum(dim=1).clamp(0.0, 1.0)
    return images.unsqueeze(1)


def render_density_variant(
    params: torch.Tensor,
    *,
    args: argparse.Namespace,
    native_range: tuple[float, float],
    sample_index: int,
    variant_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(
        int(args.seed) + 10_003 * int(sample_index) + int(variant_offset)
    )
    points = iterate_affine_vector_points(
        params.to(device=device, dtype=torch.float32),
        num_trajectories=args.density_num_trajectories,
        num_steps=args.density_num_steps,
        burn_in=args.density_burn_in,
        generator=generator,
    )
    display_points = rescale_points(points, source_range=native_range, target_range=tuple(args.fixed_range))
    density = points_to_density_map(
        display_points,
        resolution=args.density_resolution,
        fixed_range=tuple(args.fixed_range),
        smoothing_sigma=args.density_smoothing_sigma,
    )
    return display_points, density


def evaluate_tu32(
    targets: torch.Tensor,
    variants: dict[str, torch.Tensor],
    native_ranges: dict[str, tuple[float, float]],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    records = []
    best_images: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    for sample_index in range(targets.shape[0]):
        target = targets[sample_index : sample_index + 1].to(device=device)
        for variant_index, (name, params) in enumerate(variants.items()):
            images = render_tu32_images(
                params[sample_index],
                args=args,
                native_range=native_ranges[name],
                sample_index=sample_index,
                variant_offset=13_000 * (variant_index + 1),
                device=device,
            )
            mses = (images - target).square().mean(dim=(1, 2, 3))
            best_index = int(torch.argmin(mses).item())
            best_images[name].append(images[best_index, 0].detach().cpu())
            records.append(
                {
                    "sample_index": sample_index,
                    "variant": name,
                    "tu32_mse_min": float(mses.min().detach().cpu().item()),
                    "tu32_mse_mean": float(mses.mean().detach().cpu().item()),
                    "tu32_mse_median": float(mses.median().detach().cpu().item()),
                    "tu32_mse_p90": float(torch.quantile(mses.detach().cpu(), 0.90).item()),
                    "tu32_best_sequence": best_index,
                }
            )
    return records, {name: torch.stack(images, dim=0) for name, images in best_images.items()}


def evaluate_density128(
    images: torch.Tensor,
    variants: dict[str, torch.Tensor],
    native_ranges: dict[str, tuple[float, float]],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    target_density = target_density_for_reconstruction(images, resolution=args.density_resolution)
    coverage = _coverage_thresholds(
        fixed_range=tuple(args.fixed_range),
        resolution=args.density_resolution,
        pixel_thresholds=args.coverage_pixel_thresholds,
    )
    rendered: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    records = []
    for sample_index in range(images.shape[0]):
        target_points = density_images_to_point_samples(
            target_density[sample_index : sample_index + 1],
            resolution=args.density_resolution,
            fixed_range=tuple(args.fixed_range),
            num_points=args.chamfer_max_points,
            seed=args.seed + 71_000 + sample_index,
        )[0]
        for variant_index, (name, params) in enumerate(variants.items()):
            points, density = render_density_variant(
                params[sample_index],
                args=args,
                native_range=native_ranges[name],
                sample_index=sample_index,
                variant_offset=17_000 * (variant_index + 1),
                device=device,
            )
            rendered[name].append(density[0].detach().cpu())
            generator = torch.Generator(device="cpu").manual_seed(args.seed + 91_000 + 1009 * variant_index + sample_index)
            sampled_points = _filter_and_subsample(
                points,
                fixed_range=tuple(args.fixed_range),
                max_points=args.chamfer_max_points,
                generator=generator,
            )
            pmetrics = point_metrics(sampled_points, target_points, device=device, coverage_thresholds=coverage)
            diff = density.detach().cpu().float() - target_density[sample_index : sample_index + 1].cpu().float()
            record = {
                "sample_index": sample_index,
                "variant": name,
                "density_sse": float(diff.square().sum().item()),
                "density_l1": float(diff.abs().sum().item()),
                "num_points_used": int(sampled_points.shape[0]),
            }
            for key, value in pmetrics.items():
                record[key] = float(value)
            records.append(record)
    return records, {name: torch.stack(values, dim=0) for name, values in rendered.items()}


def summary_rows(records: list[dict], *, metric_keys: list[str]) -> list[dict]:
    variants = list(dict.fromkeys(record["variant"] for record in records))
    rows = []
    for variant in variants:
        selected = [record for record in records if record["variant"] == variant]
        row = {"variant": variant}
        for metric in metric_keys:
            stats = summarize([float(record[metric]) for record in selected])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_median"] = stats["median"]
            row[f"{metric}_p90"] = stats["p90"]
        rows.append(row)
    return rows


def save_grid(
    target_images: torch.Tensor,
    variant_images: dict[str, torch.Tensor],
    labels: list[int],
    *,
    output_path: Path,
    cmap: str,
    vmax: float | None,
    block_count: int = 1,
    font_size: float = 18.0,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    def display_name(name: str, *, compact: bool = False) -> str:
        if name.startswith("tu-n"):
            label = f"Tu(N={name.removeprefix('tu-n')})"
        elif name.startswith("ours-tu32gd-model-0-"):
            label = f"occ-GD-{name.removeprefix('ours-tu32gd-model-0-')}"
        else:
            label = name
        if not compact:
            return label
        if label == "ours-100":
            return "ours\n100"
        if label.startswith("occ-GD-"):
            return f"occ-GD\n{label.removeprefix('occ-GD-')}"
        if label.startswith("Tu("):
            return label.replace("(", "\n(")
        return label

    names = list(variant_images.keys())
    num_samples = len(labels)
    cols_per_block = 1 + len(names)
    block_count = max(1, min(int(block_count), max(1, num_samples)))
    block_sizes = []
    remaining = num_samples
    remaining_blocks = block_count
    while remaining_blocks > 0:
        size = (remaining + remaining_blocks - 1) // remaining_blocks
        block_sizes.append(size)
        remaining -= size
        remaining_blocks -= 1
    rows = max(block_sizes)
    cols = cols_per_block * block_count
    cell_size = 1.55 if block_count == 1 else 1.42
    fig, axes = plt.subplots(rows, cols, figsize=(cell_size * cols, cell_size * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    start = 0
    for block_index, block_size in enumerate(block_sizes):
        col_offset = block_index * cols_per_block
        for block_row in range(block_size):
            row = start + block_row
            axes[block_row, col_offset].imshow(
                target_images[row, 0].cpu(),
                cmap=cmap,
                origin="upper",
                vmin=0.0,
                vmax=vmax,
            )
            axes[block_row, col_offset].set_ylabel(
                f"{labels[row]}",
                fontsize=font_size,
                rotation=0,
                labelpad=14,
                va="center",
            )
            if block_row == 0:
                axes[block_row, col_offset].set_title("target", fontsize=font_size)
            for col, name in enumerate(names, start=1):
                axes[block_row, col_offset + col].imshow(
                    variant_images[name][row].cpu(),
                    cmap=cmap,
                    origin="upper",
                    vmin=0.0,
                    vmax=vmax,
                )
                if block_row == 0:
                    axes[block_row, col_offset + col].set_title(
                        display_name(name, compact=block_count > 1),
                        fontsize=font_size,
                    )
        start += block_size
    fig.tight_layout(pad=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=360)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    images, tu32_targets, labels, mnist_indices, variants, native_ranges, missing_paths = load_variants(args)
    tu32_records, tu32_best_images = evaluate_tu32(tu32_targets, variants, native_ranges, args=args, device=device)
    density_records, density_images = evaluate_density128(images, variants, native_ranges, args=args, device=device)

    tu32_summary = summary_rows(
        tu32_records,
        metric_keys=["tu32_mse_min", "tu32_mse_mean", "tu32_mse_median", "tu32_mse_p90"],
    )
    density_summary = summary_rows(
        density_records,
        metric_keys=[
            "density_sse",
            "density_l1",
            "chamfer",
            "hausdorff_p95",
            "coverage_symmetric_1px",
            "coverage_symmetric_2px",
            "coverage_symmetric_4px",
            "modified_hausdorff_mean",
        ],
    )
    write_csv(tu32_records, output_dir / "tu32_per_sample.csv")
    write_csv(tu32_summary, output_dir / "tu32_summary.csv")
    write_csv(density_records, output_dir / "density128_per_sample.csv")
    write_csv(density_summary, output_dir / "density128_summary.csv")

    target_density = target_density_for_reconstruction(images, resolution=args.density_resolution)[:, 0].cpu()
    save_grid(
        tu32_targets,
        tu32_best_images,
        labels,
        output_path=output_dir / "mnist_common_tu32_grid",
        cmap="gray",
        vmax=1.0,
    )
    save_grid(
        target_density.unsqueeze(1),
        density_images,
        labels,
        output_path=output_dir / "mnist_common_density128_grid",
        cmap="magma",
        vmax=None,
    )
    save_grid(
        target_density.unsqueeze(1),
        density_images,
        labels,
        output_path=output_dir / "mnist_common_density128_grid_gray",
        cmap="gray",
        vmax=None,
    )
    density_stack = torch.cat([target_density, *density_images.values()], dim=0)
    density_vmax_shared = float(torch.quantile(density_stack.flatten(), 0.995).item())
    save_grid(
        target_density.unsqueeze(1),
        density_images,
        labels,
        output_path=output_dir / "mnist_common_density128_grid_gray_shared",
        cmap="gray",
        vmax=density_vmax_shared,
    )
    density_all50_two_column = None
    if args.density_all50_two_column_output is not None:
        two_column_path = Path(args.density_all50_two_column_output)
        save_grid(
            target_density.unsqueeze(1),
            density_images,
            labels,
            output_path=two_column_path,
            cmap="gray",
            vmax=density_vmax_shared,
            block_count=2,
            font_size=20.0,
        )
        left_rows = (len(labels) + 1) // 2
        density_all50_two_column = {
            "output_png": str(two_column_path.with_suffix(".png")),
            "output_pdf": str(two_column_path.with_suffix(".pdf")),
            "block_count": 2,
            "left_rows": left_rows,
            "right_rows": len(labels) - left_rows,
            "left_sample_positions": list(range(0, left_rows)),
            "right_sample_positions": list(range(left_rows, len(labels))),
        }
    if int(args.grid_preview_rows) > 0 and int(args.grid_preview_rows) < len(labels):
        num_rows = int(args.grid_preview_rows)
        preview_labels = labels[:num_rows]
        tu32_preview = {name: value[:num_rows] for name, value in tu32_best_images.items()}
        density_preview = {name: value[:num_rows] for name, value in density_images.items()}
        save_grid(
            tu32_targets[:num_rows],
            tu32_preview,
            preview_labels,
            output_path=output_dir / f"mnist_common_tu32_grid_first{num_rows}",
            cmap="gray",
            vmax=1.0,
        )
        save_grid(
            target_density[:num_rows].unsqueeze(1),
            density_preview,
            preview_labels,
            output_path=output_dir / f"mnist_common_density128_grid_gray_first{num_rows}",
            cmap="gray",
            vmax=None,
        )
        save_grid(
            target_density[:num_rows].unsqueeze(1),
            density_preview,
            preview_labels,
            output_path=output_dir / f"mnist_common_density128_grid_gray_shared_first{num_rows}",
            cmap="gray",
            vmax=density_vmax_shared,
        )

    manifest = {
        "ours_output": args.ours_output,
        "dataset": resolve_dataset_name(args, torch.load(args.ours_output, map_location="cpu", weights_only=False)),
        "tu_target": resolve_tu_target(
            args,
            resolve_dataset_name(args, torch.load(args.ours_output, map_location="cpu", weights_only=False)),
        ),
        "tu_root": args.tu_root,
        "mnist_indices": mnist_indices,
        "labels": labels,
        "digit_sample_indices": digit_sample_indices(labels),
        "variants": list(variants.keys()),
        "native_ranges": {name: list(value) for name, value in native_ranges.items()},
        "missing_tu_checkpoints": missing_paths,
        "args": vars(args),
        "density128_gray_shared_vmax_p995": density_vmax_shared,
        "density_all50_two_column": density_all50_two_column,
        "tu32_summary": tu32_summary,
        "density128_summary": density_summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"variants": list(variants.keys()), "missing": len(missing_paths)}, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
