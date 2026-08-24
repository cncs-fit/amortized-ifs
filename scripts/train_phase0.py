"""Phase 0 training: on-the-fly synthetic data with fixed validation/test sets."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
import yaml

from data.dataset import (
    IFSIterableDataset,
    IFSIterableDatasetConfig,
    default_train_cache_path,
    generate_train_cache_batched_from_iterable_config,
    generate_train_cache_from_iterable_config,
    load_train_cache,
    make_fixed_dataset_from_iterable_config,
    save_train_cache_batched_from_iterable_config,
    save_train_cache_from_iterable_config,
)
from data.renderer import render_density_from_affine_vector, render_density_from_params
from losses.hungarian import (
    hungarian_matching_loss,
    hungarian_matching_loss_affine,
    hungarian_metrics,
    hungarian_metrics_affine,
)
from losses.reconstruction import (
    density_reconstruction_loss_affine,
    density_reconstruction_loss_svd,
    point_chamfer_loss_affine,
    point_chamfer_loss_svd,
)
from models.set_head import TinyCNNAffineSetEstimator, TinyCNNSetEstimator

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--num-transforms", type=int, default=2)
    parser.add_argument("--phase", default="phase0")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--num-trajectories", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--density-smoothing-sigma", type=float, default=0.0)
    parser.add_argument("--validity-resolution", type=int, default=None)
    parser.add_argument("--validity-num-trajectories", type=int, default=None)
    parser.add_argument("--validity-num-steps", type=int, default=None)
    parser.add_argument("--validity-burn-in", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-seed", type=int, default=3100)
    parser.add_argument("--test-seed", type=int, default=4100)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--lr-decay-start-step", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pool-grid", type=int, default=4)
    parser.add_argument(
        "--encoder-type",
        choices=("tiny", "residual", "residual_wide", "residual_wide_attn"),
        default="tiny",
    )
    parser.add_argument("--coord-channels", action="store_true")
    parser.add_argument("--density-feature-mode", choices=("none", "moments"), default="none")
    parser.add_argument("--global-moments", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--head-type", choices=("mlp", "query_attention"), default="mlp")
    parser.add_argument("--query-num-heads", type=int, default=8)
    parser.add_argument("--query-layers", type=int, default=2)
    parser.add_argument("--target-representation", choices=("svd", "affine"), default="svd")
    parser.add_argument("--linear-loss-weight", type=float, default=1.0)
    parser.add_argument("--bias-loss-weight", type=float, default=1.0)
    parser.add_argument("--fixed-point-loss-weight", type=float, default=0.0)
    parser.add_argument("--fixed-point-cost-weight", type=float, default=0.0)
    parser.add_argument("--spectral-upper-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-upper-bound", type=float, default=0.75)
    parser.add_argument("--det-positive-loss-weight", type=float, default=0.0)
    parser.add_argument("--singular-value-loss-weight", type=float, default=0.0)
    parser.add_argument("--determinant-loss-weight", type=float, default=0.0)
    parser.add_argument("--target-spectral-linear-extra-weight", type=float, default=0.0)
    parser.add_argument("--target-spectral-linear-threshold", type=float, default=0.55)
    parser.add_argument("--target-spectral-linear-upper", type=float, default=0.70)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=0.0)
    parser.add_argument("--reconstruction-loss-interval", type=int, default=1)
    parser.add_argument("--reconstruction-loss-batch-size", type=int, default=0)
    parser.add_argument("--reconstruction-resolution", type=int, default=64)
    parser.add_argument("--reconstruction-num-trajectories", type=int, default=4)
    parser.add_argument("--reconstruction-num-steps", type=int, default=96)
    parser.add_argument("--reconstruction-burn-in", type=int, default=16)
    parser.add_argument("--reconstruction-smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--reconstruction-seed", type=int, default=12345)
    parser.add_argument(
        "--reconstruction-map-probability-mode",
        choices=("uniform", "determinant"),
        default="uniform",
    )
    parser.add_argument("--reconstruction-match-render-config", action="store_true")
    parser.add_argument("--point-chamfer-loss-weight", type=float, default=0.0)
    parser.add_argument("--point-chamfer-loss-interval", type=int, default=1)
    parser.add_argument("--point-chamfer-batch-size", type=int, default=0)
    parser.add_argument("--point-chamfer-num-pred-points", type=int, default=512)
    parser.add_argument("--point-chamfer-num-target-points", type=int, default=512)
    parser.add_argument("--point-chamfer-seed", type=int, default=24680)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-model-path", default=None)
    parser.add_argument("--output-dir", default="outputs/phase0")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comparison-samples", type=int, default=4)
    parser.add_argument("--point-cloud-max-points", type=int, default=3000)
    parser.add_argument("--train-cache-samples", type=int, default=0)
    parser.add_argument("--train-cache-dir", default="cache/phase0_train")
    parser.add_argument("--train-cache-path", default=None)
    parser.add_argument("--train-cache-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-train-cache", action="store_true")
    parser.add_argument(
        "--cache-generation-mode",
        choices=("iterable", "batched"),
        default="iterable",
    )
    parser.add_argument("--cache-generation-device", default=None)
    parser.add_argument("--cache-build-batch-size", type=int, default=64)
    parser.add_argument("--cache-num-workers", type=int, default=None)
    parser.add_argument("--train-cache-refresh-interval", type=int, default=0)
    parser.add_argument("--train-cache-refresh-seed-stride", type=int, default=1_000_003)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-monitor-interval", type=float, default=0.0)
    if config_args.config is not None:
        config_path = Path(config_args.config)
        with config_path.open("r", encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle) or {}
        parser.set_defaults(**defaults)
    return parser.parse_args()


def _loader_kwargs(
    args: argparse.Namespace,
    *,
    for_train: bool,
    for_cache: bool = False,
) -> dict:
    kwargs = {
        "batch_size": args.batch_size if for_train else args.eval_batch_size,
        "num_workers": args.num_workers if for_train else 0,
        "pin_memory": args.device == "cuda",
    }
    if for_train and args.num_workers > 0 and not for_cache:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def infinite_batches(loader: DataLoader):
    """Yield batches forever, restarting finite loaders at epoch boundaries."""
    while True:
        for batch in loader:
            yield batch


def step_learning_rate(args: argparse.Namespace, step: int) -> float:
    """Return the learning rate for a 1-based training step."""
    if args.lr_scheduler == "none":
        return float(args.lr)
    if args.lr_scheduler == "cosine":
        decay_start = max(1, int(args.lr_decay_start_step))
        if step < decay_start:
            return float(args.lr)
        if args.train_steps <= decay_start:
            return float(args.min_lr)
        progress = float(step - decay_start) / float(args.train_steps - decay_start)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(args.min_lr + (args.lr - args.min_lr) * cosine)
    raise ValueError(f"unsupported lr scheduler: {args.lr_scheduler}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def apply_reconstruction_match_render_config(
    args: argparse.Namespace,
    train_config: IFSIterableDatasetConfig,
) -> None:
    """Use the dataset renderer settings for the differentiable recon loss."""
    if not args.reconstruction_match_render_config:
        return
    args.reconstruction_resolution = int(train_config.resolution)
    args.reconstruction_num_trajectories = int(train_config.num_trajectories)
    args.reconstruction_num_steps = int(train_config.num_steps)
    args.reconstruction_burn_in = int(train_config.burn_in)
    args.reconstruction_smoothing_sigma = float(train_config.density_smoothing_sigma)
    args.reconstruction_map_probability_mode = "determinant"


def build_or_load_train_cache(
    args: argparse.Namespace,
    train_config: IFSIterableDatasetConfig,
) -> tuple[torch.utils.data.TensorDataset, Path, dict, float]:
    """Build or load a finite train cache and return timing metadata."""
    cache_workers = args.num_workers if args.cache_num_workers is None else args.cache_num_workers
    generation_device = args.cache_generation_device
    if generation_device is None:
        generation_device = args.device if args.cache_generation_mode == "batched" else None
    save_cache = bool(getattr(args, "train_cache_save", True))
    if args.train_cache_path is None:
        cache_path = default_train_cache_path(
            train_config,
            num_samples=args.train_cache_samples,
            cache_dir=args.train_cache_dir,
            generation_num_workers=cache_workers,
            generation_mode=args.cache_generation_mode
            if args.cache_generation_mode != "iterable"
            else None,
            generation_device=(
                generation_device if args.cache_generation_mode != "iterable" else None
            ),
        )
    else:
        cache_path = Path(args.train_cache_path)

    started = time.perf_counter()
    if args.rebuild_train_cache or not cache_path.exists():
        if save_cache:
            print(
                f"building train cache: samples={args.train_cache_samples} "
                f"workers={cache_workers} path={cache_path}",
                flush=True,
            )
            if args.cache_generation_mode == "iterable":
                dataset = save_train_cache_from_iterable_config(
                    train_config,
                    num_samples=args.train_cache_samples,
                    path=cache_path,
                    batch_size=args.cache_build_batch_size,
                    num_workers=cache_workers,
                )
            else:
                dataset = save_train_cache_batched_from_iterable_config(
                    train_config,
                    num_samples=args.train_cache_samples,
                    path=cache_path,
                    batch_size=args.cache_build_batch_size,
                    device=generation_device,
                )
            cache_status = "built"
        else:
            print(
                f"building unsaved train cache: samples={args.train_cache_samples} "
                f"workers={cache_workers} planned_path={cache_path}",
                flush=True,
            )
            if args.cache_generation_mode == "iterable":
                dataset = generate_train_cache_from_iterable_config(
                    train_config,
                    num_samples=args.train_cache_samples,
                    batch_size=args.cache_build_batch_size,
                    num_workers=cache_workers,
                )
            else:
                dataset = generate_train_cache_batched_from_iterable_config(
                    train_config,
                    num_samples=args.train_cache_samples,
                    batch_size=args.cache_build_batch_size,
                    device=generation_device,
                )
            cache_status = "built_unsaved"
        metadata = {
            "cache_path": str(cache_path),
            "cache_samples": len(dataset),
            "cache_status": cache_status,
            "cache_saved": save_cache,
            "cache_num_workers": cache_workers,
            "cache_generation_mode": args.cache_generation_mode,
            "cache_generation_device": generation_device,
        }
    else:
        print(f"loading train cache: path={cache_path}", flush=True)
        dataset, saved_metadata = load_train_cache(cache_path)
        metadata = {
            "cache_path": str(cache_path),
            "cache_samples": len(dataset),
            "cache_status": "loaded",
            "cache_saved": True,
            "cache_num_workers": saved_metadata.get("generation_num_workers", cache_workers),
            "saved_metadata": saved_metadata,
        }

    elapsed = time.perf_counter() - started
    print(
        f"train cache ready: samples={len(dataset)} elapsed_sec={elapsed:.2f}",
        flush=True,
    )
    return dataset, cache_path, metadata, elapsed


def make_cached_train_loader(
    args: argparse.Namespace,
    train_config: IFSIterableDatasetConfig,
) -> tuple[DataLoader, dict, float]:
    """Build/load a finite cache and wrap it in a train DataLoader."""
    train_dataset, _, cache_metadata, cache_elapsed_sec = build_or_load_train_cache(
        args,
        train_config,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **_loader_kwargs(args, for_train=True, for_cache=True),
    )
    return train_loader, cache_metadata, cache_elapsed_sec


def start_gpu_monitor(output_dir: Path, *, interval_sec: float):
    """Start lightweight nvidia-smi CSV logging for long GPU runs."""
    if interval_sec <= 0.0:
        return None
    log_path = output_dir / "gpu_monitor.csv"
    query = (
        "timestamp,index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,power.draw,temperature.gpu"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,nounits",
        "-l",
        str(max(1, int(round(interval_sec)))),
    ]
    handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.DEVNULL, text=True)
    except FileNotFoundError:
        handle.close()
        return None
    monitor = {"process": process, "handle": handle, "path": log_path, "stopped": False}
    atexit.register(stop_gpu_monitor, monitor)
    return monitor


def stop_gpu_monitor(monitor) -> None:
    if monitor is None or monitor.get("stopped"):
        return
    process = monitor["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    monitor["handle"].close()
    monitor["stopped"] = True


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    target_representation: str,
) -> Dict[str, float]:
    model.eval()
    batch_metrics: List[Dict[str, float]] = []
    with torch.no_grad():
        for images, target_params in loader:
            images = images.to(device)
            target_params = target_params.to(device)
            pred_params = model(images)
            if target_representation == "affine":
                batch_metrics.append(hungarian_metrics_affine(pred_params, target_params))
            else:
                batch_metrics.append(hungarian_metrics(pred_params, target_params))
    model.train()
    keys = batch_metrics[0].keys()
    return {key: sum(metrics[key] for metrics in batch_metrics) / len(batch_metrics) for key in keys}


def constant_baseline_metrics(
    dataset,
    *,
    target_representation: str,
    scale: float = 0.45,
) -> Dict[str, float]:
    """Evaluate a constant centered IFS prediction on a fixed TensorDataset."""
    target_params = dataset.tensors[1]
    pred = torch.zeros_like(target_params)
    if target_representation == "affine":
        pred[..., 0] = scale
        pred[..., 3] = scale
        return hungarian_metrics_affine(pred, target_params)

    pred[..., 2:4] = scale
    return hungarian_metrics(pred, target_params)


def save_curves(history: Iterable[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    records = list(history)
    train_steps = [record["step"] for record in records if "train_loss" in record]
    train_loss = [record["train_loss"] for record in records if "train_loss" in record]
    eval_records = [record for record in records if "val_loss" in record]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].plot(train_steps, train_loss, label="train")
    if eval_records:
        axes[0].plot(
            [record["step"] for record in eval_records],
            [record["val_loss"] for record in eval_records],
            marker="o",
            label="val",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    if eval_records:
        steps = [record["step"] for record in eval_records]
        axes[1].plot(steps, [record["val_w_fro"] for record in eval_records], marker="o")
        axes[1].set_ylabel("val W Frobenius")
        axes[1].set_xlabel("step")
        axes[2].plot(steps, [record["val_b_l2"] for record in eval_records], marker="o")
        axes[2].set_ylabel("val b L2")
        axes[2].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=360)
    fig.savefig(output_dir / "training_curves.pdf")
    plt.close(fig)


def save_comparison_figure(
    model: torch.nn.Module,
    dataset,
    *,
    config: IFSIterableDatasetConfig,
    output_dir: Path,
    device: torch.device,
    num_samples: int,
    target_representation: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    images = dataset.tensors[0][:num_samples].to(device)
    with torch.no_grad():
        pred_params = model(images).cpu()

    generator = torch.Generator(device="cpu").manual_seed(config.seed + 9999)
    pred_images = []
    for params in pred_params:
        if target_representation == "affine":
            _, density = render_density_from_affine_vector(
                params,
                resolution=config.resolution,
                fixed_range=config.fixed_range,
                num_trajectories=config.num_trajectories,
                num_steps=config.num_steps,
                burn_in=config.burn_in,
                density_smoothing_sigma=config.density_smoothing_sigma,
                generator=generator,
            )
        else:
            _, density = render_density_from_params(
                params,
                resolution=config.resolution,
                fixed_range=config.fixed_range,
                num_trajectories=config.num_trajectories,
                num_steps=config.num_steps,
                burn_in=config.burn_in,
                density_smoothing_sigma=config.density_smoothing_sigma,
                generator=generator,
            )
        pred_images.append(density)
    pred_images = torch.stack(pred_images)
    target_images = dataset.tensors[0][:num_samples]

    fig, axes = plt.subplots(
        2,
        num_samples,
        figsize=(2.1 * num_samples, 4.2),
        squeeze=False,
    )
    for col in range(num_samples):
        axes[0, col].imshow(target_images[col, 0], cmap="magma", origin="lower")
        axes[0, col].set_title("input")
        axes[1, col].imshow(pred_images[col, 0], cmap="magma", origin="lower")
        axes[1, col].set_title("pred")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=360)
    fig.savefig(output_dir / "comparison.pdf")
    plt.close(fig)


def _filter_points_for_plot(
    points: torch.Tensor,
    *,
    fixed_range: Tuple[float, float],
    max_points: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [num_points, 2]")

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
        indices = torch.randperm(points.shape[0], generator=generator)[:max_points]
        points = points[indices]
    return points


def save_point_cloud_figure(
    model: torch.nn.Module,
    dataset,
    *,
    config: IFSIterableDatasetConfig,
    output_dir: Path,
    device: torch.device,
    num_samples: int,
    target_representation: str,
    max_points: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    images = dataset.tensors[0][:num_samples].to(device)
    target_params = dataset.tensors[1][:num_samples].cpu()
    with torch.no_grad():
        pred_params = model(images).cpu()

    target_clouds = []
    pred_clouds = []
    for sample_idx in range(num_samples):
        target_generator = torch.Generator(device="cpu").manual_seed(
            config.seed + 17_000 + sample_idx
        )
        pred_generator = torch.Generator(device="cpu").manual_seed(
            config.seed + 29_000 + sample_idx
        )
        downsample_generator = torch.Generator(device="cpu").manual_seed(
            config.seed + 41_000 + sample_idx
        )

        target_points, _ = render_density_from_params(
            target_params[sample_idx],
            resolution=config.resolution,
            fixed_range=config.fixed_range,
            num_trajectories=config.num_trajectories,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            density_smoothing_sigma=0.0,
            generator=target_generator,
        )
        if target_representation == "affine":
            pred_points, _ = render_density_from_affine_vector(
                pred_params[sample_idx],
                resolution=config.resolution,
                fixed_range=config.fixed_range,
                num_trajectories=config.num_trajectories,
                num_steps=config.num_steps,
                burn_in=config.burn_in,
                density_smoothing_sigma=0.0,
                generator=pred_generator,
            )
        else:
            pred_points, _ = render_density_from_params(
                pred_params[sample_idx],
                resolution=config.resolution,
                fixed_range=config.fixed_range,
                num_trajectories=config.num_trajectories,
                num_steps=config.num_steps,
                burn_in=config.burn_in,
                density_smoothing_sigma=0.0,
                generator=pred_generator,
            )

        target_clouds.append(
            _filter_points_for_plot(
                target_points,
                fixed_range=config.fixed_range,
                max_points=max_points,
                generator=downsample_generator,
            )
        )
        pred_clouds.append(
            _filter_points_for_plot(
                pred_points,
                fixed_range=config.fixed_range,
                max_points=max_points,
                generator=downsample_generator,
            )
        )

    fig, axes = plt.subplots(2, num_samples, figsize=(2.2 * num_samples, 4.4))
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    low, high = config.fixed_range
    for col in range(num_samples):
        for row, clouds, color, title in (
            (0, target_clouds, "#1f2937", "target"),
            (1, pred_clouds, "#b45309", "prediction"),
        ):
            ax = axes[row, col]
            points = clouds[col]
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
            ax.set_title(title, fontsize=8, pad=2)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "point_clouds.png", dpi=360)
    fig.savefig(output_dir / "point_clouds.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.train_cache_refresh_interval > 0 and args.train_cache_samples <= 0:
        raise ValueError("train cache refresh requires --train-cache-samples > 0")
    if args.train_cache_refresh_interval > 0 and args.train_cache_path is not None:
        raise ValueError("train cache refresh requires automatic cache paths")
    if args.reconstruction_loss_weight > 0.0 and args.reconstruction_loss_interval <= 0:
        raise ValueError("--reconstruction-loss-interval must be positive")
    if args.reconstruction_loss_batch_size < 0:
        raise ValueError("--reconstruction-loss-batch-size must be non-negative")
    if args.point_chamfer_loss_weight > 0.0 and args.point_chamfer_loss_interval <= 0:
        raise ValueError("--point-chamfer-loss-interval must be positive")
    if args.point_chamfer_batch_size < 0:
        raise ValueError("--point-chamfer-batch-size must be non-negative")
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_monitor = None
    if args.device == "cuda":
        gpu_monitor = start_gpu_monitor(output_dir, interval_sec=args.gpu_monitor_interval)

    train_config = IFSIterableDatasetConfig(
        num_transforms=args.num_transforms,
        phase=args.phase,
        resolution=args.resolution,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        burn_in=args.burn_in,
        seed=args.seed,
        density_smoothing_sigma=args.density_smoothing_sigma,
        validity_resolution=args.validity_resolution,
        validity_num_trajectories=args.validity_num_trajectories,
        validity_num_steps=args.validity_num_steps,
        validity_burn_in=args.validity_burn_in,
    )
    apply_reconstruction_match_render_config(args, train_config)
    cache_metadata = None
    cache_elapsed_sec = 0.0
    cache_metadata_history = []
    if args.train_cache_samples > 0:
        train_loader, cache_metadata, cache_elapsed_sec = make_cached_train_loader(
            args,
            train_config,
        )
        cache_metadata_history.append(
            {
                "step": 0,
                "seed": train_config.seed,
                "elapsed_sec": cache_elapsed_sec,
                "metadata": cache_metadata,
            }
        )
        train_data_source = (
            "rolling_cache" if args.train_cache_refresh_interval > 0 else "cache"
        )
    else:
        train_dataset = IFSIterableDataset(train_config)
        train_loader = DataLoader(
            train_dataset,
            **_loader_kwargs(args, for_train=True),
        )
        train_data_source = "on_the_fly"

    val_dataset = make_fixed_dataset_from_iterable_config(
        train_config,
        num_samples=args.val_samples,
        seed=args.val_seed,
    )
    test_dataset = make_fixed_dataset_from_iterable_config(
        train_config,
        num_samples=args.test_samples,
        seed=args.test_seed,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    device = torch.device(args.device)
    if args.target_representation == "affine":
        model = TinyCNNAffineSetEstimator(
            num_transforms=args.num_transforms,
            hidden_dim=args.hidden_dim,
            pool_grid=args.pool_grid,
            encoder_type=args.encoder_type,
            coord_channels=args.coord_channels,
            density_feature_mode=args.density_feature_mode,
            global_moments=args.global_moments,
            head_type=args.head_type,
            query_num_heads=args.query_num_heads,
            query_layers=args.query_layers,
        ).to(device)
        def loss_fn(pred_params, target_params):
            return hungarian_matching_loss_affine(
                pred_params,
                target_params,
                linear_weight=args.linear_loss_weight,
                bias_weight=args.bias_loss_weight,
                fixed_point_loss_weight=args.fixed_point_loss_weight,
                fixed_point_cost_weight=args.fixed_point_cost_weight,
                spectral_upper_loss_weight=args.spectral_upper_loss_weight,
                spectral_upper_bound=args.spectral_upper_bound,
                det_positive_loss_weight=args.det_positive_loss_weight,
                singular_value_loss_weight=args.singular_value_loss_weight,
                determinant_loss_weight=args.determinant_loss_weight,
                target_spectral_linear_extra_weight=args.target_spectral_linear_extra_weight,
                target_spectral_linear_threshold=args.target_spectral_linear_threshold,
                target_spectral_linear_upper=args.target_spectral_linear_upper,
            )
    else:
        model = TinyCNNSetEstimator(
            num_transforms=args.num_transforms,
            hidden_dim=args.hidden_dim,
            pool_grid=args.pool_grid,
            encoder_type=args.encoder_type,
            coord_channels=args.coord_channels,
            density_feature_mode=args.density_feature_mode,
            global_moments=args.global_moments,
            head_type=args.head_type,
            query_num_heads=args.query_num_heads,
            query_layers=args.query_layers,
        ).to(device)
        def loss_fn(pred_params, target_params):
            return hungarian_matching_loss(
                pred_params,
                target_params,
                linear_weight=args.linear_loss_weight,
                bias_weight=args.bias_loss_weight,
                fixed_point_loss_weight=args.fixed_point_loss_weight,
                fixed_point_cost_weight=args.fixed_point_cost_weight,
                spectral_upper_loss_weight=args.spectral_upper_loss_weight,
                spectral_upper_bound=args.spectral_upper_bound,
                det_positive_loss_weight=args.det_positive_loss_weight,
                singular_value_loss_weight=args.singular_value_loss_weight,
                determinant_loss_weight=args.determinant_loss_weight,
                target_spectral_linear_extra_weight=args.target_spectral_linear_extra_weight,
                target_spectral_linear_threshold=args.target_spectral_linear_threshold,
                target_spectral_linear_upper=args.target_spectral_linear_upper,
            )
    if args.init_model_path is not None:
        init_model_path = Path(args.init_model_path)
        if not init_model_path.exists():
            init_model_path = ROOT / init_model_path
        if not init_model_path.exists():
            raise FileNotFoundError(f"init model not found: {args.init_model_path}")
        model.load_state_dict(torch.load(init_model_path, map_location=device))
        print(f"loaded init model: {init_model_path}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[dict] = []
    perf_history: List[dict] = []
    train_iter = infinite_batches(train_loader)
    non_blocking = device.type == "cuda"
    interval_started = time.perf_counter()
    interval_data_wait_sec = 0.0
    interval_samples = 0
    last_perf_step = 0
    best_val_loss = float("inf")
    best_val_record = None
    for step in range(1, args.train_steps + 1):
        if (
            args.train_cache_refresh_interval > 0
            and step > 1
            and (step - 1) % args.train_cache_refresh_interval == 0
        ):
            refresh_index = (step - 1) // args.train_cache_refresh_interval
            refreshed_config = replace(
                train_config,
                seed=args.seed + refresh_index * args.train_cache_refresh_seed_stride,
            )
            print(
                f"refreshing train cache before step={step:05d} "
                f"refresh_index={refresh_index} seed={refreshed_config.seed}",
                flush=True,
            )
            train_loader, refresh_metadata, refresh_elapsed_sec = make_cached_train_loader(
                args,
                refreshed_config,
            )
            train_iter = infinite_batches(train_loader)
            cache_metadata_history.append(
                {
                    "step": step - 1,
                    "seed": refreshed_config.seed,
                    "elapsed_sec": refresh_elapsed_sec,
                    "metadata": refresh_metadata,
                }
            )
            interval_started = time.perf_counter()
            interval_data_wait_sec = 0.0
            interval_samples = 0
            last_perf_step = step - 1

        current_lr = step_learning_rate(args, step)
        set_optimizer_lr(optimizer, current_lr)

        fetch_started = time.perf_counter()
        images, target_params = next(train_iter)
        fetch_finished = time.perf_counter()
        interval_data_wait_sec += fetch_finished - fetch_started
        interval_samples += int(images.shape[0])

        images = images.to(device, non_blocking=non_blocking)
        target_params = target_params.to(device, non_blocking=non_blocking)
        pred_params = model(images)
        loss = loss_fn(pred_params, target_params)
        reconstruction_loss_item = None
        point_chamfer_loss_item = None
        if (
            args.reconstruction_loss_weight > 0.0
            and step % args.reconstruction_loss_interval == 0
        ):
            recon_pred = pred_params
            recon_images = images
            if (
                args.reconstruction_loss_batch_size > 0
                and args.reconstruction_loss_batch_size < images.shape[0]
            ):
                recon_pred = recon_pred[: args.reconstruction_loss_batch_size]
                recon_images = recon_images[: args.reconstruction_loss_batch_size]
            if args.target_representation == "affine":
                reconstruction_loss = density_reconstruction_loss_affine(
                    recon_pred,
                    recon_images,
                    resolution=args.reconstruction_resolution,
                    fixed_range=train_config.fixed_range,
                    num_trajectories=args.reconstruction_num_trajectories,
                    num_steps=args.reconstruction_num_steps,
                    burn_in=args.reconstruction_burn_in,
                    smoothing_sigma=args.reconstruction_smoothing_sigma,
                    seed=args.reconstruction_seed,
                    map_probability_mode=args.reconstruction_map_probability_mode,
                )
            else:
                reconstruction_loss = density_reconstruction_loss_svd(
                    recon_pred,
                    recon_images,
                    resolution=args.reconstruction_resolution,
                    fixed_range=train_config.fixed_range,
                    num_trajectories=args.reconstruction_num_trajectories,
                    num_steps=args.reconstruction_num_steps,
                    burn_in=args.reconstruction_burn_in,
                    smoothing_sigma=args.reconstruction_smoothing_sigma,
                    seed=args.reconstruction_seed,
                    map_probability_mode=args.reconstruction_map_probability_mode,
                )
            reconstruction_loss_item = reconstruction_loss.item()
            loss = loss + args.reconstruction_loss_weight * reconstruction_loss
        if (
            args.point_chamfer_loss_weight > 0.0
            and step % args.point_chamfer_loss_interval == 0
        ):
            chamfer_pred = pred_params
            chamfer_images = images
            if (
                args.point_chamfer_batch_size > 0
                and args.point_chamfer_batch_size < images.shape[0]
            ):
                chamfer_pred = chamfer_pred[: args.point_chamfer_batch_size]
                chamfer_images = chamfer_images[: args.point_chamfer_batch_size]
            point_chamfer_kwargs = {
                "resolution": args.reconstruction_resolution,
                "fixed_range": train_config.fixed_range,
                "num_trajectories": args.reconstruction_num_trajectories,
                "num_steps": args.reconstruction_num_steps,
                "burn_in": args.reconstruction_burn_in,
                "seed": args.reconstruction_seed,
                "map_probability_mode": args.reconstruction_map_probability_mode,
                "num_target_points": args.point_chamfer_num_target_points,
                "max_pred_points": args.point_chamfer_num_pred_points,
                "target_seed": args.point_chamfer_seed + step,
            }
            if args.target_representation == "affine":
                point_chamfer_loss = point_chamfer_loss_affine(
                    chamfer_pred,
                    chamfer_images,
                    **point_chamfer_kwargs,
                )
            else:
                point_chamfer_loss = point_chamfer_loss_svd(
                    chamfer_pred,
                    chamfer_images,
                    **point_chamfer_kwargs,
                )
            point_chamfer_loss_item = point_chamfer_loss.item()
            loss = loss + args.point_chamfer_loss_weight * point_chamfer_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_eval = step == 1 or step % args.eval_interval == 0 or step == args.train_steps
        should_log_perf = (
            args.log_interval > 0
            and (step == 1 or step % args.log_interval == 0 or step == args.train_steps)
        )
        record = {"step": step}
        train_loss_item = None

        if should_eval or should_log_perf:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            interval_steps = step - last_perf_step
            interval_wall_sec = max(now - interval_started, 1e-12)
            avg_step_sec = interval_wall_sec / float(interval_steps)
            avg_data_wait_sec = interval_data_wait_sec / float(interval_steps)
            samples_per_sec = float(interval_samples) / interval_wall_sec
            train_loss_item = loss.item()
            record["train_loss"] = train_loss_item
            record["lr"] = current_lr
            if reconstruction_loss_item is not None:
                record["train_reconstruction_loss"] = reconstruction_loss_item
            if point_chamfer_loss_item is not None:
                record["train_point_chamfer_loss"] = point_chamfer_loss_item

            if should_log_perf:
                perf_record = {
                    "step": step,
                    "lr": current_lr,
                    "interval_steps": interval_steps,
                    "interval_wall_sec": interval_wall_sec,
                    "avg_step_sec": avg_step_sec,
                    "avg_data_wait_sec": avg_data_wait_sec,
                    "avg_compute_plus_transfer_sec": max(
                        avg_step_sec - avg_data_wait_sec,
                        0.0,
                    ),
                    "samples_per_sec": samples_per_sec,
                }
                if device.type == "cuda":
                    perf_record["cuda_max_memory_mb"] = (
                        torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
                    )
                perf_history.append(perf_record)
                print(
                    f"perf step={step:05d} source={train_data_source} "
                    f"lr={current_lr:.6g} "
                    f"avg_step_sec={avg_step_sec:.4f} "
                    f"avg_data_wait_sec={avg_data_wait_sec:.4f} "
                    f"samples_per_sec={samples_per_sec:.1f}",
                    flush=True,
                )
            interval_started = time.perf_counter()
            interval_data_wait_sec = 0.0
            interval_samples = 0
            last_perf_step = step

        if should_eval:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                target_representation=args.target_representation,
            )
            record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_w_fro": val_metrics["w_fro"],
                    "val_b_l2": val_metrics["b_l2"],
                }
            )
            if "fixed_point_l2" in val_metrics:
                record["val_fixed_point_l2"] = val_metrics["fixed_point_l2"]
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_record = {
                    "step": step,
                    "train_loss": train_loss_item,
                    "val_loss": val_metrics["loss"],
                    "val_w_fro": val_metrics["w_fro"],
                    "val_b_l2": val_metrics["b_l2"],
                }
                if "fixed_point_l2" in val_metrics:
                    best_val_record["val_fixed_point_l2"] = val_metrics["fixed_point_l2"]
                torch.save(model.state_dict(), output_dir / "best_model.pt")
            fixed_point_text = ""
            if "fixed_point_l2" in val_metrics:
                fixed_point_text = f" val_fp_l2={val_metrics['fixed_point_l2']:.6f}"
            print(
                f"step={step:05d} train_loss={train_loss_item:.6f} "
                f"lr={current_lr:.6g} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_w_fro={val_metrics['w_fro']:.6f} "
                f"val_b_l2={val_metrics['b_l2']:.6f}"
                f"{fixed_point_text}",
                flush=True,
            )
        if "train_loss" in record or "val_loss" in record:
            history.append(record)

    val_final = evaluate_model(
        model,
        val_loader,
        device=device,
        target_representation=args.target_representation,
    )
    test_final = evaluate_model(
        model,
        test_loader,
        device=device,
        target_representation=args.target_representation,
    )
    baseline = {
        "constant_scale_0.45_val": constant_baseline_metrics(
            val_dataset,
            target_representation=args.target_representation,
        ),
        "constant_scale_0.45_test": constant_baseline_metrics(
            test_dataset,
            target_representation=args.target_representation,
        ),
    }
    final_model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), final_model_path)

    best_model_path = output_dir / "best_model.pt"
    best_val_eval = None
    best_test_eval = None
    if best_val_record is not None and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_val_eval = evaluate_model(
            model,
            val_loader,
            device=device,
            target_representation=args.target_representation,
        )
        best_test_eval = evaluate_model(
            model,
            test_loader,
            device=device,
            target_representation=args.target_representation,
        )
        model.load_state_dict(torch.load(final_model_path, map_location=device))

    result = {
        "args": vars(args),
        "train_config": train_config.__dict__,
        "val": val_final,
        "test": test_final,
        "best_val_eval": best_val_eval,
        "best_test": best_test_eval,
        "baseline": baseline,
        "history": history,
        "perf_history": perf_history,
        "train_data_source": train_data_source,
        "cache_metadata": cache_metadata,
        "cache_metadata_history": cache_metadata_history,
        "cache_elapsed_sec": cache_elapsed_sec,
        "best_val": best_val_record,
        "best_model_path": str(best_model_path) if best_val_record is not None else None,
        "final_model_path": str(final_model_path),
        "gpu_monitor_path": str(gpu_monitor["path"]) if gpu_monitor is not None else None,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.plot:
        save_curves(history, output_dir)
        save_comparison_figure(
            model,
            val_dataset,
            config=train_config,
            output_dir=output_dir,
            device=device,
            num_samples=min(args.comparison_samples, args.val_samples),
            target_representation=args.target_representation,
        )
        save_point_cloud_figure(
            model,
            val_dataset,
            config=train_config,
            output_dir=output_dir,
            device=device,
            num_samples=min(args.comparison_samples, args.val_samples),
            target_representation=args.target_representation,
            max_points=args.point_cloud_max_points,
        )

    print(
        json.dumps(
            {
                "val": val_final,
                "test": test_final,
                "best_val_eval": best_val_eval,
                "best_test": best_test_eval,
                "baseline": baseline,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {output_dir}", flush=True)
    stop_gpu_monitor(gpu_monitor)


if __name__ == "__main__":
    main()
