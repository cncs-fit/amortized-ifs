"""Per-instance reconstruction optimization oracle for Phase 1.

The oracle optimizes IFS parameters for each input density map directly.  The
optimization objective uses reconstruction loss and optional image-sampled
Chamfer loss against the input image; ground-truth parameters are used only for
reporting synthetic-data metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.dataset import make_fixed_dataset_from_iterable_config
from data.renderer import render_density_from_affine_vector
from data.sampler import (
    affine_matrices_to_vector,
    affine_vector_to_matrices,
    params_to_affine,
    sample_ifs_parameters_batch,
)
from losses.hungarian import hungarian_metrics_affine
from losses.reconstruction import (
    differentiable_density_from_affine_vector,
    differentiable_points_from_affine_vector,
    target_density_for_reconstruction,
)
from scripts.analyze_checkpoint_errors import _analyze_one, predict_all
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
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-scale-range", type=float, nargs=2, default=None)
    parser.add_argument("--sample-fixed-point-range", type=float, default=None)
    parser.add_argument("--sample-label", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--init-mode",
        choices=("model", "target", "constant", "random", "mixed"),
        default="model",
    )
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--init-noise", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reconstruction-resolution", type=int, default=64)
    parser.add_argument("--reconstruction-num-trajectories", type=int, default=4)
    parser.add_argument("--reconstruction-num-steps", type=int, default=96)
    parser.add_argument("--reconstruction-burn-in", type=int, default=16)
    parser.add_argument("--reconstruction-smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--reconstruction-seed", type=int, default=12345)
    parser.add_argument("--eval-reconstruction-seed", type=int, default=None)
    parser.add_argument(
        "--reconstruction-map-probability-mode",
        choices=("uniform", "determinant"),
        default="uniform",
    )
    parser.add_argument("--reconstruction-match-render-config", action="store_true")
    parser.add_argument("--point-chamfer-loss-weight", type=float, default=0.0)
    parser.add_argument("--point-chamfer-num-pred-points", type=int, default=512)
    parser.add_argument("--point-chamfer-num-target-points", type=int, default=512)
    parser.add_argument("--point-chamfer-seed", type=int, default=24680)
    parser.add_argument("--spectral-upper-bound", type=float, default=0.75)
    parser.add_argument("--spectral-penalty-weight", type=float, default=0.1)
    parser.add_argument("--det-negative-penalty-weight", type=float, default=0.1)
    parser.add_argument("--translation-bound", type=float, default=1.5)
    parser.add_argument("--translation-penalty-weight", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--plot-samples", type=int, default=6)
    parser.add_argument("--point-cloud-max-points", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def apply_reconstruction_match_render_config(args: argparse.Namespace, config) -> None:
    """Use the dataset renderer settings for the differentiable recon renderer."""
    if not args.reconstruction_match_render_config:
        return
    args.reconstruction_resolution = int(config.resolution)
    args.reconstruction_num_trajectories = int(config.num_trajectories)
    args.reconstruction_num_steps = int(config.num_steps)
    args.reconstruction_burn_in = int(config.burn_in)
    args.reconstruction_smoothing_sigma = float(config.density_smoothing_sigma)
    args.reconstruction_map_probability_mode = "determinant"


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def reconstruction_losses(
    affine_params: torch.Tensor,
    images: torch.Tensor,
    *,
    resolution: int,
    fixed_range: tuple[float, float],
    num_trajectories: int,
    num_steps: int,
    burn_in: int,
    smoothing_sigma: float,
    seed: int,
    map_probability_mode: str,
) -> torch.Tensor:
    pred_density = differentiable_density_from_affine_vector(
        affine_params,
        resolution=resolution,
        fixed_range=fixed_range,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        burn_in=burn_in,
        smoothing_sigma=smoothing_sigma,
        seed=seed,
        map_probability_mode=map_probability_mode,
    )
    target_density = target_density_for_reconstruction(images, resolution=resolution)
    return (pred_density - target_density).square().sum(dim=(1, 2, 3))


def density_images_to_point_samples(
    images: torch.Tensor,
    *,
    resolution: int,
    fixed_range: tuple[float, float],
    num_points: int,
    seed: int,
) -> torch.Tensor:
    """Sample fixed point clouds from observed density images."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    density = target_density_for_reconstruction(images, resolution=resolution)
    batch_size = density.shape[0]
    flat_density = density.reshape(batch_size, -1).clamp_min(0.0)
    flat_density = flat_density / flat_density.sum(dim=1, keepdim=True).clamp_min(1e-12)
    generator = torch.Generator(device=images.device).manual_seed(int(seed))
    indices = torch.multinomial(flat_density, num_points, replacement=True, generator=generator)
    ix = torch.div(indices, resolution, rounding_mode="floor")
    iy = indices.remainder(resolution)
    low, high = fixed_range
    pixel_width = float(high - low) / float(resolution)
    x = float(low) + (ix.to(dtype=images.dtype) + 0.5) * pixel_width
    y = float(low) + (iy.to(dtype=images.dtype) + 0.5) * pixel_width
    return torch.stack([x, y], dim=-1)


def point_chamfer_losses(
    affine_params: torch.Tensor,
    target_points: torch.Tensor,
    *,
    num_trajectories: int,
    num_steps: int,
    burn_in: int,
    seed: int,
    map_probability_mode: str,
    max_pred_points: int,
) -> torch.Tensor:
    """Squared symmetric Chamfer loss against point samples from the input image."""
    pred_points = differentiable_points_from_affine_vector(
        affine_params,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        burn_in=burn_in,
        seed=seed,
        map_probability_mode=map_probability_mode,
    )
    if max_pred_points > 0 and pred_points.shape[1] > max_pred_points:
        generator = torch.Generator(device=pred_points.device).manual_seed(int(seed) + 4099)
        selected = torch.randperm(
            pred_points.shape[1],
            device=pred_points.device,
            generator=generator,
        )[:max_pred_points]
        pred_points = pred_points[:, selected]
    distances = torch.cdist(
        pred_points,
        target_points.to(device=pred_points.device, dtype=pred_points.dtype),
        p=2,
    )
    pred_to_target = distances.min(dim=2).values.square().mean(dim=1)
    target_to_pred = distances.min(dim=1).values.square().mean(dim=1)
    return 0.5 * (pred_to_target + target_to_pred)


def objective_losses(
    affine_params: torch.Tensor,
    images: torch.Tensor,
    *,
    args: argparse.Namespace,
    config,
    target_points: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    recon = reconstruction_losses(
        affine_params,
        images,
        resolution=args.reconstruction_resolution,
        fixed_range=config.fixed_range,
        num_trajectories=args.reconstruction_num_trajectories,
        num_steps=args.reconstruction_num_steps,
        burn_in=args.reconstruction_burn_in,
        smoothing_sigma=args.reconstruction_smoothing_sigma,
        seed=args.reconstruction_seed,
        map_probability_mode=args.reconstruction_map_probability_mode,
    )
    chamfer_weight = float(getattr(args, "point_chamfer_loss_weight", 0.0))
    if chamfer_weight <= 0.0:
        return recon, recon.new_zeros(recon.shape)
    if target_points is None:
        target_points = density_images_to_point_samples(
            images,
            resolution=args.reconstruction_resolution,
            fixed_range=config.fixed_range,
            num_points=int(getattr(args, "point_chamfer_num_target_points", 512)),
            seed=int(getattr(args, "point_chamfer_seed", args.reconstruction_seed)),
        )
    chamfer = point_chamfer_losses(
        affine_params,
        target_points,
        num_trajectories=args.reconstruction_num_trajectories,
        num_steps=args.reconstruction_num_steps,
        burn_in=args.reconstruction_burn_in,
        seed=args.reconstruction_seed,
        map_probability_mode=args.reconstruction_map_probability_mode,
        max_pred_points=int(getattr(args, "point_chamfer_num_pred_points", 512)),
    )
    return recon, chamfer_weight * chamfer


def structural_penalties(
    affine_params: torch.Tensor,
    *,
    spectral_upper_bound: float,
    spectral_penalty_weight: float,
    det_negative_penalty_weight: float,
    translation_bound: float,
    translation_penalty_weight: float,
) -> torch.Tensor:
    w, b = affine_vector_to_matrices(affine_params)
    penalties = affine_params.new_zeros(affine_params.shape[0])
    if spectral_penalty_weight > 0.0:
        singular = torch.linalg.svdvals(w)
        spectral = torch.relu(singular - float(spectral_upper_bound)).square().mean(dim=(1, 2))
        penalties = penalties + float(spectral_penalty_weight) * spectral
    if det_negative_penalty_weight > 0.0:
        det = torch.linalg.det(w)
        det_penalty = torch.relu(-det).square().mean(dim=1)
        penalties = penalties + float(det_negative_penalty_weight) * det_penalty
    if translation_penalty_weight > 0.0:
        translation = torch.relu(b.abs() - float(translation_bound)).square().mean(dim=(1, 2))
        penalties = penalties + float(translation_penalty_weight) * translation
    return penalties


def _constant_init(batch_size: int, num_transforms: int) -> torch.Tensor:
    params = torch.zeros(batch_size, num_transforms, 6)
    params[:, :, 0] = 0.45
    params[:, :, 3] = 0.45
    return params


def _random_init(
    batch_size: int,
    num_transforms: int,
    *,
    phase: str,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    params = sample_ifs_parameters_batch(
        batch_size,
        num_transforms,
        phase=phase,
        generator=generator,
        device=torch.device("cpu"),
    )
    w, b = params_to_affine(params)
    return affine_matrices_to_vector(w, b)


def make_initial_restarts(
    *,
    model_init: torch.Tensor | None = None,
    target_init: torch.Tensor | None = None,
    batch_size: int,
    num_transforms: int,
    phase: str,
    init_mode: str,
    restarts: int,
    init_noise: float,
    seed: int,
) -> torch.Tensor:
    if restarts <= 0:
        raise ValueError("restarts must be positive")
    if init_mode == "model" and model_init is None:
        raise ValueError("model init requires a direct-affine checkpoint")
    if init_mode == "target" and target_init is None:
        raise ValueError("target init requires target parameters")

    restart_values = []
    for restart_idx in range(restarts):
        use_model = init_mode == "model" or (init_mode == "mixed" and restart_idx == 0)
        if use_model and model_init is not None:
            init = model_init.clone()
        elif init_mode == "target" and target_init is not None:
            init = target_init.clone()
        elif init_mode == "constant" or (init_mode == "mixed" and model_init is None and restart_idx == 0):
            init = _constant_init(batch_size, num_transforms)
        else:
            init = _random_init(
                batch_size,
                num_transforms,
                phase=phase,
                seed=seed + 10_003 * restart_idx,
            )
        if init_noise > 0.0 and (restart_idx > 0 or init_mode != "random"):
            generator = torch.Generator(device="cpu").manual_seed(seed + 20_003 * restart_idx)
            init = init + torch.randn(init.shape, generator=generator) * float(init_noise)
        restart_values.append(init)
    return torch.stack(restart_values, dim=1)


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


def write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def optimize_batch(
    images: torch.Tensor,
    target_params: torch.Tensor,
    init_restarts: torch.Tensor,
    *,
    args: argparse.Namespace,
    config,
    device: torch.device,
    batch_start_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    batch_size, restarts, num_transforms, _ = init_restarts.shape
    images = images.to(device)
    target_images = images[:, None, :, :, :].expand(-1, restarts, -1, -1, -1)
    target_images = target_images.reshape(batch_size * restarts, *images.shape[1:])
    target_points = None
    if float(getattr(args, "point_chamfer_loss_weight", 0.0)) > 0.0:
        target_points = density_images_to_point_samples(
            target_images,
            resolution=args.reconstruction_resolution,
            fixed_range=config.fixed_range,
            num_points=int(getattr(args, "point_chamfer_num_target_points", 512)),
            seed=int(getattr(args, "point_chamfer_seed", args.reconstruction_seed)) + int(batch_start_index),
        )
    params = torch.nn.Parameter(init_restarts.reshape(batch_size * restarts, num_transforms, 6).to(device))
    optimizer = torch.optim.AdamW(
        [params],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    with torch.no_grad():
        initial_losses = reconstruction_losses(
            params,
            target_images,
            resolution=args.reconstruction_resolution,
            fixed_range=config.fixed_range,
            num_trajectories=args.reconstruction_num_trajectories,
            num_steps=args.reconstruction_num_steps,
            burn_in=args.reconstruction_burn_in,
            smoothing_sigma=args.reconstruction_smoothing_sigma,
            seed=args.reconstruction_seed,
            map_probability_mode=args.reconstruction_map_probability_mode,
        ).reshape(batch_size, restarts)

    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        recon, point_chamfer = objective_losses(
            params,
            target_images,
            args=args,
            config=config,
            target_points=target_points,
        )
        penalty = structural_penalties(
            params,
            spectral_upper_bound=args.spectral_upper_bound,
            spectral_penalty_weight=args.spectral_penalty_weight,
            det_negative_penalty_weight=args.det_negative_penalty_weight,
            translation_bound=args.translation_bound,
            translation_penalty_weight=args.translation_penalty_weight,
        )
        objective = (recon + point_chamfer + penalty).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([params], args.grad_clip)
        optimizer.step()
        if args.log_interval > 0 and (step == 1 or step % args.log_interval == 0 or step == args.steps):
            history.append(
                {
                    "batch_start_index": int(batch_start_index),
                    "step": int(step),
                    "objective": _to_float(objective),
                    "reconstruction": _to_float(recon.mean()),
                    "point_chamfer": _to_float(point_chamfer.mean()),
                    "penalty": _to_float(penalty.mean()),
                    "elapsed_sec": time.perf_counter() - started,
                }
            )

    with torch.no_grad():
        final_recon, final_point_chamfer = objective_losses(
            params,
            target_images,
            args=args,
            config=config,
            target_points=target_points,
        )
        final_penalty = structural_penalties(
            params,
            spectral_upper_bound=args.spectral_upper_bound,
            spectral_penalty_weight=args.spectral_penalty_weight,
            det_negative_penalty_weight=args.det_negative_penalty_weight,
            translation_bound=args.translation_bound,
            translation_penalty_weight=args.translation_penalty_weight,
        )
        final_losses = final_recon.reshape(batch_size, restarts)
        final_selection_losses = (final_recon + final_point_chamfer + final_penalty).reshape(
            batch_size, restarts
        )
        final_params = params.detach().reshape(batch_size, restarts, num_transforms, 6).cpu()
        best_restart = final_selection_losses.argmin(dim=1).detach().cpu()
        selected = final_params[torch.arange(batch_size), best_restart]
    return (
        selected,
        initial_losses.detach().cpu(),
        final_losses.detach().cpu(),
        final_selection_losses.detach().cpu(),
        history,
    )


def _recon_loss_values(
    params: torch.Tensor,
    images: torch.Tensor,
    *,
    args,
    config,
    device,
    seed: int,
) -> list[float]:
    losses = []
    for start in range(0, params.shape[0], max(1, args.batch_size)):
        end = min(params.shape[0], start + max(1, args.batch_size))
        loss = reconstruction_losses(
            params[start:end].to(device),
            images[start:end].to(device),
            resolution=args.reconstruction_resolution,
            fixed_range=config.fixed_range,
            num_trajectories=args.reconstruction_num_trajectories,
            num_steps=args.reconstruction_num_steps,
            burn_in=args.reconstruction_burn_in,
            smoothing_sigma=args.reconstruction_smoothing_sigma,
            seed=seed,
            map_probability_mode=args.reconstruction_map_probability_mode,
        )
        losses.extend(loss.detach().cpu().tolist())
    return losses


def _mean_recon_loss(
    params: torch.Tensor,
    images: torch.Tensor,
    *,
    args,
    config,
    device,
    seed: int,
) -> float:
    losses = _recon_loss_values(params, images, args=args, config=config, device=device, seed=seed)
    return float(torch.tensor(losses, dtype=torch.float32).mean().item())


def _save_loss_curve(history: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history:
        return
    grouped: dict[int, dict[str, list[float]]] = {}
    for record in history:
        bucket = grouped.setdefault(
            int(record["step"]),
            {"objective": [], "reconstruction": [], "point_chamfer": []},
        )
        bucket["objective"].append(float(record["objective"]))
        bucket["reconstruction"].append(float(record["reconstruction"]))
        bucket["point_chamfer"].append(float(record.get("point_chamfer", 0.0)))
    steps = sorted(grouped)
    recon = [
        sum(grouped[step]["reconstruction"]) / len(grouped[step]["reconstruction"])
        for step in steps
    ]
    objective = [
        sum(grouped[step]["objective"]) / len(grouped[step]["objective"])
        for step in steps
    ]
    point_chamfer = [
        sum(grouped[step]["point_chamfer"]) / len(grouped[step]["point_chamfer"])
        for step in steps
    ]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(steps, objective, label="objective")
    ax.plot(steps, recon, label="reconstruction")
    if any(value != 0.0 for value in point_chamfer):
        ax.plot(steps, point_chamfer, label="point chamfer")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_loss_curve.png", dpi=360)
    fig.savefig(output_dir / "oracle_loss_curve.pdf")
    plt.close(fig)


def _filter_points(points: torch.Tensor, *, fixed_range, max_points: int, generator: torch.Generator) -> torch.Tensor:
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


def save_oracle_figure(
    images: torch.Tensor,
    init_params: torch.Tensor,
    oracle_params: torch.Tensor,
    per_sample: list[dict],
    *,
    config,
    args,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if args.plot_samples <= 0:
        return
    selected = sorted(per_sample, key=lambda record: record["oracle_recon_loss"])[: args.plot_samples]
    if not selected:
        return
    fig, axes = plt.subplots(4, len(selected), figsize=(2.1 * len(selected), 8.2), squeeze=False)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 151_000)
    for col, record in enumerate(selected):
        idx = int(record["index"])
        _, init_density = render_density_from_affine_vector(
            init_params[idx],
            resolution=config.resolution,
            fixed_range=config.fixed_range,
            num_trajectories=config.num_trajectories,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            density_smoothing_sigma=config.density_smoothing_sigma,
            generator=generator,
        )
        _, oracle_density = render_density_from_affine_vector(
            oracle_params[idx],
            resolution=config.resolution,
            fixed_range=config.fixed_range,
            num_trajectories=config.num_trajectories,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            density_smoothing_sigma=config.density_smoothing_sigma,
            generator=generator,
        )
        panels = (
            (images[idx, 0], "input"),
            (init_density[0], f"init\nW={record['init_w_fro']:.3f}"),
            (oracle_density[0], f"oracle\nW={record['oracle_w_fro']:.3f}"),
            ((images[idx, 0] - oracle_density[0]).abs(), f"abs diff\nrec={record['oracle_recon_loss']:.4f}"),
        )
        for row, (image, title) in enumerate(panels):
            axes[row, col].imshow(image, cmap="magma", origin="lower")
            axes[row, col].set_title(f"{title}\nidx={idx}", fontsize=7)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_density_examples.png", dpi=360)
    fig.savefig(output_dir / "oracle_density_examples.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("steps must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.restarts <= 0:
        raise ValueError("restarts must be positive")

    run_dir = Path(args.run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    args_payload = result["args"]
    if args_payload["target_representation"] != "affine":
        raise ValueError("oracle currently expects a direct-affine checkpoint")
    config = _config_from_payload(result["train_config"])
    if args.sample_scale_range is not None or args.sample_fixed_point_range is not None:
        config = replace(
            config,
            scale_range=(
                tuple(args.sample_scale_range)
                if args.sample_scale_range is not None
                else config.scale_range
            ),
            fixed_point_range=(
                float(args.sample_fixed_point_range)
                if args.sample_fixed_point_range is not None
                else config.fixed_point_range
            ),
        )
    apply_reconstruction_match_render_config(args, config)
    seed_key = "val_seed" if args.split == "val" else "test_seed"
    seed = int(args_payload[seed_key] if args.seed is None else args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    noise_tag = "" if args.init_noise == 0.0 else f"_noise{args.init_noise:g}"
    if args.reconstruction_match_render_config:
        recon_tag = "_matchrender"
    elif args.reconstruction_map_probability_mode == "determinant":
        recon_tag = "_probdet"
    else:
        recon_tag = ""
    if args.output_dir is None:
        output_dir = (
            ROOT
            / "outputs"
            / "oracle"
            / f"{run_dir.parent.name}_{run_dir.name}_{args.split}_n{args.num_samples}_"
            f"{args.init_mode}_r{args.restarts}_s{args.steps}{noise_tag}{recon_tag}"
        )
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    dataset = make_fixed_dataset_from_iterable_config(config, num_samples=args.num_samples, seed=seed)
    images, target_params = dataset.tensors
    dataset_elapsed = time.perf_counter() - started

    model_predictions = None
    if args.init_mode in {"model", "mixed"}:
        model = _build_model(args_payload, device=device)
        checkpoint_paths = _checkpoint_paths(result, run_dir=run_dir, checkpoint=args.checkpoint)
        checkpoint_path = checkpoint_paths[args.checkpoint]
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model_predictions = predict_all(
            model,
            dataset,
            batch_size=int(args_payload["eval_batch_size"]),
            device=device,
        )

    selected_batches = []
    init_first_batches = []
    all_initial_losses = []
    all_final_losses = []
    all_selection_losses = []
    histories = []
    opt_started = time.perf_counter()
    num_transforms = int(args_payload["num_transforms"])
    for start in range(0, args.num_samples, args.batch_size):
        end = min(args.num_samples, start + args.batch_size)
        model_init = None if model_predictions is None else model_predictions[start:end]
        target_w, target_b = params_to_affine(target_params[start:end])
        target_init = affine_matrices_to_vector(target_w, target_b)
        init_restarts = make_initial_restarts(
            model_init=model_init,
            target_init=target_init,
            batch_size=end - start,
            num_transforms=num_transforms,
            phase=config.phase,
            init_mode=args.init_mode,
            restarts=args.restarts,
            init_noise=args.init_noise,
            seed=seed + 31_000 + start,
        )
        selected, initial_losses, final_losses, selection_losses, history = optimize_batch(
            images[start:end],
            target_params[start:end],
            init_restarts,
            args=args,
            config=config,
            device=device,
            batch_start_index=start,
        )
        selected_batches.append(selected)
        init_first_batches.append(init_restarts[:, 0])
        all_initial_losses.append(initial_losses)
        all_final_losses.append(final_losses)
        all_selection_losses.append(selection_losses)
        histories.extend(history)
        selection_best = selection_losses.argmin(dim=1)
        selected_recon = final_losses[torch.arange(final_losses.shape[0]), selection_best]
        selected_objective = selection_losses[torch.arange(selection_losses.shape[0]), selection_best]
        print(
            (
                f"batch {start}:{end} init_recon={initial_losses[:, 0].mean().item():.6f} "
                f"selected_recon={selected_recon.mean().item():.6f} "
                f"selected_objective={selected_objective.mean().item():.6f}"
            ),
            flush=True,
        )
    optimize_elapsed = time.perf_counter() - opt_started

    oracle_params = torch.cat(selected_batches, dim=0)
    init_params = torch.cat(init_first_batches, dim=0)
    initial_losses = torch.cat(all_initial_losses, dim=0)
    final_losses = torch.cat(all_final_losses, dim=0)
    selection_losses = torch.cat(all_selection_losses, dim=0)
    best_restart = selection_losses.argmin(dim=1)
    best_final_recon = final_losses[torch.arange(final_losses.shape[0]), best_restart]
    best_final_objective = selection_losses[torch.arange(selection_losses.shape[0]), best_restart]
    eval_reconstruction_seed = (
        int(args.eval_reconstruction_seed)
        if args.eval_reconstruction_seed is not None
        else int(args.reconstruction_seed) + 9_973
    )

    target_w, target_b = params_to_affine(target_params)
    target_affine = affine_matrices_to_vector(target_w, target_b)
    init_metrics = hungarian_metrics_affine(init_params, target_params)
    oracle_metrics = hungarian_metrics_affine(oracle_params, target_params)
    target_metrics = hungarian_metrics_affine(target_affine, target_params)
    init_recon = _mean_recon_loss(
        init_params,
        images,
        args=args,
        config=config,
        device=device,
        seed=args.reconstruction_seed,
    )
    oracle_recon = _mean_recon_loss(
        oracle_params,
        images,
        args=args,
        config=config,
        device=device,
        seed=args.reconstruction_seed,
    )
    init_holdout_losses = _recon_loss_values(
        init_params,
        images,
        args=args,
        config=config,
        device=device,
        seed=eval_reconstruction_seed,
    )
    target_recon = _mean_recon_loss(
        target_affine,
        images,
        args=args,
        config=config,
        device=device,
        seed=args.reconstruction_seed,
    )
    target_holdout_losses = _recon_loss_values(
        target_affine,
        images,
        args=args,
        config=config,
        device=device,
        seed=eval_reconstruction_seed,
    )
    oracle_holdout_losses = _recon_loss_values(
        oracle_params,
        images,
        args=args,
        config=config,
        device=device,
        seed=eval_reconstruction_seed,
    )

    per_sample = []
    for idx in range(args.num_samples):
        init_record = _analyze_one(init_params[idx], target_params[idx], idx)
        oracle_record = _analyze_one(oracle_params[idx], target_params[idx], idx)
        per_sample.append(
            {
                "index": int(idx),
                "best_restart": int(best_restart[idx].item()),
                "init_recon_loss": float(initial_losses[idx, 0].item()),
                "oracle_recon_loss": float(best_final_recon[idx].item()),
                "oracle_selection_loss": float(best_final_objective[idx].item()),
                "init_holdout_recon_loss": float(init_holdout_losses[idx]),
                "oracle_holdout_recon_loss": float(oracle_holdout_losses[idx]),
                "target_holdout_recon_loss": float(target_holdout_losses[idx]),
                "init_w_fro": init_record["w_fro_mean"],
                "oracle_w_fro": oracle_record["w_fro_mean"],
                "init_b_l2": init_record["b_l2_mean"],
                "oracle_b_l2": oracle_record["b_l2_mean"],
                "init_fixed_point_l2": init_record["fixed_point_l2_mean"],
                "oracle_fixed_point_l2": oracle_record["fixed_point_l2_mean"],
                "init_param_loss": init_record["loss"],
                "oracle_param_loss": oracle_record["loss"],
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": seed,
        "init_mode": args.init_mode,
        "restarts": args.restarts,
        "steps": args.steps,
        "lr": args.lr,
        "dataset_elapsed_sec": dataset_elapsed,
        "optimize_elapsed_sec": optimize_elapsed,
        "seconds_per_sample": optimize_elapsed / max(1, args.num_samples),
        "reconstruction_config": {
            "resolution": args.reconstruction_resolution,
            "num_trajectories": args.reconstruction_num_trajectories,
            "num_steps": args.reconstruction_num_steps,
            "burn_in": args.reconstruction_burn_in,
            "smoothing_sigma": args.reconstruction_smoothing_sigma,
            "seed": args.reconstruction_seed,
            "eval_seed": eval_reconstruction_seed,
            "map_probability_mode": args.reconstruction_map_probability_mode,
            "match_render_config": bool(args.reconstruction_match_render_config),
        },
        "sample_config": {
            "label": args.sample_label,
            "scale_range": list(config.scale_range),
            "fixed_point_range": float(config.fixed_point_range),
            "fixed_range": list(config.fixed_range),
        },
        "objective_config": {
            "point_chamfer_loss_weight": float(args.point_chamfer_loss_weight),
            "point_chamfer_num_pred_points": int(args.point_chamfer_num_pred_points),
            "point_chamfer_num_target_points": int(args.point_chamfer_num_target_points),
            "point_chamfer_seed": int(args.point_chamfer_seed),
        },
        "initial": {
            "reconstruction_loss": init_recon,
            "holdout_reconstruction_loss": float(
                torch.tensor(init_holdout_losses, dtype=torch.float32).mean().item()
            ),
            **init_metrics,
        },
        "oracle": {
            "reconstruction_loss": oracle_recon,
            "holdout_reconstruction_loss": float(
                torch.tensor(oracle_holdout_losses, dtype=torch.float32).mean().item()
            ),
            **oracle_metrics,
        },
        "target": {
            "reconstruction_loss": target_recon,
            "holdout_reconstruction_loss": float(
                torch.tensor(target_holdout_losses, dtype=torch.float32).mean().item()
            ),
            **target_metrics,
        },
        "per_sample_summaries": {
            "init_recon_loss": _summarize([record["init_recon_loss"] for record in per_sample]),
            "oracle_recon_loss": _summarize([record["oracle_recon_loss"] for record in per_sample]),
            "oracle_selection_loss": _summarize(
                [record["oracle_selection_loss"] for record in per_sample]
            ),
            "init_holdout_recon_loss": _summarize(
                [record["init_holdout_recon_loss"] for record in per_sample]
            ),
            "oracle_holdout_recon_loss": _summarize(
                [record["oracle_holdout_recon_loss"] for record in per_sample]
            ),
            "target_holdout_recon_loss": _summarize(
                [record["target_holdout_recon_loss"] for record in per_sample]
            ),
            "init_w_fro": _summarize([record["init_w_fro"] for record in per_sample]),
            "oracle_w_fro": _summarize([record["oracle_w_fro"] for record in per_sample]),
        },
        "mean_recon_improvement": float(
            torch.tensor([r["init_recon_loss"] - r["oracle_recon_loss"] for r in per_sample]).mean().item()
        ),
        "mean_w_change": float(
            torch.tensor([r["oracle_w_fro"] - r["init_w_fro"] for r in per_sample]).mean().item()
        ),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "optimization_history.json").write_text(
        json.dumps(histories, indent=2),
        encoding="utf-8",
    )
    write_csv(per_sample, output_dir / "per_sample.csv")
    torch.save(
        {
            "oracle_params": oracle_params,
            "init_params": init_params,
            "target_params": target_params,
            "images": images,
            "summary": summary,
        },
        output_dir / "oracle_outputs.pt",
    )
    _save_loss_curve(histories, output_dir)
    save_oracle_figure(
        images,
        init_params,
        oracle_params,
        per_sample,
        config=config,
        args=args,
        output_dir=output_dir,
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
