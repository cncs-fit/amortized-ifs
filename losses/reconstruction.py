"""Differentiable reconstruction losses for predicted affine IFS maps."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from data.renderer import smooth_density_maps
from data.sampler import (
    affine_matrices_to_vector,
    affine_vector_probabilities,
    affine_vector_to_matrices,
    params_to_affine,
)


def soft_points_to_density_maps(
    points: torch.Tensor,
    *,
    resolution: int,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    smoothing_sigma: float = 0.0,
) -> torch.Tensor:
    """Bilinearly splat point clouds to normalized density maps."""
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [batch, num_points, 2]")
    if resolution <= 1:
        raise ValueError("resolution must be greater than one")

    batch_size, _, _ = points.shape
    low, high = fixed_range
    # Match the hard histogram renderer's bin convention while keeping a
    # differentiable triangular kernel around each bin center.
    scale = float(resolution) / float(high - low)
    x_raw = points[..., 0]
    y_raw = points[..., 1]
    x = (x_raw - float(low)) * scale - 0.5
    y = (y_raw - float(low)) * scale - 0.5
    valid = (x_raw > float(low)) & (x_raw <= float(high)) & (y_raw > float(low)) & (y_raw <= float(high))

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = x - x0.to(dtype=x.dtype)
    wy1 = y - y0.to(dtype=y.dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    flat = points.new_zeros(batch_size * resolution * resolution)
    batch_offsets = (
        torch.arange(batch_size, device=points.device, dtype=torch.long).view(batch_size, 1)
        * resolution
        * resolution
    )

    for xi, yi, weight in (
        (x0, y0, wx0 * wy0),
        (x1, y0, wx1 * wy0),
        (x0, y1, wx0 * wy1),
        (x1, y1, wx1 * wy1),
    ):
        corner_valid = valid & (xi >= 0) & (xi < resolution) & (yi >= 0) & (yi < resolution)
        linear = batch_offsets + xi.clamp(0, resolution - 1) * resolution + yi.clamp(0, resolution - 1)
        flat.scatter_add_(0, linear.reshape(-1), (weight * corner_valid.to(weight.dtype)).reshape(-1))

    density = flat.reshape(batch_size, 1, resolution, resolution)
    density = density / density.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return smooth_density_maps(density, sigma=smoothing_sigma)


def differentiable_points_from_affine_vector(
    params: torch.Tensor,
    *,
    num_trajectories: int = 4,
    num_steps: int = 96,
    burn_in: int = 16,
    initial_range: float = 1.0,
    seed: int = 12345,
    map_probability_mode: str = "uniform",
) -> torch.Tensor:
    """Return a fixed differentiable point trace for predicted direct-affine IFS maps."""
    if params.ndim != 3 or params.shape[-1] != 6:
        raise ValueError("params must have shape [batch, num_transforms, 6]")
    if burn_in >= num_steps:
        raise ValueError("burn_in must be smaller than num_steps")
    if num_trajectories <= 0 or num_steps <= 0:
        raise ValueError("num_trajectories and num_steps must be positive")
    if map_probability_mode not in {"uniform", "determinant"}:
        raise ValueError("map_probability_mode must be 'uniform' or 'determinant'")

    batch_size, num_transforms, _ = params.shape
    device = params.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    w, b = affine_vector_to_matrices(params)
    current = torch.empty(
        batch_size,
        num_trajectories,
        2,
        device=device,
        dtype=params.dtype,
    ).uniform_(-float(initial_range), float(initial_range), generator=generator)
    if map_probability_mode == "uniform":
        ids = torch.randint(
            num_transforms,
            (num_trajectories, num_steps),
            device=device,
            generator=generator,
        )
    else:
        probs = affine_vector_probabilities(params)
        cdf = probs.cumsum(dim=-1)
        cdf[..., -1] = 1.0
        uniforms = torch.rand(
            batch_size,
            num_trajectories,
            num_steps,
            device=device,
            generator=generator,
        )
        ids = (uniforms[..., None] > cdf[:, None, None, :]).sum(dim=-1).clamp(max=num_transforms - 1)
    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1)

    kept = []
    for step in range(num_steps):
        if ids.ndim == 2:
            step_ids = ids[:, step].view(1, num_trajectories).expand(batch_size, num_trajectories)
        else:
            step_ids = ids[:, :, step]
        selected_w = w[batch_index, step_ids]
        selected_b = b[batch_index, step_ids]
        current = torch.matmul(current.unsqueeze(-2), selected_w).squeeze(-2) + selected_b
        if step >= burn_in:
            kept.append(current)
    return torch.cat(kept, dim=1)


def differentiable_density_from_affine_vector(
    params: torch.Tensor,
    *,
    resolution: int,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    num_trajectories: int = 4,
    num_steps: int = 96,
    burn_in: int = 16,
    smoothing_sigma: float = 0.0,
    initial_range: float = 1.0,
    seed: int = 12345,
    map_probability_mode: str = "uniform",
) -> torch.Tensor:
    """Render predicted direct-affine IFS maps with a fixed differentiable point trace."""
    points = differentiable_points_from_affine_vector(
        params,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        burn_in=burn_in,
        initial_range=initial_range,
        seed=seed,
        map_probability_mode=map_probability_mode,
    )
    return soft_points_to_density_maps(
        points,
        resolution=resolution,
        fixed_range=fixed_range,
        smoothing_sigma=smoothing_sigma,
    )


def target_density_for_reconstruction(images: torch.Tensor, *, resolution: int) -> torch.Tensor:
    """Resize and renormalize input density maps for reconstruction loss."""
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must have shape [batch, 1, height, width]")
    if images.shape[-2:] != (resolution, resolution):
        images = F.interpolate(images, size=(resolution, resolution), mode="area")
    return images / images.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)


def density_reconstruction_loss_affine(
    pred_affine: torch.Tensor,
    target_images: torch.Tensor,
    *,
    resolution: int = 64,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    num_trajectories: int = 4,
    num_steps: int = 96,
    burn_in: int = 16,
    smoothing_sigma: float = 1.0,
    seed: int = 12345,
    map_probability_mode: str = "uniform",
) -> torch.Tensor:
    """Return density reconstruction loss from direct-affine predictions."""
    pred_density = differentiable_density_from_affine_vector(
        pred_affine,
        resolution=resolution,
        fixed_range=fixed_range,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        burn_in=burn_in,
        smoothing_sigma=smoothing_sigma,
        seed=seed,
        map_probability_mode=map_probability_mode,
    )
    target_density = target_density_for_reconstruction(target_images, resolution=resolution)
    return (pred_density - target_density).square().sum(dim=(1, 2, 3)).mean()


def density_images_to_point_samples(
    images: torch.Tensor,
    *,
    resolution: int,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    num_points: int = 512,
    seed: int = 24680,
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


def point_chamfer_losses_affine(
    pred_affine: torch.Tensor,
    target_points: torch.Tensor,
    *,
    num_trajectories: int = 4,
    num_steps: int = 96,
    burn_in: int = 16,
    seed: int = 12345,
    map_probability_mode: str = "uniform",
    max_pred_points: int = 512,
) -> torch.Tensor:
    """Return per-sample squared symmetric Chamfer losses."""
    pred_points = differentiable_points_from_affine_vector(
        pred_affine,
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


def point_chamfer_loss_affine(
    pred_affine: torch.Tensor,
    target_images: torch.Tensor,
    *,
    resolution: int = 64,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    num_trajectories: int = 4,
    num_steps: int = 96,
    burn_in: int = 16,
    seed: int = 12345,
    map_probability_mode: str = "uniform",
    num_target_points: int = 512,
    max_pred_points: int = 512,
    target_seed: int = 24680,
) -> torch.Tensor:
    """Return mean squared symmetric Chamfer loss against input-image samples."""
    target_points = density_images_to_point_samples(
        target_images,
        resolution=resolution,
        fixed_range=fixed_range,
        num_points=num_target_points,
        seed=target_seed,
    )
    losses = point_chamfer_losses_affine(
        pred_affine,
        target_points,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        burn_in=burn_in,
        seed=seed,
        map_probability_mode=map_probability_mode,
        max_pred_points=max_pred_points,
    )
    return losses.mean()


def density_reconstruction_loss_svd(
    pred_params: torch.Tensor,
    target_images: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Return density reconstruction loss from SVD-format predictions."""
    w, b = params_to_affine(pred_params)
    pred_affine = affine_matrices_to_vector(w, b)
    return density_reconstruction_loss_affine(pred_affine, target_images, **kwargs)


def point_chamfer_loss_svd(
    pred_params: torch.Tensor,
    target_images: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Return point Chamfer loss from SVD-format predictions."""
    w, b = params_to_affine(pred_params)
    pred_affine = affine_matrices_to_vector(w, b)
    return point_chamfer_loss_affine(pred_affine, target_images, **kwargs)
