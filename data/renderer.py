"""Point-cloud rendering utilities for IFS density maps."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from data.sampler import (
    iterate_affine_vector_points,
    iterate_ifs_points,
    iterate_ifs_points_batch,
)


def smooth_density_map(density: torch.Tensor, *, sigma: float = 0.0) -> torch.Tensor:
    """Apply Gaussian smoothing to a normalized density map and renormalize."""
    if sigma <= 0.0:
        return density
    if density.ndim != 3 or density.shape[0] != 1:
        raise ValueError("density must have shape [1, H, W]")

    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=density.device, dtype=density.dtype)
    kernel = torch.exp(-0.5 * (coords / float(sigma)).square())
    kernel = kernel / kernel.sum()

    x = density.unsqueeze(0)
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    x = F.conv2d(x, kernel_x, padding=(0, radius))
    x = F.conv2d(x, kernel_y, padding=(radius, 0))
    smoothed = x.squeeze(0)
    total = smoothed.sum()
    if total > 0:
        smoothed = smoothed / total
    return smoothed


def smooth_density_maps(densities: torch.Tensor, *, sigma: float = 0.0) -> torch.Tensor:
    """Apply Gaussian smoothing to a batch of normalized density maps."""
    if sigma <= 0.0:
        return densities
    if densities.ndim != 4 or densities.shape[1] != 1:
        raise ValueError("densities must have shape [B, 1, H, W]")

    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=densities.device, dtype=densities.dtype)
    kernel = torch.exp(-0.5 * (coords / float(sigma)).square())
    kernel = kernel / kernel.sum()

    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    x = F.conv2d(densities, kernel_x, padding=(0, radius))
    x = F.conv2d(x, kernel_y, padding=(radius, 0))
    totals = x.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return x / totals


def points_to_density_map(
    points: torch.Tensor,
    *,
    resolution: int = 32,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    smoothing_sigma: float = 0.0,
) -> torch.Tensor:
    """Convert points to a normalized 2D histogram with shape ``[1, H, W]``."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [num_points, 2]")

    low, high = fixed_range
    bins = torch.linspace(low, high, resolution + 1, device=points.device)
    ix = torch.bucketize(points[:, 0].contiguous(), bins) - 1
    iy = torch.bucketize(points[:, 1].contiguous(), bins) - 1
    valid = (ix >= 0) & (ix < resolution) & (iy >= 0) & (iy < resolution)
    linear_idx = ix[valid] * resolution + iy[valid]
    density = torch.bincount(
        linear_idx,
        minlength=resolution * resolution,
    ).reshape(resolution, resolution).float()

    total = density.sum()
    if total > 0:
        density = density / total
    return smooth_density_map(density.unsqueeze(0), sigma=smoothing_sigma)


def points_to_density_maps(
    points: torch.Tensor,
    *,
    resolution: int = 32,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    smoothing_sigma: float = 0.0,
) -> torch.Tensor:
    """Convert batched point clouds to normalized density maps ``[B, 1, H, W]``."""
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [batch, num_points, 2]")

    batch_size = points.shape[0]
    low, high = fixed_range
    bins = torch.linspace(low, high, resolution + 1, device=points.device)
    ix = torch.bucketize(points[..., 0].contiguous(), bins) - 1
    iy = torch.bucketize(points[..., 1].contiguous(), bins) - 1
    valid = (ix >= 0) & (ix < resolution) & (iy >= 0) & (iy < resolution)
    batch_offsets = (
        torch.arange(batch_size, device=points.device).view(batch_size, 1)
        * resolution
        * resolution
    )
    linear_idx = batch_offsets + ix * resolution + iy
    flat_idx = linear_idx[valid]
    density = torch.bincount(
        flat_idx,
        minlength=batch_size * resolution * resolution,
    ).reshape(batch_size, resolution, resolution).float()

    totals = density.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    density = density / totals
    return smooth_density_maps(density.unsqueeze(1), sigma=smoothing_sigma)


def is_valid_sample(
    points: torch.Tensor,
    density: torch.Tensor,
    *,
    resolution: int,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    min_filling_rate: float = 0.005,
    max_out_of_range_ratio: float = 0.20,
    max_pixel_mass: float = 0.25,
) -> bool:
    """Reject numerically invalid or degenerate rendered samples."""
    if torch.isnan(points).any() or torch.isinf(points).any():
        return False
    low, high = fixed_range
    out_of_range = ((points < low) | (points > high)).any(dim=1).float().mean().item()
    if out_of_range > max_out_of_range_ratio:
        return False
    filling_rate = (density > 0).sum().item() / float(resolution * resolution)
    if filling_rate < min_filling_rate:
        return False
    if density.max().item() > max_pixel_mass:
        return False
    return True


def valid_sample_mask(
    points: torch.Tensor,
    density: torch.Tensor,
    *,
    resolution: int,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    min_filling_rate: float = 0.005,
    max_out_of_range_ratio: float = 0.20,
    max_pixel_mass: float = 0.25,
) -> torch.Tensor:
    """Return a validity mask for batched rendered samples."""
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [batch, num_points, 2]")
    if density.ndim != 4 or density.shape[1] != 1:
        raise ValueError("density must have shape [batch, 1, H, W]")

    finite = ~(torch.isnan(points).any(dim=(1, 2)) | torch.isinf(points).any(dim=(1, 2)))
    low, high = fixed_range
    out_of_range = ((points < low) | (points > high)).any(dim=-1).float().mean(dim=1)
    filling_rate = (density[:, 0] > 0).sum(dim=(1, 2)).float() / float(resolution * resolution)
    max_mass = density.amax(dim=(1, 2, 3))
    return (
        finite
        & (out_of_range <= max_out_of_range_ratio)
        & (filling_rate >= min_filling_rate)
        & (max_mass <= max_pixel_mass)
    )


def render_density_from_params(
    params: torch.Tensor,
    *,
    resolution: int = 32,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    num_trajectories: int = 8,
    num_steps: int = 512,
    burn_in: int = 64,
    density_smoothing_sigma: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate points and a density map from one parameter set."""
    points = iterate_ifs_points(
        params,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        burn_in=burn_in,
        generator=generator,
    )
    return points, points_to_density_map(
        points,
        resolution=resolution,
        fixed_range=fixed_range,
        smoothing_sigma=density_smoothing_sigma,
    )


def render_density_from_params_batch(
    params: torch.Tensor,
    *,
    resolution: int = 32,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    num_trajectories: int = 8,
    num_steps: int = 512,
    burn_in: int = 64,
    density_smoothing_sigma: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate batched points and density maps from SVD-format parameter sets."""
    points = iterate_ifs_points_batch(
        params,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        burn_in=burn_in,
        generator=generator,
    )
    return points, points_to_density_maps(
        points,
        resolution=resolution,
        fixed_range=fixed_range,
        smoothing_sigma=density_smoothing_sigma,
    )


def render_density_from_affine_vector(
    params: torch.Tensor,
    *,
    resolution: int = 32,
    fixed_range: Tuple[float, float] = (-1.5, 1.5),
    num_trajectories: int = 8,
    num_steps: int = 512,
    burn_in: int = 64,
    density_smoothing_sigma: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate points and a density map from direct affine-vector parameters."""
    points = iterate_affine_vector_points(
        params,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        burn_in=burn_in,
        generator=generator,
    )
    return points, points_to_density_map(
        points,
        resolution=resolution,
        fixed_range=fixed_range,
        smoothing_sigma=density_smoothing_sigma,
    )
