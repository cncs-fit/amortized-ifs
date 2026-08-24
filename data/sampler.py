"""Sampling and affine-parameter utilities for synthetic IFS data."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from models.ifs import AffineIFS


PARAM_DIM = 6
PARAM_ORDER = ("phi1", "phi2", "sx", "sy", "tx", "ty")
AFFINE_VECTOR_ORDER = ("w00", "w01", "w10", "w11", "tx", "ty")


def _uniform(
    shape: Tuple[int, ...],
    low: float,
    high: float,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(low, high, generator=generator)


def _random_signs(
    shape: Tuple[int, ...],
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    values = torch.randint(0, 2, shape, generator=generator, device=device)
    return values.float().mul_(2.0).sub_(1.0)


def rotation_matrices(theta: torch.Tensor) -> torch.Tensor:
    """Return 2D rotation matrices with shape ``theta.shape + (2, 2)``."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    row0 = torch.stack((cos_t, -sin_t), dim=-1)
    row1 = torch.stack((sin_t, cos_t), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def params_to_affine(params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert SVD-format parameters to row-vector affine matrices.

    Args:
        params: Tensor with trailing dimension 6:
            ``(phi1, phi2, sx, sy, tx, ty)``. ``sx`` and ``sy`` are effective
            scales after tanh, not raw unconstrained values.

    Returns:
        ``(w, b)`` where points are updated as ``x @ w + b``. This matches
        ``AffineIFS.transform_points``.
    """
    if params.shape[-1] != PARAM_DIM:
        raise ValueError(f"expected trailing dim {PARAM_DIM}, got {params.shape[-1]}")

    phi1 = params[..., 0]
    phi2 = params[..., 1]
    sx = params[..., 2]
    sy = params[..., 3]
    b = params[..., 4:6]

    r1 = rotation_matrices(phi1)
    r2 = rotation_matrices(phi2)
    zeros = torch.zeros_like(sx)
    scale = torch.stack(
        (
            torch.stack((sx, zeros), dim=-1),
            torch.stack((zeros, sy), dim=-1),
        ),
        dim=-2,
    )
    column_matrix = r1 @ scale @ r2
    row_matrix = column_matrix.transpose(-1, -2)
    return row_matrix, b


def affine_matrices_to_vector(w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pack row-vector affine matrices and translations into trailing dim 6."""
    if w.shape[-2:] != (2, 2):
        raise ValueError("w must have trailing shape [2, 2]")
    if b.shape[-1] != 2:
        raise ValueError("b must have trailing shape [2]")
    if w.shape[:-2] != b.shape[:-1]:
        raise ValueError("w and b leading dimensions must match")
    return torch.cat((w.reshape(*w.shape[:-2], 4), b), dim=-1)


def affine_vector_to_matrices(params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack direct affine vectors into row-vector matrices and translations."""
    if params.shape[-1] != PARAM_DIM:
        raise ValueError(f"expected trailing dim {PARAM_DIM}, got {params.shape[-1]}")
    w = params[..., 0:4].reshape(*params.shape[:-1], 2, 2)
    b = params[..., 4:6]
    return w, b


def affine_probabilities(params: torch.Tensor, prob_floor: float = 0.02) -> torch.Tensor:
    """Return map-selection probabilities from effective scale determinants."""
    weights = torch.abs(params[..., 2] * params[..., 3]).clamp_min(prob_floor)
    return weights / weights.sum(dim=-1, keepdim=True)


def affine_vector_probabilities(params: torch.Tensor, prob_floor: float = 0.02) -> torch.Tensor:
    """Return map-selection probabilities from direct affine determinants."""
    w, _ = affine_vector_to_matrices(params)
    det = w[..., 0, 0] * w[..., 1, 1] - w[..., 0, 1] * w[..., 1, 0]
    weights = torch.abs(det).clamp_min(prob_floor)
    return weights / weights.sum(dim=-1, keepdim=True)


def sample_ifs_parameters(
    num_transforms: int = 2,
    *,
    phase: str = "phase_minus1",
    scale_range: Tuple[float, float] = (0.20, 0.70),
    fixed_point_range: float = 0.75,
    allow_negative_scales: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample one IFS parameter set in effective SVD space.

    Phase -1 intentionally uses positive scales and fixed-point based
    translations to reduce avoidable degeneracy while debugging the pipeline.
    """
    if phase not in {"phase_minus1", "phase0"}:
        raise ValueError(f"unsupported sampling phase: {phase}")

    phi1 = _uniform((num_transforms,), -math.pi, math.pi, generator=generator, device=device)
    phi2 = _uniform((num_transforms,), -math.pi, math.pi, generator=generator, device=device)
    sx = _uniform((num_transforms,), scale_range[0], scale_range[1], generator=generator, device=device)
    sy = _uniform((num_transforms,), scale_range[0], scale_range[1], generator=generator, device=device)

    if allow_negative_scales:
        sx = sx * _random_signs((num_transforms,), generator=generator, device=device)
        sy = sy * _random_signs((num_transforms,), generator=generator, device=device)

    provisional = torch.stack(
        (phi1, phi2, sx, sy, torch.zeros_like(sx), torch.zeros_like(sy)), dim=-1
    )
    w, _ = params_to_affine(provisional)
    fixed_points = _uniform(
        (num_transforms, 2),
        -fixed_point_range,
        fixed_point_range,
        generator=generator,
        device=device,
    )
    b = fixed_points - torch.bmm(fixed_points.unsqueeze(1), w).squeeze(1)
    return torch.cat((provisional[:, :4], b), dim=-1)


def sample_ifs_parameters_batch(
    batch_size: int,
    num_transforms: int = 2,
    *,
    phase: str = "phase_minus1",
    scale_range: Tuple[float, float] = (0.20, 0.70),
    fixed_point_range: float = 0.75,
    allow_negative_scales: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample a batch of IFS parameter sets in effective SVD space."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if phase not in {"phase_minus1", "phase0"}:
        raise ValueError(f"unsupported sampling phase: {phase}")

    shape = (batch_size, num_transforms)
    phi1 = _uniform(shape, -math.pi, math.pi, generator=generator, device=device)
    phi2 = _uniform(shape, -math.pi, math.pi, generator=generator, device=device)
    sx = _uniform(shape, scale_range[0], scale_range[1], generator=generator, device=device)
    sy = _uniform(shape, scale_range[0], scale_range[1], generator=generator, device=device)

    if allow_negative_scales:
        sx = sx * _random_signs(shape, generator=generator, device=device)
        sy = sy * _random_signs(shape, generator=generator, device=device)

    provisional = torch.stack(
        (phi1, phi2, sx, sy, torch.zeros_like(sx), torch.zeros_like(sy)), dim=-1
    )
    w, _ = params_to_affine(provisional)
    fixed_points = _uniform(
        (batch_size, num_transforms, 2),
        -fixed_point_range,
        fixed_point_range,
        generator=generator,
        device=device,
    )
    b = fixed_points - torch.matmul(fixed_points.unsqueeze(-2), w).squeeze(-2)
    return torch.cat((provisional[..., :4], b), dim=-1)


def create_ifs_from_params(params: torch.Tensor, prob_floor: float = 0.02) -> AffineIFS:
    """Create an ``AffineIFS`` instance from effective SVD parameters."""
    if params.ndim != 2 or params.shape[-1] != PARAM_DIM:
        raise ValueError("params must have shape [num_transforms, 6]")

    params_cpu = params.detach().cpu().float()
    ifs = AffineIFS(
        num_transforms=params_cpu.shape[0],
        use_htan=True,
        contractive_init=False,
        name="sampled",
    )
    scales = params_cpu[:, 2:4].clamp(-0.999999, 0.999999)
    raw_scales = torch.atanh(scales)
    with torch.no_grad():
        ifs.ifs_w.weight.copy_(torch.cat((params_cpu[:, 0:2], raw_scales), dim=-1))
        ifs.ifs_b.weight.copy_(params_cpu[:, 4:6])
    ifs.set_probs(affine_probabilities(params_cpu, prob_floor=prob_floor))
    return ifs


def iterate_ifs_points(
    params: torch.Tensor,
    *,
    num_trajectories: int = 8,
    num_steps: int = 512,
    burn_in: int = 64,
    initial_range: float = 1.0,
    prob_floor: float = 0.02,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate point samples from an IFS using the same row-vector convention."""
    if params.ndim != 2 or params.shape[-1] != PARAM_DIM:
        raise ValueError("params must have shape [num_transforms, 6]")
    if burn_in >= num_steps:
        raise ValueError("burn_in must be smaller than num_steps")

    device = params.device
    w, b = params_to_affine(params)
    probs = affine_probabilities(params, prob_floor=prob_floor)
    seq = torch.multinomial(
        probs,
        num_trajectories * num_steps,
        replacement=True,
        generator=generator,
    ).reshape(num_trajectories, num_steps)
    current = _uniform(
        (num_trajectories, 2),
        -initial_range,
        initial_range,
        generator=generator,
        device=device,
    )

    kept = []
    for step in range(num_steps):
        ids = seq[:, step].to(device=device)
        current = torch.bmm(current.unsqueeze(1), w[ids]).squeeze(1) + b[ids]
        if step >= burn_in:
            kept.append(current)
    return torch.stack(kept, dim=1).reshape(-1, 2)


def iterate_ifs_points_batch(
    params: torch.Tensor,
    *,
    num_trajectories: int = 8,
    num_steps: int = 512,
    burn_in: int = 64,
    initial_range: float = 1.0,
    prob_floor: float = 0.02,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate batched point samples from SVD-format IFS parameters."""
    if params.ndim != 3 or params.shape[-1] != PARAM_DIM:
        raise ValueError("params must have shape [batch, num_transforms, 6]")
    if burn_in >= num_steps:
        raise ValueError("burn_in must be smaller than num_steps")

    batch_size = params.shape[0]
    device = params.device
    w, b = params_to_affine(params)
    probs = affine_probabilities(params, prob_floor=prob_floor)
    seq = torch.multinomial(
        probs,
        num_trajectories * num_steps,
        replacement=True,
        generator=generator,
    ).reshape(batch_size, num_trajectories, num_steps)
    current = _uniform(
        (batch_size, num_trajectories, 2),
        -initial_range,
        initial_range,
        generator=generator,
        device=device,
    )

    kept = []
    for step in range(num_steps):
        ids = seq[:, :, step].to(device=device)
        w_idx = ids[:, :, None, None].expand(-1, -1, 2, 2)
        b_idx = ids[:, :, None].expand(-1, -1, 2)
        selected_w = torch.gather(w, dim=1, index=w_idx)
        selected_b = torch.gather(b, dim=1, index=b_idx)
        current = torch.matmul(current.unsqueeze(-2), selected_w).squeeze(-2) + selected_b
        if step >= burn_in:
            kept.append(current)
    return torch.stack(kept, dim=2).reshape(batch_size, -1, 2)


def iterate_affine_vector_points(
    params: torch.Tensor,
    *,
    num_trajectories: int = 8,
    num_steps: int = 512,
    burn_in: int = 64,
    initial_range: float = 1.0,
    prob_floor: float = 0.02,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate point samples from direct affine-vector parameters."""
    if params.ndim != 2 or params.shape[-1] != PARAM_DIM:
        raise ValueError("params must have shape [num_transforms, 6]")
    if burn_in >= num_steps:
        raise ValueError("burn_in must be smaller than num_steps")

    device = params.device
    w, b = affine_vector_to_matrices(params)
    probs = affine_vector_probabilities(params, prob_floor=prob_floor)
    seq = torch.multinomial(
        probs,
        num_trajectories * num_steps,
        replacement=True,
        generator=generator,
    ).reshape(num_trajectories, num_steps)
    current = _uniform(
        (num_trajectories, 2),
        -initial_range,
        initial_range,
        generator=generator,
        device=device,
    )

    kept = []
    for step in range(num_steps):
        ids = seq[:, step].to(device=device)
        current = torch.bmm(current.unsqueeze(1), w[ids]).squeeze(1) + b[ids]
        if step >= burn_in:
            kept.append(current)
    return torch.stack(kept, dim=1).reshape(-1, 2)
