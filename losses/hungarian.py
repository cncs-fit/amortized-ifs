"""Hungarian matching loss in affine matrix space."""

from __future__ import annotations

import itertools
from typing import List, Tuple

import torch

from data.sampler import affine_matrices_to_vector, affine_vector_to_matrices, params_to_affine

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only in minimal envs
    linear_sum_assignment = None


def pairwise_affine_distance(
    pred_params: torch.Tensor,
    target_params: torch.Tensor,
    *,
    linear_weight: float = 1.0,
    bias_weight: float = 1.0,
    fixed_point_weight: float = 0.0,
) -> torch.Tensor:
    """Return pairwise squared distances between affine maps."""
    pred_w, pred_b = params_to_affine(pred_params)
    target_w, target_b = params_to_affine(target_params)
    linear = (pred_w[:, None, :, :] - target_w[None, :, :, :]).square().sum(dim=(-1, -2))
    bias = (pred_b[:, None, :] - target_b[None, :, :]).square().sum(dim=-1)
    distance = linear_weight * linear + bias_weight * bias
    if fixed_point_weight > 0.0:
        pred_fp = fixed_points_from_matrices(pred_w, pred_b)
        target_fp = fixed_points_from_matrices(target_w, target_b)
        fixed_point = (pred_fp[:, None, :] - target_fp[None, :, :]).square().sum(dim=-1)
        distance = distance + fixed_point_weight * fixed_point
    return distance


def pairwise_direct_affine_distance(
    pred_affine: torch.Tensor,
    target_affine: torch.Tensor,
    *,
    linear_weight: float = 1.0,
    bias_weight: float = 1.0,
    fixed_point_weight: float = 0.0,
) -> torch.Tensor:
    """Return pairwise squared distances between direct affine-vector maps."""
    pred_w, pred_b = affine_vector_to_matrices(pred_affine)
    target_w, target_b = affine_vector_to_matrices(target_affine)
    linear = (pred_w[:, None, :, :] - target_w[None, :, :, :]).square().sum(dim=(-1, -2))
    bias = (pred_b[:, None, :] - target_b[None, :, :]).square().sum(dim=-1)
    distance = linear_weight * linear + bias_weight * bias
    if fixed_point_weight > 0.0:
        pred_fp = fixed_points_from_matrices(pred_w, pred_b)
        target_fp = fixed_points_from_matrices(target_w, target_b)
        fixed_point = (pred_fp[:, None, :] - target_fp[None, :, :]).square().sum(dim=-1)
        distance = distance + fixed_point_weight * fixed_point
    return distance


def fixed_points_from_matrices(w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return fixed points for row-vector maps ``x -> x @ W + b``."""
    if w.shape[-2:] != (2, 2):
        raise ValueError("w must have trailing shape [2, 2]")
    if b.shape[-1] != 2:
        raise ValueError("b must have trailing shape [2]")
    if w.shape[:-2] != b.shape[:-1]:
        raise ValueError("w and b leading dimensions must match")

    identity = torch.eye(2, device=w.device, dtype=w.dtype).expand_as(w)
    system = (identity - w).transpose(-1, -2)
    return torch.linalg.solve(system, b.unsqueeze(-1)).squeeze(-1)


def affine_matrix_regularization(
    w: torch.Tensor,
    *,
    spectral_upper_bound: float = 0.75,
    spectral_upper_loss_weight: float = 0.0,
    det_positive_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Return weak structural penalties for contraction-like affine maps."""
    if spectral_upper_loss_weight <= 0.0 and det_positive_loss_weight <= 0.0:
        return w.new_zeros(())
    if w.shape[-2:] != (2, 2):
        raise ValueError("w must have trailing shape [2, 2]")

    penalties = []
    if spectral_upper_loss_weight > 0.0:
        singular_values = torch.linalg.svdvals(w)
        upper_violation = torch.relu(singular_values - float(spectral_upper_bound)).square().mean()
        penalties.append(float(spectral_upper_loss_weight) * upper_violation)
    if det_positive_loss_weight > 0.0:
        det = torch.linalg.det(w)
        det_violation = torch.relu(-det).square().mean()
        penalties.append(float(det_positive_loss_weight) * det_violation)
    return torch.stack(penalties).sum()


def affine_shape_auxiliary_loss(
    pred_w: torch.Tensor,
    target_w: torch.Tensor,
    *,
    singular_value_loss_weight: float = 0.0,
    determinant_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Return supervised spectral/area penalties for matched affine maps."""
    if singular_value_loss_weight <= 0.0 and determinant_loss_weight <= 0.0:
        return pred_w.new_zeros(())
    if pred_w.shape != target_w.shape or pred_w.shape[-2:] != (2, 2):
        raise ValueError("pred_w and target_w must have matching trailing shape [2, 2]")

    penalties = []
    if singular_value_loss_weight > 0.0:
        pred_singular = torch.linalg.svdvals(pred_w)
        target_singular = torch.linalg.svdvals(target_w)
        singular_loss = (pred_singular - target_singular).square().sum(dim=-1).mean()
        penalties.append(float(singular_value_loss_weight) * singular_loss)
    if determinant_loss_weight > 0.0:
        pred_det = torch.linalg.det(pred_w)
        target_det = torch.linalg.det(target_w)
        determinant_loss = (pred_det - target_det).square().mean()
        penalties.append(float(determinant_loss_weight) * determinant_loss)
    return torch.stack(penalties).sum()


def target_spectral_linear_weights(
    target_params: torch.Tensor,
    *,
    extra_weight: float = 0.0,
    threshold: float = 0.55,
    upper: float = 0.70,
) -> torch.Tensor:
    """Return target-dependent weights for high-spectral affine maps."""
    if extra_weight <= 0.0:
        return torch.ones(target_params.shape[:-1], device=target_params.device)
    if target_params.shape[-1] != 6:
        raise ValueError("target_params must have trailing shape [6]")
    denom = max(float(upper) - float(threshold), 1e-6)
    target_spectral_max = target_params[..., 2:4].abs().amax(dim=-1)
    relative = ((target_spectral_max - float(threshold)) / denom).clamp(0.0, 1.0)
    return 1.0 + float(extra_weight) * relative


def _fallback_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_pred, num_target = cost.shape
    if num_target > num_pred:
        raise ValueError("number of targets cannot exceed number of predictions")
    best_cols = None
    best_value = None
    for cols in itertools.permutations(range(num_pred), num_target):
        rows = torch.tensor(cols, dtype=torch.long)
        target_cols = torch.arange(num_target, dtype=torch.long)
        value = cost[rows, target_cols].sum().item()
        if best_value is None or value < best_value:
            best_value = value
            best_cols = rows
    return best_cols, torch.arange(num_target, dtype=torch.long)


def hungarian_indices(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve a rectangular assignment problem for one cost matrix."""
    if linear_sum_assignment is None:
        return _fallback_assignment(cost.detach().cpu())
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)


def hungarian_matching_loss(
    pred_params: torch.Tensor,
    target_params: torch.Tensor,
    *,
    linear_weight: float = 1.0,
    bias_weight: float = 1.0,
    fixed_point_loss_weight: float = 0.0,
    fixed_point_cost_weight: float = 0.0,
    spectral_upper_loss_weight: float = 0.0,
    spectral_upper_bound: float = 0.75,
    det_positive_loss_weight: float = 0.0,
    singular_value_loss_weight: float = 0.0,
    determinant_loss_weight: float = 0.0,
    target_spectral_linear_extra_weight: float = 0.0,
    target_spectral_linear_threshold: float = 0.55,
    target_spectral_linear_upper: float = 0.70,
) -> torch.Tensor:
    """Compute batch-mean Hungarian matching loss in affine map space."""
    if pred_params.ndim != 3 or target_params.ndim != 3:
        raise ValueError("pred_params and target_params must be [B, N, 6]")
    if target_params.shape[1] > pred_params.shape[1]:
        raise ValueError("number of targets cannot exceed number of predictions")

    losses: List[torch.Tensor] = []
    for batch_idx in range(pred_params.shape[0]):
        cost = pairwise_affine_distance(
            pred_params[batch_idx],
            target_params[batch_idx],
            linear_weight=linear_weight,
            bias_weight=bias_weight,
            fixed_point_weight=fixed_point_cost_weight,
        )
        row_ind, col_ind = hungarian_indices(cost)
        row_ind = row_ind.to(device=cost.device)
        col_ind = col_ind.to(device=cost.device)
        if target_spectral_linear_extra_weight > 0.0:
            pred_w, pred_b = params_to_affine(pred_params[batch_idx, row_ind])
            target_w, target_b = params_to_affine(target_params[batch_idx, col_ind])
            linear_losses = (pred_w - target_w).square().sum(dim=(-1, -2))
            bias_losses = (pred_b - target_b).square().sum(dim=-1)
            weights = target_spectral_linear_weights(
                target_params[batch_idx, col_ind],
                extra_weight=target_spectral_linear_extra_weight,
                threshold=target_spectral_linear_threshold,
                upper=target_spectral_linear_upper,
            )
            weighted_linear = (weights * linear_losses).sum() / weights.sum().clamp_min(1e-8)
            selected_loss = linear_weight * weighted_linear + bias_weight * bias_losses.mean()
        else:
            base_cost = pairwise_affine_distance(
                pred_params[batch_idx],
                target_params[batch_idx],
                linear_weight=linear_weight,
                bias_weight=bias_weight,
            )
            selected_loss = base_cost[row_ind, col_ind].mean()
        if fixed_point_loss_weight > 0.0:
            pred_w, pred_b = params_to_affine(pred_params[batch_idx, row_ind])
            target_w, target_b = params_to_affine(target_params[batch_idx, col_ind])
            pred_fp = fixed_points_from_matrices(pred_w, pred_b)
            target_fp = fixed_points_from_matrices(target_w, target_b)
            fp_loss = (pred_fp - target_fp).square().sum(dim=-1).mean()
            selected_loss = selected_loss + fixed_point_loss_weight * fp_loss
        if spectral_upper_loss_weight > 0.0 or det_positive_loss_weight > 0.0:
            pred_w, _ = params_to_affine(pred_params[batch_idx, row_ind])
            selected_loss = selected_loss + affine_matrix_regularization(
                pred_w,
                spectral_upper_bound=spectral_upper_bound,
                spectral_upper_loss_weight=spectral_upper_loss_weight,
                det_positive_loss_weight=det_positive_loss_weight,
            )
        if singular_value_loss_weight > 0.0 or determinant_loss_weight > 0.0:
            pred_w, _ = params_to_affine(pred_params[batch_idx, row_ind])
            target_w, _ = params_to_affine(target_params[batch_idx, col_ind])
            selected_loss = selected_loss + affine_shape_auxiliary_loss(
                pred_w,
                target_w,
                singular_value_loss_weight=singular_value_loss_weight,
                determinant_loss_weight=determinant_loss_weight,
            )
        losses.append(selected_loss)
    return torch.stack(losses).mean()


def hungarian_matching_loss_affine(
    pred_affine: torch.Tensor,
    target_params: torch.Tensor,
    *,
    linear_weight: float = 1.0,
    bias_weight: float = 1.0,
    fixed_point_loss_weight: float = 0.0,
    fixed_point_cost_weight: float = 0.0,
    spectral_upper_loss_weight: float = 0.0,
    spectral_upper_bound: float = 0.75,
    det_positive_loss_weight: float = 0.0,
    singular_value_loss_weight: float = 0.0,
    determinant_loss_weight: float = 0.0,
    target_spectral_linear_extra_weight: float = 0.0,
    target_spectral_linear_threshold: float = 0.55,
    target_spectral_linear_upper: float = 0.70,
) -> torch.Tensor:
    """Compute Hungarian matching loss for direct affine-vector predictions."""
    if pred_affine.ndim != 3 or target_params.ndim != 3:
        raise ValueError("pred_affine and target_params must be [B, N, 6]")
    if target_params.shape[1] > pred_affine.shape[1]:
        raise ValueError("number of targets cannot exceed number of predictions")

    target_w, target_b = params_to_affine(target_params)
    target_affine = affine_matrices_to_vector(target_w, target_b)

    losses: List[torch.Tensor] = []
    for batch_idx in range(pred_affine.shape[0]):
        cost = pairwise_direct_affine_distance(
            pred_affine[batch_idx],
            target_affine[batch_idx],
            linear_weight=linear_weight,
            bias_weight=bias_weight,
            fixed_point_weight=fixed_point_cost_weight,
        )
        row_ind, col_ind = hungarian_indices(cost)
        row_ind = row_ind.to(device=cost.device)
        col_ind = col_ind.to(device=cost.device)
        if target_spectral_linear_extra_weight > 0.0:
            pred_w, pred_b = affine_vector_to_matrices(pred_affine[batch_idx, row_ind])
            matched_target_w = target_w[batch_idx, col_ind]
            matched_target_b = target_b[batch_idx, col_ind]
            linear_losses = (pred_w - matched_target_w).square().sum(dim=(-1, -2))
            bias_losses = (pred_b - matched_target_b).square().sum(dim=-1)
            weights = target_spectral_linear_weights(
                target_params[batch_idx, col_ind],
                extra_weight=target_spectral_linear_extra_weight,
                threshold=target_spectral_linear_threshold,
                upper=target_spectral_linear_upper,
            )
            weighted_linear = (weights * linear_losses).sum() / weights.sum().clamp_min(1e-8)
            selected_loss = linear_weight * weighted_linear + bias_weight * bias_losses.mean()
        else:
            base_cost = pairwise_direct_affine_distance(
                pred_affine[batch_idx],
                target_affine[batch_idx],
                linear_weight=linear_weight,
                bias_weight=bias_weight,
            )
            selected_loss = base_cost[row_ind, col_ind].mean()
        if fixed_point_loss_weight > 0.0:
            pred_w, pred_b = affine_vector_to_matrices(pred_affine[batch_idx, row_ind])
            matched_target_w = target_w[batch_idx, col_ind]
            matched_target_b = target_b[batch_idx, col_ind]
            pred_fp = fixed_points_from_matrices(pred_w, pred_b)
            target_fp = fixed_points_from_matrices(matched_target_w, matched_target_b)
            fp_loss = (pred_fp - target_fp).square().sum(dim=-1).mean()
            selected_loss = selected_loss + fixed_point_loss_weight * fp_loss
        if spectral_upper_loss_weight > 0.0 or det_positive_loss_weight > 0.0:
            pred_w, _ = affine_vector_to_matrices(pred_affine[batch_idx, row_ind])
            selected_loss = selected_loss + affine_matrix_regularization(
                pred_w,
                spectral_upper_bound=spectral_upper_bound,
                spectral_upper_loss_weight=spectral_upper_loss_weight,
                det_positive_loss_weight=det_positive_loss_weight,
            )
        if singular_value_loss_weight > 0.0 or determinant_loss_weight > 0.0:
            pred_w, _ = affine_vector_to_matrices(pred_affine[batch_idx, row_ind])
            matched_target_w = target_w[batch_idx, col_ind]
            selected_loss = selected_loss + affine_shape_auxiliary_loss(
                pred_w,
                matched_target_w,
                singular_value_loss_weight=singular_value_loss_weight,
                determinant_loss_weight=determinant_loss_weight,
            )
        losses.append(selected_loss)
    return torch.stack(losses).mean()


@torch.no_grad()
def hungarian_metrics(
    pred_params: torch.Tensor,
    target_params: torch.Tensor,
) -> dict:
    """Return interpretable affine-space metrics after optimal matching."""
    linear_errors = []
    bias_errors = []
    fixed_point_errors = []
    spectral_max_values = []
    spectral_upper_violations = []
    det_negative_violations = []
    total_losses = []
    for batch_idx in range(pred_params.shape[0]):
        cost = pairwise_affine_distance(pred_params[batch_idx], target_params[batch_idx])
        row_ind, col_ind = hungarian_indices(cost)
        row_ind = row_ind.to(device=cost.device)
        col_ind = col_ind.to(device=cost.device)

        pred_w, pred_b = params_to_affine(pred_params[batch_idx, row_ind])
        target_w, target_b = params_to_affine(target_params[batch_idx, col_ind])
        linear_errors.append((pred_w - target_w).square().sum(dim=(-1, -2)).sqrt().mean())
        bias_errors.append((pred_b - target_b).square().sum(dim=-1).sqrt().mean())
        pred_fp = fixed_points_from_matrices(pred_w, pred_b)
        target_fp = fixed_points_from_matrices(target_w, target_b)
        fixed_point_errors.append((pred_fp - target_fp).square().sum(dim=-1).sqrt().mean())
        pred_singular_values = torch.linalg.svdvals(pred_w)
        spectral_max_values.append(pred_singular_values.amax(dim=-1).mean())
        spectral_upper_violations.append(torch.relu(pred_singular_values - 0.75).mean())
        det_negative_violations.append(torch.relu(-torch.linalg.det(pred_w)).mean())
        total_losses.append(cost[row_ind, col_ind].mean())

    return {
        "loss": torch.stack(total_losses).mean().item(),
        "w_fro": torch.stack(linear_errors).mean().item(),
        "b_l2": torch.stack(bias_errors).mean().item(),
        "fixed_point_l2": torch.stack(fixed_point_errors).mean().item(),
        "pred_spectral_max": torch.stack(spectral_max_values).mean().item(),
        "pred_spectral_upper_violation": torch.stack(spectral_upper_violations).mean().item(),
        "pred_det_negative_violation": torch.stack(det_negative_violations).mean().item(),
    }


@torch.no_grad()
def hungarian_metrics_affine(
    pred_affine: torch.Tensor,
    target_params: torch.Tensor,
) -> dict:
    """Return affine-space metrics for direct affine-vector predictions."""
    target_w, target_b = params_to_affine(target_params)
    target_affine = affine_matrices_to_vector(target_w, target_b)

    linear_errors = []
    bias_errors = []
    fixed_point_errors = []
    spectral_max_values = []
    spectral_upper_violations = []
    det_negative_violations = []
    total_losses = []
    for batch_idx in range(pred_affine.shape[0]):
        cost = pairwise_direct_affine_distance(pred_affine[batch_idx], target_affine[batch_idx])
        row_ind, col_ind = hungarian_indices(cost)
        row_ind = row_ind.to(device=cost.device)
        col_ind = col_ind.to(device=cost.device)

        pred_w, pred_b = affine_vector_to_matrices(pred_affine[batch_idx, row_ind])
        matched_target_w = target_w[batch_idx, col_ind]
        matched_target_b = target_b[batch_idx, col_ind]
        linear_errors.append((pred_w - matched_target_w).square().sum(dim=(-1, -2)).sqrt().mean())
        bias_errors.append((pred_b - matched_target_b).square().sum(dim=-1).sqrt().mean())
        pred_fp = fixed_points_from_matrices(pred_w, pred_b)
        target_fp = fixed_points_from_matrices(matched_target_w, matched_target_b)
        fixed_point_errors.append((pred_fp - target_fp).square().sum(dim=-1).sqrt().mean())
        pred_singular_values = torch.linalg.svdvals(pred_w)
        spectral_max_values.append(pred_singular_values.amax(dim=-1).mean())
        spectral_upper_violations.append(torch.relu(pred_singular_values - 0.75).mean())
        det_negative_violations.append(torch.relu(-torch.linalg.det(pred_w)).mean())
        total_losses.append(cost[row_ind, col_ind].mean())

    return {
        "loss": torch.stack(total_losses).mean().item(),
        "w_fro": torch.stack(linear_errors).mean().item(),
        "b_l2": torch.stack(bias_errors).mean().item(),
        "fixed_point_l2": torch.stack(fixed_point_errors).mean().item(),
        "pred_spectral_max": torch.stack(spectral_max_values).mean().item(),
        "pred_spectral_upper_violation": torch.stack(spectral_upper_violations).mean().item(),
        "pred_det_negative_violation": torch.stack(det_negative_violations).mean().item(),
    }
