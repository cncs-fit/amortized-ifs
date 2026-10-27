"""Revision-1 training variants (reviewer items I1-I4).

This script reuses the data pipeline, helpers, and argument parser of
``scripts/train_phase0.py`` unchanged and adds training-signal variants that the
reviewers asked for.  The original script is left untouched; with all revision
options at their defaults this script reproduces the ``paper_base50k`` recipe
(Hungarian matching + fixed-point term) through the original loss function.

Revision options are read from the YAML config (``parser.set_defaults`` passes
unknown keys through to the namespace):

  target_matching:   hungarian | ordered_fixed_point          (I1)
  supervision:       theta | recon_only | collage              (I2)
  num_hypotheses:    K >= 1, relaxed winner-takes-all          (I2)
  wta_epsilon:       weight shared by non-winning hypotheses
  output_constraint: none | spectral, spectral_radius          (I3)
  head_init:         identity | spread (symmetry breaking for GT-free training)
  selection_metric:  theta | recon (checkpoint selection on the validation set)
  reflection_prior:  false | true: every generated set (train cache, val, test) draws each
                     singular value with a random sign, i.e. orientation-reversing maps (I4b);
                     requires rebuild_train_cache so that no reflection-free cache is reused
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from data.dataset import (
    IFSIterableDataset,
    IFSIterableDatasetConfig,
    make_fixed_dataset_from_iterable_config,
)
from data.sampler import affine_matrices_to_vector, affine_vector_to_matrices, params_to_affine
from losses.hungarian import (
    fixed_points_from_matrices,
    hungarian_matching_loss_affine,
    hungarian_metrics_affine,
)
from losses.reconstruction import (
    differentiable_density_from_affine_vector,
    target_density_for_reconstruction,
)
from models.set_head import TinyCNNAffineSetEstimator, _atanh_float
from scripts import train_phase0 as base
from scripts.optimize_oracle import objective_losses, structural_penalties

REV1_DEFAULTS = {
    "target_matching": "hungarian",
    "supervision": "theta",
    "num_hypotheses": 1,
    "wta_epsilon": 0.05,
    "output_constraint": "none",
    "spectral_radius": 0.95,
    "head_init": "identity",
    "head_init_radius": 0.5,
    "selection_metric": "theta",
    "reflection_prior": False,
    # GT-free objectives (recon_only / collage); defaults mirror the refinement objective.
    "objective_point_chamfer_weight": 0.10,
    "objective_num_points": 512,
    "objective_resolution": None,
    "objective_num_trajectories": None,
    "objective_num_steps": None,
    "objective_burn_in": None,
    "objective_smoothing_sigma": None,
    "objective_seed": 12345,
    "spectral_penalty_weight": 0.1,
    "spectral_penalty_bound": 0.75,
    "det_negative_penalty_weight": 0.1,
    "translation_penalty_weight": 0.01,
    "collage_density_weight": 1.0,
    "collage_chamfer_weight": 0.10,
    # Validation reconstruction metric (all variants; used for selection when selection_metric=recon).
    "val_recon_seed": 777,
}


def rev1_options(args) -> SimpleNamespace:
    values = {key: getattr(args, key, default) for key, default in REV1_DEFAULTS.items()}
    opts = SimpleNamespace(**values)
    if opts.target_matching not in {"hungarian", "ordered_fixed_point"}:
        raise ValueError(f"unknown target_matching: {opts.target_matching}")
    if opts.supervision not in {"theta", "recon_only", "collage"}:
        raise ValueError(f"unknown supervision: {opts.supervision}")
    if opts.output_constraint not in {"none", "spectral"}:
        raise ValueError(f"unknown output_constraint: {opts.output_constraint}")
    if opts.head_init not in {"identity", "spread"}:
        raise ValueError(f"unknown head_init: {opts.head_init}")
    if int(opts.num_hypotheses) < 1:
        raise ValueError("num_hypotheses must be >= 1")
    if int(opts.num_hypotheses) > 1 and opts.supervision != "theta":
        raise ValueError("multi-hypothesis training is defined for theta supervision only")
    if int(opts.num_hypotheses) > 1 and opts.target_matching != "hungarian":
        raise ValueError("multi-hypothesis training uses Hungarian matching")
    return opts


def enable_reflection_prior() -> None:
    """I4b: make the data generator draw random singular-value signs (data/ itself is unchanged)."""
    import data.dataset as dataset_module
    from data.sampler import sample_ifs_parameters, sample_ifs_parameters_batch

    dataset_module.sample_ifs_parameters = partial(sample_ifs_parameters, allow_negative_scales=True)
    dataset_module.sample_ifs_parameters_batch = partial(sample_ifs_parameters_batch, allow_negative_scales=True)


# --------------------------------------------------------------------------- outputs


def spectral_rescale(affine: torch.Tensor, radius: float, eps: float = 1e-12) -> torch.Tensor:
    """Scale each linear part so that its largest singular value is at most ``radius``.

    For a 2x2 matrix, s_max^2 = (|W|_F^2 + sqrt(|W|_F^4 - 4 det^2)) / 2.
    """
    w = affine[..., 0:4]
    fro2 = w.square().sum(dim=-1)
    det = w[..., 0] * w[..., 3] - w[..., 1] * w[..., 2]
    disc = (fro2.square() - 4.0 * det.square()).clamp_min(0.0)
    smax = torch.sqrt(0.5 * (fro2 + torch.sqrt(disc + eps)) + eps)
    factor = torch.clamp(float(radius) / smax, max=1.0)
    return torch.cat((w * factor.unsqueeze(-1), affine[..., 4:6]), dim=-1)


def apply_output_constraint(pred: torch.Tensor, opts: SimpleNamespace) -> torch.Tensor:
    if opts.output_constraint == "spectral":
        return spectral_rescale(pred, opts.spectral_radius)
    return pred


def split_hypotheses(pred: torch.Tensor, num_transforms: int, num_hypotheses: int) -> torch.Tensor:
    """[B, K*n, 6] -> [B, K, n, 6]."""
    return pred.reshape(pred.shape[0], num_hypotheses, num_transforms, 6)


def spread_head_init(model: TinyCNNAffineSetEstimator, *, num_transforms: int, radius: float) -> None:
    """Start the n maps at distinct fixed points on a circle (W = 0.45 I).

    The default head starts all maps at the identical map 0.45 I, b = 0, which is
    harmless under Hungarian supervision but leaves GT-free objectives with an
    exactly symmetric start; spreading the fixed points breaks that symmetry.
    """
    final = model.head[-1]
    total = final.out_features // 6
    scale = 0.45
    raw_scale = _atanh_float(scale / model.linear_bound)
    bias = torch.zeros(total, 6)
    for slot in range(total):
        j = slot % num_transforms
        angle = 2.0 * math.pi * j / num_transforms + math.pi / 4.0
        fixed_point = torch.tensor([radius * math.cos(angle), radius * math.sin(angle)])
        b = fixed_point * (1.0 - scale)
        bias[slot, 0] = raw_scale
        bias[slot, 3] = raw_scale
        bias[slot, 4] = _atanh_float(float(b[0]) / model.translation_bound)
        bias[slot, 5] = _atanh_float(float(b[1]) / model.translation_bound)
    with torch.no_grad():
        final.bias.copy_(bias.reshape(-1))


# --------------------------------------------------------------------------- theta losses


def target_affine_vectors(target_params: torch.Tensor) -> torch.Tensor:
    target_w, target_b = params_to_affine(target_params)
    return affine_matrices_to_vector(target_w, target_b)


def ordered_fixed_point_loss(
    pred: torch.Tensor,
    target_params: torch.Tensor,
    *,
    fixed_point_loss_weight: float,
) -> torch.Tensor:
    """Canonical-order regression: slot i <-> i-th GT map sorted by fixed point (x, then y)."""
    target_affine = target_affine_vectors(target_params)
    tw, tb = affine_vector_to_matrices(target_affine)
    target_fp = fixed_points_from_matrices(tw, tb)
    # Lexicographic order via two stable sorts: secondary key (y) first, primary (x) second.
    order = torch.argsort(target_fp[..., 1], dim=1, stable=True)
    order = order.gather(1, torch.argsort(target_fp[..., 0].gather(1, order), dim=1, stable=True))
    sorted_target = target_affine.gather(1, order.unsqueeze(-1).expand(-1, -1, 6))
    diff = pred - sorted_target
    per_map = diff[..., 0:4].square().sum(dim=-1) + diff[..., 4:6].square().sum(dim=-1)
    loss = per_map.mean(dim=1)
    if fixed_point_loss_weight > 0.0:
        pw, pb = affine_vector_to_matrices(pred)
        sw, sb = affine_vector_to_matrices(sorted_target)
        fp = (fixed_points_from_matrices(pw, pb) - fixed_points_from_matrices(sw, sb)).square().sum(dim=-1)
        loss = loss + fixed_point_loss_weight * fp.mean(dim=1)
    return loss.mean()


_PERM_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _all_permutations(n: int, device: torch.device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = torch.tensor(list(itertools.permutations(range(n))), device=device)
    return _PERM_CACHE[key]


def per_sample_matching_losses(
    pred: torch.Tensor,
    target_affine: torch.Tensor,
    *,
    fixed_point_loss_weight: float,
) -> torch.Tensor:
    """Exact optimal-assignment loss per sample by enumerating permutations (small n).

    Same definition as ``hungarian_matching_loss_affine`` (assignment on the
    (W, b) cost; fixed-point term added on the matched pairs), but returns one
    value per sample so that hypotheses can compete.
    """
    n = pred.shape[1]
    if n > 6:
        raise ValueError("permutation enumeration is intended for n <= 6")
    cost = (pred[:, :, None, :] - target_affine[:, None, :, :]).square().sum(dim=-1)  # [B, n, n]
    perms = _all_permutations(n, pred.device)  # [P, n]
    rows = torch.arange(n, device=pred.device)
    perm_costs = cost[:, rows[None, :], perms].sum(dim=-1)  # [B, P]
    best = perm_costs.argmin(dim=1)
    matched_cols = perms[best]  # [B, n]: target index for prediction slot i
    loss = perm_costs.gather(1, best[:, None]).squeeze(1) / float(n)
    if fixed_point_loss_weight > 0.0:
        matched_target = target_affine.gather(1, matched_cols.unsqueeze(-1).expand(-1, -1, 6))
        pw, pb = affine_vector_to_matrices(pred)
        tw, tb = affine_vector_to_matrices(matched_target)
        fp = (fixed_points_from_matrices(pw, pb) - fixed_points_from_matrices(tw, tb)).square().sum(dim=-1)
        loss = loss + fixed_point_loss_weight * fp.mean(dim=1)
    return loss


def multi_hypothesis_loss(
    pred_hyp: torch.Tensor,
    target_params: torch.Tensor,
    *,
    fixed_point_loss_weight: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relaxed winner-takes-all over K hypotheses (Rupprecht et al., 2017)."""
    batch, k, n, _ = pred_hyp.shape
    target_affine = target_affine_vectors(target_params)
    flat_pred = pred_hyp.reshape(batch * k, n, 6)
    flat_target = target_affine.unsqueeze(1).expand(batch, k, n, 6).reshape(batch * k, n, 6)
    losses = per_sample_matching_losses(
        flat_pred,
        flat_target,
        fixed_point_loss_weight=fixed_point_loss_weight,
    ).reshape(batch, k)
    winner = losses.argmin(dim=1)
    weights = torch.full_like(losses, float(epsilon) / max(k - 1, 1))
    weights.scatter_(1, winner[:, None], 1.0 - float(epsilon) if k > 1 else 1.0)
    return (weights * losses).sum(dim=1).mean(), winner


# --------------------------------------------------------------------------- GT-free losses


def objective_namespace(args, opts: SimpleNamespace, train_config, *, step: int) -> SimpleNamespace:
    """Refinement-objective settings (matched renderer unless overridden)."""
    return SimpleNamespace(
        reconstruction_resolution=int(opts.objective_resolution or train_config.resolution),
        reconstruction_num_trajectories=int(
            opts.objective_num_trajectories or train_config.num_trajectories
        ),
        reconstruction_num_steps=int(opts.objective_num_steps or train_config.num_steps),
        reconstruction_burn_in=int(opts.objective_burn_in or train_config.burn_in),
        reconstruction_smoothing_sigma=float(
            train_config.density_smoothing_sigma
            if opts.objective_smoothing_sigma is None
            else opts.objective_smoothing_sigma
        ),
        reconstruction_map_probability_mode="determinant",
        reconstruction_seed=int(opts.objective_seed) + int(step),
        point_chamfer_loss_weight=float(opts.objective_point_chamfer_weight),
        point_chamfer_num_pred_points=int(opts.objective_num_points),
        point_chamfer_num_target_points=int(opts.objective_num_points),
        point_chamfer_seed=int(opts.objective_seed) + 500_000 + int(step),
    )


def penalties(pred: torch.Tensor, opts: SimpleNamespace, translation_bound: float) -> torch.Tensor:
    return structural_penalties(
        pred,
        spectral_upper_bound=opts.spectral_penalty_bound,
        spectral_penalty_weight=opts.spectral_penalty_weight,
        det_negative_penalty_weight=opts.det_negative_penalty_weight,
        translation_bound=translation_bound,
        translation_penalty_weight=opts.translation_penalty_weight,
    )


def recon_only_loss(pred, images, *, args, opts, train_config, step) -> tuple[torch.Tensor, dict]:
    ns = objective_namespace(args, opts, train_config, step=step)
    recon, chamfer = objective_losses(pred, images, args=ns, config=train_config)
    pen = penalties(pred, opts, translation_bound=float(train_config.fixed_range[1]))
    loss = (recon + chamfer + pen).mean()
    return loss, {"density": recon.mean().item(), "chamfer": chamfer.mean().item(), "penalty": pen.mean().item()}


def _pixel_centers(resolution: int, fixed_range, device, dtype) -> torch.Tensor:
    low, high = fixed_range
    width = float(high - low) / float(resolution)
    centers = float(low) + (torch.arange(resolution, device=device, dtype=dtype) + 0.5) * width
    gx, gy = torch.meshgrid(centers, centers, indexing="ij")  # density[ix, iy] convention
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)  # [R*R, 2]


def weighted_soft_splat(points, weights, *, resolution: int, fixed_range) -> torch.Tensor:
    """Bilinear splat of weighted points (same bin convention as the soft renderer)."""
    batch = points.shape[0]
    low, high = fixed_range
    scale = float(resolution) / float(high - low)
    x = (points[..., 0] - float(low)) * scale - 0.5
    y = (points[..., 1] - float(low)) * scale - 0.5
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    wx1 = x - x0.to(x.dtype)
    wy1 = y - y0.to(y.dtype)
    flat = points.new_zeros(batch * resolution * resolution)
    offsets = torch.arange(batch, device=points.device).view(batch, 1) * resolution * resolution
    for dx, dy, wgt in (
        (0, 0, (1 - wx1) * (1 - wy1)),
        (1, 0, wx1 * (1 - wy1)),
        (0, 1, (1 - wx1) * wy1),
        (1, 1, wx1 * wy1),
    ):
        xi = x0 + dx
        yi = y0 + dy
        ok = (xi >= 0) & (xi < resolution) & (yi >= 0) & (yi < resolution)
        index = offsets + xi.clamp(0, resolution - 1) * resolution + yi.clamp(0, resolution - 1)
        flat.scatter_add_(0, index.reshape(-1), (wgt * weights * ok.to(wgt.dtype)).reshape(-1))
    density = flat.reshape(batch, 1, resolution, resolution)
    return density / density.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)


def collage_loss(pred, images, *, opts, train_config, step) -> tuple[torch.Tensor, dict]:
    """Collage-theorem objective for the invariant measure (amortized-collage variant).

    density term:  || y - sum_i p_i (f_i)_# y ||^2 on the pixel grid, with p_i from |det W_i|
    point term:    symmetric squared Chamfer between X ~ y and the union of f_i(X)
    Neither term needs the chaos game; both are cheap and differentiable.
    """
    resolution = int(train_config.resolution)
    fixed_range = train_config.fixed_range
    y = target_density_for_reconstruction(images, resolution=resolution)  # [B,1,R,R]
    batch, n = pred.shape[0], pred.shape[1]
    w, b = affine_vector_to_matrices(pred)  # [B,n,2,2], [B,n,2]
    det = w[..., 0, 0] * w[..., 1, 1] - w[..., 0, 1] * w[..., 1, 0]
    prob = det.abs().clamp_min(0.02)
    prob = prob / prob.sum(dim=1, keepdim=True)

    centers = _pixel_centers(resolution, fixed_range, pred.device, pred.dtype)  # [P,2]
    mass = y.reshape(batch, -1)  # [B,P], matches meshgrid(ix, iy) flattening
    keep = mass.sum(dim=0) > 0  # drop pixels empty for the whole batch (cheap pruning)
    centers_k = centers[keep]
    mass_k = mass[:, keep]
    mapped = torch.einsum("pc,bncd->bnpd", centers_k, w) + b[:, :, None, :]  # [B,n,Pk,2]
    weights = prob[:, :, None] * mass_k[:, None, :]
    collage_density = weighted_soft_splat(
        mapped.reshape(batch, -1, 2),
        weights.reshape(batch, -1),
        resolution=resolution,
        fixed_range=fixed_range,
    )
    density_term = (collage_density - y).square().sum(dim=(1, 2, 3))

    num_points = int(opts.objective_num_points)
    generator = torch.Generator(device=pred.device).manual_seed(int(opts.objective_seed) + 900_000 + int(step))
    flat = mass.clamp_min(0.0)
    flat = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    idx = torch.multinomial(flat, num_points, replacement=True, generator=generator)
    x_pts = centers[idx]  # [B,M,2]
    sub = max(1, num_points // n)
    images_of_x = torch.einsum("bmc,bncd->bnmd", x_pts[:, :sub], w) + b[:, :, None, :]
    images_of_x = images_of_x.reshape(batch, -1, 2)
    dist = torch.cdist(images_of_x, x_pts)
    chamfer = 0.5 * (dist.min(dim=2).values.square().mean(dim=1) + dist.min(dim=1).values.square().mean(dim=1))

    pen = penalties(pred, opts, translation_bound=float(fixed_range[1]))
    loss = (
        float(opts.collage_density_weight) * density_term
        + float(opts.collage_chamfer_weight) * chamfer
        + pen
    ).mean()
    return loss, {
        "collage_density": density_term.mean().item(),
        "collage_chamfer": chamfer.mean().item(),
        "penalty": pen.mean().item(),
    }


# --------------------------------------------------------------------------- evaluation


@torch.no_grad()
def predict_hypotheses(model, images, *, num_transforms: int, opts: SimpleNamespace) -> torch.Tensor:
    pred = model(images)
    pred = split_hypotheses(pred, num_transforms, int(opts.num_hypotheses))
    return apply_output_constraint(pred, opts)


@torch.no_grad()
def recon_sse(pred: torch.Tensor, images: torch.Tensor, *, train_config, seed: int) -> torch.Tensor:
    """Per-sample density SSE with the (matched-setting) soft renderer and a fixed eval seed."""
    density = differentiable_density_from_affine_vector(
        pred,
        resolution=int(train_config.resolution),
        fixed_range=train_config.fixed_range,
        num_trajectories=int(train_config.num_trajectories),
        num_steps=int(train_config.num_steps),
        burn_in=int(train_config.burn_in),
        smoothing_sigma=float(train_config.density_smoothing_sigma),
        seed=int(seed),
        map_probability_mode="determinant",
    )
    target = target_density_for_reconstruction(images, resolution=int(train_config.resolution))
    return (density - target).square().sum(dim=(1, 2, 3))


@torch.no_grad()
def evaluate_rev1(model, loader, *, device, num_transforms, opts, train_config) -> dict:
    """Hungarian (W,b) metrics + reconstruction SSE.

    For K hypotheses: ``loss`` is the oracle (min over hypotheses) matching loss and
    the reconstruction metrics use the hypothesis selected by reconstruction SSE
    on the observed image (the deployable, GT-free rule).
    """
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for images, target_params in loader:
        images = images.to(device)
        target_params = target_params.to(device)
        pred_hyp = predict_hypotheses(model, images, num_transforms=num_transforms, opts=opts)
        batch, k = pred_hyp.shape[:2]
        sse = recon_sse(
            pred_hyp.reshape(batch * k, num_transforms, 6),
            images.repeat_interleave(k, dim=0),
            train_config=train_config,
            seed=opts.val_recon_seed,
        ).reshape(batch, k)
        selected = sse.argmin(dim=1)
        pred_sel = pred_hyp[torch.arange(batch, device=device), selected]
        metrics = hungarian_metrics_affine(pred_sel, target_params)
        batch_values = {f"sel_{key}": value for key, value in metrics.items()}
        target_affine = target_affine_vectors(target_params)
        if k > 1:
            hyp_losses = per_sample_matching_losses(
                pred_hyp.reshape(batch * k, num_transforms, 6),
                target_affine.unsqueeze(1).expand(batch, k, num_transforms, 6).reshape(batch * k, num_transforms, 6),
                fixed_point_loss_weight=0.0,
            ).reshape(batch, k)
            batch_values["loss"] = hyp_losses.min(dim=1).values.mean().item()
            batch_values["first_loss"] = hyp_losses[:, 0].mean().item()
            batch_values["selected_loss"] = hyp_losses[torch.arange(batch, device=device), selected].mean().item()
        else:
            batch_values["loss"] = metrics["loss"]
        batch_values["w_fro"] = metrics["w_fro"]
        batch_values["b_l2"] = metrics["b_l2"]
        batch_values["fixed_point_l2"] = metrics["fixed_point_l2"]
        batch_values["recon_sse"] = sse.gather(1, selected[:, None]).mean().item()
        for key, value in batch_values.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch
        count += batch
    model.train()
    return {key: value / count for key, value in totals.items()}


# --------------------------------------------------------------------------- main


def main() -> None:
    args = base.parse_args()
    opts = rev1_options(args)
    if opts.reflection_prior:
        if not getattr(args, "rebuild_train_cache", False):
            raise ValueError("reflection_prior requires rebuild_train_cache: true")
        enable_reflection_prior()
    if args.target_representation != "affine":
        raise ValueError("train_rev1 supports target_representation=affine only")
    if args.reconstruction_loss_weight > 0.0 or args.point_chamfer_loss_weight > 0.0:
        raise ValueError("use supervision=recon_only/collage instead of the phase0 auxiliary losses")
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_monitor = base.start_gpu_monitor(output_dir, interval_sec=args.gpu_monitor_interval) if args.device == "cuda" else None

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
    cache_metadata_history = []
    if args.train_cache_samples > 0:
        train_loader, cache_metadata, cache_elapsed_sec = base.make_cached_train_loader(args, train_config)
        cache_metadata_history.append({"step": 0, "seed": train_config.seed, "elapsed_sec": cache_elapsed_sec, "metadata": cache_metadata})
        train_data_source = "rolling_cache" if args.train_cache_refresh_interval > 0 else "cache"
    else:
        train_loader = DataLoader(IFSIterableDataset(train_config), **base._loader_kwargs(args, for_train=True))
        train_data_source = "on_the_fly"

    val_dataset = make_fixed_dataset_from_iterable_config(train_config, num_samples=args.val_samples, seed=args.val_seed)
    test_dataset = make_fixed_dataset_from_iterable_config(train_config, num_samples=args.test_samples, seed=args.test_seed)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    device = torch.device(args.device)
    n = int(args.num_transforms)
    k = int(opts.num_hypotheses)
    model = TinyCNNAffineSetEstimator(
        num_transforms=n * k,
        hidden_dim=args.hidden_dim,
        pool_grid=args.pool_grid,
        encoder_type=args.encoder_type,
        coord_channels=args.coord_channels,
        density_feature_mode=args.density_feature_mode,
        global_moments=args.global_moments,
        head_type=args.head_type,
        query_num_heads=args.query_num_heads,
        query_layers=args.query_layers,
    )
    if opts.head_init == "spread":
        spread_head_init(model, num_transforms=n, radius=float(opts.head_init_radius))
    model = model.to(device)
    if args.init_model_path is not None:
        model.load_state_dict(torch.load(args.init_model_path, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def training_loss(pred_raw, images, target_params, step):
        pred_hyp = apply_output_constraint(split_hypotheses(pred_raw, n, k), opts)
        extras: dict = {}
        if opts.supervision == "recon_only":
            loss, extras = recon_only_loss(pred_hyp[:, 0], images, args=args, opts=opts, train_config=train_config, step=step)
        elif opts.supervision == "collage":
            loss, extras = collage_loss(pred_hyp[:, 0], images, opts=opts, train_config=train_config, step=step)
        elif k > 1:
            loss, winner = multi_hypothesis_loss(
                pred_hyp, target_params,
                fixed_point_loss_weight=args.fixed_point_loss_weight, epsilon=opts.wta_epsilon,
            )
            extras = {"winner_hist": torch.bincount(winner, minlength=k).tolist()}
        elif opts.target_matching == "ordered_fixed_point":
            loss = ordered_fixed_point_loss(pred_hyp[:, 0], target_params, fixed_point_loss_weight=args.fixed_point_loss_weight)
        else:
            loss = hungarian_matching_loss_affine(
                pred_hyp[:, 0], target_params,
                linear_weight=args.linear_loss_weight,
                bias_weight=args.bias_loss_weight,
                fixed_point_loss_weight=args.fixed_point_loss_weight,
                fixed_point_cost_weight=args.fixed_point_cost_weight,
            )
        return loss, extras

    history: list[dict] = []
    perf_history: list[dict] = []
    train_iter = base.infinite_batches(train_loader)
    non_blocking = device.type == "cuda"
    interval_started = time.perf_counter()
    interval_samples = 0
    last_perf_step = 0
    best_metric = float("inf")
    best_record = None
    selection_key = "recon_sse" if opts.selection_metric == "recon" else "loss"
    for step in range(1, args.train_steps + 1):
        if args.train_cache_refresh_interval > 0 and step > 1 and (step - 1) % args.train_cache_refresh_interval == 0:
            refresh_index = (step - 1) // args.train_cache_refresh_interval
            refreshed = replace(train_config, seed=args.seed + refresh_index * args.train_cache_refresh_seed_stride)
            train_loader, meta, elapsed = base.make_cached_train_loader(args, refreshed)
            train_iter = base.infinite_batches(train_loader)
            cache_metadata_history.append({"step": step - 1, "seed": refreshed.seed, "elapsed_sec": elapsed, "metadata": meta})
            interval_started = time.perf_counter()
            interval_samples = 0
            last_perf_step = step - 1

        current_lr = base.step_learning_rate(args, step)
        base.set_optimizer_lr(optimizer, current_lr)
        images, target_params = next(train_iter)
        interval_samples += int(images.shape[0])
        images = images.to(device, non_blocking=non_blocking)
        target_params = target_params.to(device, non_blocking=non_blocking)
        loss, extras = training_loss(model(images), images, target_params, step)
        if not torch.isfinite(loss):
            print(f"non-finite loss at step={step}; skipping update", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if opts.supervision != "theta":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        should_eval = step == 1 or step % args.eval_interval == 0 or step == args.train_steps
        should_log = args.log_interval > 0 and (step == 1 or step % args.log_interval == 0 or step == args.train_steps)
        if not (should_eval or should_log):
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        interval_steps = step - last_perf_step
        record = {"step": step, "train_loss": loss.item(), "lr": current_lr}
        record.update({f"train_{key}": value for key, value in extras.items()})
        if should_log:
            perf = {
                "step": step,
                "avg_step_sec": (now - interval_started) / interval_steps,
                "samples_per_sec": interval_samples / max(now - interval_started, 1e-12),
            }
            if device.type == "cuda":
                perf["cuda_max_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
            perf_history.append(perf)
            print(f"perf step={step:05d} avg_step_sec={perf['avg_step_sec']:.4f} samples_per_sec={perf['samples_per_sec']:.1f}", flush=True)
        if should_eval:
            val = evaluate_rev1(model, val_loader, device=device, num_transforms=n, opts=opts, train_config=train_config)
            record.update({f"val_{key}": value for key, value in val.items()})
            if val[selection_key] < best_metric:
                best_metric = val[selection_key]
                best_record = {"step": step, "selection_key": selection_key, **{f"val_{key}": value for key, value in val.items()}}
                torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(
                f"step={step:05d} train_loss={loss.item():.6f} lr={current_lr:.6g} "
                f"val_loss={val['loss']:.6f} val_w_fro={val['w_fro']:.6f} val_b_l2={val['b_l2']:.6f} "
                f"val_recon_sse={val['recon_sse']:.6e}",
                flush=True,
            )
        history.append(record)
        interval_started = time.perf_counter()
        interval_samples = 0
        last_perf_step = step

    final_model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), final_model_path)
    val_final = evaluate_rev1(model, val_loader, device=device, num_transforms=n, opts=opts, train_config=train_config)
    test_final = evaluate_rev1(model, test_loader, device=device, num_transforms=n, opts=opts, train_config=train_config)
    best_model_path = output_dir / "best_model.pt"
    best_test = None
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_test = evaluate_rev1(model, test_loader, device=device, num_transforms=n, opts=opts, train_config=train_config)

    args_payload = vars(args).copy()
    args_payload.update({key: getattr(opts, key) for key in REV1_DEFAULTS})
    result = {
        "args": args_payload,
        "rev1": vars(opts),
        "train_config": train_config.__dict__,
        "val": val_final,
        "test": test_final,
        "best_test": best_test,
        "best_val": best_record,
        "history": history,
        "perf_history": perf_history,
        "train_data_source": train_data_source,
        "cache_metadata_history": cache_metadata_history,
        "best_model_path": str(best_model_path) if best_model_path.exists() else None,
        "final_model_path": str(final_model_path),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.plot:
        base.save_curves([{**r, "val_loss": r.get("val_loss"), "val_w_fro": r.get("val_w_fro"), "val_b_l2": r.get("val_b_l2")} for r in history if "val_loss" in r], output_dir)
    print(json.dumps({"val": val_final, "test": test_final, "best_test": best_test}, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)
    base.stop_gpu_monitor(gpu_monitor)


if __name__ == "__main__":
    main()
