"""Shift / robustness evaluation driver for revision items I4, I5, I6.

Builds a set of target IFSs, renders *observations* under configurable
conditions, and evaluates the amortized estimator (0-step and K-step
refinement) and the random-restart per-image baseline on them.

Target sources (``--gt-source``):
  testset   the paper's test256 (same generator and seed) with optional scale/fixed-point overrides
  reflect   same generator with random singular-value signs (orientation-reversing maps), validity-filtered
  file      external IFS codes from a .pt file {"affine": [N,n,6], optional "probs": [N,n], "names": [...]}

Selection probabilities (``--prob-mode``): det (|det W| rule, as in training),
file (from the .pt), or mix: p = (1-alpha) p_det + alpha * Dirichlet(1).

Observation renderer: trajectories / steps / burn-in / smoothing / resolution
(resampled to 128 for the estimator) / splat type (hist or bilinear).
``--input-preblur s`` adds a test-time Gaussian blur to the observation (used as
both the estimator input and the refinement target, for every method), e.g. to
bring a sharper observation (obs sigma < 2) back to the training smoothing.

Evaluation never uses the observation renderer for the point metrics: the
predicted attractor and the true attractor (with its true probabilities) are
both drawn with the same high-fidelity chaos game, so Chamfer / HD95 /
coverage measure recovery of the attractor itself.  Density SSE is reported
twice: against the observation (prediction rendered with the observation
renderer) and on the standard renderer (prediction vs. truth, both standard).

The refinement objective, optimizer, restarts and penalties are those of
``scripts/optimize_oracle.py`` (imported unchanged).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from data.dataset import make_fixed_dataset_from_iterable_config
from data.renderer import points_to_density_maps, smooth_density_maps, valid_sample_mask
from data.sampler import (
    affine_matrices_to_vector,
    affine_vector_probabilities,
    affine_vector_to_matrices,
    iterate_ifs_points_batch,
    params_to_affine,
    sample_ifs_parameters_batch,
)
from losses.reconstruction import soft_points_to_density_maps
from scripts.analyze_identifiability import affine_set_distance
from scripts.evaluate_checkpoint import _build_model, _checkpoint_paths, _config_from_payload
from scripts.evaluate_oracle_fidelity import point_metrics
from scripts.optimize_oracle import make_initial_restarts, optimize_batch

PIXEL = 3.0 / 128.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="checkpoints/paper_base50k")
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gt-source", choices=("testset", "reflect", "file"), default="testset")
    p.add_argument("--gt-file", default=None)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=4100)
    p.add_argument("--scale-range", type=float, nargs=2, default=None)
    p.add_argument("--fixed-point-range", type=float, default=None)
    p.add_argument("--prob-mode", choices=("det", "file", "mix"), default="det")
    p.add_argument("--prob-alpha", type=float, default=0.0)
    p.add_argument("--prob-seed", type=int, default=777)
    p.add_argument("--obs-num-trajectories", type=int, default=16)
    p.add_argument("--obs-num-steps", type=int, default=1024)
    p.add_argument("--obs-burn-in", type=int, default=128)
    p.add_argument("--obs-sigma", type=float, default=2.0, help="smoothing in units of 128-grid pixels")
    p.add_argument("--obs-resolution", type=int, default=128)
    p.add_argument("--obs-splat", choices=("hist", "bilinear"), default="hist")
    p.add_argument("--obs-seed", type=int, default=91_000)
    p.add_argument("--input-preblur", type=float, default=0.0,
                   help="test-time Gaussian blur (128-grid pixels) applied to every observation before use; 0 = off")
    p.add_argument("--steps", type=int, nargs="+", default=[30])
    p.add_argument("--methods", nargs="+", default=["model", "random"])
    p.add_argument("--random-restarts", type=int, default=4)
    p.add_argument("--refine-renderer", choices=("train", "obs"), default="train")
    p.add_argument("--det-negative-penalty-weight", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-seed", type=int, default=53_100)
    p.add_argument("--chamfer-max-points", type=int, default=2048)
    p.add_argument("--external-preds", action="store_true", help="also score external_predictions stored in --gt-file")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# --------------------------------------------------------------------------- targets


def _affine_from_svd(params: torch.Tensor) -> torch.Tensor:
    w, b = params_to_affine(params)
    return affine_matrices_to_vector(w, b)


def build_targets(args, config) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
    names: list[str] = []
    file_probs = None
    if args.gt_source == "testset":
        cfg = config
        if args.scale_range is not None or args.fixed_point_range is not None:
            from dataclasses import replace

            cfg = replace(
                config,
                scale_range=tuple(args.scale_range) if args.scale_range else config.scale_range,
                fixed_point_range=float(args.fixed_point_range) if args.fixed_point_range else config.fixed_point_range,
            )
        dataset = make_fixed_dataset_from_iterable_config(cfg, num_samples=args.num_samples, seed=args.seed)
        affine = _affine_from_svd(dataset.tensors[1].float())
    elif args.gt_source == "reflect":
        affine = sample_reflection_targets(args, config)
    else:
        payload = torch.load(args.gt_file, map_location="cpu", weights_only=False)
        affine = payload["affine"].float()[: args.num_samples]
        if "probs" in payload:
            file_probs = payload["probs"].float()[: args.num_samples]
        names = list(payload.get("names", []))[: affine.shape[0]]
    return affine, file_probs, names


def sample_reflection_targets(args, config) -> torch.Tensor:
    """Training generator with random singular-value signs, validity-filtered like the test set."""
    generator = torch.Generator().manual_seed(int(args.seed) + 777_001)
    render_gen = torch.Generator().manual_seed(int(args.seed) + 777_002)
    scale_range = tuple(args.scale_range) if args.scale_range else config.scale_range
    fp_range = float(args.fixed_point_range) if args.fixed_point_range else config.fixed_point_range
    kept = []
    total = 0
    while total < args.num_samples:
        params = sample_ifs_parameters_batch(
            256, config.num_transforms, phase=config.phase, scale_range=scale_range,
            fixed_point_range=fp_range, allow_negative_scales=True, generator=generator,
        )
        # Keep only systems that actually contain at least one orientation-reversing map.
        has_reflection = ((params[..., 2] * params[..., 3]) < 0).any(dim=1)
        points = iterate_ifs_points_batch(
            params, num_trajectories=config.validity_num_trajectories or 4,
            num_steps=config.validity_num_steps or 256, burn_in=config.validity_burn_in or 32,
            generator=render_gen,
        )
        density = points_to_density_maps(points, resolution=config.validity_resolution or 32, fixed_range=config.fixed_range)
        ok = valid_sample_mask(points, density, resolution=config.validity_resolution or 32, fixed_range=config.fixed_range)
        ok = ok & has_reflection
        kept.append(params[ok])
        total += int(ok.sum())
    return _affine_from_svd(torch.cat(kept)[: args.num_samples].float())


def build_probs(args, affine: torch.Tensor, file_probs: torch.Tensor | None) -> torch.Tensor:
    det_probs = affine_vector_probabilities(affine)
    if args.prob_mode == "det":
        return det_probs
    if args.prob_mode == "file":
        if file_probs is None:
            raise ValueError("--prob-mode file requires probs in --gt-file")
        return file_probs / file_probs.sum(dim=1, keepdim=True)
    generator = torch.Generator().manual_seed(int(args.prob_seed))
    gamma = torch._standard_gamma(torch.ones(affine.shape[:2]), generator=generator)
    dirichlet = gamma / gamma.sum(dim=1, keepdim=True)
    return (1.0 - args.prob_alpha) * det_probs + args.prob_alpha * dirichlet


# --------------------------------------------------------------------------- rendering


def chaos_points(affine, probs, *, num_trajectories, num_steps, burn_in, generator, chunk=32):
    """Batched chaos game with explicit probabilities. Returns [N, T*(S-burn), 2] on affine.device."""
    out = []
    w_all, b_all = affine_vector_to_matrices(affine)
    for start in range(0, affine.shape[0], chunk):
        w = w_all[start : start + chunk]
        b = b_all[start : start + chunk]
        p = probs[start : start + chunk].to(affine.device)
        batch = w.shape[0]
        seq = torch.multinomial(p.cpu(), num_trajectories * num_steps, replacement=True, generator=generator)
        seq = seq.reshape(batch, num_trajectories, num_steps).to(affine.device)
        current = (torch.rand(batch, num_trajectories, 2, generator=generator) * 2.0 - 1.0).to(affine.device)
        kept = []
        for step in range(num_steps):
            ids = seq[:, :, step]
            sw = torch.gather(w, 1, ids[:, :, None, None].expand(-1, -1, 2, 2))
            sb = torch.gather(b, 1, ids[:, :, None].expand(-1, -1, 2))
            current = torch.matmul(current.unsqueeze(-2), sw).squeeze(-2) + sb
            if step >= burn_in:
                kept.append(current)
        out.append(torch.stack(kept, dim=2).reshape(batch, -1, 2))
    return torch.cat(out)


def points_to_observation(points, *, resolution, sigma128, splat, fixed_range, out_resolution=128):
    sigma = float(sigma128) * float(resolution) / float(out_resolution)  # keep physical blur fixed
    if splat == "hist":
        density = points_to_density_maps(points, resolution=resolution, fixed_range=fixed_range, smoothing_sigma=sigma)
    else:
        density = soft_points_to_density_maps(points, resolution=resolution, fixed_range=fixed_range, smoothing_sigma=sigma)
    if resolution > out_resolution:
        density = F.interpolate(density, size=(out_resolution, out_resolution), mode="area")
    elif resolution < out_resolution:
        density = F.interpolate(density, size=(out_resolution, out_resolution), mode="bilinear", align_corners=False).clamp_min(0.0)
    return density / density.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)


def render_observation(affine, probs, args, *, seed, fixed_range, device):
    generator = torch.Generator().manual_seed(int(seed))
    images = []
    for start in range(0, affine.shape[0], 16):
        pts = chaos_points(
            affine[start : start + 16].to(device), probs[start : start + 16],
            num_trajectories=args.obs_num_trajectories, num_steps=args.obs_num_steps,
            burn_in=args.obs_burn_in, generator=generator,
        )
        images.append(points_to_observation(
            pts, resolution=args.obs_resolution, sigma128=args.obs_sigma, splat=args.obs_splat,
            fixed_range=fixed_range,
        ).cpu())
    return smooth_density_maps(torch.cat(images), sigma=float(args.input_preblur))


def standard_density(affine, probs, *, seed, fixed_range, device):
    generator = torch.Generator().manual_seed(int(seed))
    pts = chaos_points(affine.to(device), probs, num_trajectories=16, num_steps=1024, burn_in=128, generator=generator)
    return points_to_density_maps(pts, resolution=128, fixed_range=fixed_range, smoothing_sigma=2.0).cpu()


def eval_point_clouds(affine, probs, *, seed, fixed_range, device, max_points):
    generator = torch.Generator().manual_seed(int(seed))
    pts = chaos_points(affine.to(device), probs, num_trajectories=16, num_steps=1024, burn_in=128, generator=generator)
    low, high = fixed_range
    clouds = []
    sub_gen = torch.Generator().manual_seed(int(seed) + 1)
    for i in range(pts.shape[0]):
        p = pts[i]
        ok = torch.isfinite(p).all(dim=-1) & (p[:, 0] >= low) & (p[:, 0] <= high) & (p[:, 1] >= low) & (p[:, 1] <= high)
        p = p[ok].cpu()
        if p.shape[0] > max_points:
            p = p[torch.randperm(p.shape[0], generator=sub_gen)[:max_points]]
        clouds.append(p)
    return clouds


# --------------------------------------------------------------------------- methods


def refine_args(args, *, steps, obs_settings: bool) -> SimpleNamespace:
    if obs_settings:
        traj, nsteps, burn = args.obs_num_trajectories, args.obs_num_steps, args.obs_burn_in
        sigma = args.obs_sigma
    else:
        traj, nsteps, burn, sigma = 16, 1024, 128, 2.0
    return SimpleNamespace(
        steps=int(steps), lr=0.005, weight_decay=0.0, grad_clip=1.0, log_interval=0,
        reconstruction_resolution=128, reconstruction_num_trajectories=int(traj),
        reconstruction_num_steps=int(nsteps), reconstruction_burn_in=int(burn),
        reconstruction_smoothing_sigma=float(sigma), reconstruction_seed=12345,
        reconstruction_map_probability_mode="determinant",
        point_chamfer_loss_weight=0.10, point_chamfer_num_pred_points=512,
        point_chamfer_num_target_points=512, point_chamfer_seed=24680,
        spectral_upper_bound=0.75, spectral_penalty_weight=0.1,
        det_negative_penalty_weight=float(args.det_negative_penalty_weight),
        translation_bound=1.5, translation_penalty_weight=0.01,
    )


def run_refinement(images, init_restarts, *, rargs, config, device, batch_size):
    selected = []
    started = time.perf_counter()
    restarts = init_restarts.shape[1]
    per_batch = max(1, batch_size // restarts) if restarts > 1 else batch_size
    for start in range(0, images.shape[0], per_batch):
        end = min(images.shape[0], start + per_batch)
        sel, *_ = optimize_batch(
            images[start:end], None, init_restarts[start:end],
            args=rargs, config=config, device=device, batch_start_index=start,
        )
        selected.append(sel)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(selected), (time.perf_counter() - started) / images.shape[0]


# --------------------------------------------------------------------------- main


def summarize(values):
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(t.mean()), "median": float(t.median()), "p90": float(torch.quantile(t, 0.9))}


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run_dir = ROOT / args.run_dir
    result = json.loads((run_dir / "result.json").read_text())
    config = _config_from_payload(result["train_config"])
    fixed_range = tuple(config.fixed_range)
    n_model = int(result["args"]["num_transforms"])

    affine, file_probs, names = build_targets(args, config)
    probs = build_probs(args, affine, file_probs)
    images = render_observation(affine, probs, args, seed=args.obs_seed, fixed_range=fixed_range, device=device)

    model = _build_model(result["args"], device=device)
    ckpt = _checkpoint_paths(result, run_dir=run_dir, checkpoint=args.checkpoint)[args.checkpoint]
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        pred0 = torch.cat([model(images[i : i + 64].to(device)).cpu() for i in range(0, images.shape[0], 64)])
    t0 = (time.perf_counter() - started) / images.shape[0]

    variants: dict[str, torch.Tensor] = {"truth": affine, "model-0": pred0}
    variant_probs: dict[str, torch.Tensor] = {"truth": probs}
    timing = {"model-0": t0}
    if args.external_preds:
        payload = torch.load(args.gt_file, map_location="cpu", weights_only=False)
        for ext_name, ext in payload.get("external_predictions", {}).items():
            variants[f"ext-{ext_name}"] = ext["affine"][: affine.shape[0]].float()
            variant_probs[f"ext-{ext_name}"] = ext["probs"][: affine.shape[0]].float()
    cfg_ns = SimpleNamespace(fixed_range=fixed_range, phase=config.phase)
    obs_refine = args.refine_renderer == "obs"
    for steps in args.steps:
        rargs = refine_args(args, steps=steps, obs_settings=obs_refine)
        if "model" in args.methods:
            sel, sec = run_refinement(images, pred0[:, None], rargs=rargs, config=cfg_ns, device=device, batch_size=args.batch_size)
            variants[f"model-{steps}"] = sel
            timing[f"model-{steps}"] = sec
        if "random" in args.methods:
            init = make_initial_restarts(
                batch_size=images.shape[0], num_transforms=n_model, phase=config.phase, init_mode="random",
                restarts=args.random_restarts, init_noise=0.0, seed=args.seed + 31_000,
            )
            sel, sec = run_refinement(images, init, rargs=rargs, config=cfg_ns, device=device, batch_size=args.batch_size * args.random_restarts)
            variants[f"random-r{args.random_restarts}-{steps}"] = sel
            timing[f"random-r{args.random_restarts}-{steps}"] = sec

    # Evaluation.
    truth_clouds = eval_point_clouds(affine, probs, seed=args.eval_seed, fixed_range=fixed_range, device=device, max_points=args.chamfer_max_points)
    truth_std = standard_density(affine, probs, seed=args.eval_seed + 5, fixed_range=fixed_range, device=device)
    coverage = {"2px": 2.0 * PIXEL}
    records = []
    for v_index, (name, params) in enumerate(variants.items()):
        p_probs = variant_probs.get(name, affine_vector_probabilities(params))
        clouds = eval_point_clouds(params, p_probs, seed=args.eval_seed + 1000 * (v_index + 1), fixed_range=fixed_range, device=device, max_points=args.chamfer_max_points)
        obs_render = render_observation(params, p_probs, args, seed=args.obs_seed + 7 + v_index, fixed_range=fixed_range, device=device)
        std_render = standard_density(params, p_probs, seed=args.eval_seed + 2000 * (v_index + 1), fixed_range=fixed_range, device=device)
        w, _ = affine_vector_to_matrices(params)
        smax = torch.linalg.svdvals(w).amax(dim=-1)
        for i in range(affine.shape[0]):
            pm = point_metrics(clouds[i], truth_clouds[i], device=device, coverage_thresholds=coverage)
            rec = {
                "variant": name,
                "sample_index": i,
                "name": names[i] if i < len(names) else "",
                "chamfer": pm["chamfer"],
                "hd95": pm["hausdorff_p95"],
                "coverage_2px": pm["coverage_symmetric_2px"],
                "density_sse_obs": float((obs_render[i] - images[i]).square().sum()),
                "density_sse_std": float((std_render[i] - truth_std[i]).square().sum()),
                "smax_max": float(smax[i].max()),
                "num_violating_maps": int((smax[i] >= 1.0).sum()),
            }
            if args.gt_source != "file" and params.shape[1] == affine.shape[1]:
                rec["param_distance"] = affine_set_distance(params[i], affine[i])["param_distance"]
            records.append(rec)

    keys = sorted({k for r in records for k in r}, key=lambda k: list(records[0].keys()).index(k) if k in records[0] else 99)
    with (out / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    summary = {"args": vars(args), "timing_sec_per_sample": timing, "variants": {}}
    for name in variants:
        rows = [r for r in records if r["variant"] == name]
        summary["variants"][name] = {
            metric: summarize([r[metric] for r in rows])
            for metric in ("chamfer", "hd95", "coverage_2px", "density_sse_obs", "density_sse_std", "smax_max")
        }
        if "param_distance" in rows[0]:
            summary["variants"][name]["param_distance"] = summarize([r["param_distance"] for r in rows])
        summary["variants"][name]["frac_systems_violating"] = sum(r["num_violating_maps"] > 0 for r in rows) / len(rows)
    # Per-sample wins of model-K over random-K on Chamfer.
    wins = {}
    for steps in args.steps:
        a, b = f"model-{steps}", f"random-r{args.random_restarts}-{steps}"
        if a in variants and b in variants:
            ca = [r["chamfer"] for r in records if r["variant"] == a]
            cb = [r["chamfer"] for r in records if r["variant"] == b]
            wins[f"{a}_vs_{b}_chamfer"] = sum(x < y for x, y in zip(ca, cb))
    summary["wins"] = wins
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"images": images, "probs": probs, "names": names, "variants": variants}, out / "outputs.pt")
    for name, s in summary["variants"].items():
        print(f"{name:16s} CD={s['chamfer']['mean']:.4f} HD95={s['hd95']['mean']:.4f} cov={s['coverage_2px']['mean']:.3f} "
              f"SSEobs={s['density_sse_obs']['mean']:.2e} SSEstd={s['density_sse_std']['mean']:.2e} "
              + (f"param={s['param_distance']['mean']:.3f}" if 'param_distance' in s else ""))
    print("wins", wins, "timing", {k: round(v, 4) for k, v in timing.items()})


if __name__ == "__main__":
    main()
