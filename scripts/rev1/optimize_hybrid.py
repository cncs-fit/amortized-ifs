"""I7-a1: per-image optimization with Djeacoumar et al. (2025)-style hybrid search.

Gradient descent on the paper's refinement objective (density SSE + 0.10
Chamfer + feasibility penalties, matched renderer) interleaved with
simulated-annealing candidate search, following the official trainer
(github.com/sshadr/chaotic-fractals, base/trainers/trainer.py):

* every ``sa_interval`` steps during the first ``sa_fraction`` of the budget,
  ``sa_candidates`` candidates are drawn as Gaussian perturbations of the current
  state with std ``sa_noise * T``, T = (steps - step) / steps;
* a candidate is accepted if it lowers the energy or with Metropolis probability
  exp(-metropolis_scale * dE / T); accepted candidates become the new search state;
* the last accepted state replaces the parameters and the optimizer is reset.

Differences from the official code, stated for the paper: the perturbation acts
on our raw affine parameters (W, b) instead of their QR/SVD parameterization, and
the energy change is measured relative to the current energy (dE / E) because our
objective is several orders of magnitude smaller than their image loss.

Output directory layout matches ``scripts/optimize_oracle.py`` so that
``scripts/evaluate_oracle_fidelity.py`` can score it unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.dataset import make_fixed_dataset_from_iterable_config
from losses.reconstruction import density_images_to_point_samples
from scripts.evaluate_checkpoint import _build_model, _checkpoint_paths, _config_from_payload
from scripts.optimize_oracle import make_initial_restarts, objective_losses, structural_penalties


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="checkpoints/paper_base50k")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=4100)
    p.add_argument("--init-mode", choices=("model", "random"), default="random")
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64, help="systems per batch (restarts folded in)")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--sa-interval", type=int, default=None, help="default: steps // 60 (official: 15000 // 60 = 250)")
    p.add_argument("--sa-candidates", type=int, default=10)
    p.add_argument("--sa-noise", type=float, default=0.2)
    p.add_argument("--sa-fraction", type=float, default=0.5)
    p.add_argument("--metropolis-scale", type=float, default=10.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def objective_settings() -> SimpleNamespace:
    return SimpleNamespace(
        reconstruction_resolution=128, reconstruction_num_trajectories=16, reconstruction_num_steps=1024,
        reconstruction_burn_in=128, reconstruction_smoothing_sigma=2.0, reconstruction_seed=12345,
        reconstruction_map_probability_mode="determinant", point_chamfer_loss_weight=0.10,
        point_chamfer_num_pred_points=512, point_chamfer_num_target_points=512, point_chamfer_seed=24680,
    )


def energy(params, images, target_points, *, ns, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    recon, chamfer = objective_losses(params, images, args=ns, config=cfg, target_points=target_points)
    pen = structural_penalties(
        params, spectral_upper_bound=0.75, spectral_penalty_weight=0.1, det_negative_penalty_weight=0.1,
        translation_bound=1.5, translation_penalty_weight=0.01,
    )
    return recon + chamfer + pen, recon


def optimize_hybrid_batch(images, init, *, args, ns, cfg, device, batch_start, generator):
    batch, restarts, n, _ = init.shape
    images = images.to(device)
    tgt = images[:, None].expand(-1, restarts, -1, -1, -1).reshape(batch * restarts, *images.shape[1:])
    target_points = density_images_to_point_samples(
        tgt, resolution=128, fixed_range=cfg.fixed_range, num_points=512, seed=24680 + int(batch_start),
    )
    params = torch.nn.Parameter(init.reshape(batch * restarts, n, 6).to(device))
    opt = torch.optim.AdamW([params], lr=args.lr, weight_decay=0.0)
    interval = args.sa_interval or max(1, args.steps // 60)
    sa_events = 0
    accepted = 0
    for step in range(1, args.steps + 1):
        total, _ = energy(params, tgt, target_points, ns=ns, cfg=cfg)
        opt.zero_grad(set_to_none=True)
        total.mean().backward()
        torch.nn.utils.clip_grad_norm_([params], 1.0)
        opt.step()
        if step % interval == 0 and step < int(args.steps * args.sa_fraction):
            temperature = (args.steps - step) / args.steps
            with torch.no_grad():
                current = params.detach().clone()
                search_state = current.clone()
                search_energy, _ = energy(search_state, tgt, target_points, ns=ns, cfg=cfg)
                for _ in range(args.sa_candidates):
                    noise = torch.randn(current.shape, generator=generator).to(device) * (args.sa_noise * temperature)
                    candidate = current + noise
                    cand_energy, _ = energy(candidate, tgt, target_points, ns=ns, cfg=cfg)
                    rel = (cand_energy - search_energy) / search_energy.clamp_min(1e-12)
                    metropolis = torch.exp(-args.metropolis_scale * rel / max(temperature, 1e-6))
                    accept = (rel <= 0) | (torch.rand(rel.shape, generator=generator).to(device) < metropolis)
                    search_state[accept] = candidate[accept]
                    search_energy = torch.where(accept, cand_energy, search_energy)
                    accepted += int(accept.sum())
                params.data.copy_(search_state)
            opt = torch.optim.AdamW([params], lr=args.lr, weight_decay=0.0)
            sa_events += 1
    with torch.no_grad():
        final_total, final_recon = energy(params, tgt, target_points, ns=ns, cfg=cfg)
        best = final_total.reshape(batch, restarts).argmin(dim=1).cpu()
        final = params.detach().reshape(batch, restarts, n, 6).cpu()
        selected = final[torch.arange(batch), best]
    return selected, final_recon.reshape(batch, restarts).cpu(), sa_events, accepted


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run_dir = ROOT / args.run_dir
    result = json.loads((run_dir / "result.json").read_text())
    config = _config_from_payload(result["train_config"])
    dataset = make_fixed_dataset_from_iterable_config(config, num_samples=args.num_samples, seed=args.seed)
    images, target_params = dataset.tensors
    n = int(result["args"]["num_transforms"])
    model_pred = None
    if args.init_mode == "model":
        model = _build_model(result["args"], device=device)
        model.load_state_dict(torch.load(_checkpoint_paths(result, run_dir=run_dir, checkpoint=args.checkpoint)[args.checkpoint], map_location=device))
        model.eval()
        with torch.no_grad():
            model_pred = torch.cat([model(images[i : i + 64].to(device)).cpu() for i in range(0, images.shape[0], 64)])
    ns = objective_settings()
    cfg = SimpleNamespace(fixed_range=tuple(config.fixed_range), phase=config.phase)
    generator = torch.Generator().manual_seed(args.seed + 424_242)
    restarts = 1 if args.init_mode == "model" else args.restarts
    per_batch = max(1, args.batch_size // restarts)
    selected_all, init_first = [], []
    sa_events = accepted = 0
    started = time.perf_counter()
    for start in range(0, images.shape[0], per_batch):
        end = min(images.shape[0], start + per_batch)
        init = make_initial_restarts(
            model_init=None if model_pred is None else model_pred[start:end],
            batch_size=end - start, num_transforms=n, phase=config.phase, init_mode=args.init_mode,
            restarts=restarts, init_noise=0.0, seed=args.seed + 31_000 + start,
        )
        sel, _, events, acc = optimize_hybrid_batch(images[start:end], init, args=args, ns=ns, cfg=cfg, device=device, batch_start=start, generator=generator)
        selected_all.append(sel)
        init_first.append(init[:, 0])
        sa_events += events
        accepted += acc
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    torch.save(
        {
            "oracle_params": torch.cat(selected_all),
            "init_params": torch.cat(init_first),
            "target_params": target_params,
            "images": images,
            "summary": {},
        },
        out / "oracle_outputs.pt",
    )
    summary = {
        "args": vars(args),
        "seconds_per_sample": elapsed / images.shape[0],
        "sa_events_per_batch": sa_events / max(1, len(selected_all)),
        "accepted_candidates": accepted,
        "note": "wall-clock times depend on other load on the GPU",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
