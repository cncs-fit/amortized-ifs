"""Tu32-objective refinement from amortized MNIST/Fashion-MNIST IFS predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision import datasets, transforms

from data.sampler import affine_vector_probabilities, affine_vector_to_matrices
from scripts.evaluate_mnist_common_rendering import dataset_class, resolve_dataset_name, rescale_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-output", default="outputs/mnist_p0/base50k_best_n4_balanced10/mnist_p0_outputs.pt")
    parser.add_argument("--dataset", choices=("auto", "mnist", "fashion-mnist"), default="auto")
    parser.add_argument("--data-root", default="refs/LearningFractals/data")
    parser.add_argument("--output-dir", default="outputs/mnist_tu32_refine/base50k_best_n4_balanced10")
    parser.add_argument("--init-variant", default="model-0")
    parser.add_argument("--steps", type=int, nargs="+", default=(30, 100))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-sequences", type=int, default=50)
    parser.add_argument("--num-coords", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--source-range", type=float, nargs=2, default=(-1.5, 1.5))
    parser.add_argument("--target-range", type=float, nargs=2, default=(-5.0, 5.0))
    parser.add_argument("--spectral-upper-bound", type=float, default=0.98)
    parser.add_argument("--spectral-penalty-weight", type=float, default=0.01)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--translation-penalty-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=82000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_targets(*, dataset_name: str, data_root: str, mnist_indices: list[int], image_size: int) -> torch.Tensor:
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
    return torch.stack([dataset[index][0].float() for index in mnist_indices], dim=0)


def fixed_points_first_map(params: torch.Tensor) -> torch.Tensor:
    w, b = affine_vector_to_matrices(params)
    eye = torch.eye(2, dtype=params.dtype, device=params.device).view(1, 2, 2).expand(params.shape[0], -1, -1)
    try:
        return torch.linalg.solve((eye - w[:, 0].transpose(-1, -2)).float(), b[:, 0].float()).to(dtype=params.dtype)
    except RuntimeError:
        return torch.zeros(params.shape[0], 2, dtype=params.dtype, device=params.device)


def sample_sequences(
    params: torch.Tensor,
    *,
    num_sequences: int,
    num_coords: int,
    seed: int,
) -> torch.Tensor:
    probs = affine_vector_probabilities(params.detach(), prob_floor=0.0)
    bad = (~torch.isfinite(probs).all(dim=1)) | (probs.sum(dim=1) <= 0)
    if bad.any():
        probs = probs.clone()
        probs[bad] = 1.0 / float(probs.shape[1])
    generator = torch.Generator(device=params.device).manual_seed(int(seed))
    seqs = [
        torch.multinomial(prob, num_sequences * num_coords, replacement=True, generator=generator).reshape(
            num_sequences, num_coords
        )
        for prob in probs
    ]
    return torch.stack(seqs, dim=0)


def differentiable_tu32_images(
    params: torch.Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
) -> torch.Tensor:
    batch_size = params.shape[0]
    w, b = affine_vector_to_matrices(params)
    with torch.no_grad():
        seqs = sample_sequences(
            params,
            num_sequences=args.num_sequences,
            num_coords=args.num_coords,
            seed=seed,
        )
        start = fixed_points_first_map(params.detach())
    current = start[:, None, :].expand(batch_size, args.num_sequences, 2).contiguous()
    coords = []
    for step in range(args.num_coords):
        ids = seqs[:, :, step]
        selected_w = torch.gather(w, dim=1, index=ids[:, :, None, None].expand(-1, -1, 2, 2))
        selected_b = torch.gather(b, dim=1, index=ids[:, :, None].expand(-1, -1, 2))
        current = torch.matmul(current.unsqueeze(-2), selected_w).squeeze(-2) + selected_b
        coords.append(current)
    points = torch.stack(coords, dim=2).reshape(batch_size * args.num_sequences, args.num_coords, 2)
    points = rescale_points(points, source_range=tuple(args.source_range), target_range=tuple(args.target_range))
    low, high = tuple(float(value) for value in args.target_range)
    size = int(args.image_size)
    norm = ((points - low) / (high - low)).clamp(0.0, 1.0) * float(size - 1)
    grid = torch.arange(size, device=params.device, dtype=params.dtype)
    x_coords = grid.view(1, 1, size, 1)
    y_coords = grid.view(1, 1, 1, size)
    dist = (norm[:, :, 0].unsqueeze(-1).unsqueeze(-1) - x_coords).square()
    dist = dist + (norm[:, :, 1].unsqueeze(-1).unsqueeze(-1) - y_coords).square()
    images = torch.exp(-dist / float(args.sigma)).sum(dim=1).clamp(0.0, 1.0)
    return images.view(batch_size, args.num_sequences, 1, size, size)


def penalties(params: torch.Tensor, *, args: argparse.Namespace) -> torch.Tensor:
    w, b = affine_vector_to_matrices(params)
    spectral = torch.linalg.svdvals(w).amax(dim=-1)
    spectral_penalty = torch.relu(spectral - float(args.spectral_upper_bound)).square().mean()
    translation_penalty = torch.relu(b.abs() - float(args.translation_bound)).square().mean()
    return float(args.spectral_penalty_weight) * spectral_penalty + float(args.translation_penalty_weight) * translation_penalty


def optimize_to_steps(
    init_params: torch.Tensor,
    targets: torch.Tensor,
    *,
    args: argparse.Namespace,
    batch_start: int,
) -> tuple[dict[int, torch.Tensor], list[dict], float]:
    device = targets.device
    max_steps = max(args.steps)
    params = init_params.to(device=device, dtype=torch.float32).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=float(args.lr), weight_decay=float(args.weight_decay))
    wanted = set(int(step) for step in args.steps)
    saved: dict[int, torch.Tensor] = {0: params.detach().cpu()} if 0 in wanted else {}
    history: list[dict] = []
    started = time.perf_counter()
    target_seq = targets[:, None].expand(-1, args.num_sequences, -1, -1, -1)
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        images = differentiable_tu32_images(params, args=args, seed=args.seed + 1009 * step + batch_start)
        mse = (images - target_seq).square().mean()
        penalty = penalties(params, args=args)
        loss = mse + penalty
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([params], float(args.grad_clip))
        optimizer.step()
        history.append(
            {
                "batch_start": int(batch_start),
                "step": int(step),
                "loss": float(loss.detach().cpu().item()),
                "mse": float(mse.detach().cpu().item()),
                "penalty": float(penalty.detach().cpu().item()),
            }
        )
        if step in wanted:
            saved[step] = params.detach().cpu()
    return saved, history, (time.perf_counter() - started) / max(1, init_params.shape[0])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    payload = torch.load(args.ours_output, map_location="cpu", weights_only=False)
    if args.init_variant not in payload["params_by_label"]:
        raise KeyError(f"missing init variant {args.init_variant}")
    labels = [int(value) for value in payload["labels"]]
    mnist_indices = [int(value) for value in payload["mnist_indices"]]
    dataset_name = resolve_dataset_name(args, payload)
    init_params = payload["params_by_label"][args.init_variant].float()
    targets = load_targets(
        dataset_name=dataset_name,
        data_root=args.data_root,
        mnist_indices=mnist_indices,
        image_size=args.image_size,
    )

    saved_by_step: dict[int, list[torch.Tensor]] = {int(step): [] for step in args.steps}
    histories: list[dict] = []
    timing_rows: list[dict] = []
    for start in range(0, init_params.shape[0], args.batch_size):
        end = min(init_params.shape[0], start + args.batch_size)
        saved, history, sec_per_sample = optimize_to_steps(
            init_params[start:end],
            targets[start:end].to(device=device),
            args=args,
            batch_start=start,
        )
        histories.extend(history)
        for step in args.steps:
            saved_by_step[int(step)].append(saved[int(step)])
            timing_rows.append(
                {
                    "variant": f"tu32gd-{args.init_variant}-{int(step)}",
                    "batch_start": start,
                    "batch_end": end,
                    "seconds_per_sample": sec_per_sample,
                }
            )

    params_by_label = dict(payload["params_by_label"])
    for step, chunks in saved_by_step.items():
        params_by_label[f"tu32gd-{args.init_variant}-{step}"] = torch.cat(chunks, dim=0)

    out_payload = {
        **payload,
        "params_by_label": params_by_label,
        "tu32gd_args": vars(args),
        "dataset": dataset_name,
    }
    torch.save(out_payload, output_dir / "mnist_p0_outputs_with_tu32gd.pt")
    write_csv(histories, output_dir / "tu32gd_history.csv")
    write_csv(timing_rows, output_dir / "tu32gd_timings.csv")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "ours_output": args.ours_output,
                "dataset": dataset_name,
                "output": str(output_dir / "mnist_p0_outputs_with_tu32gd.pt"),
                "labels": labels,
                "mnist_indices": mnist_indices,
                "added_variants": [f"tu32gd-{args.init_variant}-{int(step)}" for step in args.steps],
                "args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_dir / "mnist_p0_outputs_with_tu32gd.pt"),
                "added_variants": [f"tu32gd-{args.init_variant}-{int(step)}" for step in args.steps],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
