"""P0 MNIST/Fashion-MNIST OOD pilot for trained amortized IFS models.

The script evaluates a trained direct-affine model on a small balanced image
pilot set, then runs the same image reconstruction refinement objective used in
the synthetic E1 experiments.  These image datasets have no ground-truth IFS
parameters, so all metrics are reconstruction metrics against the input image
and image-sampled point clouds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from data.renderer import render_density_from_affine_vector
from losses.reconstruction import density_images_to_point_samples, target_density_for_reconstruction
from scripts.analyze_checkpoint_errors import predict_all
from scripts.evaluate_checkpoint import _build_model, _checkpoint_paths, _config_from_payload
from scripts.evaluate_oracle_fidelity import _coverage_thresholds, _filter_and_subsample, point_metrics
from scripts.optimize_oracle import (
    apply_reconstruction_match_render_config,
    make_initial_restarts,
    optimize_batch,
)

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/paper_base50k/20260614_224608")
    parser.add_argument("--checkpoint", choices=("final", "best"), default="best")
    parser.add_argument("--dataset", choices=("mnist", "fashion-mnist"), default="mnist")
    parser.add_argument("--data-root", default="refs/LearningFractals/data")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--selection", choices=("balanced", "first"), default="balanced")
    parser.add_argument("--mnist-resolution", type=int, default=32)
    parser.add_argument("--steps", type=int, nargs="+", default=(0, 10, 20, 30))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--reconstruction-resolution", type=int, default=128)
    parser.add_argument("--reconstruction-num-trajectories", type=int, default=16)
    parser.add_argument("--reconstruction-num-steps", type=int, default=1024)
    parser.add_argument("--reconstruction-burn-in", type=int, default=128)
    parser.add_argument("--reconstruction-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--reconstruction-seed", type=int, default=12345)
    parser.add_argument("--reconstruction-map-probability-mode", choices=("uniform", "determinant"), default="determinant")
    parser.add_argument("--reconstruction-match-render-config", action="store_true", default=True)
    parser.add_argument("--point-chamfer-loss-weight", type=float, default=0.10)
    parser.add_argument("--point-chamfer-num-pred-points", type=int, default=512)
    parser.add_argument("--point-chamfer-num-target-points", type=int, default=512)
    parser.add_argument("--point-chamfer-seed", type=int, default=24680)
    parser.add_argument("--spectral-upper-bound", type=float, default=0.75)
    parser.add_argument("--spectral-penalty-weight", type=float, default=0.1)
    parser.add_argument("--det-negative-penalty-weight", type=float, default=0.1)
    parser.add_argument("--translation-bound", type=float, default=1.5)
    parser.add_argument("--translation-penalty-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--fidelity-resolution", type=int, default=128)
    parser.add_argument("--fidelity-num-trajectories", type=int, default=16)
    parser.add_argument("--fidelity-num-steps", type=int, default=1024)
    parser.add_argument("--fidelity-burn-in", type=int, default=128)
    parser.add_argument("--fidelity-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--fidelity-seed", type=int, default=53100)
    parser.add_argument("--chamfer-max-points", type=int, default=2048)
    parser.add_argument("--coverage-pixel-thresholds", type=float, nargs="+", default=(1.0, 2.0, 4.0))
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/mnist_p0/base50k_best_n4_test10")
    return parser.parse_args()


def dataset_class(name: str):
    if name == "mnist":
        return datasets.MNIST
    if name == "fashion-mnist":
        return datasets.FashionMNIST
    raise ValueError(f"unsupported dataset: {name}")


def select_indices(dataset, *, num_samples: int, selection: str) -> list[int]:
    if selection == "first":
        return list(range(num_samples))
    by_digit: dict[int, list[int]] = {digit: [] for digit in range(10)}
    targets = [int(value) for value in dataset.targets]
    for index, label in enumerate(targets):
        if label in by_digit:
            by_digit[label].append(index)
    indices: list[int] = []
    while len(indices) < num_samples:
        added = False
        for digit in range(10):
            bucket = by_digit[digit]
            offset = len([idx for idx in indices if targets[idx] == digit])
            if offset < len(bucket):
                indices.append(bucket[offset])
                added = True
                if len(indices) >= num_samples:
                    break
        if not added:
            break
    if len(indices) < num_samples:
        raise RuntimeError(f"could only select {len(indices)} samples")
    return indices


def load_images(args: argparse.Namespace, *, model_resolution: int) -> tuple[torch.Tensor, list[int], list[int]]:
    transform = transforms.Compose(
        [
            transforms.Resize(args.mnist_resolution),
            transforms.ToTensor(),
        ]
    )
    dataset = dataset_class(args.dataset)(
        args.data_root,
        train=False,
        download=True,
        transform=transform,
    )
    indices = select_indices(dataset, num_samples=args.num_samples, selection=args.selection)
    images = []
    labels = []
    for index in indices:
        image, label = dataset[index]
        images.append(image)
        labels.append(int(label))
    batch = torch.stack(images).float()
    if batch.shape[-2:] != (model_resolution, model_resolution):
        batch = F.interpolate(batch, size=(model_resolution, model_resolution), mode="bilinear", align_corners=False)
    batch = batch / batch.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return batch.contiguous(), labels, indices


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_variant(
    params: torch.Tensor,
    *,
    args: argparse.Namespace,
    sample_index: int,
    variant_offset: int,
    fixed_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=params.device).manual_seed(
        int(args.fidelity_seed) + 10_003 * int(sample_index) + int(variant_offset)
    )
    return render_density_from_affine_vector(
        params,
        resolution=args.fidelity_resolution,
        fixed_range=fixed_range,
        num_trajectories=args.fidelity_num_trajectories,
        num_steps=args.fidelity_num_steps,
        burn_in=args.fidelity_burn_in,
        density_smoothing_sigma=args.fidelity_smoothing_sigma,
        generator=generator,
    )


def image_points_for_display(points: torch.Tensor, *, fixed_range: tuple[float, float]) -> torch.Tensor:
    """Convert row/column image-coordinate points to an upright Cartesian display."""
    low, high = fixed_range
    x = points[:, 1]
    y = float(low) + float(high) - points[:, 0]
    return torch.stack([x, y], dim=-1)


def evaluate_variants(
    images: torch.Tensor,
    params_by_label: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    fixed_range: tuple[float, float],
    device: torch.device,
) -> tuple[list[dict], dict[str, dict[str, dict[str, float]]]]:
    target_density = target_density_for_reconstruction(images, resolution=args.fidelity_resolution)
    coverage = _coverage_thresholds(
        fixed_range=fixed_range,
        resolution=args.fidelity_resolution,
        pixel_thresholds=args.coverage_pixel_thresholds,
    )
    records: list[dict] = []
    for sample_index in range(images.shape[0]):
        target_points = density_images_to_point_samples(
            target_density[sample_index : sample_index + 1],
            resolution=args.fidelity_resolution,
            fixed_range=fixed_range,
            num_points=args.chamfer_max_points,
            seed=args.fidelity_seed + 71_000 + sample_index,
        )[0]
        for variant_index, (label, params) in enumerate(params_by_label.items()):
            points, density = render_variant(
                params[sample_index].cpu(),
                args=args,
                sample_index=sample_index,
                variant_offset=17_000 * (variant_index + 1),
                fixed_range=fixed_range,
            )
            generator = torch.Generator(device="cpu").manual_seed(
                args.fidelity_seed + 91_000 + 1009 * variant_index + sample_index
            )
            sampled_points = _filter_and_subsample(
                points,
                fixed_range=fixed_range,
                max_points=args.chamfer_max_points,
                generator=generator,
            )
            pmetrics = point_metrics(
                sampled_points,
                target_points,
                device=device,
                coverage_thresholds=coverage,
            )
            diff = density.float() - target_density[sample_index].cpu().float()
            record = {
                "sample_index": int(sample_index),
                "variant": label,
                "density_sse_to_input": float(diff.square().sum().item()),
                "density_l1_to_input": float(diff.abs().sum().item()),
                "num_points": int(points.shape[0]),
                "num_points_used": int(sampled_points.shape[0]),
            }
            for key, value in pmetrics.items():
                record[f"{key}_to_image_points"] = float(value)
            records.append(record)

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    metric_keys = [key for key in records[0] if key not in {"sample_index", "variant"}]
    for label in params_by_label:
        selected = [record for record in records if record["variant"] == label]
        summaries[label] = {
            key: summarize([float(record[key]) for record in selected])
            for key in metric_keys
        }
    return records, summaries


def save_density_examples(
    images: torch.Tensor,
    params_by_label: dict[str, torch.Tensor],
    labels: list[int],
    *,
    args: argparse.Namespace,
    fixed_range: tuple[float, float],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    count = min(args.plot_samples, images.shape[0])
    target_density = target_density_for_reconstruction(images[:count], resolution=args.fidelity_resolution)
    row_labels = ["target"] + list(params_by_label.keys())
    fig, axes = plt.subplots(len(row_labels), count, figsize=(2.0 * count, 2.0 * len(row_labels)), squeeze=False)
    for col in range(count):
        axes[0, col].imshow(target_density[col, 0].cpu(), cmap="magma", origin="upper")
        axes[0, col].set_title(f"digit {labels[col]}", fontsize=8)
        axes[0, col].axis("off")
    for row, label in enumerate(params_by_label, start=1):
        for col in range(count):
            _, density = render_variant(
                params_by_label[label][col].cpu(),
                args=args,
                sample_index=col,
                variant_offset=17_000 * row,
                fixed_range=fixed_range,
            )
            axes[row, col].imshow(density[0].cpu(), cmap="magma", origin="upper")
            axes[row, col].set_title(label, fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "mnist_density_examples.png", dpi=360)
    fig.savefig(output_dir / "mnist_density_examples.pdf")
    plt.close(fig)


def save_point_examples(
    images: torch.Tensor,
    params_by_label: dict[str, torch.Tensor],
    labels: list[int],
    *,
    args: argparse.Namespace,
    fixed_range: tuple[float, float],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    shown_labels = [label for label in ("model-0", "model-30") if label in params_by_label]
    if not shown_labels:
        shown_labels = list(params_by_label.keys())[:2]
    count = min(args.plot_samples, images.shape[0])
    target_density = target_density_for_reconstruction(images[:count], resolution=args.fidelity_resolution)
    rows = ["target"] + shown_labels
    fig, axes = plt.subplots(len(rows), count, figsize=(2.0 * count, 2.0 * len(rows)), squeeze=False)
    low, high = fixed_range
    for col in range(count):
        target_points = density_images_to_point_samples(
            target_density[col : col + 1],
            resolution=args.fidelity_resolution,
            fixed_range=fixed_range,
            num_points=args.chamfer_max_points,
            seed=args.fidelity_seed + 131_000 + col,
        )[0]
        display_points = image_points_for_display(target_points, fixed_range=fixed_range)
        axes[0, col].scatter(display_points[:, 0], display_points[:, 1], s=2.5, c="#111827", alpha=0.55)
        axes[0, col].set_title(f"target {labels[col]}", fontsize=8)
        axes[0, col].set_xlim(low, high)
        axes[0, col].set_ylim(low, high)
        axes[0, col].set_aspect("equal")
        axes[0, col].axis("off")
        for row, label in enumerate(shown_labels, start=1):
            points, _ = render_variant(
                params_by_label[label][col].cpu(),
                args=args,
                sample_index=col,
                variant_offset=19_000 * row,
                fixed_range=fixed_range,
            )
            generator = torch.Generator(device="cpu").manual_seed(args.fidelity_seed + 151_000 + row * 1009 + col)
            points = _filter_and_subsample(
                points,
                fixed_range=fixed_range,
                max_points=args.chamfer_max_points,
                generator=generator,
            )
            display_points = image_points_for_display(points, fixed_range=fixed_range)
            axes[row, col].scatter(display_points[:, 0], display_points[:, 1], s=2.5, c="#2563eb", alpha=0.55)
            axes[row, col].set_title(label, fontsize=8)
            axes[row, col].set_xlim(low, high)
            axes[row, col].set_ylim(low, high)
            axes[row, col].set_aspect("equal")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "mnist_point_examples.png", dpi=360)
    fig.savefig(output_dir / "mnist_point_examples.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    run_dir = Path(args.run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    args_payload = result["args"]
    if args_payload["target_representation"] != "affine":
        raise ValueError("MNIST P0 expects a direct-affine checkpoint")
    config = _config_from_payload(result["train_config"])
    apply_reconstruction_match_render_config(args, config)
    fixed_range = tuple(config.fixed_range)

    started = time.perf_counter()
    images, labels, mnist_indices = load_images(args, model_resolution=int(config.resolution))
    dataset_elapsed = time.perf_counter() - started
    dataset = TensorDataset(images, torch.zeros(images.shape[0], int(args_payload["num_transforms"]), 6))

    model = _build_model(args_payload, device=device)
    checkpoint_path = _checkpoint_paths(result, run_dir=run_dir, checkpoint=args.checkpoint)[args.checkpoint]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_predictions = predict_all(
        model,
        dataset,
        batch_size=int(args_payload["eval_batch_size"]),
        device=device,
    )

    params_by_label: dict[str, torch.Tensor] = {}
    histories_by_step: dict[str, list[dict]] = {}
    timings_by_step: dict[str, float] = {}
    num_transforms = int(args_payload["num_transforms"])
    for steps in args.steps:
        step_args = SimpleNamespace(**vars(args))
        step_args.steps = int(steps)
        step_args.init_mode = "model"
        step_args.restarts = 1
        selected_batches = []
        histories = []
        opt_started = time.perf_counter()
        for start in range(0, images.shape[0], args.batch_size):
            end = min(images.shape[0], start + args.batch_size)
            init_restarts = make_initial_restarts(
                model_init=model_predictions[start:end],
                target_init=None,
                batch_size=end - start,
                num_transforms=num_transforms,
                phase=config.phase,
                init_mode="model",
                restarts=1,
                init_noise=0.0,
                seed=4100 + 31_000 + start,
            )
            selected, _, _, _, history = optimize_batch(
                images[start:end],
                torch.zeros(end - start, num_transforms, 6),
                init_restarts,
                args=step_args,
                config=config,
                device=device,
                batch_start_index=start,
            )
            selected_batches.append(selected)
            histories.extend(history)
        timings_by_step[f"model-{steps}"] = (time.perf_counter() - opt_started) / max(1, images.shape[0])
        histories_by_step[f"model-{steps}"] = histories
        params_by_label[f"model-{steps}"] = torch.cat(selected_batches, dim=0)

    records, summaries = evaluate_variants(
        images,
        params_by_label,
        args=args,
        fixed_range=fixed_range,
        device=device,
    )
    for label in summaries:
        summaries[label]["seconds_per_sample"] = {
            "mean": timings_by_step[label],
            "median": timings_by_step[label],
            "p90": timings_by_step[label],
            "p95": timings_by_step[label],
            "min": timings_by_step[label],
            "max": timings_by_step[label],
        }

    rows = []
    for label, summary in summaries.items():
        rows.append(
            {
                "label": label,
                "seconds_per_sample": timings_by_step[label],
                "density_sse": summary["density_sse_to_input"]["mean"],
                "chamfer": summary["chamfer_to_image_points"]["mean"],
                "hd95": summary["hausdorff_p95_to_image_points"]["mean"],
                "coverage_1px": summary["coverage_symmetric_1px_to_image_points"]["mean"],
                "coverage_2px": summary["coverage_symmetric_2px_to_image_points"]["mean"],
                "coverage_4px": summary["coverage_symmetric_4px_to_image_points"]["mean"],
                "modified_hausdorff_mean": summary["modified_hausdorff_mean_to_image_points"]["mean"],
            }
        )
    write_csv(rows, output_dir / "summary.csv")
    write_csv(records, output_dir / "per_sample_metrics.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint": args.checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "dataset": args.dataset,
                "mnist_indices": mnist_indices,
                "mnist_labels": labels,
                "dataset_elapsed_sec": dataset_elapsed,
                "timings_by_step": timings_by_step,
                "summaries": summaries,
                "args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "images": images,
            "labels": labels,
            "mnist_indices": mnist_indices,
            "dataset": args.dataset,
            "params_by_label": params_by_label,
            "summary_rows": rows,
        },
        output_dir / "mnist_p0_outputs.pt",
    )
    save_density_examples(images, params_by_label, labels, args=args, fixed_range=fixed_range, output_dir=output_dir)
    save_point_examples(images, params_by_label, labels, args=args, fixed_range=fixed_range, output_dir=output_dir)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
