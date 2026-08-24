"""Evaluate saved Phase 0/1 checkpoints on newly generated fixed sets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from data.dataset import IFSIterableDatasetConfig, make_fixed_dataset_from_iterable_config
from models.set_head import TinyCNNAffineSetEstimator, TinyCNNSetEstimator
from scripts.train_phase0 import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--checkpoint", choices=("final", "best", "both"), default="both")
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--val-seed", type=int, default=None)
    parser.add_argument("--test-seed", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def _resolve_path(path_value: str | None, *, run_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    candidates = [path, ROOT / path, run_dir / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _config_from_payload(payload: dict) -> IFSIterableDatasetConfig:
    field_names = {field.name for field in fields(IFSIterableDatasetConfig)}
    kwargs = {key: value for key, value in payload.items() if key in field_names}
    if "fixed_range" in kwargs:
        kwargs["fixed_range"] = tuple(kwargs["fixed_range"])
    if "scale_range" in kwargs:
        kwargs["scale_range"] = tuple(kwargs["scale_range"])
    return IFSIterableDatasetConfig(**kwargs)


def _build_model(args_payload: dict, *, device: torch.device) -> torch.nn.Module:
    args = SimpleNamespace(**args_payload)
    common = {
        "num_transforms": args.num_transforms,
        "hidden_dim": args.hidden_dim,
        "pool_grid": args.pool_grid,
        "encoder_type": args.encoder_type,
        "coord_channels": args.coord_channels,
        "density_feature_mode": args.density_feature_mode,
        "global_moments": args.global_moments,
        "head_type": args.head_type,
        "query_num_heads": args.query_num_heads,
        "query_layers": args.query_layers,
    }
    if args.target_representation == "affine":
        return TinyCNNAffineSetEstimator(**common).to(device)
    return TinyCNNSetEstimator(**common).to(device)


def _checkpoint_paths(result: dict, *, run_dir: Path, checkpoint: str) -> dict[str, Path]:
    paths = {}
    if checkpoint in ("final", "both"):
        final_path = _resolve_path(result.get("final_model_path"), run_dir=run_dir)
        if final_path is None:
            final_path = run_dir / "model.pt"
        paths["final"] = final_path
    if checkpoint in ("best", "both"):
        best_path = _resolve_path(result.get("best_model_path"), run_dir=run_dir)
        if best_path is not None:
            paths["best"] = best_path
    return paths


def evaluate_run(
    run_dir: Path,
    *,
    checkpoint: str,
    val_samples: int,
    test_samples: int,
    val_seed: int | None,
    test_seed: int | None,
    eval_batch_size: int | None,
    device: torch.device,
    output_name: str | None,
) -> dict:
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    args_payload = result["args"]
    train_config = _config_from_payload(result["train_config"])
    val_seed = int(args_payload["val_seed"] if val_seed is None else val_seed)
    test_seed = int(args_payload["test_seed"] if test_seed is None else test_seed)
    eval_batch_size = int(
        args_payload["eval_batch_size"] if eval_batch_size is None else eval_batch_size
    )

    started = time.perf_counter()
    val_dataset = make_fixed_dataset_from_iterable_config(
        train_config,
        num_samples=val_samples,
        seed=val_seed,
    )
    test_dataset = make_fixed_dataset_from_iterable_config(
        train_config,
        num_samples=test_samples,
        seed=test_seed,
    )
    dataset_elapsed = time.perf_counter() - started

    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    metrics_by_checkpoint = {}
    for name, checkpoint_path in _checkpoint_paths(result, run_dir=run_dir, checkpoint=checkpoint).items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        model = _build_model(args_payload, device=device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        metrics_by_checkpoint[name] = {
            "path": str(checkpoint_path),
            "val": evaluate_model(
                model,
                val_loader,
                device=device,
                target_representation=args_payload["target_representation"],
            ),
            "test": evaluate_model(
                model,
                test_loader,
                device=device,
                target_representation=args_payload["target_representation"],
            ),
        }

    output = {
        "run_dir": str(run_dir),
        "source_result": str(result_path),
        "eval_config": {
            "val_samples": val_samples,
            "test_samples": test_samples,
            "val_seed": val_seed,
            "test_seed": test_seed,
            "eval_batch_size": eval_batch_size,
            "device": str(device),
        },
        "dataset_elapsed_sec": dataset_elapsed,
        "checkpoints": metrics_by_checkpoint,
    }

    if output_name is None:
        output_name = (
            f"eval_{checkpoint}_val{val_samples}_test{test_samples}_"
            f"seed{val_seed}_{test_seed}.json"
        )
    (run_dir / output_name).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    outputs = []
    for run_dir_value in args.run_dir:
        run_dir = Path(run_dir_value)
        print(f"evaluating {run_dir}", flush=True)
        outputs.append(
            evaluate_run(
                run_dir,
                checkpoint=args.checkpoint,
                val_samples=args.val_samples,
                test_samples=args.test_samples,
                val_seed=args.val_seed,
                test_seed=args.test_seed,
                eval_batch_size=args.eval_batch_size,
                device=device,
                output_name=args.output_name,
            )
        )

    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
