"""Summarize Hungarian-matched direct-affine parameter errors for oracle outputs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import affine_matrices_to_vector, affine_vector_to_matrices, params_to_affine
from losses.hungarian import (
    fixed_points_from_matrices,
    hungarian_indices,
    pairwise_direct_affine_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "ORACLE_DIR"),
        required=True,
        help="Label and oracle output directory containing oracle_outputs.pt.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="param_error")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _summarize(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": _to_float(tensor.mean()),
        "median": _to_float(tensor.median()),
        "p05": _to_float(torch.quantile(tensor, 0.05)),
        "p10": _to_float(torch.quantile(tensor, 0.10)),
        "p90": _to_float(torch.quantile(tensor, 0.90)),
        "p95": _to_float(torch.quantile(tensor, 0.95)),
        "min": _to_float(tensor.min()),
        "max": _to_float(tensor.max()),
    }


def _target_affine(target_params: torch.Tensor) -> torch.Tensor:
    target_w, target_b = params_to_affine(target_params)
    return affine_matrices_to_vector(target_w, target_b)


@lru_cache(maxsize=16)
def _permutation_indices(num_items: int) -> torch.Tensor:
    return torch.tensor(list(itertools.permutations(range(num_items))), dtype=torch.long)


def _affine_set_distance(source_affine: torch.Tensor, target_affine: torch.Tensor) -> dict[str, float]:
    cost = pairwise_direct_affine_distance(source_affine, target_affine)
    num_source, num_target = cost.shape
    if num_source == num_target and num_source <= 6:
        perms = _permutation_indices(num_source).to(cost.device)
        rows = torch.arange(num_source, device=cost.device)
        values = cost[rows[None, :], perms].mean(dim=1)
        best_idx = int(torch.argmin(values).item())
        row_ind = rows
        col_ind = perms[best_idx]
    else:
        row_ind, col_ind = hungarian_indices(cost)
        row_ind = row_ind.to(cost.device)
        col_ind = col_ind.to(cost.device)

    source_w, source_b = affine_vector_to_matrices(source_affine[row_ind])
    target_w, target_b = affine_vector_to_matrices(target_affine[col_ind])
    w_errors = (source_w - target_w).square().sum(dim=(-1, -2)).sqrt()
    b_errors = (source_b - target_b).square().sum(dim=-1).sqrt()
    source_fp = fixed_points_from_matrices(source_w, source_b)
    target_fp = fixed_points_from_matrices(target_w, target_b)
    fp_errors = (source_fp - target_fp).square().sum(dim=-1).sqrt()
    selected_loss = cost[row_ind, col_ind].mean()
    return {
        "param_loss": _to_float(selected_loss),
        "param_distance": _to_float(selected_loss.sqrt()),
        "w_fro": _to_float(w_errors.mean()),
        "b_l2": _to_float(b_errors.mean()),
        "fixed_point_l2": _to_float(fp_errors.mean()),
    }


def _load_distances(label: str, oracle_dir: Path) -> tuple[list[dict], dict]:
    payload_path = oracle_dir / "oracle_outputs.pt"
    if not payload_path.exists():
        raise FileNotFoundError(f"oracle output not found: {payload_path}")
    payload = torch.load(payload_path, map_location="cpu")
    pred = payload["oracle_params"].float().contiguous()
    target = _target_affine(payload["target_params"].float().contiguous())
    if pred.shape != target.shape:
        raise ValueError(f"{label}: pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    rows = []
    for sample_index in range(pred.shape[0]):
        distances = _affine_set_distance(pred[sample_index], target[sample_index])
        rows.append(
            {
                "label": label,
                "sample_index": int(sample_index),
                **distances,
            }
        )
    summary = {
        "label": label,
        "oracle_dir": str(oracle_dir),
        "num_samples": int(pred.shape[0]),
        "num_transforms": int(pred.shape[1]),
    }
    for key in ("param_loss", "param_distance", "w_fro", "b_l2", "fixed_point_l2"):
        stats = _summarize(row[key] for row in rows)
        for stat_key, value in stats.items():
            summary[f"{key}_{stat_key}"] = value
    return rows, summary


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows: list[dict] = []
    summary_rows: list[dict] = []
    for label, oracle_dir_raw in args.run:
        rows, summary = _load_distances(label, _resolve(oracle_dir_raw))
        per_sample_rows.extend(rows)
        summary_rows.append(summary)

    _write_csv(summary_rows, output_dir / f"{args.output_prefix}_summary.csv")
    _write_csv(per_sample_rows, output_dir / f"{args.output_prefix}_per_sample.csv")
    payload = {
        "definition": (
            "Hungarian-matched direct-affine IFS set distance. param_distance is "
            "sqrt(mean(||W_hat-W||_F^2 + ||b_hat-b||_2^2)) over matched maps, "
            "matching the affine_set_distance diagnostic used in analyze_identifiability.py."
        ),
        "runs": summary_rows,
    }
    (output_dir / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
