"""Compare per-sample fidelity metrics between oracle/refinement runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


LOWER_IS_BETTER = (
    "density_sse_to_input",
    "chamfer_to_target_points",
    "hausdorff_p95_to_target_points",
    "hausdorff_to_target_points",
    "modified_hausdorff_mean_to_target_points",
)
HIGHER_IS_BETTER = ("coverage_symmetric_2px_to_target_points",)
DEFAULT_METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "ORACLE_DIR"),
        required=True,
        help="Run label and oracle output directory. Can be repeated.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        nargs=3,
        metavar=("COMPARISON_LABEL", "LEFT_LABEL", "RIGHT_LABEL"),
        required=True,
        help="Compare LEFT against RIGHT. Win counts are wins by LEFT.",
    )
    parser.add_argument(
        "--fidelity-name",
        default="fidelity_res128_traj16_steps1024_sigma2_robust2048",
    )
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def step_from_label(label: str) -> int | None:
    match = re.search(r"-(\d+)$", label)
    return int(match.group(1)) if match else None


def load_oracle_records(oracle_dir: Path, fidelity_name: str) -> dict[int, dict[str, float]]:
    path = oracle_dir / f"{fidelity_name}_per_sample.csv"
    records: dict[int, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "oracle":
                continue
            sample_index = int(row["sample_index"])
            records[sample_index] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"sample_index", "variant"}
            }
    return records


def metric_is_higher_better(metric: str) -> bool:
    return metric.startswith("coverage_")


def one_sided_sign_test_p_value(*, wins: int, ties: int, num_samples: int) -> float:
    """Exact P[X >= wins] for non-tied samples under a fair coin null."""
    trials = num_samples - ties
    if trials <= 0:
        return 1.0
    wins = min(max(int(wins), 0), trials)
    numerator = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    return float(numerator / (2**trials))


def summarize_comparison(
    *,
    comparison_label: str,
    left_label: str,
    right_label: str,
    left: dict[int, dict[str, float]],
    right: dict[int, dict[str, float]],
    metrics: list[str],
) -> dict[str, object]:
    sample_indices = sorted(set(left) & set(right))
    if not sample_indices:
        raise ValueError(f"no overlapping sample indices for {comparison_label}")
    row: dict[str, object] = {
        "comparison": comparison_label,
        "left_label": left_label,
        "right_label": right_label,
        "step": step_from_label(left_label),
        "num_samples": len(sample_indices),
    }
    for metric in metrics:
        wins = 0
        ties = 0
        deltas = []
        higher_better = metric_is_higher_better(metric)
        for sample_index in sample_indices:
            left_value = left[sample_index][metric]
            right_value = right[sample_index][metric]
            delta = left_value - right_value
            deltas.append(delta)
            if left_value == right_value:
                ties += 1
            elif (left_value > right_value) if higher_better else (left_value < right_value):
                wins += 1
        row[f"{metric}_wins"] = wins
        row[f"{metric}_ties"] = ties
        row[f"{metric}_win_rate"] = wins / max(1, len(sample_indices) - ties)
        row[f"{metric}_sign_p_left_better"] = one_sided_sign_test_p_value(
            wins=wins,
            ties=ties,
            num_samples=len(sample_indices),
        )
        row[f"{metric}_mean_delta"] = sum(deltas) / len(deltas)
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs = {
        label: load_oracle_records(Path(path), args.fidelity_name)
        for label, path in args.run
    }
    rows = [
        summarize_comparison(
            comparison_label=comparison_label,
            left_label=left_label,
            right_label=right_label,
            left=runs[left_label],
            right=runs[right_label],
            metrics=args.metrics,
        )
        for comparison_label, left_label, right_label in args.comparison
    ]
    write_csv(rows, Path(args.output_csv))
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
