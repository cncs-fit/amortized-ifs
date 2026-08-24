"""Paired bootstrap confidence intervals for mean metric deltas between two oracle runs.

Reproduces the base-vs-aux significance check reported in the paper: for each
metric, resample the per-sample paired deltas (left minus right) with
replacement, and report the 95% percentile interval of the mean delta together
with the fraction of resamples in which the left run is better.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

DEFAULT_METRICS = (
    "density_sse_to_input",
    "chamfer_to_target_points",
    "hausdorff_p95_to_target_points",
    "coverage_symmetric_2px_to_target_points",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True, help="Oracle output dir of the left run.")
    parser.add_argument("--right-dir", required=True, help="Oracle output dir of the right run.")
    parser.add_argument("--comparison-label", default="left_vs_right")
    parser.add_argument(
        "--fidelity-name",
        default="fidelity_res128_traj16_steps1024_sigma2_robust2048",
    )
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--num-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_oracle_records(oracle_dir: Path, fidelity_name: str) -> dict[int, dict[str, float]]:
    path = oracle_dir / f"{fidelity_name}_per_sample.csv"
    records: dict[int, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "oracle":
                continue
            records[int(row["sample_index"])] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"sample_index", "variant"}
            }
    return records


def metric_is_higher_better(metric: str) -> bool:
    return metric.startswith("coverage_")


def percentile(sorted_values: list[float], q: float) -> float:
    h = (len(sorted_values) - 1) * q
    low = int(h)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (h - low) * (sorted_values[high] - sorted_values[low])


def main() -> None:
    args = parse_args()
    left = load_oracle_records(Path(args.left_dir), args.fidelity_name)
    right = load_oracle_records(Path(args.right_dir), args.fidelity_name)
    sample_indices = sorted(set(left) & set(right))
    if not sample_indices:
        raise SystemExit("no overlapping sample indices between the two runs")

    rng = random.Random(args.seed)
    result: dict[str, object] = {
        "comparison": args.comparison_label,
        "num_samples": len(sample_indices),
        "bootstrap_samples": args.num_resamples,
        "seed": args.seed,
        "metrics": {},
    }
    for metric in args.metrics:
        deltas = [left[i][metric] - right[i][metric] for i in sample_indices]
        higher_better = metric_is_higher_better(metric)
        means = []
        better = 0
        n = len(deltas)
        for _ in range(args.num_resamples):
            resample_mean = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
            means.append(resample_mean)
            if (resample_mean > 0) == higher_better and resample_mean != 0:
                better += 1
        means.sort()
        result["metrics"][metric] = {
            "mean_delta": sum(deltas) / n,
            "ci95_low": percentile(means, 0.025),
            "ci95_high": percentile(means, 0.975),
            "left_better_direction": "positive" if higher_better else "negative",
            "bootstrap_fraction_left_better": better / args.num_resamples,
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
