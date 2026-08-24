"""Aggregate seed-robustness results for the central Pareto experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


LOWER_IS_BETTER = ("density_sse", "chamfer", "hd95")
HIGHER_IS_BETTER = ("coverage_2px",)
BASE_LABEL_RE = re.compile(r"^seed(?P<seed>\d+)_base-(?P<step>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--param-summary-csv", default=None)
    parser.add_argument("--per-sample-wins-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_csv(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value == "" else float(value)


def _sample_std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) <= 1:
        return 0.0
    return stdev(finite)


def _summary(values: list[float]) -> dict[str, float | int]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": len(finite),
        "mean": mean(finite),
        "std": _sample_std(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _load_param_by_label(path: str | Path | None) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        label = row["label"]
        out[label] = {
            "param_distance_mean": _f(row, "param_distance_mean"),
            "param_distance_median": _f(row, "param_distance_median"),
            "w_fro_param_mean": _f(row, "w_fro_mean"),
            "b_l2_param_mean": _f(row, "b_l2_mean"),
        }
    return out


def _base_rows(summary_rows: list[dict[str, str]], param_by_label: dict[str, dict[str, float]]) -> list[dict]:
    out = []
    for row in summary_rows:
        match = BASE_LABEL_RE.match(row["label"])
        if not match:
            continue
        label = row["label"]
        out.append(
            {
                "label": label,
                "seed": int(match.group("seed")),
                "step": int(match.group("step")),
                "seconds_per_sample": _f(row, "seconds_per_sample"),
                "density_sse": _f(row, "density_sse"),
                "chamfer": _f(row, "chamfer"),
                "hd95": _f(row, "hd95"),
                "coverage_2px": _f(row, "coverage_2px"),
                "w_fro": _f(row, "w_fro"),
                "b_l2": _f(row, "b_l2"),
                **param_by_label.get(label, {}),
            }
        )
    return sorted(out, key=lambda row: (row["seed"], row["step"]))


def _aggregate_base_rows(base_rows: list[dict]) -> list[dict]:
    by_step: dict[int, list[dict]] = defaultdict(list)
    for row in base_rows:
        by_step[int(row["step"])].append(row)

    metrics = [
        "seconds_per_sample",
        "density_sse",
        "chamfer",
        "hd95",
        "coverage_2px",
        "w_fro",
        "b_l2",
        "param_distance_mean",
    ]
    out = []
    for step, rows in sorted(by_step.items()):
        entry: dict[str, float | int | str] = {
            "step": step,
            "num_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])),
        }
        for metric in metrics:
            if metric not in rows[0]:
                continue
            stats = _summary([float(row.get(metric, math.nan)) for row in rows])
            for key, value in stats.items():
                entry[f"{metric}_{key}"] = value
        out.append(entry)
    return out


def _random_rows(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out = {}
    for row in summary_rows:
        if not row["label"].startswith("random_r4-"):
            continue
        out[row["label"]] = {
            "density_sse": _f(row, "density_sse"),
            "chamfer": _f(row, "chamfer"),
            "hd95": _f(row, "hd95"),
            "coverage_2px": _f(row, "coverage_2px"),
        }
    return out


def _dominates(base: dict, random: dict[str, float]) -> bool:
    return all(float(base[m]) < random[m] for m in LOWER_IS_BETTER) and all(
        float(base[m]) > random[m] for m in HIGHER_IS_BETTER
    )


def _dominance_rows(base_rows: list[dict], random_by_label: dict[str, dict[str, float]]) -> list[dict]:
    out = []
    base30_rows = [row for row in base_rows if int(row["step"]) == 30]
    for row in base30_rows:
        for random_label in ("random_r4-30", "random_r4-60"):
            if random_label not in random_by_label:
                continue
            random = random_by_label[random_label]
            entry = {
                "seed": row["seed"],
                "base_label": row["label"],
                "random_label": random_label,
                "pareto_dominates": _dominates(row, random),
            }
            for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER:
                base_value = float(row[metric])
                random_value = float(random[metric])
                entry[f"base_{metric}"] = base_value
                entry[f"random_{metric}"] = random_value
                if metric in HIGHER_IS_BETTER:
                    entry[f"{metric}_advantage"] = base_value - random_value
                else:
                    entry[f"{metric}_advantage"] = random_value - base_value
            out.append(entry)
    return out


def _wins_summary(rows: list[dict[str, str]]) -> list[dict]:
    out = []
    for row in rows:
        entry = {
            "comparison": row["comparison"],
            "left_label": row["left_label"],
            "right_label": row["right_label"],
            "num_samples": int(row["num_samples"]),
        }
        for metric in (
            "density_sse_to_input",
            "chamfer_to_target_points",
            "hausdorff_p95_to_target_points",
            "coverage_symmetric_2px_to_target_points",
        ):
            entry[f"{metric}_win_rate"] = _f(row, f"{metric}_win_rate")
            entry[f"{metric}_mean_delta"] = _f(row, f"{metric}_mean_delta")
        out.append(entry)
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _read_csv(args.summary_csv)
    param_by_label = _load_param_by_label(args.param_summary_csv)
    base = _base_rows(summary_rows, param_by_label)
    aggregated = _aggregate_base_rows(base)
    dominance = _dominance_rows(base, _random_rows(summary_rows))
    wins = _wins_summary(_read_csv(args.per_sample_wins_csv))

    _write_csv(base, output_dir / "base_seed_metrics.csv")
    _write_csv(aggregated, output_dir / "base_seed_mean_std.csv")
    _write_csv(dominance, output_dir / "base30_vs_random_dominance.csv")
    _write_csv(wins, output_dir / "per_sample_wins_summary.csv")
    _write_json(
        {
            "definition": (
                "Seed robustness aggregation for base50k. "
                "base30_vs_random_dominance uses lower density/Chamfer/HD95 and higher coverage@2px."
            ),
            "base_seed_metrics": base,
            "base_seed_mean_std": aggregated,
            "base30_vs_random_dominance": dominance,
            "per_sample_wins_summary": wins,
        },
        output_dir / "seed_robustness_summary.json",
    )
    print(json.dumps({"output_dir": str(output_dir), "num_base_rows": len(base)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
