"""Analyze Pareto frontiers for oracle/refinement quality-speed summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ERROR_METRICS = ("density_sse", "chamfer", "hd95")
COVERAGE_METRICS = ("coverage_2px",)
DEFAULT_METRICS = ERROR_METRICS + COVERAGE_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"label", "oracle_dir", "init_mode"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
        match = re.search(r"-(\d+)$", str(row["label"]))
        row["refinement_steps"] = int(match.group(1)) if match else math.nan
        row["curve"] = re.sub(r"-\d+$", "", str(row["label"]))
    return rows


def _quality_value(row: dict, metric: str) -> float:
    value = float(row[metric])
    return -value if metric.startswith("coverage_") else value


def dominates(a: dict, b: dict, metrics: tuple[str, ...]) -> bool:
    """Return True when ``a`` Pareto-dominates ``b`` in time plus metrics."""
    keys = ("seconds_per_sample",) + metrics
    strict = False
    for key in keys:
        av = float(a[key]) if key == "seconds_per_sample" else _quality_value(a, key)
        bv = float(b[key]) if key == "seconds_per_sample" else _quality_value(b, key)
        if av > bv:
            return False
        strict = strict or av < bv
    return strict


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pareto_rows(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    frontier = []
    for row in rows:
        if not any(other is not row and dominates(other, row, metrics) for other in rows):
            frontier.append(row)
    return sorted(frontier, key=lambda item: (float(item["seconds_per_sample"]), str(item["label"])))


def row_dominance(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    records = []
    for target in rows:
        dominators = [
            str(candidate["label"])
            for candidate in rows
            if candidate is not target and dominates(candidate, target, metrics)
        ]
        records.append(
            {
                "label": target["label"],
                "curve": target["curve"],
                "refinement_steps": target["refinement_steps"],
                "dominated": bool(dominators),
                "num_dominators": len(dominators),
                "dominators": ";".join(dominators),
            }
        )
    return records


def curve_dominance(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    curves = sorted({str(row["curve"]) for row in rows})
    by_curve = {curve: [row for row in rows if row["curve"] == curve] for curve in curves}
    records = []
    for candidate_curve in curves:
        for target_curve in curves:
            if candidate_curve == target_curve:
                continue
            target_rows = by_curve[target_curve]
            dominated_count = sum(
                any(dominates(candidate, target, metrics) for candidate in by_curve[candidate_curve])
                for target in target_rows
            )
            records.append(
                {
                    "candidate_curve": candidate_curve,
                    "target_curve": target_curve,
                    "metrics": "+".join(metrics),
                    "target_points": len(target_rows),
                    "dominated_points": dominated_count,
                    "curve_dominates": dominated_count == len(target_rows),
                }
            )
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(Path(args.summary_csv))
    metrics = tuple(args.metrics)

    payload = {
        "source": str(args.summary_csv),
        "metrics": metrics,
        "frontiers": {},
    }

    for metric in metrics:
        frontier = pareto_rows(rows, (metric,))
        _write_csv(frontier, output_dir / f"pareto_{metric}.csv")
        payload["frontiers"][metric] = [row["label"] for row in frontier]

    multi_frontier = pareto_rows(rows, metrics)
    _write_csv(multi_frontier, output_dir / "pareto_multi_metric.csv")
    _write_csv(row_dominance(rows, metrics), output_dir / "row_dominance_multi_metric.csv")
    _write_csv(curve_dominance(rows, metrics), output_dir / "curve_dominance_multi_metric.csv")
    payload["frontiers"]["multi_metric"] = [row["label"] for row in multi_frontier]

    (output_dir / "pareto_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
