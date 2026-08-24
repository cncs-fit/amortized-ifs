"""Analyze long-horizon oracle refinement distributions.

This script consumes oracle fidelity per-sample CSVs and optional
Hungarian-matched parameter-error diagnostics, then writes distribution
tables and figures for the amortized-vs-random long-horizon experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    from matplotlib_config import configure_matplotlib_pdf_fonts
except ImportError:
    from scripts.matplotlib_config import configure_matplotlib_pdf_fonts

configure_matplotlib_pdf_fonts()


METRICS = [
    "density_sse_to_input",
    "chamfer_to_target_points",
    "hausdorff_p95_to_target_points",
    "coverage_symmetric_2px_to_target_points",
]
OPTIONAL_PARAM_METRICS = ["param_distance", "w_fro", "b_l2", "fixed_point_l2"]


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
    parser.add_argument("--fidelity-name", required=True)
    parser.add_argument("--param-error-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _summarize(values: Iterable[float]) -> dict[str, float | int]:
    vals = sorted(_finite(values))
    if not vals:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p05": math.nan,
            "p10": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "median": _quantile(vals, 0.50),
        "p05": _quantile(vals, 0.05),
        "p10": _quantile(vals, 0.10),
        "p90": _quantile(vals, 0.90),
        "p95": _quantile(vals, 0.95),
        "min": vals[0],
        "max": vals[-1],
    }


def _float_or_nan(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def _parse_label(label: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)-(\d+)$", label)
    if not match:
        return label, None
    return match.group(1), int(match.group(2))


def _load_summary_meta(oracle_dir: Path) -> dict[str, float | int | str]:
    path = oracle_dir / "summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "init_mode": payload.get("init_mode", ""),
        "restarts": payload.get("restarts", ""),
        "steps_summary": payload.get("steps", ""),
        "seconds_per_sample": payload.get("seconds_per_sample", math.nan),
    }


def _load_param_errors(path: Path | None) -> dict[tuple[str, int], dict[str, float]]:
    if path is None or not path.exists():
        return {}
    result: dict[tuple[str, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row["label"]
            sample_index = int(row["sample_index"])
            result[(label, sample_index)] = {
                key: _float_or_nan(row, key)
                for key in OPTIONAL_PARAM_METRICS
                if key in row
            }
    return result


def _load_run(label: str, oracle_dir: Path, fidelity_name: str, param_errors: dict) -> list[dict]:
    fidelity_path = oracle_dir / f"{fidelity_name}_per_sample.csv"
    if not fidelity_path.exists():
        raise FileNotFoundError(f"missing fidelity per-sample CSV: {fidelity_path}")

    family, step = _parse_label(label)
    meta = _load_summary_meta(oracle_dir)
    rows: list[dict] = []
    with fidelity_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("variant") != "oracle":
                continue
            sample_index = int(row["sample_index"])
            out = {
                "label": label,
                "family": family,
                "step": step if step is not None else "",
                "sample_index": sample_index,
                "oracle_dir": str(oracle_dir),
                **meta,
            }
            for metric in METRICS:
                out[metric] = _float_or_nan(row, metric)
            for key, value in param_errors.get((label, sample_index), {}).items():
                out[key] = value
            rows.append(out)
    if not rows:
        raise ValueError(f"no oracle rows found in {fidelity_path}")
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summary_rows(rows: list[dict]) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)

    out: list[dict] = []
    for label, group in sorted(
        by_label.items(),
        key=lambda item: (str(item[1][0]["family"]), int(item[1][0]["step"])),
    ):
        base = {
            "label": label,
            "family": group[0]["family"],
            "step": group[0]["step"],
            "count": len(group),
            "init_mode": group[0].get("init_mode", ""),
            "restarts": group[0].get("restarts", ""),
            "seconds_per_sample": group[0].get("seconds_per_sample", math.nan),
        }
        for metric in METRICS + OPTIONAL_PARAM_METRICS:
            if metric not in group[0]:
                continue
            stats = _summarize(row.get(metric, math.nan) for row in group)
            for stat_key, value in stats.items():
                base[f"{metric}_{stat_key}"] = value
        out.append(base)
    return out


def _thresholds(rows: list[dict]) -> dict[str, float]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)

    thresholds: dict[str, float] = {}
    for label, suffix in (
        ("model_r1-0", "model0_median"),
        ("model_r1-30", "model30_median"),
        ("model_r1-30", "model30_mean"),
        ("model_r1-100", "model100_median"),
    ):
        if label not in by_label:
            continue
        values = [row["chamfer_to_target_points"] for row in by_label[label]]
        stats = _summarize(values)
        key = "mean" if suffix.endswith("_mean") else "median"
        thresholds[f"tau_{suffix}"] = float(stats[key])
    if not thresholds:
        thresholds["tau_all_median"] = float(_summarize(row["chamfer_to_target_points"] for row in rows)["median"])
    return thresholds


def _success_rows(rows: list[dict], thresholds: dict[str, float]) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)

    out: list[dict] = []
    for label, group in sorted(
        by_label.items(),
        key=lambda item: (str(item[1][0]["family"]), int(item[1][0]["step"])),
    ):
        for threshold_name, tau in thresholds.items():
            values = _finite(row["chamfer_to_target_points"] for row in group)
            successes = sum(1 for value in values if value <= tau)
            out.append(
                {
                    "label": label,
                    "family": group[0]["family"],
                    "step": group[0]["step"],
                    "threshold": threshold_name,
                    "tau": tau,
                    "count": len(values),
                    "successes": successes,
                    "success_rate": successes / len(values) if values else math.nan,
                }
            )
    return out


def _group_metric(rows: list[dict], metric: str) -> dict[str, dict[int, list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("step") == "":
            continue
        grouped[str(row["family"])][int(row["step"])].append(float(row[metric]))
    return grouped


def _color(family: str) -> str:
    return {
        "model_r1": "#2563eb",
        "random_r1": "#dc2626",
        "random_r4": "#7c3aed",
    }.get(family, "#475569")


def _label(family: str) -> str:
    return {
        "model_r1": "model init r1",
        "random_r1": "random r1",
        "random_r4": "random r4",
    }.get(family, family)


def _save_fig(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=360)
    fig.savefig(output_dir / f"{stem}.pdf")


def _plot_hist_at_step(rows: list[dict], metric: str, step: int, output_dir: Path, stem: str, xlabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for family, by_step in sorted(_group_metric(rows, metric).items()):
        values = _finite(by_step.get(step, []))
        if not values:
            continue
        ax.hist(
            values,
            bins=30,
            alpha=0.38,
            density=True,
            color=_color(family),
            label=_label(family),
            linewidth=0.0,
        )
        ax.axvline(_summarize(values)["median"], color=_color(family), linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    if metric != "coverage_symmetric_2px_to_target_points":
        ax.set_xscale("log")
    _save_fig(fig, output_dir, stem)
    plt.close(fig)


def _plot_quantiles(rows: list[dict], metric: str, output_dir: Path, stem: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    grouped = _group_metric(rows, metric)
    for family, by_step in sorted(grouped.items()):
        steps = sorted(by_step)
        medians = []
        p10s = []
        p90s = []
        for step in steps:
            stats = _summarize(by_step[step])
            medians.append(stats["median"])
            p10s.append(stats["p10"])
            p90s.append(stats["p90"])
        ax.plot(steps, medians, marker="o", color=_color(family), label=_label(family))
        ax.fill_between(steps, p10s, p90s, color=_color(family), alpha=0.16, linewidth=0.0)
    ax.set_xlabel("refinement steps")
    ax.set_ylabel(ylabel)
    if metric != "coverage_symmetric_2px_to_target_points":
        ax.set_yscale("log")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    _save_fig(fig, output_dir, stem)
    plt.close(fig)


def _plot_success(success_rows: list[dict], thresholds: dict[str, float], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not success_rows:
        return

    preferred = "tau_model30_mean"
    threshold_name = preferred if preferred in thresholds else next(iter(thresholds))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in success_rows:
        if row["threshold"] == threshold_name:
            grouped[str(row["family"])].append(row)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for family, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["step"]))
        ax.plot(
            [int(row["step"]) for row in group],
            [float(row["success_rate"]) for row in group],
            marker="o",
            color=_color(family),
            label=_label(family),
        )
    ax.set_xlabel("refinement steps")
    ax.set_ylabel(f"P(Chamfer <= {threshold_name})")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    _save_fig(fig, output_dir, "success_rate_by_step")
    plt.close(fig)


def _plot_pair_scatter(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_label_sample: dict[tuple[str, int], float] = {}
    for row in rows:
        by_label_sample[(str(row["label"]), int(row["sample_index"]))] = float(row["chamfer_to_target_points"])

    model_values = {
        sample: value
        for (label, sample), value in by_label_sample.items()
        if label == "model_r1-1000"
    }
    if not model_values:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharex=True, sharey=True)
    for ax, random_label in zip(axes, ["random_r1-1000", "random_r4-1000"], strict=True):
        pairs = [
            (model_value, by_label_sample[(random_label, sample)])
            for sample, model_value in model_values.items()
            if (random_label, sample) in by_label_sample
        ]
        if not pairs:
            continue
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        ax.scatter(xs, ys, s=16, alpha=0.58, color=_color(random_label.rsplit("-", 1)[0]), edgecolors="none")
        finite_values = _finite(xs + ys)
        lo = max(min(finite_values) * 0.85, 1e-8)
        hi = max(finite_values) * 1.15
        ax.plot([lo, hi], [lo, hi], color="#334155", linewidth=1.0, linestyle="--")
        ax.set_title(_label(random_label.rsplit("-", 1)[0]), fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.22)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    axes[0].set_ylabel("random Chamfer at 1000")
    for ax in axes:
        ax.set_xlabel("model-init Chamfer at 1000")
    _save_fig(fig, output_dir, "pair_scatter_model_vs_random_s1000")
    plt.close(fig)


def _save_plots(rows: list[dict], success_rows: list[dict], thresholds: dict[str, float], output_dir: Path) -> None:
    _plot_hist_at_step(
        rows,
        "chamfer_to_target_points",
        1000,
        output_dir,
        "chamfer_hist_s1000",
        "Chamfer to target points",
    )
    _plot_hist_at_step(
        rows,
        "density_sse_to_input",
        1000,
        output_dir,
        "density_hist_s1000",
        "density SSE to input",
    )
    _plot_quantiles(
        rows,
        "chamfer_to_target_points",
        output_dir,
        "chamfer_quantiles_by_step",
        "Chamfer to target points",
    )
    _plot_quantiles(
        rows,
        "density_sse_to_input",
        output_dir,
        "density_quantiles_by_step",
        "density SSE to input",
    )
    _plot_success(success_rows, thresholds, output_dir)
    _plot_pair_scatter(rows, output_dir)
    if any("param_distance" in row for row in rows):
        _plot_quantiles(rows, "param_distance", output_dir, "param_distance_by_step", "affine set distance")


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_errors = _load_param_errors(_resolve(args.param_error_csv) if args.param_error_csv else None)
    rows: list[dict] = []
    for label, oracle_dir in args.run:
        rows.extend(_load_run(label, _resolve(oracle_dir), args.fidelity_name, param_errors))

    summary_rows = _summary_rows(rows)
    thresholds = _thresholds(rows)
    success_rows = _success_rows(rows, thresholds)

    _write_csv(rows, output_dir / "per_sample_metrics.csv")
    _write_csv(summary_rows, output_dir / "summary_quantiles.csv")
    _write_csv(success_rows, output_dir / "success_rates.csv")
    _write_json(
        {
            "metrics": METRICS,
            "optional_param_metrics": OPTIONAL_PARAM_METRICS,
            "thresholds": thresholds,
            "runs": summary_rows,
        },
        output_dir / "summary_quantiles.json",
    )
    _write_json(
        {
            "definition": "success is measured by per-sample Chamfer <= tau",
            "thresholds": thresholds,
            "rows": success_rows,
        },
        output_dir / "success_rates.json",
    )
    _save_plots(rows, success_rows, thresholds, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "thresholds": thresholds}, indent=2), flush=True)


if __name__ == "__main__":
    main()
