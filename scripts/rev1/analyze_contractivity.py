"""I3(a): contractivity statistics of predicted / refined IFS maps in every setting.

Reviewer R2-W3 asked for the proportion and magnitude of contractivity
violations (s_max(W_i) >= 1) across all settings and refinement stages.  All
statistics are computed from parameters already saved by the paper's runs and by
the revision experiments (spectral variant, reflection / probability shifts,
external and hand-specified IFS); no optimization is rerun.  Rows whose source file is missing are skipped and
listed in the output JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import affine_matrices_to_vector, affine_vector_to_matrices, params_to_affine

ORACLE = ROOT / "outputs/oracle"
LONG = ORACLE / "advice027_long_horizon_test256_chamfer010_p512_bs256"
MNIST = ROOT / "outputs/mnist_tu32_refine/n10_balanced50/mnist_p0_outputs_with_tu32gd.pt"
FMNIST = ROOT / "outputs/fashion_mnist_tu32_refine/n10_balanced50/mnist_p0_outputs_with_tu32gd.pt"
REV1 = ROOT / "outputs/rev1"  # revision experiments (eval_shift.py outputs and the variant evaluation)


def oracle_source(path: Path, key: str = "oracle_params"):
    return ("oracle", path / "oracle_outputs.pt", key)


def sources() -> list[tuple[str, str, str, tuple]]:
    rows: list[tuple[str, str, str, tuple]] = []
    test = "in-distribution (test256, n=4)"
    rows.append((test, "ground truth", "-", ("target", ORACLE / "paper_base50k_fresh_best_s0_test256_chamfer010_p512/oracle_outputs.pt", "target_params")))
    for steps in (0, 10, 20, 30):
        rows.append((test, "base", str(steps), oracle_source(ORACLE / f"paper_base50k_fresh_best_s{steps}_test256_chamfer010_p512")))
    for steps in (100, 300, 1000):
        rows.append((test, "base", str(steps), oracle_source(LONG / f"model_r1_s{steps}")))
    for steps in (0, 10, 20, 30, 60):
        rows.append((test, "random-r4", str(steps), oracle_source(ORACLE / f"paper50k_e1_random_r4_s{steps}_test256_chamfer010_p512")))
    for family in ("random_r1", "random_r4"):
        rows.append((test, family.replace("_", "-"), "1000", oracle_source(LONG / f"{family}_s1000")))
    for scenario, label in (("t1_wide_fixed_point", "OOD T1 (wide translation)"), ("s1_scale_high", "OOD S1 (large singular values)")):
        rows.append((label, "ground truth", "-", ("target", ORACLE / f"ood_{scenario}_model_s0_n256_chamfer010_p512/oracle_outputs.pt", "target_params")))
        for steps in (0, 10, 20, 30):
            rows.append((label, "base", str(steps), oracle_source(ORACLE / f"ood_{scenario}_model_s{steps}_n256_chamfer010_p512")))
        for steps in (30, 60):
            rows.append((label, "random-r4", str(steps), oracle_source(ORACLE / f"ood_{scenario}_random_r4_s{steps}_n256_chamfer010_p512")))
    for label, path, variants in (
        ("MNIST (n=10)", MNIST, ("model-0", "model-30", "model-100", "tu32gd-model-0-100")),
        ("Fashion-MNIST (n=10)", FMNIST, ("model-0", "model-30", "model-100", "tu32gd-model-0-100")),
    ):
        for variant in variants:
            method = "ours-init + occupancy-GD" if variant.startswith("tu32gd") else "ours"
            steps = "100" if variant.startswith("tu32gd") else variant.removeprefix("model-")
            rows.append((label, method, steps, ("payload", path, variant)))
    # Revision rows (Appendix A of the revised manuscript).
    for steps in (0, 30):
        rows.append((test, "spectral (s_max<=0.95)", str(steps),
                     oracle_source(REV1 / f"variant_eval/I3_spectral_s{steps}_test256")))
    # 100- and 1000-step runs of the training-signal comparison (Table D.1; run_variant_eval.sh protocol)
    for label, method in (("base", "base (Table D.1 run)"), ("I1_hung_nofp", "Hungarian, no fixed point"),
                          ("I1_ordered", "ordered"), ("I2_multihyp4", "multi-hypothesis"),
                          ("I2_recon_only", "reconstruction only")):
        for steps in ("100", "1000"):
            rows.append((test, method, steps, oracle_source(REV1 / f"variant_eval/{label}_s{steps}_test256")))
    for label, model, path, variants in (
        ("R (reflections)", "base", REV1 / "I4_prior/reflect_detpen0.1/outputs.pt", ("model-0", "model-30", "model-60", "random-r4-60")),
        ("P (alpha=1)", "base", REV1 / "I4_prior/probmix_a1.0/outputs.pt", ("model-0", "model-30", "model-60", "random-r4-60")),
        ("External (n=10)", "ours", REV1 / "I6_external/djeacoumar_all_n10/outputs.pt", ("model-0", "model-100", "model-1000", "random-r4-1000")),
        ("Hand-specified (n=4)", "base", REV1 / "I6_external/classical_n4/outputs.pt", ("model-0", "model-100", "random-r4-1000")),
    ):
        for variant in variants:
            method, steps = variant.rsplit("-", 1)
            rows.append((label, model if method == "model" else method, steps, ("shift", path, variant)))
    return rows


def load_affine(kind: str, path: Path, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "target":
        w, b = params_to_affine(payload[key].float())
        return affine_matrices_to_vector(w, b)
    if kind == "oracle":
        return payload[key].float()
    if kind == "shift":  # scripts/rev1/eval_shift.py outputs.pt
        return payload["variants"][key].float()
    return payload["params_by_label"][key].float()


def stats(affine: torch.Tensor) -> dict:
    w, _ = affine_vector_to_matrices(affine)
    smax = torch.linalg.svdvals(w).amax(dim=-1)  # [B, n]
    det = torch.linalg.det(w)
    violation = smax >= 1.0
    excess = (smax - 1.0)[violation]
    return {
        "num_systems": int(smax.shape[0]),
        "num_maps": int(smax.numel()),
        "smax_mean": float(smax.mean()),
        "smax_p99": float(torch.quantile(smax.reshape(-1), 0.99)),
        "smax_max": float(smax.max()),
        "frac_maps_smax_gt_0.75": float((smax > 0.75).float().mean()),
        "frac_maps_smax_gt_0.9": float((smax > 0.9).float().mean()),
        "frac_maps_violating": float(violation.float().mean()),
        "frac_systems_violating": float(violation.any(dim=1).float().mean()),
        "mean_excess_over_1": float(excess.mean()) if excess.numel() else 0.0,
        "max_excess_over_1": float(excess.max()) if excess.numel() else 0.0,
        "frac_maps_det_negative": float((det < 0).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/rev1/I3_contractivity"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    missing = []
    for setting, method, steps, (kind, path, key) in sources():
        if not Path(path).exists():
            missing.append(str(path))
            continue
        try:
            affine = load_affine(kind, Path(path), key)
        except KeyError:  # the run exists but was made without this variant / step count
            missing.append(f"{path} [{key}]")
            continue
        record = {"setting": setting, "method": method, "steps": steps, "source": str(Path(path).relative_to(ROOT))}
        record.update(stats(affine))
        records.append(record)
    with (out / "contractivity_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (out / "contractivity_stats.json").write_text(json.dumps({"rows": records, "missing": missing}, indent=2))
    for r in records:
        print(f"{r['setting'][:30]:30s} {r['method'][:24]:24s} {r['steps']:>5s} "
              f"smax_mean={r['smax_mean']:.3f} p99={r['smax_p99']:.3f} max={r['smax_max']:.3f} "
              f">0.9={r['frac_maps_smax_gt_0.9']:.4f} viol_maps={r['frac_maps_violating']:.4f} "
              f"viol_sys={r['frac_systems_violating']:.4f} max_excess={r['max_excess_over_1']:.3f} det<0={r['frac_maps_det_negative']:.3f}")
    if missing:
        print("missing:", missing)


if __name__ == "__main__":
    main()
