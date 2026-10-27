"""I4/I5: LaTeX rows for the robustness tables (main-text summary and appendix sweep).

Reads ``outputs/rev1/{I5_renderer,I4_prior}/<condition>/summary.json`` written by
``scripts/rev1/eval_shift.py`` and prints/saves table rows with the Chamfer mean of
ours-0, ours-30, random-r4-30 (and ours-60 / random-r4-60 for prior shifts), HD95 and
coverage@2px of ours-30, and the per-sample count of ours-30 beating random-r4-30.
Output: ``outputs/rev1/paper/robustness_{main,appendix}_rows.tex`` and a JSON dump.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/rev1/paper"

RENDERER = [  # (directory, label, in main table)
    ("default", r"training renderer ($16{\times}1024$, $128^2$, $\sigma{=}2$)", True),
    ("traj2", r"$2$ trajectories", True),
    ("traj4", r"$4$ trajectories", False),
    ("traj8", r"$8$ trajectories", False),
    ("traj64", r"$64$ trajectories", False),
    ("traj256", r"$256$ trajectories", False),
    ("dense_t256_s4096", r"$256{\times}4096$ points", False),
    ("res64", r"resolution $64^2$", True),
    ("res96", r"resolution $96^2$", False),
    ("res256", r"resolution $256^2$", True),
    ("splat_bilinear", r"bilinear splatting", True),
    ("burnin0", r"no burn-in", True),
    ("sigma0", r"$\sigma{=}0$", True),
    ("sigma1", r"$\sigma{=}1$", True),
    ("sigma3", r"$\sigma{=}3$", False),
    ("sigma4", r"$\sigma{=}4$", True),
    ("sigma1_preblur2", r"$\sigma{=}1$, pre-blurred by $2$", True),
    ("sigma1_preblur1.73", r"$\sigma{=}1$, pre-blurred by $\sqrt3$", False),
    ("sigma0_preblur2", r"$\sigma{=}0$, pre-blurred by $2$", False),
    ("traj2_refobs", r"$2$ trajectories, refined with obs.\ renderer", False),
    ("traj4_refobs", r"$4$ trajectories, refined with obs.\ renderer", False),
    ("sigma0_refobs", r"$\sigma{=}0$, refined with obs.\ renderer", False),
    ("sigma4_refobs", r"$\sigma{=}4$, refined with obs.\ renderer", False),
]
PRIOR = [
    ("reflect_detpen0.1", r"reflections (det.\ penalty $0.1$)"),
    ("reflect_detpen0", r"reflections (det.\ penalty $0$)"),
    # the estimator trained on a prior with reflections (checkpoints/rev1/I4b_reflect)
    ("I4b_reflect_detpen0", r"reflections, reflection prior (det.\ penalty $0$)"),
    ("I4b_reflect_detpen0.1", r"reflections, reflection prior (det.\ penalty $0.1$)"),
    ("probmix_a0.25", r"probabilities, $\alpha{=}0.25$"),
    ("probmix_a0.5", r"probabilities, $\alpha{=}0.5$"),
    ("probmix_a0.75", r"probabilities, $\alpha{=}0.75$"),
    ("probmix_a1.0", r"probabilities, $\alpha{=}1$"),
]


def load(group: str, name: str) -> dict | None:
    path = ROOT / "outputs/rev1" / group / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def cd(summary: dict, variant: str) -> float:
    return summary["variants"][variant]["chamfer"]["mean"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows, app_rows, dump = [], [], {}
    for name, label, in_main in RENDERER:
        s = load("I5_renderer", name)
        if s is None:
            print(f"missing {name}")
            continue
        v = s["variants"]
        rec = {
            "ours0": cd(s, "model-0"), "ours30": cd(s, "model-30"), "r4_30": cd(s, "random-r4-30"),
            "hd95_30": v["model-30"]["hd95"]["mean"], "cov_30": v["model-30"]["coverage_2px"]["mean"],
            "wins": s["wins"]["model-30_vs_random-r4-30_chamfer"],
        }
        dump[name] = rec
        row = (f"    {label} & {rec['ours0']:.4f} & {rec['ours30']:.4f} & {rec['r4_30']:.4f} & "
               f"{rec['wins']} \\\\")
        if in_main:
            main_rows.append(row)
        app_rows.append(f"    {label} & {rec['ours0']:.4f} & {rec['ours30']:.4f} & {rec['hd95_30']:.4f} & "
                        f"{rec['cov_30']:.3f} & {rec['r4_30']:.4f} & {rec['wins']} \\\\")
    prior_rows = []
    for name, label in PRIOR:
        s = load("I4_prior", name)
        if s is None:
            print(f"missing {name}")
            continue
        rec = {k: cd(s, k) for k in ("model-0", "model-30", "model-60", "random-r4-30", "random-r4-60")}
        rec["wins30"] = s["wins"]["model-30_vs_random-r4-30_chamfer"]
        rec["wins60"] = s["wins"]["model-60_vs_random-r4-60_chamfer"]
        dump[name] = rec
        prior_rows.append(
            f"    {label} & {rec['model-0']:.4f} & {rec['model-30']:.4f} & {rec['model-60']:.4f} & "
            f"{rec['random-r4-30']:.4f} & {rec['random-r4-60']:.4f} & {rec['wins30']} \\\\")
    (OUT / "robustness_main_rows.tex").write_text("\n".join(main_rows) + "\n")
    (OUT / "robustness_appendix_rows.tex").write_text("\n".join(app_rows) + "\n")
    (OUT / "robustness_prior_rows.tex").write_text("\n".join(prior_rows) + "\n")
    (OUT / "robustness_values.json").write_text(json.dumps(dump, indent=2))
    print("\n".join(main_rows), "\n---\n" + "\n".join(app_rows), "\n---\n" + "\n".join(prior_rows))


if __name__ == "__main__":
    main()
