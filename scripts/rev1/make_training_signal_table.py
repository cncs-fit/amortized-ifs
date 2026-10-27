"""I1/I2/I3b: LaTeX rows for the training-signal tables (main-text summary and appendix).

Reads the fidelity outputs of the variant evaluation (``scripts/rev1/run_variant_eval.sh``,
test256, matched renderer) through ``summarize_variants.load`` and writes
  * main rows: CD at 0/30/100 steps, HD95 and coverage@2px at 30 steps, base better at 30;
  * appendix rows: CD at 0/10/20/30/100/1000 steps, density SSE at 30 steps, (W,b) error at 0 and
    30 steps, and the number of samples (of 256) on which base has the lower Chamfer at 0/30/100/1000.
The JSON dump also counts, for ordered regression, the samples on which Hungarian matching without
the fixed-point term (the variant that differs only in the matching) has the lower Chamfer.
Missing evaluations are written as ``--``.  The base and +aux entries at 0-30 steps are the
evaluations of the original submission (same protocol); every other entry is in variant_eval.
Output: ``outputs/rev1/paper/training_signal_{main,appendix}_rows.tex`` and a JSON dump.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rev1.summarize_variants import BASE, load, mean

OUT = ROOT / "outputs/rev1/paper"
VE = ROOT / "outputs/rev1/variant_eval"
AUX_OLD = ROOT / "outputs/oracle/paper50k_md4_3k_final_s{step}_test256_chamfer010_p512"
STEPS = (0, 10, 20, 30, 100, 1000)
VARIANTS = [  # (label, LaTeX name, uses GT theta)
    ("base", r"\texttt{base}: Hungarian + fixed point", "yes"),
    ("I1_hung_nofp", r"Hungarian, no fixed point", "yes"),
    ("I1_ordered", r"ordered (fixed-point order)", "yes"),
    ("I2_multihyp4", r"multi-hypothesis ($K{=}4$)", "yes"),
    ("I3_spectral", r"spectrally constrained output", "yes"),
    ("aux_md4_3k", r"\texttt{+aux}: \texttt{base} + reconstruction", "yes"),
    ("I2_recon_only", r"reconstruction only", "no"),
    # collage runs: final checkpoint (their "best" by validation SSE was the untrained step-1 network)
    ("I2_collage_final", r"collage only", "no"),
    ("I2_collage_c070_final", r"collage only, $s_{\max}\le 0.70$", "no"),
]
# variant -> reference that differs from it only in the matching (per-sample counts in the JSON dump,
# quoted in Section 6.5 and the response to Comment 2.2)
MATCHING_ONLY = {"I1_ordered": "I1_hung_nofp"}


def entry(label: str, step: int) -> dict | None:
    if label == "base":
        return load(BASE[step]) if step in BASE else None
    data = load(VE / f"{label}_s{step}_test256")
    if data is None and label == "aux_md4_3k" and step <= 30:
        data = load(Path(str(AUX_OLD).format(step=step)))
    return data


def fmt(value, spec: str) -> str:
    return "--" if value is None else format(value, spec)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = {s: entry("base", s) for s in STEPS}
    dump, main_rows, app_rows = {}, [], []
    for label, name, uses_theta in VARIANTS:
        rec = {}
        for s in STEPS:
            d = entry(label, s)
            if d is None:
                continue
            rec[s] = {k: mean(v) for k, v in d.items()}
            if label != "base" and base.get(s) is not None:
                rec[s]["base_better"] = sum(a < b for a, b in zip(base[s]["chamfer"], d["chamfer"]))
            ref = entry(MATCHING_ONLY[label], s) if label in MATCHING_ONLY else None
            if ref is not None:
                rec[s][f"{MATCHING_ONLY[label]}_better"] = sum(
                    a < b for a, b in zip(ref["chamfer"], d["chamfer"]))
        if not rec:
            print(f"missing {label}")
            continue
        dump[label] = rec
        get = lambda s, k: rec.get(s, {}).get(k)
        better = lambda s: "--" if label == "base" else fmt(get(s, "base_better"), "d")
        main_rows.append(
            f"    {name} & {uses_theta} & {fmt(get(0, 'chamfer'), '.4f')} & {fmt(get(30, 'chamfer'), '.4f')} & "
            f"{fmt(get(100, 'chamfer'), '.4f')} & {fmt(get(30, 'hd95'), '.4f')} & {fmt(get(30, 'coverage_2px'), '.3f')} & "
            f"{better(30)} \\\\")
        sse30 = get(30, "density_sse")
        app_rows.append(
            f"    {name} & " + " & ".join(fmt(get(s, "chamfer"), ".4f") for s in STEPS)
            + f" & {fmt(None if sse30 is None else sse30 * 1e4, '.2f')} & {fmt(get(0, 'param_distance'), '.3f')} & "
            f"{fmt(get(30, 'param_distance'), '.3f')} & "
            + ("--" if label == "base" else " / ".join(better(s) for s in (0, 30, 100, 1000))) + r" \\")
    (OUT / "training_signal_main_rows.tex").write_text("\n".join(main_rows) + "\n")
    (OUT / "training_signal_appendix_rows.tex").write_text("\n".join(app_rows) + "\n")
    (OUT / "training_signal_values.json").write_text(json.dumps(dump, indent=2))
    print("\n".join(main_rows), "\n---\n" + "\n".join(app_rows))


if __name__ == "__main__":
    main()
