"""I2-V4c diagnosis: does the collage objective prefer the true IFS on our observations?

For the test256 images we evaluate the training objective of the amortized-collage variants
(``collage_loss`` in ``train_rev1.py``: density collage term + 0.1 x point collage term, penalties
excluded) at the ground truth and at the 0-step predictions of several networks.  If a prediction
scores below the ground truth, the failure is in the objective, not in the optimization.
CPU only (the GPU is reserved for timed runs).

    python scripts/rev1/analyze_collage.py [--codes label=<variant-eval dir> ...] [--out DIR] [--config-run DIR]

Each directory must hold ``oracle_outputs.pt`` (``init_params`` = 0-step prediction) for test256.
Default output: ``outputs/rev1/I2_collage_c070/analysis/{per_sample.csv,summary.json}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import params_to_affine
from scripts.rev1.train_rev1 import collage_loss

EVAL = ROOT / "outputs/rev1/variant_eval"
OUT = ROOT / "outputs/rev1/I2_collage_c070/analysis"
CODES = {  # label -> variant-eval directory whose init_params are the 0-step predictions
    "collage": EVAL / "I2_collage_final_s0_test256",
    "collage_c070": EVAL / "I2_collage_c070_final_s0_test256",
    "base": EVAL / "base_s100_test256",
}


def terms(affine: torch.Tensor, images: torch.Tensor, opts, train_config) -> tuple[list[float], list[float]]:
    """Per-image terms (collage_loss reports batch means, so it is called one image at a time)."""
    density, chamfer = [], []
    for i in range(affine.shape[0]):
        _, e = collage_loss(affine[i:i + 1], images[i:i + 1], opts=opts, train_config=train_config, step=0)
        density.append(e["collage_density"])
        chamfer.append(e["collage_chamfer"])
    return density, chamfer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", default=None, help="label=dir pairs (default: CODES)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--config-run", type=Path, default=ROOT / "checkpoints/rev1/I2_collage_c070",
                        help="collage training run (or its parent) whose result.json gives the objective settings")
    args = parser.parse_args()
    code_dirs = dict(CODES) if not args.codes else {k: Path(v) for k, v in (c.split("=", 1) for c in args.codes)}
    out_dir = args.out
    torch.set_grad_enabled(False)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_run = ROOT / args.config_run
    run = json.loads(next(iter(sorted(config_run.glob("result.json")) or sorted(config_run.glob("*/result.json")))).read_text())
    train_config = SimpleNamespace(**run["train_config"])
    opts = SimpleNamespace(**run["rev1"])
    opts.spectral_penalty_weight = opts.det_negative_penalty_weight = opts.translation_penalty_weight = 0.0
    ref = torch.load(EVAL / "base_s100_test256" / "oracle_outputs.pt", map_location="cpu", weights_only=False)
    images = ref["images"].float()
    w, b = params_to_affine(ref["target_params"].float())
    codes = {"truth": torch.cat((w.reshape(*w.shape[:-2], 4), b), dim=-1)}
    for label, path in code_dirs.items():
        payload = torch.load(path / "oracle_outputs.pt", map_location="cpu", weights_only=False)
        assert torch.equal(payload["images"].float(), images), path
        codes[label] = payload["init_params"].float()
    n = images.shape[0]
    rows = [{"index": i} for i in range(n)]
    for label, affine in codes.items():
        density, chamfer = terms(affine, images, opts, train_config)
        for i in range(n):
            rows[i][f"{label}_density"] = density[i]
            rows[i][f"{label}_chamfer"] = chamfer[i]
            rows[i][f"{label}_objective"] = density[i] + opts.collage_chamfer_weight * chamfer[i]
        print(label, "done", flush=True)
    with (out_dir / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"num_images": n, "chamfer_weight": opts.collage_chamfer_weight}
    for label in codes:
        summary[label] = {k: statistics.fmean(r[f"{label}_{k}"] for r in rows) for k in ("density", "chamfer", "objective")}
        summary[label]["median_objective"] = statistics.median(r[f"{label}_objective"] for r in rows)
        if label != "truth":
            summary[label]["below_truth"] = sum(r[f"{label}_objective"] < r["truth_objective"] for r in rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
