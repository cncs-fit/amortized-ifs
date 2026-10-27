"""I2 check: are the reconstruction-only predictions near stationary points of the refinement objective?

For the test256 systems, evaluates at several parameter sets the gradient of exactly the objective that
refinement minimizes (``optimize_oracle.objective_losses`` + ``structural_penalties`` with the
arguments of ``run_variant_eval.sh``, same target point samples per batch of 32), and the distance
(Hungarian-matched (W,b) set distance) that refinement moves each prediction.

Parameter sets: ground truth, base and reconstruction-only at 0 / 100 / 1000 steps (stored outputs).
Output: ``outputs/rev1/I2_stationarity/{per_sample.csv,summary.json}``.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import scripts.optimize_oracle as oo
from data.sampler import affine_matrices_to_vector, params_to_affine
from scripts.analyze_identifiability import affine_set_distance
from scripts.evaluate_checkpoint import _config_from_payload

OUT = ROOT / "outputs/rev1/I2_stationarity"
VE = ROOT / "outputs/rev1/variant_eval"
SETS = {  # label -> (oracle_outputs.pt, key)
    "base-0": (ROOT / "outputs/oracle/paper_base50k_fresh_best_s0_test256_chamfer010_p512", "init_params"),
    "base-100": (VE / "base_s100_test256", "oracle_params"),
    "base-1000": (VE / "base_s1000_test256", "oracle_params"),
    "recon_only-0": (VE / "I2_recon_only_s0_test256", "init_params"),
    "recon_only-100": (VE / "I2_recon_only_s100_test256", "oracle_params"),
    "recon_only-1000": (VE / "I2_recon_only_s1000_test256", "oracle_params"),
}
ARGV = [  # the refinement arguments of run_variant_eval.sh
    "--run-dir", "checkpoints/paper_base50k", "--split", "test", "--num-samples", "256",
    "--seed", "4100", "--batch-size", "32", "--init-mode", "model", "--steps", "1", "--lr", "0.005",
    "--reconstruction-resolution", "128", "--reconstruction-num-trajectories", "16",
    "--reconstruction-num-steps", "1024", "--reconstruction-burn-in", "128",
    "--reconstruction-smoothing-sigma", "2.0", "--reconstruction-seed", "12345",
    "--eval-reconstruction-seed", "22318", "--reconstruction-map-probability-mode", "determinant",
    "--reconstruction-match-render-config", "--point-chamfer-loss-weight", "0.10",
    "--point-chamfer-num-pred-points", "512", "--point-chamfer-num-target-points", "512",
    "--point-chamfer-seed", "24680", "--device", "cuda", "--output-dir", "/dev/null",
]


def gradient_norms(params: torch.Tensor, images: torch.Tensor, *, args, config, device) -> tuple[list[float], list[float]]:
    norms, values = [], []
    for start in range(0, params.shape[0], int(args.batch_size)):
        img = images[start : start + args.batch_size].to(device)
        target_points = oo.density_images_to_point_samples(
            img, resolution=args.reconstruction_resolution, fixed_range=config.fixed_range,
            num_points=int(args.point_chamfer_num_target_points), seed=int(args.point_chamfer_seed) + start,
        )  # as in optimize_batch (one restart)
        p = params[start : start + args.batch_size].to(device).clone().requires_grad_(True)
        recon, chamfer = oo.objective_losses(p, img, args=args, config=config, target_points=target_points)
        penalty = oo.structural_penalties(
            p, spectral_upper_bound=args.spectral_upper_bound, spectral_penalty_weight=args.spectral_penalty_weight,
            det_negative_penalty_weight=args.det_negative_penalty_weight, translation_bound=args.translation_bound,
            translation_penalty_weight=args.translation_penalty_weight,
        )
        objective = recon + chamfer + penalty  # per sample; samples are independent
        (grad,) = torch.autograd.grad(objective.sum(), p)
        norms += grad.reshape(grad.shape[0], -1).norm(dim=1).tolist()
        values += objective.detach().tolist()
    return norms, values


def main() -> None:
    sys.argv = [sys.argv[0]] + ARGV
    args = oo.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = json.loads((ROOT / args.run_dir / "result.json").read_text())
    config = _config_from_payload(result["train_config"])
    oo.apply_reconstruction_match_render_config(args, config)
    OUT.mkdir(parents=True, exist_ok=True)
    ref = torch.load(SETS["base-0"][0] / "oracle_outputs.pt", map_location="cpu", weights_only=False)
    images = ref["images"].float()
    w, b = params_to_affine(ref["target_params"].float())
    sets = {"truth": affine_matrices_to_vector(w, b)}
    for label, (path, key) in SETS.items():
        payload = torch.load(path / "oracle_outputs.pt", map_location="cpu", weights_only=False)
        assert torch.equal(payload["images"].float(), images), path
        sets[label] = payload[key].float()
    rows = [{"index": i} for i in range(images.shape[0])]
    for label, params in sets.items():
        norms, values = gradient_norms(params, images, args=args, config=config, device=device)
        for i in range(len(rows)):
            rows[i][f"{label}_grad_norm"] = norms[i]
            rows[i][f"{label}_objective"] = values[i]
        print(label, "median grad norm", statistics.median(norms), flush=True)
    for method in ("base", "recon_only"):
        for steps in (100, 1000):
            for i in range(len(rows)):
                rows[i][f"{method}_moved_0_{steps}"] = affine_set_distance(
                    sets[f"{method}-0"][i], sets[f"{method}-{steps}"][i])["param_distance"]
    with (OUT / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for key in rows[0]:
        if key == "index":
            continue
        v = [r[key] for r in rows]
        summary[key] = {"median": statistics.median(v), "mean": statistics.fmean(v)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: round(v["median"], 6) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
