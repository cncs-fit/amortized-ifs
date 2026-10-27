"""I2: per-hypothesis analysis of the multi-hypothesis (K=4, relaxed WTA) estimator on test256.

For every test image, all K hypotheses are scored against the ground truth: matched (W,b)
distance (Hungarian, as in Table ident_dist) and point Chamfer / HD95 of the attractor drawn
with the standard chaos game (16 x 1024, burn-in 128, p ~ |det W|).  Reports
  * the hypothesis selected from the observation only (density SSE, as in optimize_rev1.py),
  * the oracle best-of-K (by Chamfer and by (W,b) distance, uses the ground truth),
  * how often the observation-only choice equals the Chamfer oracle, slot usage,
  * diversity: mean pairwise (W,b) distance between the K hypotheses of one image.
Images / targets come from the 0-step variant evaluation so the test set is identical.
Output: ``outputs/rev1/I2_multihyp4/analysis/{summary.json,per_sample.csv}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import affine_matrices_to_vector, affine_vector_probabilities, params_to_affine
from scripts.analyze_identifiability import affine_set_distance
from scripts.evaluate_checkpoint import _build_model, _checkpoint_paths, _config_from_payload
from scripts.evaluate_oracle_fidelity import point_metrics
from scripts.rev1.eval_shift import eval_point_clouds
from scripts.rev1.train_rev1 import recon_sse

RUN = ROOT / "checkpoints/rev1/I2_multihyp4"
SOURCE = ROOT / "outputs/rev1/variant_eval/I2_multihyp4_s0_test256/oracle_outputs.pt"
OUT = ROOT / "outputs/rev1/I2_multihyp4/analysis"
POINT_SEED = 53100
MAX_POINTS = 2048


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN, help="multi-hypothesis training run")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    cli = parser.parse_args()
    run, out_dir = cli.run_dir if cli.run_dir.is_absolute() else ROOT / cli.run_dir, cli.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = json.loads((run / "result.json").read_text())
    args = dict(result["args"])
    n, k = int(args["num_transforms"]), int(args["num_hypotheses"])
    config = _config_from_payload(result["train_config"])
    fixed_range = tuple(config.fixed_range)
    model = _build_model({**args, "num_transforms": n * k}, device=device)
    ckpt = _checkpoint_paths(result, run_dir=run, checkpoint="best")["best"]
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    images = payload["images"].float()
    w, b = params_to_affine(payload["target_params"].float())
    target = affine_matrices_to_vector(w, b)
    batch = images.shape[0]
    with torch.no_grad():
        raw = torch.cat([model(images[i : i + 64].to(device)).cpu() for i in range(0, batch, 64)])
    hyp = raw.reshape(batch, k, n, 6)
    with torch.no_grad():
        sse = torch.cat([
            recon_sse(hyp[i : i + 16].reshape(-1, n, 6).to(device), images[i : i + 16].repeat_interleave(k, dim=0).to(device),
                      train_config=config, seed=int(args.get("val_recon_seed", 777))).cpu()
            for i in range(0, batch, 16)
        ]).reshape(batch, k)
    selected = sse.argmin(dim=1)
    # hypothesis used by the variant evaluation (its SSE noise depends on the batch, so near ties can differ)
    saved = (hyp - payload["init_params"].float()[:, None]).abs().amax(dim=(2, 3)).argmin(dim=1)

    target_clouds = eval_point_clouds(target, affine_vector_probabilities(target), seed=POINT_SEED,
                                      fixed_range=fixed_range, device=device, max_points=MAX_POINTS)
    flat = hyp.reshape(batch * k, n, 6)
    pred_clouds = eval_point_clouds(flat, affine_vector_probabilities(flat), seed=POINT_SEED + 7,
                                    fixed_range=fixed_range, device=device, max_points=MAX_POINTS)
    chamfer = torch.zeros(batch, k)
    hd95 = torch.zeros(batch, k)
    param = torch.zeros(batch, k)
    spread = torch.zeros(batch)
    for i in range(batch):
        for j in range(k):
            m = point_metrics(pred_clouds[i * k + j], target_clouds[i], device=device)
            chamfer[i, j], hd95[i, j] = m["chamfer"], m["hausdorff_p95"]
            param[i, j] = affine_set_distance(hyp[i, j], target[i])["param_distance"]
        spread[i] = torch.tensor([affine_set_distance(hyp[i, a], hyp[i, c])["param_distance"]
                                  for a, c in combinations(range(k), 2)]).mean()

    rows = torch.arange(batch)
    best_cd = chamfer.argmin(dim=1)
    best_param = param.argmin(dim=1)
    pick = lambda t, idx: t[rows, idx]
    summary = {
        "num_samples": batch, "num_hypotheses": k, "selection_agrees_with_variant_eval": float((selected == saved).float().mean()),
        "selected": {"chamfer": float(pick(chamfer, selected).mean()), "hd95": float(pick(hd95, selected).mean()),
                     "param": float(pick(param, selected).mean())},
        "oracle_chamfer": {"chamfer": float(pick(chamfer, best_cd).mean()), "hd95": float(pick(hd95, best_cd).mean()),
                           "param": float(pick(param, best_cd).mean())},
        "oracle_param": {"chamfer": float(pick(chamfer, best_param).mean()), "param": float(pick(param, best_param).mean())},
        "per_slot": {"chamfer": chamfer.mean(dim=0).tolist(), "param": param.mean(dim=0).tolist(),
                     "selected_count": torch.bincount(selected, minlength=k).tolist(),
                     "oracle_chamfer_count": torch.bincount(best_cd, minlength=k).tolist()},
        "worst_of_k_chamfer": float(chamfer.max(dim=1).values.mean()),
        "selection_equals_chamfer_oracle": float((selected == best_cd).float().mean()),
        "selection_equals_param_oracle": float((selected == best_param).float().mean()),
        "chamfer_oracle_equals_param_oracle": float((best_cd == best_param).float().mean()),
        "pairwise_hypothesis_param_distance": {"mean": float(spread.mean()), "median": float(spread.median())},
        "protocol": f"standard chaos game 16x1024 burn-in 128, p~|det|, {MAX_POINTS} points, seeds {POINT_SEED}/{POINT_SEED + 7}",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "selected", "oracle_chamfer", "oracle_param", "spread"]
                        + [f"chamfer_{j}" for j in range(k)] + [f"param_{j}" for j in range(k)] + [f"sse_{j}" for j in range(k)])
        for i in range(batch):
            writer.writerow([i, int(selected[i]), int(best_cd[i]), int(best_param[i]), f"{float(spread[i]):.5f}"]
                            + [f"{float(v):.6f}" for v in chamfer[i]] + [f"{float(v):.5f}" for v in param[i]]
                            + [f"{float(v):.6e}" for v in sse[i]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
