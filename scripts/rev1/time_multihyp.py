"""I2: inference time of the multi-hypothesis head, with and without the observation-only selection.

Measures, on the test256 images and an otherwise idle GPU, the wall-clock time per image of
  * base: one forward pass of the standard estimator;
  * MH forward: one forward pass of the K=4 head (all candidates);
  * MH + selection: forward pass plus rendering every candidate once and choosing the one with the
    lowest density SSE against the observation (``Rev1Estimator.forward`` in ``optimize_rev1.py``),
at batch size 1 (latency) and 32 (the evaluation batch).  CUDA is synchronized around each call;
the median over repeats is reported.
Output: ``outputs/rev1/timing/multihyp.json`` (``--device cpu --num-images 8 --repeats 1`` for a
smoke test, written to ``multihyp_smoke.json``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.evaluate_checkpoint import _build_model, _config_from_payload
from scripts.rev1.optimize_rev1 import Rev1Estimator

BASE_RUN = ROOT / "checkpoints/paper_base50k"
MH_RUN = ROOT / "checkpoints/rev1/I2_multihyp4"
IMAGES = ROOT / "outputs/rev1/variant_eval/base_s100_test256/oracle_outputs.pt"
OUT = ROOT / "outputs/rev1/timing"


def load(run: Path, device) -> tuple[torch.nn.Module, dict]:
    result = json.loads((run / "result.json").read_text())
    args = dict(result["args"])
    k = int(args.get("num_hypotheses", 1))
    payload = dict(args, num_transforms=int(args["num_transforms"]) * k)
    inner = _build_model(payload, device=device)
    inner.load_state_dict(torch.load(run / "best_model.pt", map_location=device))
    inner.eval()
    if k == 1:
        return inner, result
    model = Rev1Estimator(inner, num_transforms=int(args["num_transforms"]), num_hypotheses=k,
                          constraint=args.get("output_constraint", "none"), radius=args.get("spectral_radius", 0.95),
                          train_config=_config_from_payload(result["train_config"]),
                          seed=int(args.get("val_recon_seed", 777))).to(device)
    return model, result


@torch.no_grad()
def per_image_ms(fn, images: torch.Tensor, batch: int, repeats: int) -> float:
    chunks = [images[i:i + batch] for i in range(0, images.shape[0], batch)]
    sync = torch.cuda.synchronize if images.is_cuda else (lambda: None)
    for c in chunks[:2]:  # warm-up
        fn(c)
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for c in chunks:
            fn(c)
        sync()
        times.append((time.perf_counter() - start) / images.shape[0] * 1e3)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--base-run", type=Path, default=BASE_RUN)
    parser.add_argument("--mh-run", type=Path, default=MH_RUN)
    args = parser.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    images = torch.load(IMAGES, map_location="cpu", weights_only=False)["images"].float()[: args.num_images].to(device)
    base, _ = load(ROOT / args.base_run, device)  # ROOT / absolute path keeps the absolute path
    mh, _ = load(ROOT / args.mh_run, device)
    report = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "num_images": int(images.shape[0]), "repeats": args.repeats}
    for batch in (1, 32):
        report[f"batch{batch}"] = {
            "base_forward_ms": per_image_ms(base, images, batch, args.repeats),
            "mh_forward_ms": per_image_ms(mh.inner, images, batch, args.repeats),
            "mh_forward_select_ms": per_image_ms(mh, images, batch, args.repeats),
        }
    name = "multihyp.json" if args.num_images == 256 and device.type == "cuda" else "multihyp_smoke.json"
    (OUT / name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
