"""I7-a2: run the official optimizer of Djeacoumar et al. (2025) on test256 codes.

Run in the chaotic-fractals env (third_party/setup_chaotic_env.sh) (authors' code in refs/chaotic-fractals, unchanged except
for the one-line <cstdint> build fix):

    conda run -n chaotic-fractals python scripts/rev1/i7a2_run_official.py [--names t256_000 ...]

For each code written by ``i7a2_make_inputs.py``:
  1. the target image is drawn exactly as by the authors' ground-truth generator
     (``data_gen/ground_truth_generator.py``: 5e7 points, 1024^2, 16x supersampling);
  2. it is optimized with the authors' default configuration (``configs/test_suite_ours.json``:
     10 maps, 15000 iterations, seed 5) through the logic of ``scripts/chaotic_fractals.py``;
  3. the optimized code is exported to JSON (column-vector convention, per-map opacity) together
     with the wall-clock time of the optimization.
Codes that already have an exported result are skipped, so the script can be resumed.
Protocol checks (revision, I7-a2b): ``--num-maps 4 --suffix _n4`` runs their optimizer with the true
number of maps; ``--benchmark --suffix _bench`` runs it on systems of their own benchmark (their target
images), to compare with the results they publish.
Output: ``outputs/rev1/I7_official/{targets/<name>.png, optimized/<name>.json, timing.jsonl}``;
the authors' logs go to ``refs/chaotic-fractals/base/log/rev1_i7a2/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "refs/chaotic-fractals/base"
BENCH_IMAGES = ROOT / "refs/chaotic_data/full_dataset/ifs"  # the benchmark's own target images
OUT = ROOT / "outputs/rev1/I7_official"
os.chdir(REPO)  # the authors' code resolves configs/ and log/ relative to its base directory
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from cf.images.image_io import load_image, save_image  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402
from system.model_inference import ModelInfer, batch_point_gen  # noqa: E402
from system.renderer import SuperSampleRender  # noqa: E402
from trainers.trainer import FractalTrainer  # noqa: E402
from utils.optim_utils import Parser, safe_state  # noqa: E402


def render_target(code: dict, path: Path) -> None:
    safe_state(1)  # as in ground_truth_generator.py
    generator = ModelInfer(optimized_code_path=edict(code), lf=False, naive=True)
    points = batch_point_gen(generator, int(1e8 / 2))
    image = SuperSampleRender(points, 1024, 16, np.ones(3, dtype=np.float32), 0)
    save_image(image, str(path))


def export(pth: Path, path: Path) -> None:
    code = torch.load(pth, map_location="cpu", weights_only=False)
    ms, ts, ops = [], [], []
    for fn in code.contractive_functions:
        ll = fn.ll
        with torch.no_grad():
            ms.append(ll.weights().detach().cpu().reshape(2, 2).tolist())
            bias = ll.bias if torch.is_tensor(ll.bias) else ll.bias()
            ts.append(bias.detach().cpu().reshape(2).tolist())
            ops.append(float(torch.clamp(ll.opacity.detach().cpu(), 0.0, 1.0)))
    path.write_text(json.dumps({"ifs_m": ms, "ifs_t": ts, "opacity": ops}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--config", default="test_suite_ours")
    parser.add_argument("--iters", type=int, default=None, help="override training_iters (smoke tests only)")
    parser.add_argument("--tag", default="rev1_i7a2")
    parser.add_argument("--num-maps", type=int, default=None, help="override num_matrices (default: config, 10)")
    parser.add_argument("--suffix", default="", help="write optimized<suffix>/ and timing<suffix>.jsonl")
    parser.add_argument("--benchmark", action="store_true",
                        help="names are benchmark systems (fdb_*); optimize their own target images")
    args = parser.parse_args()
    names = args.names or json.loads((OUT / "subset.json").read_text())["names"]
    optimized_dir, timing_file = OUT / f"optimized{args.suffix}", OUT / f"timing{args.suffix}.jsonl"
    for d in (OUT / "targets", optimized_dir):
        d.mkdir(parents=True, exist_ok=True)
    for name in names:
        result = optimized_dir / f"{name}.json"
        if result.exists() and args.iters is None:
            print(f"skip {name}", flush=True)
            continue
        if args.benchmark:
            target = BENCH_IMAGES / f"{name}.png"
        else:
            target = OUT / "targets" / f"{name}.png"
            if not target.exists():
                render_target(json.loads((OUT / "codes" / f"{name}.json").read_text()), target)
        cfg = Parser(os.path.join("configs", f"{args.config}.json"))
        safe_state(cfg.config.optimizer.seed)
        if args.iters is not None:
            cfg.config.optimizer.training_iters = int(args.iters)
        if args.num_maps is not None:
            cfg.config.optimizer.num_matrices = int(args.num_maps)
        n_maps = cfg.config.optimizer.num_matrices
        cfg.config.logdirs.expt_name = f"{args.tag}/N_{n_maps}/{name}"
        started = time.time()
        trainer = FractalTrainer(cfg, load_image(str(target)))
        trainer.train()
        elapsed = time.time() - started
        pth = REPO / "log" / cfg.config.logdirs.expt_name / "output" / "optimized_ifs_code.pth"
        if args.iters is None:
            export(pth, result)
            with timing_file.open("a") as handle:
                handle.write(json.dumps({"name": name, "seconds": elapsed, "num_maps": n_maps,
                                         "iters": cfg.config.optimizer.training_iters,
                                         "gpu": torch.cuda.get_device_name(0)}) + "\n")
        else:
            export(pth, OUT / f"smoke_{name}.json")
        print(f"done {name} {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
