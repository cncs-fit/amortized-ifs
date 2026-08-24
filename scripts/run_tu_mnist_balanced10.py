"""Run Tu et al. public image reconstruction on a balanced digit/class subset."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yyama/anaconda3/envs/dlifs/bin/python"

TARGET_SPECS = {
    "mnist": ("mnist_images", "generate_mnist.py"),
    "fmnist": ("fmnist_images", "generate_fmnist.py"),
    "kmnist": ("kmnist_images", "generate_kmnist.py"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tu-root", default="refs/LearningFractals")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-log-dir", default="outputs/tu_mnist_p0/logs")
    parser.add_argument("--target", choices=tuple(TARGET_SPECS.keys()), default="mnist")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--labels", type=int, nargs="+", default=tuple(range(10)))
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=None)
    parser.add_argument("--samples-per-digit", type=int, default=None)
    parser.add_argument("--num-transforms", type=int, nargs="+", default=(4, 10))
    parser.add_argument("--num-coords", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--tar-batch-size", type=int, default=1)
    parser.add_argument("--gen-batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--init-seed", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def inner_dir(args: argparse.Namespace, num_transforms: int) -> str:
    return (
        f"img{args.image_size}"
        f"_nc{args.num_coords}"
        f"_tbs{args.tar_batch_size}"
        f"_gbs{args.gen_batch_size}"
        f"_nt{num_transforms}"
        f"_lr{args.lr}"
        f"_std{args.std}"
        f"_n{args.noise}"
        f"_initseed{args.init_seed}"
    )


def sample_indices(args: argparse.Namespace) -> list[int]:
    if args.sample_indices is not None:
        return [int(value) for value in args.sample_indices]
    if args.samples_per_digit is not None:
        return list(range(int(args.samples_per_digit)))
    return [int(args.sample_idx)]


def checkpoint_path(args: argparse.Namespace, *, num_transforms: int, label: int, sample_idx: int) -> Path:
    return (
        Path(args.tu_root)
        / f"IMAGEMATCH_{args.target.upper()}"
        / inner_dir(args, num_transforms)
        / f"IDX{label}-Sample{sample_idx}"
        / "iter1000_opti_ifs_code.pth"
    )


def env_for_tu(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    stub = ROOT / "scripts" / "tu_cv2_stub"
    tu_root = ROOT / args.tu_root
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(stub), str(tu_root), existing]) if existing else os.pathsep.join([str(stub), str(tu_root)])
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return env


def ensure_target_images(args: argparse.Namespace, *, env: dict[str, str]) -> None:
    tu_root = ROOT / args.tu_root
    image_dir, generator_script = TARGET_SPECS[str(args.target)]
    existing = list((tu_root / image_dir).glob("*/*.png"))
    if len(existing) >= 10_000:
        return
    subprocess.run(
        [PYTHON, generator_script, args.data_root],
        cwd=tu_root,
        env=env,
        check=True,
    )


def run_one(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    num_transforms: int,
    label: int,
    sample_idx: int,
    log_path: Path,
) -> tuple[str, float]:
    ckpt = checkpoint_path(args, num_transforms=num_transforms, label=label, sample_idx=sample_idx)
    if ckpt.exists() and not args.force:
        return "skipped", 0.0
    command = [
        PYTHON,
        "train_deep_fractal.py",
        "--target",
        str(args.target),
        "--idx",
        str(label),
        "--sample_idx",
        str(sample_idx),
        "--num_coords",
        str(args.num_coords),
        "--image_size",
        str(args.image_size),
        "--num_transforms",
        str(num_transforms),
        "--tar_batch_size",
        str(args.tar_batch_size),
        "--gen_batch_size",
        str(args.gen_batch_size),
        "--lr",
        str(args.lr),
        "--std",
        str(args.std),
        "--noise",
        str(args.noise),
        "--init_seed",
        str(args.init_seed),
    ]
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT / args.tu_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return "ran", time.perf_counter() - started


def main() -> None:
    args = parse_args()
    env = env_for_tu(args)
    ensure_target_images(args, env=env)
    rows = []
    for num_transforms in args.num_transforms:
        for label in args.labels:
            for sample_idx in sample_indices(args):
                log_path = (
                    ROOT
                    / args.output_log_dir
                    / f"tu_{args.target}_nt{num_transforms}_digit{label}_sample{sample_idx}.log"
                )
                print(
                    f"Tu {str(args.target).upper()} nt={num_transforms} "
                    f"label={label} sample={sample_idx}",
                    flush=True,
                )
                status, elapsed = run_one(
                    args,
                    env=env,
                    num_transforms=int(num_transforms),
                    label=int(label),
                    sample_idx=int(sample_idx),
                    log_path=log_path,
                )
                rows.append(
                    {
                        "num_transforms": int(num_transforms),
                        "label": int(label),
                        "sample_idx": int(sample_idx),
                        "status": status,
                        "elapsed_sec": elapsed,
                        "checkpoint": str(
                            checkpoint_path(
                                args,
                                num_transforms=int(num_transforms),
                                label=int(label),
                                sample_idx=int(sample_idx),
                            )
                        ),
                        "log": str(log_path),
                    }
                )
                print(f"  {status} elapsed={elapsed:.1f}s", flush=True)
    suffix = f"balanced{len(args.labels) * len(sample_indices(args))}"
    summary_path = ROOT / args.output_log_dir / f"tu_{args.target}_{suffix}_runs.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit code {exc.returncode}", file=sys.stderr)
        raise
