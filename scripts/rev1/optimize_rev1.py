"""Run ``scripts/optimize_oracle.py`` unchanged on revision-1 checkpoints.

Checkpoints from ``scripts/rev1/train_rev1.py`` that use the standard output (ordered,
Hungarian without fixed-point term, reconstruction-only, collage) already work with
``optimize_oracle.py``.  This wrapper additionally handles
  * multi-hypothesis models (head emits K*n maps): the hypothesis used as the
    initialization is selected per image by reconstruction SSE against the observed
    image only (soft renderer with matched settings, fixed seed) -- no ground truth;
  * the spectrally constrained output (``output_constraint: spectral``).
Usage: identical to optimize_oracle.py (same CLI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import scripts.optimize_oracle as oo
from scripts.evaluate_checkpoint import _config_from_payload
from scripts.rev1.train_rev1 import recon_sse, spectral_rescale

_ORIGINAL_BUILD = oo._build_model


class Rev1Estimator(torch.nn.Module):
    def __init__(self, inner, *, num_transforms, num_hypotheses, constraint, radius, train_config, seed):
        super().__init__()
        self.inner = inner
        self.n = int(num_transforms)
        self.k = int(num_hypotheses)
        self.constraint = constraint
        self.radius = float(radius)
        self.train_config = train_config
        self.seed = int(seed)
        self.last_selection: list[torch.Tensor] = []

    def load_state_dict(self, state_dict, strict: bool = True):  # checkpoints store the bare estimator
        return self.inner.load_state_dict(state_dict, strict=strict)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raw = self.inner(images).reshape(images.shape[0], self.k, self.n, 6)
        if self.constraint == "spectral":
            raw = spectral_rescale(raw, self.radius)
        if self.k == 1:
            return raw[:, 0]
        batch = images.shape[0]
        sse = recon_sse(
            raw.reshape(batch * self.k, self.n, 6),
            images.repeat_interleave(self.k, dim=0),
            train_config=self.train_config,
            seed=self.seed,
        ).reshape(batch, self.k)
        choice = sse.argmin(dim=1)
        self.last_selection.append(choice.cpu())
        return raw[torch.arange(batch, device=raw.device), choice]


def build(args_payload: dict, *, device: torch.device) -> torch.nn.Module:
    k = int(args_payload.get("num_hypotheses", 1))
    payload = dict(args_payload)
    payload["num_transforms"] = int(args_payload["num_transforms"]) * k
    inner = _ORIGINAL_BUILD(payload, device=device)
    run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1])
    result = json.loads((ROOT / run_dir / "result.json").read_text() if not run_dir.is_absolute() else (run_dir / "result.json").read_text())
    return Rev1Estimator(
        inner,
        num_transforms=int(args_payload["num_transforms"]),
        num_hypotheses=k,
        constraint=args_payload.get("output_constraint", "none"),
        radius=args_payload.get("spectral_radius", 0.95),
        train_config=_config_from_payload(result["train_config"]),
        seed=int(args_payload.get("val_recon_seed", 777)),
    ).to(device)


if __name__ == "__main__":
    oo._build_model = build
    oo.main()
