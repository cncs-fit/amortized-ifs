"""Checks for the revision-1 evaluation code (multi-hypothesis selection, external-code conventions)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import affine_vector_probabilities, sample_ifs_parameters_batch
from losses.reconstruction import differentiable_density_from_affine_vector
from scripts.rev1.i7a2_score import align, bbox, code_affine
from scripts.rev1.optimize_rev1 import Rev1Estimator
from scripts.rev1.prepare_external_ifs import code_to_affine
from scripts.rev1.train_rev1 import collage_loss, target_affine_vectors

RENDER = SimpleNamespace(resolution=32, fixed_range=(-1.5, 1.5), num_trajectories=4, num_steps=128, burn_in=16,
                         density_smoothing_sigma=1.0)


def _targets(batch: int, seed: int) -> torch.Tensor:
    params = sample_ifs_parameters_batch(batch, 4, phase="phase0", generator=torch.Generator().manual_seed(seed))
    return target_affine_vectors(params)


@torch.no_grad()
def _observe(affine: torch.Tensor, *, seed: int, num_trajectories: int = 4, num_steps: int = 128) -> torch.Tensor:
    return differentiable_density_from_affine_vector(
        affine, resolution=32, fixed_range=(-1.5, 1.5), num_trajectories=num_trajectories, num_steps=num_steps,
        burn_in=16, smoothing_sigma=1.0, seed=seed, map_probability_mode="determinant")


class _Fixed(torch.nn.Module):
    def __init__(self, out: torch.Tensor):
        super().__init__()
        self.out = out

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.out


class Rev1EvalTest(unittest.TestCase):
    def test_multi_hypothesis_selection_uses_observation_error(self) -> None:
        truth = _targets(3, seed=1)
        decoy = _targets(3, seed=2)
        obs = _observe(truth, seed=7)  # rendered exactly as the selection renders its candidates
        # candidate order differs per sample: the truth is hypothesis 1 for sample 0, hypothesis 0 otherwise
        hyp = torch.stack((decoy, truth), dim=1)
        hyp[1:] = torch.stack((truth[1:], decoy[1:]), dim=1)
        model = Rev1Estimator(_Fixed(hyp.reshape(3, 8, 6)), num_transforms=4, num_hypotheses=2, constraint="none",
                              radius=0.95, train_config=RENDER, seed=7)
        with torch.no_grad():
            chosen = model(obs)
        self.assertTrue(torch.equal(model.last_selection[-1], torch.tensor([1, 0, 0])))
        self.assertTrue(torch.equal(chosen, truth))

    def test_external_code_convention(self) -> None:
        code = {"ifs_m": [[[0.5, 0.1], [0.0, 0.4]]], "ifs_t": [[0.2, -0.3]]}  # x' = M x + t (column vectors)
        affine, p = code_to_affine(code, benchmark_rule=True)
        w = affine[0, :4].reshape(2, 2)
        x = torch.tensor([0.7, -0.2])
        self.assertTrue(torch.allclose(x @ w + affine[0, 4:], torch.tensor(code["ifs_m"][0]) @ x + torch.tensor(code["ifs_t"][0])))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        self.assertTrue(torch.equal(code_affine(code), affine))

    def test_bounding_box_alignment(self) -> None:
        affine = _targets(1, seed=3)[0]
        probs = affine_vector_probabilities(affine[None])[0]
        center, half = bbox(affine, probs)
        self.assertTrue(torch.allclose(align(affine, probs, center, half), affine, atol=1e-5))  # self-alignment
        moved = align(affine, probs, center + torch.tensor([0.3, -0.2]), 0.5 * half)
        self.assertTrue(torch.equal(moved[:, :4], affine[:, :4]))  # conjugation keeps the linear parts
        new_center, new_half = bbox(moved, probs)
        self.assertTrue(torch.allclose(new_center, center + torch.tensor([0.3, -0.2]), atol=1e-3))
        self.assertAlmostEqual(new_half, 0.5 * half, places=3)

    def test_collage_loss_is_finite_and_prefers_truth_over_noise(self) -> None:
        truth = _targets(2, seed=4)
        obs = _observe(truth, seed=5, num_trajectories=8, num_steps=256)
        opts = SimpleNamespace(objective_num_points=256, objective_seed=0, collage_density_weight=1.0,
                               collage_chamfer_weight=0.1, spectral_penalty_weight=0.0, spectral_penalty_bound=0.75,
                               det_negative_penalty_weight=0.0, translation_penalty_weight=0.0)
        config = SimpleNamespace(resolution=32, fixed_range=(-1.5, 1.5))
        at_truth, _ = collage_loss(truth, obs, opts=opts, train_config=config, step=0)
        at_noise, _ = collage_loss(truth + 0.5 * torch.randn(truth.shape, generator=torch.Generator().manual_seed(6)),
                                   obs, opts=opts, train_config=config, step=0)
        self.assertTrue(bool(torch.isfinite(at_truth)))
        self.assertLess(float(at_truth), float(at_noise))


if __name__ == "__main__":
    unittest.main()
