"""Checks for the revision-1 training variants (scripts/rev1/train_rev1.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.sampler import sample_ifs_parameters_batch
from losses.hungarian import hungarian_matching_loss_affine
from scripts.rev1.train_rev1 import (
    REV1_DEFAULTS,
    multi_hypothesis_loss,
    ordered_fixed_point_loss,
    per_sample_matching_losses,
    spectral_rescale,
    target_affine_vectors,
)


def _random_targets(batch: int, n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return sample_ifs_parameters_batch(batch, n, phase="phase0", generator=generator)


class Rev1LossTest(unittest.TestCase):
    def test_permutation_matching_equals_hungarian(self) -> None:
        torch.manual_seed(0)
        target = _random_targets(16, 4, seed=1)
        pred = target_affine_vectors(target) + 0.3 * torch.randn(16, 4, 6)
        for fp_weight in (0.0, 0.05):
            ours = per_sample_matching_losses(
                pred, target_affine_vectors(target), fixed_point_loss_weight=fp_weight
            ).mean()
            ref = hungarian_matching_loss_affine(pred, target, fixed_point_loss_weight=fp_weight)
            self.assertAlmostEqual(float(ours), float(ref), places=5)

    def test_ordered_loss_zero_at_sorted_target_and_not_permutation_invariant(self) -> None:
        target = _random_targets(8, 4, seed=2)
        affine = target_affine_vectors(target)
        w = affine[..., 0:4].reshape(8, 4, 2, 2)
        b = affine[..., 4:6]
        eye = torch.eye(2).expand_as(w)
        fp = torch.linalg.solve((eye - w).transpose(-1, -2), b.unsqueeze(-1)).squeeze(-1)
        order = torch.argsort(fp[..., 0], dim=1)
        sorted_affine = affine.gather(1, order.unsqueeze(-1).expand(-1, -1, 6))
        self.assertLess(float(ordered_fixed_point_loss(sorted_affine, target, fixed_point_loss_weight=0.05)), 1e-10)
        shuffled = sorted_affine[:, torch.tensor([1, 0, 3, 2])]
        self.assertGreater(float(ordered_fixed_point_loss(shuffled, target, fixed_point_loss_weight=0.0)), 1e-3)

    def test_multi_hypothesis_picks_correct_hypothesis(self) -> None:
        target = _random_targets(4, 4, seed=3)
        affine = target_affine_vectors(target)
        noise = torch.randn(4, 3, 4, 6)
        hyp = affine.unsqueeze(1) + noise
        hyp[:, 2] = affine  # hypothesis 2 is exact
        loss, winner = multi_hypothesis_loss(hyp, target, fixed_point_loss_weight=0.0, epsilon=0.0)
        self.assertTrue(torch.equal(winner, torch.full((4,), 2)))
        self.assertLess(float(loss), 1e-10)

    def test_spectral_rescale_bounds_singular_values(self) -> None:
        torch.manual_seed(4)
        affine = torch.randn(64, 4, 6)
        out = spectral_rescale(affine, 0.95)
        smax = torch.linalg.svdvals(out[..., 0:4].reshape(64, 4, 2, 2)).amax(dim=-1)
        self.assertTrue(bool((smax <= 0.95 + 1e-5).all()))
        small = 0.1 * affine
        small_smax = torch.linalg.svdvals(small[..., 0:4].reshape(64, 4, 2, 2)).amax(dim=-1)
        keep = small_smax < 0.95
        self.assertTrue(torch.allclose(spectral_rescale(small, 0.95)[keep], small[keep], atol=1e-6))
        self.assertTrue(torch.equal(spectral_rescale(affine, 0.95)[..., 4:6], affine[..., 4:6]))

    def test_defaults_reproduce_base_options(self) -> None:
        self.assertEqual(REV1_DEFAULTS["target_matching"], "hungarian")
        self.assertEqual(REV1_DEFAULTS["supervision"], "theta")
        self.assertEqual(REV1_DEFAULTS["num_hypotheses"], 1)
        self.assertEqual(REV1_DEFAULTS["output_constraint"], "none")
        self.assertEqual(REV1_DEFAULTS["head_init"], "identity")


if __name__ == "__main__":
    unittest.main()
