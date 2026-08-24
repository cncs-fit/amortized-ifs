import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from data.dataset import (
    IFSDatasetConfig,
    IFSIterableDataset,
    IFSIterableDatasetConfig,
    default_train_cache_path,
    generate_fixed_dataset,
    load_train_cache,
    make_fixed_dataset_from_iterable_config,
    save_train_cache_batched_from_iterable_config,
    save_train_cache_from_iterable_config,
)
from data.renderer import points_to_density_map, points_to_density_maps, smooth_density_map
from data.sampler import (
    affine_matrices_to_vector,
    affine_vector_to_matrices,
    create_ifs_from_params,
    iterate_affine_vector_points,
    iterate_ifs_points,
    iterate_ifs_points_batch,
    params_to_affine,
    sample_ifs_parameters,
    sample_ifs_parameters_batch,
)
from losses.hungarian import (
    affine_shape_auxiliary_loss,
    affine_matrix_regularization,
    hungarian_matching_loss,
    hungarian_matching_loss_affine,
    target_spectral_linear_weights,
)
from losses.hungarian import fixed_points_from_matrices, hungarian_metrics_affine
from losses.reconstruction import (
    density_reconstruction_loss_affine,
    differentiable_density_from_affine_vector,
    point_chamfer_loss_affine,
    point_chamfer_losses_affine,
    soft_points_to_density_maps,
)
from models.set_head import TinyCNNAffineSetEstimator, TinyCNNSetEstimator
from scripts.analyze_identifiability import affine_set_distance
from scripts.optimize_oracle import (
    density_images_to_point_samples,
    make_initial_restarts,
    optimize_batch,
    point_chamfer_losses,
    reconstruction_losses,
    structural_penalties,
)
from scripts.evaluate_oracle_fidelity import point_metrics


class PhaseMinusOneTests(unittest.TestCase):
    def test_params_to_affine_matches_affine_ifs(self):
        generator = torch.Generator().manual_seed(1)
        params = sample_ifs_parameters(3, generator=generator)
        expected_w, expected_b = params_to_affine(params)
        ifs = create_ifs_from_params(params)
        actual_w, _ = ifs.make_matrices_from_svdformat()
        self.assertTrue(torch.allclose(expected_w, actual_w, atol=1e-5))
        self.assertTrue(torch.allclose(expected_b, ifs.ifs_b.weight, atol=1e-6))

    def test_hungarian_loss_is_permutation_invariant(self):
        generator = torch.Generator().manual_seed(2)
        params = sample_ifs_parameters(4, generator=generator).unsqueeze(0)
        permuted = params[:, torch.tensor([2, 0, 3, 1])]
        loss = hungarian_matching_loss(permuted, params)
        self.assertLess(loss.item(), 1e-10)

    def test_direct_affine_hungarian_loss_is_permutation_invariant(self):
        generator = torch.Generator().manual_seed(20)
        params = sample_ifs_parameters(4, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).unsqueeze(0)
        permuted = affine[:, torch.tensor([2, 0, 3, 1])]
        loss = hungarian_matching_loss_affine(permuted, params.unsqueeze(0))
        self.assertLess(loss.item(), 1e-10)

    def test_direct_affine_vector_roundtrip(self):
        generator = torch.Generator().manual_seed(21)
        params = sample_ifs_parameters(3, generator=generator)
        w, b = params_to_affine(params)
        actual_w, actual_b = affine_vector_to_matrices(affine_matrices_to_vector(w, b))
        self.assertTrue(torch.allclose(w, actual_w))
        self.assertTrue(torch.allclose(b, actual_b))

    def test_identifiability_affine_set_distance_is_permutation_invariant(self):
        generator = torch.Generator().manual_seed(210)
        params = sample_ifs_parameters(4, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b)
        permuted = affine[torch.tensor([3, 1, 0, 2])]
        distances = affine_set_distance(permuted, affine)
        self.assertLess(distances["param_distance"], 1e-6)
        self.assertLess(distances["w_fro"], 1e-6)
        self.assertLess(distances["b_l2"], 1e-6)

    def test_fixed_points_satisfy_row_vector_affine_equation(self):
        generator = torch.Generator().manual_seed(22)
        params = sample_ifs_parameters(3, generator=generator)
        w, b = params_to_affine(params)
        fixed_points = fixed_points_from_matrices(w, b)
        mapped = torch.bmm(fixed_points.unsqueeze(1), w).squeeze(1) + b
        self.assertTrue(torch.allclose(fixed_points, mapped, atol=1e-5))

    def test_fixed_point_aux_loss_is_permutation_invariant(self):
        generator = torch.Generator().manual_seed(23)
        params = sample_ifs_parameters(4, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).unsqueeze(0)
        permuted = affine[:, torch.tensor([2, 0, 3, 1])]
        loss = hungarian_matching_loss_affine(
            permuted,
            params.unsqueeze(0),
            fixed_point_loss_weight=0.1,
            fixed_point_cost_weight=0.1,
        )
        self.assertLess(loss.item(), 1e-10)

    def test_affine_metrics_include_fixed_point_error(self):
        generator = torch.Generator().manual_seed(24)
        params = sample_ifs_parameters(2, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).unsqueeze(0)
        metrics = hungarian_metrics_affine(affine, params.unsqueeze(0))
        self.assertIn("fixed_point_l2", metrics)
        self.assertLess(metrics["fixed_point_l2"], 1e-6)

    def test_affine_matrix_regularization_penalizes_structural_violations(self):
        w = torch.tensor(
            [
                [[0.5, 0.0], [0.0, 0.4]],
                [[0.9, 0.0], [0.0, -0.2]],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(affine_matrix_regularization(w).item(), 0.0)
        penalty = affine_matrix_regularization(
            w,
            spectral_upper_bound=0.75,
            spectral_upper_loss_weight=1.0,
            det_positive_loss_weight=1.0,
        )
        self.assertGreater(penalty.item(), 0.0)

    def test_affine_shape_auxiliary_loss_matches_singular_values_and_determinant(self):
        target_w = torch.tensor(
            [
                [[0.5, 0.1], [0.0, 0.3]],
                [[0.2, -0.2], [0.1, 0.4]],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(
            affine_shape_auxiliary_loss(
                target_w,
                target_w,
                singular_value_loss_weight=1.0,
                determinant_loss_weight=1.0,
            ).item(),
            0.0,
        )
        pred_w = target_w * 0.5
        penalty = affine_shape_auxiliary_loss(
            pred_w,
            target_w,
            singular_value_loss_weight=1.0,
            determinant_loss_weight=1.0,
        )
        self.assertGreater(penalty.item(), 0.0)

    def test_direct_affine_shape_auxiliary_loss_is_permutation_invariant(self):
        generator = torch.Generator().manual_seed(25)
        params = sample_ifs_parameters(4, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).unsqueeze(0)
        permuted = affine[:, torch.tensor([2, 0, 3, 1])]
        loss = hungarian_matching_loss_affine(
            permuted,
            params.unsqueeze(0),
            singular_value_loss_weight=0.5,
            determinant_loss_weight=0.5,
        )
        self.assertLess(loss.item(), 1e-10)

    def test_target_spectral_linear_weights_emphasize_high_spectral_maps(self):
        params = torch.zeros(3, 6)
        params[:, 2] = torch.tensor([0.30, 0.55, 0.70])
        params[:, 3] = 0.20
        weights = target_spectral_linear_weights(
            params,
            extra_weight=1.0,
            threshold=0.55,
            upper=0.70,
        )
        self.assertTrue(torch.allclose(weights, torch.tensor([1.0, 1.0, 2.0])))

    def test_target_spectral_weighted_loss_is_zero_for_exact_match(self):
        generator = torch.Generator().manual_seed(26)
        params = sample_ifs_parameters(4, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).unsqueeze(0)
        loss = hungarian_matching_loss_affine(
            affine,
            params.unsqueeze(0),
            target_spectral_linear_extra_weight=1.0,
            target_spectral_linear_threshold=0.55,
            target_spectral_linear_upper=0.70,
        )
        self.assertLess(loss.item(), 1e-10)

    def test_render_density_shape_and_normalization(self):
        generator = torch.Generator().manual_seed(3)
        params = sample_ifs_parameters(2, generator=generator)
        points = iterate_ifs_points(params, num_trajectories=2, num_steps=32, burn_in=4, generator=generator)
        density = points_to_density_map(points, resolution=16)
        self.assertEqual(tuple(density.shape), (1, 16, 16))
        self.assertAlmostEqual(density.sum().item(), 1.0, places=5)

    def test_batched_sampling_and_render_density_shape(self):
        generator = torch.Generator().manual_seed(300)
        params = sample_ifs_parameters_batch(3, 2, generator=generator)
        self.assertEqual(tuple(params.shape), (3, 2, 6))
        points = iterate_ifs_points_batch(
            params,
            num_trajectories=2,
            num_steps=32,
            burn_in=4,
            generator=generator,
        )
        self.assertEqual(tuple(points.shape), (3, 56, 2))
        density = points_to_density_maps(points, resolution=16)
        self.assertEqual(tuple(density.shape), (3, 1, 16, 16))
        self.assertTrue(torch.allclose(density.sum(dim=(1, 2, 3)), torch.ones(3), atol=1e-5))

    def test_smooth_density_shape_and_normalization(self):
        density = torch.zeros(1, 16, 16)
        density[0, 8, 8] = 1.0
        smoothed = smooth_density_map(density, sigma=1.0)
        self.assertEqual(tuple(smoothed.shape), (1, 16, 16))
        self.assertAlmostEqual(smoothed.sum().item(), 1.0, places=5)
        self.assertLess(smoothed.max().item(), 1.0)

    def test_direct_affine_render_density_shape_and_normalization(self):
        generator = torch.Generator().manual_seed(31)
        params = sample_ifs_parameters(2, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b)
        points = iterate_affine_vector_points(
            affine,
            num_trajectories=2,
            num_steps=32,
            burn_in=4,
            generator=generator,
        )
        density = points_to_density_map(points, resolution=16)
        self.assertEqual(tuple(density.shape), (1, 16, 16))
        self.assertAlmostEqual(density.sum().item(), 1.0, places=5)

    def test_differentiable_affine_density_shape_and_normalization(self):
        generator = torch.Generator().manual_seed(32)
        params = sample_ifs_parameters(2, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).unsqueeze(0)
        density = differentiable_density_from_affine_vector(
            affine,
            resolution=16,
            num_trajectories=2,
            num_steps=24,
            burn_in=4,
            smoothing_sigma=0.5,
            seed=33,
        )
        self.assertEqual(tuple(density.shape), (1, 1, 16, 16))
        self.assertTrue(torch.allclose(density.sum(dim=(1, 2, 3)), torch.ones(1), atol=1e-5))
        det_density = differentiable_density_from_affine_vector(
            affine,
            resolution=16,
            num_trajectories=2,
            num_steps=24,
            burn_in=4,
            smoothing_sigma=0.5,
            seed=33,
            map_probability_mode="determinant",
        )
        self.assertEqual(tuple(det_density.shape), (1, 1, 16, 16))
        self.assertTrue(torch.allclose(det_density.sum(dim=(1, 2, 3)), torch.ones(1), atol=1e-5))

    def test_soft_density_matches_renderer_axis_convention_at_bin_centers(self):
        points = torch.tensor([[[-0.75, 0.75]]], dtype=torch.float32)
        soft_density = soft_points_to_density_maps(
            points,
            resolution=4,
            fixed_range=(-1.0, 1.0),
            smoothing_sigma=0.0,
        )
        hard_density = points_to_density_map(
            points[0],
            resolution=4,
            fixed_range=(-1.0, 1.0),
            smoothing_sigma=0.0,
        )
        self.assertTrue(torch.allclose(soft_density[0], hard_density, atol=1e-6))

    def test_density_reconstruction_loss_has_affine_gradients(self):
        generator = torch.Generator().manual_seed(34)
        params = sample_ifs_parameters(2, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).unsqueeze(0).clone().requires_grad_(True)
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 7:9, 7:9] = 0.25
        loss = density_reconstruction_loss_affine(
            affine,
            target,
            resolution=16,
            num_trajectories=2,
            num_steps=24,
            burn_in=4,
            smoothing_sigma=0.5,
            seed=35,
        )
        loss.backward()
        self.assertIsNotNone(affine.grad)
        self.assertGreater(affine.grad.abs().sum().item(), 0.0)

    def test_oracle_reconstruction_losses_are_per_sample(self):
        generator = torch.Generator().manual_seed(360)
        params = sample_ifs_parameters_batch(2, 2, generator=generator)
        w, b = params_to_affine(params)
        target_affine = affine_matrices_to_vector(w, b)
        target_images = differentiable_density_from_affine_vector(
            target_affine,
            resolution=8,
            num_trajectories=1,
            num_steps=12,
            burn_in=2,
            smoothing_sigma=0.25,
            seed=361,
        ).detach()
        pred_affine = (target_affine + 0.02).clone().requires_grad_(True)
        losses = reconstruction_losses(
            pred_affine,
            target_images,
            resolution=8,
            fixed_range=(-1.5, 1.5),
            num_trajectories=1,
            num_steps=12,
            burn_in=2,
            smoothing_sigma=0.25,
            seed=361,
            map_probability_mode="uniform",
        )
        self.assertEqual(tuple(losses.shape), (2,))
        losses.mean().backward()
        self.assertIsNotNone(pred_affine.grad)

    def test_density_images_to_point_samples_shape(self):
        images = torch.zeros(2, 1, 8, 8)
        images[:, :, 3:5, 3:5] = 1.0
        points = density_images_to_point_samples(
            images,
            resolution=8,
            fixed_range=(-1.0, 1.0),
            num_points=16,
            seed=368,
        )
        self.assertEqual(tuple(points.shape), (2, 16, 2))
        self.assertTrue(torch.isfinite(points).all())
        self.assertGreaterEqual(points.min().item(), -1.0)
        self.assertLessEqual(points.max().item(), 1.0)

    def test_point_chamfer_losses_have_affine_gradients(self):
        generator = torch.Generator().manual_seed(369)
        params = sample_ifs_parameters_batch(2, 2, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).clone().requires_grad_(True)
        target_points = torch.zeros(2, 12, 2)
        losses = point_chamfer_losses(
            affine,
            target_points,
            num_trajectories=1,
            num_steps=12,
            burn_in=2,
            seed=370,
            map_probability_mode="uniform",
            max_pred_points=8,
        )
        self.assertEqual(tuple(losses.shape), (2,))
        losses.mean().backward()
        self.assertIsNotNone(affine.grad)
        self.assertGreater(affine.grad.abs().sum().item(), 0.0)

    def test_reconstruction_module_point_chamfer_has_affine_gradients(self):
        generator = torch.Generator().manual_seed(371)
        params = sample_ifs_parameters_batch(2, 2, generator=generator)
        w, b = params_to_affine(params)
        affine = affine_matrices_to_vector(w, b).clone().requires_grad_(True)
        target_images = differentiable_density_from_affine_vector(
            affine.detach(),
            resolution=8,
            num_trajectories=1,
            num_steps=12,
            burn_in=2,
            smoothing_sigma=0.25,
            seed=372,
        ).detach()
        loss = point_chamfer_loss_affine(
            affine + 0.02,
            target_images,
            resolution=8,
            num_trajectories=1,
            num_steps=12,
            burn_in=2,
            seed=373,
            num_target_points=12,
            max_pred_points=8,
            target_seed=374,
        )
        loss.backward()
        self.assertIsNotNone(affine.grad)
        self.assertGreater(affine.grad.abs().sum().item(), 0.0)

    def test_reconstruction_module_point_chamfer_losses_are_per_sample(self):
        affine = torch.zeros(2, 2, 6)
        affine[:, :, 0] = 0.45
        affine[:, :, 3] = 0.45
        affine.requires_grad_(True)
        target_points = torch.zeros(2, 12, 2)
        losses = point_chamfer_losses_affine(
            affine,
            target_points,
            num_trajectories=1,
            num_steps=12,
            burn_in=2,
            seed=375,
            max_pred_points=8,
        )
        self.assertEqual(tuple(losses.shape), (2,))
        self.assertTrue(torch.isfinite(losses).all())

    def test_oracle_structural_penalties_flag_invalid_affines(self):
        affine = torch.zeros(2, 2, 6)
        affine[:, :, 0] = 0.9
        affine[:, :, 3] = 0.9
        affine[:, :, 4:6] = 2.0
        penalties = structural_penalties(
            affine,
            spectral_upper_bound=0.75,
            spectral_penalty_weight=1.0,
            det_negative_penalty_weight=1.0,
            translation_bound=1.5,
            translation_penalty_weight=1.0,
        )
        self.assertEqual(tuple(penalties.shape), (2,))
        self.assertGreater(penalties.mean().item(), 0.0)

    def test_oracle_restart_initialization_uses_model_first(self):
        generator = torch.Generator().manual_seed(362)
        params = sample_ifs_parameters_batch(2, 2, generator=generator)
        w, b = params_to_affine(params)
        model_init = affine_matrices_to_vector(w, b)
        restarts = make_initial_restarts(
            model_init=model_init,
            batch_size=2,
            num_transforms=2,
            phase="phase0",
            init_mode="mixed",
            restarts=3,
            init_noise=0.0,
            seed=363,
        )
        self.assertEqual(tuple(restarts.shape), (2, 3, 2, 6))
        self.assertTrue(torch.allclose(restarts[:, 0], model_init))

    def test_oracle_restart_initialization_can_use_target(self):
        generator = torch.Generator().manual_seed(366)
        params = sample_ifs_parameters_batch(2, 2, generator=generator)
        w, b = params_to_affine(params)
        target_init = affine_matrices_to_vector(w, b)
        restarts = make_initial_restarts(
            target_init=target_init,
            batch_size=2,
            num_transforms=2,
            phase="phase0",
            init_mode="target",
            restarts=1,
            init_noise=0.0,
            seed=367,
        )
        self.assertEqual(tuple(restarts.shape), (2, 1, 2, 6))
        self.assertTrue(torch.allclose(restarts[:, 0], target_init))

    def test_oracle_optimize_batch_returns_selected_params(self):
        generator = torch.Generator().manual_seed(364)
        params = sample_ifs_parameters_batch(2, 2, generator=generator)
        w, b = params_to_affine(params)
        target_affine = affine_matrices_to_vector(w, b)
        images = differentiable_density_from_affine_vector(
            target_affine,
            resolution=8,
            num_trajectories=1,
            num_steps=12,
            burn_in=2,
            smoothing_sigma=0.25,
            seed=365,
        ).detach()
        init_restarts = (target_affine + 0.05).unsqueeze(1)
        args = SimpleNamespace(
            lr=0.005,
            weight_decay=0.0,
            grad_clip=1.0,
            steps=2,
            reconstruction_resolution=8,
            reconstruction_num_trajectories=1,
            reconstruction_num_steps=12,
            reconstruction_burn_in=2,
            reconstruction_smoothing_sigma=0.25,
            reconstruction_seed=365,
            reconstruction_map_probability_mode="uniform",
            spectral_upper_bound=0.75,
            spectral_penalty_weight=0.1,
            det_negative_penalty_weight=0.1,
            translation_bound=1.5,
            translation_penalty_weight=0.01,
            log_interval=1,
        )
        config = SimpleNamespace(fixed_range=(-1.5, 1.5))
        selected, initial_losses, final_losses, selection_losses, history = optimize_batch(
            images,
            params,
            init_restarts,
            args=args,
            config=config,
            device=torch.device("cpu"),
            batch_start_index=0,
        )
        self.assertEqual(tuple(selected.shape), (2, 2, 6))
        self.assertEqual(tuple(initial_losses.shape), (2, 1))
        self.assertEqual(tuple(final_losses.shape), (2, 1))
        self.assertEqual(tuple(selection_losses.shape), (2, 1))
        self.assertGreaterEqual(len(history), 2)
        self.assertTrue(torch.isfinite(final_losses).all())
        self.assertTrue(torch.isfinite(selection_losses).all())

    def test_point_metrics_are_zero_for_identical_clouds(self):
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        metrics = point_metrics(
            points,
            points,
            device=torch.device("cpu"),
            coverage_thresholds={"1px": 0.1, "2px": 0.2},
        )
        self.assertLess(metrics["chamfer"], 1e-7)
        self.assertLess(metrics["hausdorff"], 1e-7)
        self.assertLess(metrics["hausdorff_p95"], 1e-7)
        self.assertLess(metrics["modified_hausdorff_mean"], 1e-7)
        self.assertAlmostEqual(metrics["coverage_symmetric_1px"], 1.0)
        self.assertAlmostEqual(metrics["coverage_target_to_pred_2px"], 1.0)

    def test_point_metrics_percentiles_are_less_outlier_sensitive(self):
        reference = torch.stack([torch.tensor([float(i), 0.0]) for i in range(20)])
        prediction = reference.clone()
        prediction[-1] = torch.tensor([100.0, 0.0])
        metrics = point_metrics(
            prediction,
            reference,
            device=torch.device("cpu"),
            coverage_thresholds={"1px": 0.5},
        )
        self.assertGreater(metrics["hausdorff"], 80.0)
        self.assertLess(metrics["hausdorff_p90"], metrics["hausdorff"])
        self.assertLess(metrics["hausdorff_p95"], metrics["hausdorff"])
        self.assertLess(metrics["coverage_pred_to_target_1px"], 1.0)
        self.assertLess(metrics["coverage_symmetric_1px"], 1.0)

    def test_generate_fixed_dataset_shapes(self):
        config = IFSDatasetConfig(
            num_samples=3,
            num_transforms=2,
            resolution=16,
            num_trajectories=2,
            num_steps=64,
            burn_in=8,
            seed=4,
        )
        dataset = generate_fixed_dataset(config)
        images, params = dataset.tensors
        self.assertEqual(tuple(images.shape), (3, 1, 16, 16))
        self.assertEqual(tuple(params.shape), (3, 2, 6))

    def test_generate_fixed_dataset_rejects_unknown_phase(self):
        config = IFSDatasetConfig(
            num_samples=1,
            num_transforms=2,
            phase="unknown",
            resolution=16,
            num_trajectories=1,
            num_steps=16,
            burn_in=4,
            seed=40,
        )
        with self.assertRaises(ValueError):
            generate_fixed_dataset(config)

    def test_iterable_dataset_yields_expected_shapes(self):
        config = IFSIterableDatasetConfig(
            num_transforms=2,
            resolution=16,
            num_trajectories=2,
            num_steps=64,
            burn_in=8,
            seed=5,
        )
        dataset = IFSIterableDataset(config)
        image, params = next(iter(dataset))
        self.assertEqual(tuple(image.shape), (1, 16, 16))
        self.assertEqual(tuple(params.shape), (2, 6))

    def test_make_fixed_dataset_from_iterable_config(self):
        config = IFSIterableDatasetConfig(
            num_transforms=2,
            resolution=16,
            num_trajectories=2,
            num_steps=64,
            burn_in=8,
            seed=6,
        )
        dataset = make_fixed_dataset_from_iterable_config(config, num_samples=2, seed=7)
        images, params = dataset.tensors
        self.assertEqual(tuple(images.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(params.shape), (2, 2, 6))

    def test_train_cache_roundtrip(self):
        config = IFSIterableDatasetConfig(
            num_transforms=2,
            resolution=16,
            num_trajectories=1,
            num_steps=32,
            burn_in=4,
            seed=41,
        )
        with TemporaryDirectory() as tmpdir:
            path = default_train_cache_path(
                config,
                num_samples=4,
                cache_dir=tmpdir,
                generation_num_workers=0,
            )
            dataset = save_train_cache_from_iterable_config(
                config,
                num_samples=4,
                path=path,
                batch_size=2,
                num_workers=0,
            )
            loaded, metadata = load_train_cache(path)

        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(loaded), 4)
        self.assertEqual(metadata["num_samples"], 4)
        self.assertTrue(torch.allclose(dataset.tensors[0], loaded.tensors[0]))
        self.assertTrue(torch.allclose(dataset.tensors[1], loaded.tensors[1]))

    def test_batched_train_cache_roundtrip(self):
        config = IFSIterableDatasetConfig(
            num_transforms=2,
            resolution=16,
            num_trajectories=1,
            num_steps=32,
            burn_in=4,
            seed=42,
        )
        with TemporaryDirectory() as tmpdir:
            path = default_train_cache_path(
                config,
                num_samples=4,
                cache_dir=tmpdir,
                generation_num_workers=0,
                generation_mode="batched",
                generation_device="cpu",
            )
            dataset = save_train_cache_batched_from_iterable_config(
                config,
                num_samples=4,
                path=path,
                batch_size=8,
                device="cpu",
            )
            loaded, metadata = load_train_cache(path)

        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(loaded), 4)
        self.assertEqual(metadata["generation_mode"], "batched")
        self.assertEqual(metadata["generation_device"], "cpu")
        self.assertTrue(torch.allclose(dataset.tensors[0], loaded.tensors[0]))
        self.assertTrue(torch.allclose(dataset.tensors[1], loaded.tensors[1]))

    def test_smoothed_fixed_dataset_shapes(self):
        config = IFSIterableDatasetConfig(
            num_transforms=2,
            resolution=16,
            num_trajectories=2,
            num_steps=64,
            burn_in=8,
            seed=8,
            density_smoothing_sigma=1.0,
        )
        dataset = make_fixed_dataset_from_iterable_config(config, num_samples=2, seed=9)
        images, params = dataset.tensors
        self.assertEqual(tuple(images.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(params.shape), (2, 2, 6))
        self.assertTrue(torch.allclose(images.sum(dim=(1, 2, 3)), torch.ones(2), atol=1e-5))

    def test_fixed_dataset_params_do_not_depend_on_render_settings(self):
        base_config = IFSIterableDatasetConfig(
            num_transforms=2,
            resolution=16,
            num_trajectories=2,
            num_steps=64,
            burn_in=8,
            seed=10,
            validity_resolution=16,
            validity_num_trajectories=2,
            validity_num_steps=64,
            validity_burn_in=8,
        )
        dense_config = IFSIterableDatasetConfig(
            num_transforms=2,
            resolution=16,
            num_trajectories=4,
            num_steps=96,
            burn_in=8,
            seed=10,
            density_smoothing_sigma=1.0,
            validity_resolution=16,
            validity_num_trajectories=2,
            validity_num_steps=64,
            validity_burn_in=8,
        )
        base = make_fixed_dataset_from_iterable_config(base_config, num_samples=3, seed=11)
        dense = make_fixed_dataset_from_iterable_config(dense_config, num_samples=3, seed=11)
        self.assertTrue(torch.allclose(base.tensors[1], dense.tensors[1]))

    def test_tiny_cnn_affine_set_estimator_shape(self):
        model = TinyCNNAffineSetEstimator(num_transforms=3, hidden_dim=16, pool_grid=2)
        output = model(torch.zeros(2, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_coord_tiny_affine_set_estimator_shape(self):
        model = TinyCNNAffineSetEstimator(
            num_transforms=3,
            hidden_dim=16,
            pool_grid=4,
            coord_channels=True,
        )
        output = model(torch.zeros(2, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_residual_affine_set_estimator_shape(self):
        model = TinyCNNAffineSetEstimator(
            num_transforms=3,
            hidden_dim=16,
            pool_grid=4,
            encoder_type="residual",
            coord_channels=True,
        )
        output = model(torch.zeros(2, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_residual_wide_affine_set_estimator_shape(self):
        model = TinyCNNAffineSetEstimator(
            num_transforms=3,
            hidden_dim=16,
            pool_grid=4,
            encoder_type="residual_wide",
            coord_channels=True,
        )
        output = model(torch.zeros(2, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_moment_feature_affine_set_estimator_shape(self):
        model = TinyCNNAffineSetEstimator(
            num_transforms=3,
            hidden_dim=16,
            pool_grid=4,
            encoder_type="residual_wide",
            coord_channels=True,
            density_feature_mode="moments",
            global_moments=True,
        )
        images = torch.zeros(2, 1, 16, 16)
        images[:, :, 4:12, 5:10] = 1.0
        images = images / images.sum(dim=(1, 2, 3), keepdim=True)
        output = model(images)
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_query_attention_affine_set_estimator_shape(self):
        model = TinyCNNAffineSetEstimator(
            num_transforms=3,
            hidden_dim=32,
            pool_grid=4,
            encoder_type="residual_wide",
            coord_channels=True,
            head_type="query_attention",
            query_num_heads=4,
            query_layers=1,
        )
        output = model(torch.zeros(2, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_query_attention_residual_wide_attention_affine_set_estimator_shape(self):
        model = TinyCNNAffineSetEstimator(
            num_transforms=3,
            hidden_dim=32,
            pool_grid=4,
            encoder_type="residual_wide_attn",
            coord_channels=True,
            head_type="query_attention",
            query_num_heads=4,
            query_layers=1,
        )
        output = model(torch.zeros(2, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_residual_wide_attention_affine_set_estimator_shape(self):
        model = TinyCNNAffineSetEstimator(
            num_transforms=3,
            hidden_dim=16,
            pool_grid=4,
            encoder_type="residual_wide_attn",
            coord_channels=True,
        )
        output = model(torch.zeros(2, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 3, 6))

    def test_residual_svd_set_estimator_shape(self):
        model = TinyCNNSetEstimator(
            num_transforms=3,
            hidden_dim=16,
            pool_grid=4,
            encoder_type="residual",
            coord_channels=True,
        )
        output = model(torch.zeros(2, 1, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 3, 6))


if __name__ == "__main__":
    unittest.main()
