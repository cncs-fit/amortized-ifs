"""Datasets for synthetic IFS parameter estimation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset, get_worker_info

from data.renderer import (
    is_valid_sample,
    render_density_from_params,
    render_density_from_params_batch,
    smooth_density_map,
    smooth_density_maps,
    valid_sample_mask,
)
from data.sampler import sample_ifs_parameters, sample_ifs_parameters_batch


@dataclass(frozen=True)
class IFSDatasetConfig:
    num_samples: int = 16
    num_transforms: int = 2
    phase: str = "phase_minus1"
    resolution: int = 32
    num_trajectories: int = 8
    num_steps: int = 512
    burn_in: int = 64
    seed: int = 1234
    max_attempts: int = 10000
    fixed_range: Tuple[float, float] = (-1.5, 1.5)
    scale_range: Tuple[float, float] = (0.20, 0.70)
    fixed_point_range: float = 0.75
    density_smoothing_sigma: float = 0.0
    validity_resolution: Optional[int] = None
    validity_num_trajectories: Optional[int] = None
    validity_num_steps: Optional[int] = None
    validity_burn_in: Optional[int] = None


@dataclass(frozen=True)
class IFSIterableDatasetConfig:
    num_transforms: int = 2
    phase: str = "phase0"
    resolution: int = 32
    num_trajectories: int = 4
    num_steps: int = 256
    burn_in: int = 32
    seed: int = 2026
    max_attempts_per_sample: int = 1000
    fixed_range: Tuple[float, float] = (-1.5, 1.5)
    scale_range: Tuple[float, float] = (0.20, 0.70)
    fixed_point_range: float = 0.75
    density_smoothing_sigma: float = 0.0
    validity_resolution: Optional[int] = None
    validity_num_trajectories: Optional[int] = None
    validity_num_steps: Optional[int] = None
    validity_burn_in: Optional[int] = None


TRAIN_CACHE_VERSION = 1


def _validity_render_kwargs(config) -> dict:
    return {
        "resolution": config.validity_resolution or config.resolution,
        "num_trajectories": config.validity_num_trajectories or config.num_trajectories,
        "num_steps": config.validity_num_steps or config.num_steps,
        "burn_in": config.validity_burn_in or config.burn_in,
    }


def generate_fixed_dataset(config: IFSDatasetConfig) -> TensorDataset:
    """Generate a deterministic in-memory dataset of density maps and params."""
    param_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    render_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1_000_003)
    validity_generator = torch.Generator(device="cpu").manual_seed(config.seed + 2_000_003)
    images = []
    params_list = []
    attempts = 0
    validity_kwargs = _validity_render_kwargs(config)

    while len(images) < config.num_samples and attempts < config.max_attempts:
        attempts += 1
        params = sample_ifs_parameters(
            config.num_transforms,
            phase=config.phase,
            scale_range=config.scale_range,
            fixed_point_range=config.fixed_point_range,
            generator=param_generator,
            device=torch.device("cpu"),
        )
        validity_points, validity_density = render_density_from_params(
            params,
            resolution=validity_kwargs["resolution"],
            fixed_range=config.fixed_range,
            num_trajectories=validity_kwargs["num_trajectories"],
            num_steps=validity_kwargs["num_steps"],
            burn_in=validity_kwargs["burn_in"],
            generator=validity_generator,
        )
        if not is_valid_sample(
            validity_points,
            validity_density,
            resolution=validity_kwargs["resolution"],
            fixed_range=config.fixed_range,
        ):
            continue
        _, raw_density = render_density_from_params(
            params,
            resolution=config.resolution,
            fixed_range=config.fixed_range,
            num_trajectories=config.num_trajectories,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            generator=render_generator,
        )
        density = smooth_density_map(raw_density, sigma=config.density_smoothing_sigma)
        images.append(density)
        params_list.append(params)

    if len(images) != config.num_samples:
        raise RuntimeError(
            f"generated {len(images)} valid samples after {attempts} attempts; "
            f"requested {config.num_samples}"
        )
    return TensorDataset(torch.stack(images), torch.stack(params_list))


def _config_cache_payload(
    config: IFSIterableDatasetConfig,
    *,
    num_samples: int,
    generation_num_workers: int,
    generation_mode: Optional[str] = None,
    generation_device: Optional[str] = None,
) -> dict:
    payload = {
        "cache_version": TRAIN_CACHE_VERSION,
        "num_samples": int(num_samples),
        "generation_num_workers": int(generation_num_workers),
        "config": config.__dict__,
    }
    if generation_mode is not None:
        payload["generation_mode"] = generation_mode
    if generation_device is not None:
        payload["generation_device"] = generation_device
    return payload


def default_train_cache_path(
    config: IFSIterableDatasetConfig,
    *,
    num_samples: int,
    cache_dir: str | Path = "cache/phase0_train",
    generation_num_workers: int = 0,
    generation_mode: Optional[str] = None,
    generation_device: Optional[str] = None,
) -> Path:
    """Return a deterministic cache path for a training-data configuration."""
    payload = _config_cache_payload(
        config,
        num_samples=num_samples,
        generation_num_workers=generation_num_workers,
        generation_mode=generation_mode,
        generation_device=generation_device,
    )
    encoded = json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return Path(cache_dir) / f"ifs_train_{num_samples}_{digest}.pt"


def generate_train_cache_from_iterable_config(
    config: IFSIterableDatasetConfig,
    *,
    num_samples: int,
    batch_size: int = 64,
    num_workers: int = 0,
) -> TensorDataset:
    """Generate a finite in-memory cache of training samples."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset = IFSIterableDataset(config)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    image_batches = []
    param_batches = []
    collected = 0
    iterator = iter(loader)
    while collected < num_samples:
        images, params = next(iterator)
        take = min(images.shape[0], num_samples - collected)
        image_batches.append(images[:take].contiguous())
        param_batches.append(params[:take].contiguous())
        collected += take

    images = torch.cat(image_batches, dim=0)
    params = torch.cat(param_batches, dim=0)
    return TensorDataset(images, params)


def save_train_cache_from_iterable_config(
    config: IFSIterableDatasetConfig,
    *,
    num_samples: int,
    path: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> TensorDataset:
    """Generate and save a finite cache of training samples."""
    dataset = generate_train_cache_from_iterable_config(
        config,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    images, params = dataset.tensors
    payload = {
        "metadata": _config_cache_payload(
            config,
            num_samples=num_samples,
            generation_num_workers=num_workers,
        ),
        "images": images,
        "params": params,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)
    return dataset


def generate_train_cache_batched_from_iterable_config(
    config: IFSIterableDatasetConfig,
    *,
    num_samples: int,
    batch_size: int = 512,
    device: str | torch.device = "cuda",
) -> TensorDataset:
    """Generate a finite in-memory cache with batched tensor rendering."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA cache generation was requested but CUDA is not available")

    param_generator = torch.Generator(device=device).manual_seed(config.seed)
    render_generator = torch.Generator(device=device).manual_seed(config.seed + 1_000_003)
    validity_generator = torch.Generator(device=device).manual_seed(config.seed + 2_000_003)
    validity_kwargs = _validity_render_kwargs(config)

    image_batches = []
    param_batches = []
    collected = 0
    attempts = 0
    required_batches = (num_samples + batch_size - 1) // batch_size
    max_candidate_batches = max(1, config.max_attempts_per_sample * required_batches)

    while collected < num_samples and attempts < max_candidate_batches:
        attempts += 1
        candidate_params = sample_ifs_parameters_batch(
            batch_size,
            config.num_transforms,
            phase=config.phase,
            scale_range=config.scale_range,
            fixed_point_range=config.fixed_point_range,
            generator=param_generator,
            device=device,
        )
        validity_points, validity_density = render_density_from_params_batch(
            candidate_params,
            resolution=validity_kwargs["resolution"],
            fixed_range=config.fixed_range,
            num_trajectories=validity_kwargs["num_trajectories"],
            num_steps=validity_kwargs["num_steps"],
            burn_in=validity_kwargs["burn_in"],
            generator=validity_generator,
        )
        mask = valid_sample_mask(
            validity_points,
            validity_density,
            resolution=validity_kwargs["resolution"],
            fixed_range=config.fixed_range,
        )
        if not mask.any():
            continue

        valid_params = candidate_params[mask]
        _, raw_density = render_density_from_params_batch(
            valid_params,
            resolution=config.resolution,
            fixed_range=config.fixed_range,
            num_trajectories=config.num_trajectories,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            generator=render_generator,
        )
        density = smooth_density_maps(raw_density, sigma=config.density_smoothing_sigma)
        take = min(density.shape[0], num_samples - collected)
        image_batches.append(density[:take].detach().cpu().contiguous())
        param_batches.append(valid_params[:take].detach().cpu().contiguous())
        collected += take

    if collected != num_samples:
        raise RuntimeError(
            f"generated {collected} valid samples after {attempts} batched attempts; "
            f"requested {num_samples}"
        )

    images = torch.cat(image_batches, dim=0)
    params = torch.cat(param_batches, dim=0)
    return TensorDataset(images, params)


def save_train_cache_batched_from_iterable_config(
    config: IFSIterableDatasetConfig,
    *,
    num_samples: int,
    path: str | Path,
    batch_size: int = 512,
    device: str | torch.device = "cuda",
) -> TensorDataset:
    """Generate and save a finite cache with batched tensor rendering."""
    device = torch.device(device)
    dataset = generate_train_cache_batched_from_iterable_config(
        config,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
    )
    images, params = dataset.tensors
    payload = {
        "metadata": _config_cache_payload(
            config,
            num_samples=num_samples,
            generation_num_workers=0,
            generation_mode="batched",
            generation_device=device.type,
        ),
        "images": images,
        "params": params,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)
    return dataset


def load_train_cache(path: str | Path) -> tuple[TensorDataset, dict]:
    """Load a finite training cache saved by ``save_train_cache_from_iterable_config``."""
    payload = torch.load(Path(path), map_location="cpu")
    images = payload["images"].contiguous()
    params = payload["params"].contiguous()
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("cached images must have shape [N, 1, H, W]")
    if params.ndim != 3 or params.shape[-1] != 6:
        raise ValueError("cached params must have shape [N, num_transforms, 6]")
    if images.shape[0] != params.shape[0]:
        raise ValueError("cached images and params have different sample counts")
    return TensorDataset(images, params), payload.get("metadata", {})


def generate_one_sample(
    config: IFSIterableDatasetConfig,
    param_generator: torch.Generator,
    render_generator: torch.Generator,
    validity_generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate one valid ``(density, params)`` sample."""
    validity_kwargs = _validity_render_kwargs(config)
    for _ in range(config.max_attempts_per_sample):
        params = sample_ifs_parameters(
            config.num_transforms,
            phase=config.phase,
            scale_range=config.scale_range,
            fixed_point_range=config.fixed_point_range,
            generator=param_generator,
            device=torch.device("cpu"),
        )
        validity_points, validity_density = render_density_from_params(
            params,
            resolution=validity_kwargs["resolution"],
            fixed_range=config.fixed_range,
            num_trajectories=validity_kwargs["num_trajectories"],
            num_steps=validity_kwargs["num_steps"],
            burn_in=validity_kwargs["burn_in"],
            generator=validity_generator,
        )
        if is_valid_sample(
            validity_points,
            validity_density,
            resolution=validity_kwargs["resolution"],
            fixed_range=config.fixed_range,
        ):
            _, raw_density = render_density_from_params(
                params,
                resolution=config.resolution,
                fixed_range=config.fixed_range,
                num_trajectories=config.num_trajectories,
                num_steps=config.num_steps,
                burn_in=config.burn_in,
                generator=render_generator,
            )
            density = smooth_density_map(raw_density, sigma=config.density_smoothing_sigma)
            return density, params
    raise RuntimeError(
        f"failed to generate a valid sample after {config.max_attempts_per_sample} attempts"
    )


class IFSIterableDataset(IterableDataset):
    """Infinite on-the-fly IFS dataset for Phase 0 training."""

    def __init__(self, config: IFSIterableDatasetConfig) -> None:
        super().__init__()
        self.config = config

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        param_generator = torch.Generator(device="cpu").manual_seed(self.config.seed + worker_id)
        render_generator = torch.Generator(device="cpu").manual_seed(
            self.config.seed + 1_000_003 + worker_id
        )
        validity_generator = torch.Generator(device="cpu").manual_seed(
            self.config.seed + 2_000_003 + worker_id
        )
        while True:
            yield generate_one_sample(
                self.config,
                param_generator,
                render_generator,
                validity_generator,
            )


def make_fixed_dataset_from_iterable_config(
    config: IFSIterableDatasetConfig,
    *,
    num_samples: int,
    seed: int,
) -> TensorDataset:
    """Generate a fixed evaluation set using Phase 0 data settings."""
    fixed_config = IFSDatasetConfig(
        num_samples=num_samples,
        num_transforms=config.num_transforms,
        phase=config.phase,
        resolution=config.resolution,
        num_trajectories=config.num_trajectories,
        num_steps=config.num_steps,
        burn_in=config.burn_in,
        seed=seed,
        fixed_range=config.fixed_range,
        scale_range=config.scale_range,
        fixed_point_range=config.fixed_point_range,
        density_smoothing_sigma=config.density_smoothing_sigma,
        validity_resolution=config.validity_resolution,
        validity_num_trajectories=config.validity_num_trajectories,
        validity_num_steps=config.validity_num_steps,
        validity_burn_in=config.validity_burn_in,
    )
    return generate_fixed_dataset(fixed_config)
