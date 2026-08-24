"""Small CNN set-head baseline for Phase -1 and Phase 0."""

from __future__ import annotations

import torch
import torch.nn as nn


def _atanh_float(value: float) -> float:
    value_tensor = torch.tensor(float(value))
    return torch.atanh(value_tensor).item()


def _spatial_coord_channels(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized x/y coordinate grids expanded to the batch size."""
    batch, _, height, width = images.shape
    ys = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.view(1, 1, height, width).expand(batch, 1, height, width)
    yy = yy.view(1, 1, height, width).expand(batch, 1, height, width)
    return xx, yy


def _input_channel_count(*, coord_channels: bool, density_feature_mode: str) -> int:
    """Return the number of image-like channels consumed by the CNN encoder."""
    count = 1
    if coord_channels:
        count += 2
    if density_feature_mode == "none":
        return count
    if density_feature_mode == "moments":
        return count + 5
    raise ValueError(f"unknown density_feature_mode: {density_feature_mode}")


def _build_input_features(
    images: torch.Tensor,
    *,
    coord_channels: bool,
    density_feature_mode: str,
) -> torch.Tensor:
    """Build deterministic image-derived input channels for the CNN encoder."""
    density = images * float(images.shape[-1] * images.shape[-2])
    channels = [density]
    if coord_channels or density_feature_mode != "none":
        xx, yy = _spatial_coord_channels(images)
    if coord_channels:
        channels.extend((xx, yy))
    if density_feature_mode == "moments":
        channels.extend(
            (
                density * xx,
                density * yy,
                density * xx.square(),
                density * yy.square(),
                density * xx * yy,
            )
        )
    elif density_feature_mode != "none":
        raise ValueError(f"unknown density_feature_mode: {density_feature_mode}")
    return torch.cat(channels, dim=1)


def _global_density_moments(images: torch.Tensor) -> torch.Tensor:
    """Return low-order global moments computed from normalized density maps."""
    xx, yy = _spatial_coord_channels(images)
    mass = images.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    weights = images / mass
    mean_x = (weights * xx).sum(dim=(2, 3))
    mean_y = (weights * yy).sum(dim=(2, 3))
    centered_x = xx - mean_x.view(-1, 1, 1, 1)
    centered_y = yy - mean_y.view(-1, 1, 1, 1)
    var_x = (weights * centered_x.square()).sum(dim=(2, 3))
    var_y = (weights * centered_y.square()).sum(dim=(2, 3))
    cov_xy = (weights * centered_x * centered_y).sum(dim=(2, 3))
    skew_x = (weights * centered_x.pow(3)).sum(dim=(2, 3))
    skew_y = (weights * centered_y.pow(3)).sum(dim=(2, 3))
    safe_weights = weights.clamp_min(1e-12)
    entropy = -(safe_weights * safe_weights.log()).sum(dim=(2, 3))
    entropy = entropy / torch.log(images.new_tensor(float(images.shape[-1] * images.shape[-2])))
    return torch.cat((mean_x, mean_y, var_x, var_y, cov_xy, skew_x, skew_y, entropy), dim=1)


class _TinyCNNEncoder(nn.Module):
    """Shared compact encoder for Phase -1/0 baselines."""

    def __init__(
        self,
        pool_grid: int = 4,
        coord_channels: bool = False,
        density_feature_mode: str = "none",
    ) -> None:
        super().__init__()
        self.coord_channels = coord_channels
        self.density_feature_mode = density_feature_mode
        input_channels = _input_channel_count(
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
        self.feature_dim = 64 * pool_grid * pool_grid
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((pool_grid, pool_grid)),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = _build_input_features(
            images,
            coord_channels=self.coord_channels,
            density_feature_mode=self.density_feature_mode,
        )
        return self.net(x)


class _ResidualBlock(nn.Module):
    """Small residual block with GroupNorm for sparse density maps."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class _ChannelSpatialGate(nn.Module):
    """Lightweight channel and spatial attention for residual feature maps."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weight = self.channel_gate(x).view(x.shape[0], x.shape[1], 1, 1)
        x = x * channel_weight
        spatial_summary = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)),
            dim=1,
        )
        return x * self.spatial_gate(spatial_summary)


class _AttentionResidualBlock(nn.Module):
    """Residual block with channel/spatial attention on the residual branch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.attention = _ChannelSpatialGate(out_channels)
        self.activation = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.attention(x)
        return self.activation(x + residual)


class _ResidualSpatialEncoder(nn.Module):
    """Residual encoder that preserves an 8x8 spatial grid for 32x32 inputs."""

    def __init__(
        self,
        pool_grid: int = 8,
        coord_channels: bool = True,
        density_feature_mode: str = "none",
    ) -> None:
        super().__init__()
        self.coord_channels = coord_channels
        self.density_feature_mode = density_feature_mode
        input_channels = _input_channel_count(
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
        self.feature_dim = 96 * pool_grid * pool_grid
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            _ResidualBlock(32, 32),
            _ResidualBlock(32, 64, stride=2),
            _ResidualBlock(64, 64),
            _ResidualBlock(64, 96, stride=2),
            _ResidualBlock(96, 96),
            nn.AdaptiveAvgPool2d((pool_grid, pool_grid)),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = _build_input_features(
            images,
            coord_channels=self.coord_channels,
            density_feature_mode=self.density_feature_mode,
        )
        return self.net(x)


class _ResidualWideSpatialEncoder(nn.Module):
    """Wider residual encoder for 64x64 density maps."""

    def __init__(
        self,
        pool_grid: int = 8,
        coord_channels: bool = True,
        density_feature_mode: str = "none",
    ) -> None:
        super().__init__()
        self.coord_channels = coord_channels
        self.density_feature_mode = density_feature_mode
        input_channels = _input_channel_count(
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
        self.feature_dim = 128 * pool_grid * pool_grid
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.ReLU(inplace=True),
            _ResidualBlock(48, 48),
            _ResidualBlock(48, 64, stride=2),
            _ResidualBlock(64, 64),
            _ResidualBlock(64, 96, stride=2),
            _ResidualBlock(96, 96),
            _ResidualBlock(96, 128, stride=2),
            _ResidualBlock(128, 128),
            nn.AdaptiveAvgPool2d((pool_grid, pool_grid)),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = _build_input_features(
            images,
            coord_channels=self.coord_channels,
            density_feature_mode=self.density_feature_mode,
        )
        return self.net(x)


class _ResidualWideAttentionSpatialEncoder(nn.Module):
    """Wide residual encoder with lightweight channel/spatial attention."""

    def __init__(
        self,
        pool_grid: int = 8,
        coord_channels: bool = True,
        density_feature_mode: str = "none",
    ) -> None:
        super().__init__()
        self.coord_channels = coord_channels
        self.density_feature_mode = density_feature_mode
        input_channels = _input_channel_count(
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
        self.feature_dim = 128 * pool_grid * pool_grid
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.ReLU(inplace=True),
            _AttentionResidualBlock(48, 48),
            _AttentionResidualBlock(48, 64, stride=2),
            _AttentionResidualBlock(64, 64),
            _AttentionResidualBlock(64, 96, stride=2),
            _AttentionResidualBlock(96, 96),
            _AttentionResidualBlock(96, 128, stride=2),
            _AttentionResidualBlock(128, 128),
            nn.AdaptiveAvgPool2d((pool_grid, pool_grid)),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = _build_input_features(
            images,
            coord_channels=self.coord_channels,
            density_feature_mode=self.density_feature_mode,
        )
        return self.net(x)


class _ResidualGridTokenEncoder(nn.Module):
    """Residual encoder that exposes pooled spatial tokens for attention heads."""

    def __init__(
        self,
        *,
        encoder_type: str,
        pool_grid: int = 8,
        coord_channels: bool = True,
        density_feature_mode: str = "none",
    ) -> None:
        super().__init__()
        self.coord_channels = coord_channels
        self.density_feature_mode = density_feature_mode
        input_channels = _input_channel_count(
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
        if encoder_type == "residual":
            self.output_channels = 96
            layers = [
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(inplace=True),
                _ResidualBlock(32, 32),
                _ResidualBlock(32, 64, stride=2),
                _ResidualBlock(64, 64),
                _ResidualBlock(64, 96, stride=2),
                _ResidualBlock(96, 96),
            ]
        elif encoder_type == "residual_wide":
            self.output_channels = 128
            layers = [
                nn.Conv2d(input_channels, 48, kernel_size=3, padding=1),
                nn.GroupNorm(8, 48),
                nn.ReLU(inplace=True),
                _ResidualBlock(48, 48),
                _ResidualBlock(48, 64, stride=2),
                _ResidualBlock(64, 64),
                _ResidualBlock(64, 96, stride=2),
                _ResidualBlock(96, 96),
                _ResidualBlock(96, 128, stride=2),
                _ResidualBlock(128, 128),
            ]
        elif encoder_type == "residual_wide_attn":
            self.output_channels = 128
            layers = [
                nn.Conv2d(input_channels, 48, kernel_size=3, padding=1),
                nn.GroupNorm(8, 48),
                nn.ReLU(inplace=True),
                _AttentionResidualBlock(48, 48),
                _AttentionResidualBlock(48, 64, stride=2),
                _AttentionResidualBlock(64, 64),
                _AttentionResidualBlock(64, 96, stride=2),
                _AttentionResidualBlock(96, 96),
                _AttentionResidualBlock(96, 128, stride=2),
                _AttentionResidualBlock(128, 128),
            ]
        else:
            raise ValueError("query attention head supports residual encoders only")
        self.pool = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        self.net = nn.Sequential(*layers)

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = _build_input_features(
            images,
            coord_channels=self.coord_channels,
            density_feature_mode=self.density_feature_mode,
        )
        x = self.pool(self.net(x))
        return x.flatten(2).transpose(1, 2).contiguous()


def _build_encoder(
    *,
    encoder_type: str,
    pool_grid: int,
    coord_channels: bool,
    density_feature_mode: str,
) -> nn.Module:
    if encoder_type == "tiny":
        return _TinyCNNEncoder(
            pool_grid=pool_grid,
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
    if encoder_type == "residual":
        return _ResidualSpatialEncoder(
            pool_grid=pool_grid,
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
    if encoder_type == "residual_wide":
        return _ResidualWideSpatialEncoder(
            pool_grid=pool_grid,
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
    if encoder_type == "residual_wide_attn":
        return _ResidualWideAttentionSpatialEncoder(
            pool_grid=pool_grid,
            coord_channels=coord_channels,
            density_feature_mode=density_feature_mode,
        )
    raise ValueError(f"unknown encoder_type: {encoder_type}")


class _QueryAttentionBlock(nn.Module):
    """A small DETR-style query block over fixed spatial tokens."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query_self_norm = nn.LayerNorm(hidden_dim)
        self.query_cross_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_self_norm(queries)
        queries = queries + self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )[0]
        queries = queries + self.cross_attention(
            self.query_cross_norm(queries),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=False,
        )[0]
        return queries + self.ffn(self.ffn_norm(queries))


class _QueryAttentionSetHead(nn.Module):
    """Predict one affine map per learned query from spatial encoder tokens."""

    def __init__(
        self,
        *,
        num_transforms: int,
        token_dim: int,
        hidden_dim: int,
        output_dim: int = 6,
        pool_grid: int = 8,
        num_heads: int = 8,
        num_layers: int = 2,
        global_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by query attention heads")
        self.query_embed = nn.Parameter(torch.randn(num_transforms, hidden_dim) * 0.02)
        self.position_embed = nn.Parameter(torch.randn(pool_grid * pool_grid, hidden_dim) * 0.02)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.global_proj = nn.Linear(global_dim, hidden_dim) if global_dim > 0 else None
        self.blocks = nn.ModuleList(
            [_QueryAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        global_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        tokens = self.token_proj(tokens)
        tokens = tokens + self.position_embed[: tokens.shape[1]].unsqueeze(0)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        if self.global_proj is not None:
            if global_features is None:
                raise ValueError("global_features are required for this query head")
            queries = queries + self.global_proj(global_features).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, tokens)
        return self.output(self.output_norm(queries))


def _initialize_affine_output_layer(
    final: nn.Linear,
    *,
    num_transforms: int,
    linear_bound: float,
    initial_scale: float,
) -> None:
    """Initialize an affine output layer near ``initial_scale * I, b=0``."""
    if abs(initial_scale) >= linear_bound:
        raise ValueError("initial_scale must be inside linear_bound")
    raw_scale = _atanh_float(initial_scale / linear_bound)
    if final.out_features == 6:
        bias = torch.zeros(6)
        bias[0] = raw_scale
        bias[3] = raw_scale
    elif final.out_features == num_transforms * 6:
        bias = torch.zeros(num_transforms, 6)
        bias[:, 0] = raw_scale
        bias[:, 3] = raw_scale
        bias = bias.reshape(-1)
    else:
        raise ValueError("affine output layer must emit 6 or num_transforms * 6 values")
    with torch.no_grad():
        final.weight.normal_(mean=0.0, std=1e-3)
        final.bias.copy_(bias)


class TinyCNNSetEstimator(nn.Module):
    """A compact baseline that predicts a fixed-size set of IFS maps."""

    def __init__(
        self,
        num_transforms: int = 2,
        hidden_dim: int = 128,
        pool_grid: int = 4,
        scale_range=(0.05, 0.90),
        translation_bound: float = 1.5,
        encoder_type: str = "tiny",
        coord_channels: bool = False,
        density_feature_mode: str = "none",
        global_moments: bool = False,
        head_type: str = "mlp",
        query_num_heads: int = 8,
        query_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_transforms = num_transforms
        self.pool_grid = pool_grid
        self.global_moments = bool(global_moments)
        self.head_type = head_type
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        self.translation_bound = float(translation_bound)
        if head_type == "mlp":
            self.encoder = _build_encoder(
                encoder_type=encoder_type,
                pool_grid=pool_grid,
                coord_channels=coord_channels,
                density_feature_mode=density_feature_mode,
            )
            head_input_dim = self.encoder.feature_dim + (8 if self.global_moments else 0)
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_transforms * 6),
            )
        elif head_type == "query_attention":
            self.encoder = _ResidualGridTokenEncoder(
                encoder_type=encoder_type,
                pool_grid=pool_grid,
                coord_channels=coord_channels,
                density_feature_mode=density_feature_mode,
            )
            self.head = _QueryAttentionSetHead(
                num_transforms=num_transforms,
                token_dim=self.encoder.output_channels,
                hidden_dim=hidden_dim,
                output_dim=6,
                pool_grid=pool_grid,
                num_heads=query_num_heads,
                num_layers=query_layers,
                global_dim=8 if self.global_moments else 0,
            )
        else:
            raise ValueError(f"unknown head_type: {head_type}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.head_type == "mlp":
            features = self.encoder(images)
            if self.global_moments:
                features = torch.cat((features, _global_density_moments(images)), dim=1)
            raw = self.head(features).reshape(images.shape[0], self.num_transforms, 6)
        else:
            global_features = _global_density_moments(images) if self.global_moments else None
            raw = self.head(self.encoder.forward_tokens(images), global_features)
        phi = raw[..., 0:2]
        scale_unit = torch.sigmoid(raw[..., 2:4])
        scales = self.scale_min + (self.scale_max - self.scale_min) * scale_unit
        translation = self.translation_bound * torch.tanh(raw[..., 4:6])
        return torch.cat((phi, scales, translation), dim=-1)


class TinyCNNAffineSetEstimator(nn.Module):
    """A compact baseline that predicts affine matrices and translations directly."""

    def __init__(
        self,
        num_transforms: int = 2,
        hidden_dim: int = 128,
        pool_grid: int = 4,
        linear_bound: float = 1.0,
        translation_bound: float = 1.5,
        encoder_type: str = "tiny",
        coord_channels: bool = False,
        initial_scale: float = 0.45,
        density_feature_mode: str = "none",
        global_moments: bool = False,
        head_type: str = "mlp",
        query_num_heads: int = 8,
        query_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_transforms = num_transforms
        self.pool_grid = pool_grid
        self.global_moments = bool(global_moments)
        self.head_type = head_type
        self.linear_bound = float(linear_bound)
        self.translation_bound = float(translation_bound)
        if head_type == "mlp":
            self.encoder = _build_encoder(
                encoder_type=encoder_type,
                pool_grid=pool_grid,
                coord_channels=coord_channels,
                density_feature_mode=density_feature_mode,
            )
            head_input_dim = self.encoder.feature_dim + (8 if self.global_moments else 0)
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_transforms * 6),
            )
        elif head_type == "query_attention":
            self.encoder = _ResidualGridTokenEncoder(
                encoder_type=encoder_type,
                pool_grid=pool_grid,
                coord_channels=coord_channels,
                density_feature_mode=density_feature_mode,
            )
            self.head = _QueryAttentionSetHead(
                num_transforms=num_transforms,
                token_dim=self.encoder.output_channels,
                hidden_dim=hidden_dim,
                output_dim=6,
                pool_grid=pool_grid,
                num_heads=query_num_heads,
                num_layers=query_layers,
                global_dim=8 if self.global_moments else 0,
            )
        else:
            raise ValueError(f"unknown head_type: {head_type}")
        self._initialize_affine_head(initial_scale)

    def _initialize_affine_head(self, initial_scale: float) -> None:
        final = self.head[-1] if self.head_type == "mlp" else self.head.output
        if not isinstance(final, nn.Linear):
            raise TypeError("expected final affine head layer to be nn.Linear")
        _initialize_affine_output_layer(
            final,
            num_transforms=self.num_transforms,
            linear_bound=self.linear_bound,
            initial_scale=initial_scale,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.head_type == "mlp":
            features = self.encoder(images)
            if self.global_moments:
                features = torch.cat((features, _global_density_moments(images)), dim=1)
            raw = self.head(features).reshape(images.shape[0], self.num_transforms, 6)
        else:
            global_features = _global_density_moments(images) if self.global_moments else None
            raw = self.head(self.encoder.forward_tokens(images), global_features)
        linear = self.linear_bound * torch.tanh(raw[..., 0:4])
        translation = self.translation_bound * torch.tanh(raw[..., 4:6])
        return torch.cat((linear, translation), dim=-1)
