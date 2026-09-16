from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class TimeBatchNorm2d(nn.Module):
    """Apply BatchNorm2d to a tensor of shape (B, L, D)."""

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features=1,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TimeBatchNorm2d expects (B, L, D), got {tuple(x.shape)}")
        return self.bn(x.unsqueeze(1)).squeeze(1)


class TimeLayerNorm2d(nn.Module):
    """Apply LayerNorm over the temporal-feature dimensions."""

    def __init__(
        self,
        seq_len: int,
        channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(
            (seq_len, channels),
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TimeLayerNorm2d expects (B, L, D), got {tuple(x.shape)}")
        return self.ln(x)


def _make_norm2d(
    norm_type: Literal["batch", "layer"],
    seq_len: int,
    channels: int,
) -> nn.Module:
    if norm_type == "batch":
        return TimeBatchNorm2d()
    if norm_type == "layer":
        return TimeLayerNorm2d(seq_len=seq_len, channels=channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


@dataclass
class MixerCfg:
    seq_len: int
    channels: int
    ff_dim: int = 64
    dropout: float = 0.1
    activation: str = "relu"
    norm_type: Literal["batch", "layer"] = "batch"
    normalize_before: bool = False


class MixerLayer(nn.Module):
    """TSMixer block with temporal mixing and feature mixing."""

    def __init__(self, cfg: MixerCfg):
        super().__init__()
        self.cfg = cfg
        self.seq_len = int(cfg.seq_len)
        self.channels = int(cfg.channels)
        self.normalize_before = bool(cfg.normalize_before)

        act = _make_activation(cfg.activation)

        self.norm_time = _make_norm2d(
            cfg.norm_type,
            self.seq_len,
            self.channels,
        )
        self.norm_feat = _make_norm2d(
            cfg.norm_type,
            self.seq_len,
            self.channels,
        )

        self.time_linear = nn.Linear(self.seq_len, self.seq_len)
        self.time_act = act
        self.time_drop = nn.Dropout(cfg.dropout)

        self.feat_fc1 = nn.Linear(self.channels, cfg.ff_dim)
        self.feat_act = act
        self.feat_drop1 = nn.Dropout(cfg.dropout)
        self.feat_fc2 = nn.Linear(cfg.ff_dim, self.channels)
        self.feat_drop2 = nn.Dropout(cfg.dropout)

    def _maybe_pre_norm(
        self,
        norm: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return norm(x) if self.normalize_before else x

    def _maybe_post_norm(
        self,
        norm: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return norm(x) if not self.normalize_before else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MixerLayer expects (B, L, D), got {tuple(x.shape)}")

        _, length, channels = x.shape
        if length != self.seq_len or channels != self.channels:
            raise ValueError(
                f"Shape mismatch: expected (B, {self.seq_len}, {self.channels}), "
                f"got {tuple(x.shape)}"
            )

        x_in = self._maybe_pre_norm(self.norm_time, x)
        time_out = self.time_linear(x_in.transpose(1, 2)).transpose(1, 2)
        time_out = self.time_act(time_out)
        time_out = self.time_drop(time_out)
        x = self._maybe_post_norm(self.norm_time, x + time_out)

        x_in = self._maybe_pre_norm(self.norm_feat, x)
        feat_out = self.feat_fc1(x_in)
        feat_out = self.feat_act(feat_out)
        feat_out = self.feat_drop1(feat_out)
        feat_out = self.feat_fc2(feat_out)
        feat_out = self.feat_drop2(feat_out)
        x = self._maybe_post_norm(self.norm_feat, x + feat_out)

        return x


class Encoder(nn.Module):
    """Encode a flattened OHLCV window into a fixed-length feature vector."""

    def __init__(
        self,
        num_classes: int,
        state_size: int,
        *,
        feature_dim: int = 5,
        e_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: Literal["batch", "layer"] = "batch",
        normalize_before: bool = False,
        pool: Literal["mean", "last", "max"] = "mean",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.state_size = int(state_size)
        self.feature_dim = int(feature_dim)

        if self.state_size % self.feature_dim != 0:
            raise ValueError(
                f"state_size={self.state_size} must be divisible by "
                f"feature_dim={self.feature_dim}"
            )

        self.window = self.state_size // self.feature_dim
        self.pool = pool

        cfg = MixerCfg(
            seq_len=self.window,
            channels=self.feature_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            normalize_before=normalize_before,
        )

        self.backbone = nn.Sequential(
            *[MixerLayer(cfg) for _ in range(int(e_layers))]
        )
        self.proj = nn.Linear(self.feature_dim, self.num_classes)

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return h.mean(dim=1)
        if self.pool == "last":
            return h[:, -1, :]
        if self.pool == "max":
            return h.max(dim=1).values
        raise ValueError(f"Unsupported pool: {self.pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.dim() != 2:
            raise ValueError(
                f"Encoder expects (B, state_size) or (state_size,), got {tuple(x.shape)}"
            )

        batch_size, state_size = x.shape
        if state_size != self.state_size:
            raise ValueError(
                f"state_size mismatch: expected {self.state_size}, got {state_size}"
            )

        x = x.view(batch_size, self.window, self.feature_dim)
        x = self.backbone(x)
        x = self._pool(x)
        return self.proj(x)
