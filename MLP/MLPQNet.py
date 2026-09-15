# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

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


class MLPEncoder(nn.Module):
    """
    Flattened market state -> fixed-length feature representation.

    Default architecture:
        state_size -> 128 -> n_classes

    With window=10, features=5, and n_classes=64:
        50 -> 128 -> 64
    """

    def __init__(
        self,
        state_size: int,
        n_classes: int = 64,
        hidden_dim: int = 128,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        self.state_size = int(state_size)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)

        if self.state_size <= 0:
            raise ValueError("state_size must be positive")
        if self.n_classes <= 0:
            raise ValueError("n_classes must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        in_dim = self.state_size

        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(_make_activation(activation))

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            in_dim = self.hidden_dim

        layers.append(
            nn.Linear(
                in_dim,
                self.n_classes,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.dim() != 2:
            raise ValueError(
                "MLPEncoder expects (B, state_size) or (state_size,), "
                f"got {tuple(x.shape)}"
            )

        if x.size(1) != self.state_size:
            raise ValueError(
                f"state_size mismatch: expected {self.state_size}, "
                f"got {x.size(1)}"
            )

        return self.network(x)


class MLPQHead(nn.Module):
    """
    Feature representation -> Q values for:
        0 = buy
        1 = hold/None
        2 = sell

    Q-head:
        n_classes -> 128 -> 256 -> 3
    """

    def __init__(
        self,
        n_classes: int = 64,
        action_dim: int = 3,
    ):
        super().__init__()

        self.policy_network = nn.Sequential(
            nn.Linear(n_classes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return self.policy_network(x)


class MLPQNet(nn.Module):
    """
    MLP feature extractor followed by a Q-value head.

    ``encoder`` and ``q_head`` may be injected by Train_Shared so that
    policy and target branches can explicitly share the same feature encoder.

    Default encoder:
        state_size -> 128 -> 64

    Q-head:
        64 -> 128 -> 256 -> 3
    """

    def __init__(
        self,
        state_size: int,
        action_dim: int = 3,
        n_classes: int = 64,
        hidden_dim: int = 128,
        hidden_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        *,
        encoder: Optional[MLPEncoder] = None,
        q_head: Optional[MLPQHead] = None,
    ):
        super().__init__()

        self.encoder = encoder or MLPEncoder(
            state_size=state_size,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
        )

        self.q_head = q_head or MLPQHead(
            n_classes=n_classes,
            action_dim=action_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_head(
            self.encoder(x)
        )
