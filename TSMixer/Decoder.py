import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Map encoded features to Q-values."""

    def __init__(self, num_classes: int, action_length: int = 3):
        super().__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.policy_network(x)
