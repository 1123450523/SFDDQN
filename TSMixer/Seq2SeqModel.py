import torch.nn as nn


class Seq2Seq(nn.Module):
    """Compose an encoder and decoder into a Q-network."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))
