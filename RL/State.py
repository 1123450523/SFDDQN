from __future__ import annotations

import numpy as np

from .TradingEnv import TradingEnv


class State(TradingEnv):
    """Build flat rolling-window OHLCV states from normalized features."""

    REQUIRED_COLS = [
        "open_norm",
        "high_norm",
        "low_norm",
        "close_norm",
        "volume_norm",
    ]

    def __init__(
        self,
        data,
        action_name,
        device,
        gamma=0.9,
        n_step=5,
        batch_size=32,
        window_size=10,
        transaction_cost=0.0001,
    ):
        missing = [c for c in self.REQUIRED_COLS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Ensure the data loader generated the required *_norm columns "
                f"(including volume_norm)."
            )

        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        start_index_reward = self.window_size - 1

        super().__init__(
            data=data,
            action_name=action_name,
            device=device,
            gamma=gamma,
            n_step=n_step,
            batch_size=batch_size,
            start_index_reward=start_index_reward,
            transaction_cost=transaction_cost,
        )


        self.data_kind = "OHLCVWindow"

        feat_dim = len(self.REQUIRED_COLS)
        self.feature_dim = feat_dim
        self.state_size = self.window_size * feat_dim


        buf = []

        for i, (_, row) in enumerate(self.data.loc[:, self.REQUIRED_COLS].iterrows()):
            buf.extend([row[c] for c in self.REQUIRED_COLS])

            if i >= self.window_size - 1:
                self.states.append(np.array(buf, dtype=np.float32))
                buf = buf[feat_dim:]

        if len(self.states) == 0:
            raise ValueError(
                f"No states were generated. "
                f"Please check data length={len(self.data)} and window_size={self.window_size}."
            )
