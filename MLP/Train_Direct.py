# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.optim as optim

from RL.BaseTrain_DDQN import BaseTrain
from MLP.MLPQNet import MLPQNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    """
    DDQN + MLP ablation.

    Online and target networks use independent MLP feature extractors and
    independent Q-heads. This is the non-shared MLP cell in the 2 x 2 ablation.
    """

    def __init__(
        self,
        data_loader,
        data_train,
        data_test,
        dataset_name,
        data_val=None,
        window_size=10,
        transaction_cost=0.0001,
        n_classes=64,
        BATCH_SIZE=32,
        GAMMA=0.9,
        ReplayMemorySize=64,
        TARGET_UPDATE=10,
        n_step=5,
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=15000,
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
        activation="relu",
        experiment_dir=None,
    ):
        super().__init__(
            data_loader=data_loader,
            data_train=data_train,
            data_val=data_val,
            data_test=data_test,
            dataset_name=dataset_name,
            model_kind="MLP-DDQN",
            window_size=window_size,
            transaction_cost=transaction_cost,
            BATCH_SIZE=BATCH_SIZE,
            GAMMA=GAMMA,
            ReplayMemorySize=ReplayMemorySize,
            TARGET_UPDATE=TARGET_UPDATE,
            n_step=n_step,
            EPS_START=EPS_START,
            EPS_END=EPS_END,
            EPS_DECAY=EPS_DECAY,
            experiment_dir=experiment_dir,
        )

        net_kwargs = dict(
            state_size=data_train.state_size,
            action_dim=3,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
        )

        # Fully independent online and target Q-networks.
        self.policy_net = MLPQNet(**net_kwargs).to(device)
        self.target_net = MLPQNet(**net_kwargs).to(device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters()
        )

        # Independent network with identical state-dict layout for evaluation.
        self.test_net = MLPQNet(**net_kwargs).to(device)

        assert self.policy_net.encoder is not self.target_net.encoder
        assert self.policy_net.q_head is not self.target_net.q_head
