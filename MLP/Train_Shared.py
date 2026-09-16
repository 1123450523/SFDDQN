# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.optim as optim

from RL.BaseTrain_DDQN import BaseTrain
from MLP.MLPQNet import MLPEncoder, MLPQHead, MLPQNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    """
    SFDDQN + MLP ablation.

    Policy and target branches share exactly one MLP feature encoder while
    keeping independent online and target Q-heads. This isolates the effect
    of shared-feature learning from the TSMixer feature extractor.
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
            model_kind="MLP-SFDDQN",
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

        shared_encoder = MLPEncoder(
            state_size=data_train.state_size,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
        ).to(device)

        policy_q_head = MLPQHead(
            n_classes=n_classes,
            action_dim=3,
        ).to(device)

        target_q_head = MLPQHead(
            n_classes=n_classes,
            action_dim=3,
        ).to(device)

        # Both Q-networks reference exactly the same feature encoder.
        self.policy_net = MLPQNet(
            state_size=data_train.state_size,
            action_dim=3,
            n_classes=n_classes,
            encoder=shared_encoder,
            q_head=policy_q_head,
        ).to(device)

        self.target_net = MLPQNet(
            state_size=data_train.state_size,
            action_dim=3,
            n_classes=n_classes,
            encoder=shared_encoder,
            q_head=target_q_head,
        ).to(device)

        # Initialize target head from online head.
        target_q_head.load_state_dict(
            policy_q_head.state_dict()
        )

        # Expose the shared components using the names expected by the
        # current BaseTrain_DDQN target-update implementation.
        self.encoder = shared_encoder
        self.policy_decoder = policy_q_head
        self.target_decoder = target_q_head

        self.target_net.eval()

        # Optimize shared encoder + online Q-head only.
        self.optimizer = optim.Adam(
            self.policy_net.parameters()
        )

        # Independent network with identical state-dict layout for evaluation.
        self.test_net = MLPQNet(
            state_size=data_train.state_size,
            action_dim=3,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
        ).to(device)

        assert self.policy_net.encoder is self.target_net.encoder
        assert self.policy_net.q_head is not self.target_net.q_head
