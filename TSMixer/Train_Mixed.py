from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.optim as optim

from RL.Transfer_DDQN_Trainer import BaseTrainMixed
from TSMixer.Decoder import Decoder
from TSMixer.Encoder import Encoder
from TSMixer.Seq2SeqModel import Seq2Seq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainMixed(BaseTrainMixed):

    def __init__(
        self,
        data_loader,
        data_train,
        data_test,
        dataset_name,
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
        EPS_DECAY=50000,
        learning_rate=1e-3,
        norm_type="batch",
        normalize_before=False,
        e_layers=2,
        ff_dim=128,
        dropout=0.1,
        activation="relu",
        pool="mean",
        feature_dim=5,
        experiment_dir=None,
    ):
        super().__init__(
            data_loader=data_loader,
            data_train=data_train,
            data_test=data_test,
            dataset_name=dataset_name,
            model_kind="TSMixer-SFDDQN-Mixed",
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

        self.shared_encoder = True

        self.encoder = Encoder(
            n_classes,
            data_train.state_size,
            feature_dim=feature_dim,
            e_layers=e_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            normalize_before=normalize_before,
            pool=pool,
        ).to(device)

        self.policy_decoder = Decoder(
            n_classes,
            3,
        ).to(device)

        self.target_decoder = Decoder(
            n_classes,
            3,
        ).to(device)

        self.policy_net = Seq2Seq(
            self.encoder,
            self.policy_decoder,
        ).to(device)

        self.target_net = Seq2Seq(
            self.encoder,
            self.target_decoder,
        ).to(device)

        self.sync_target_decoder()

        self.learning_rate = float(learning_rate)
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        self._reset_optimizer()

        test_encoder = Encoder(
            n_classes,
            data_train.state_size,
            feature_dim=feature_dim,
            e_layers=e_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            normalize_before=normalize_before,
            pool=pool,
        ).to(device)

        test_decoder = Decoder(
            n_classes,
            3,
        ).to(device)

        self.test_net = Seq2Seq(
            test_encoder,
            test_decoder,
        ).to(device)

        assert self.policy_net.encoder is self.target_net.encoder
        assert self.policy_decoder is not self.target_decoder

    def _reset_optimizer(self):
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
        )

    def sync_target_decoder(self):
        self.target_decoder.load_state_dict(
            self.policy_decoder.state_dict()
        )
        self.target_decoder.eval()

    def get_pretrained_policy_state(self, to_cpu=True):
        state = self.policy_net.state_dict()

        if to_cpu:
            return {
                key: value.detach().cpu().clone()
                for key, value in state.items()
            }

        return {
            key: value.detach().clone()
            for key, value in state.items()
        }

    def load_pretrained_policy_state(self, pretrained_state, strict=True):
        if isinstance(pretrained_state, (str, Path)):
            pretrained_state = torch.load(
                pretrained_state,
                map_location=device,
            )

        if not isinstance(pretrained_state, Mapping):
            raise TypeError(
                "pretrained_state must be a state_dict-like mapping or a file path."
            )

        incompatible = self.policy_net.load_state_dict(
            pretrained_state,
            strict=strict,
        )

        self.sync_target_decoder()

        self._reset_optimizer()

        return incompatible
