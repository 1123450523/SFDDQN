from __future__ import annotations

import torch
import torch.optim as optim

from RL.DDQN_Trainer import BaseTrain
from TSMixer.Decoder import Decoder
from TSMixer.Encoder import Encoder
from TSMixer.Seq2SeqModel import Seq2Seq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    """TSMixer-DDQN with independent online and target encoders."""

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
        EPS_DECAY=50000,
        learning_rate=3e-4,
        norm_type="batch",
        normalize_before=False,
        e_layers=2,
        ff_dim=64,
        dropout=0.1,
        activation="relu",
        pool="mean",
        feature_dim=5,
        model_kind="TSMixer-DDQN",
        experiment_dir=None,
    ):
        super().__init__(
            data_loader=data_loader,
            data_train=data_train,
            data_val=data_val,
            data_test=data_test,
            dataset_name=dataset_name,
            model_kind=model_kind,
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

        self.shared_encoder = False

        self.policy_encoder = Encoder(
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

        self.policy_net = Seq2Seq(
            self.policy_encoder,
            self.policy_decoder,
        ).to(device)

        self.target_encoder = Encoder(
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

        self.target_decoder = Decoder(
            n_classes,
            3,
        ).to(device)

        self.target_net = Seq2Seq(
            self.target_encoder,
            self.target_decoder,
        ).to(device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        self.learning_rate = float(learning_rate)
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
        )

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
