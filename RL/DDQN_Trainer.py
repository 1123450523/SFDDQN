from __future__ import annotations

import math
import os
import random
from itertools import count
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from RL.ReplayMemory import ReplayMemory, Transition
from RL.Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseTrain:

    def __init__(
        self,
        data_loader,
        data_train,
        data_test,
        dataset_name,
        model_kind,
        data_val=None,
        window_size=10,
        transaction_cost=0.0001,
        BATCH_SIZE=32,
        GAMMA=0.9,
        ReplayMemorySize=64,
        TARGET_UPDATE=10,
        n_step=5,
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=50000,
        experiment_dir: str | os.PathLike | None = None,
    ):
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.DATASET_NAME = dataset_name
        self.model_kind = model_kind

        self.window_size = window_size
        self.transaction_cost = transaction_cost

        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.ReplayMemorySize = ReplayMemorySize
        self.TARGET_UPDATE = TARGET_UPDATE

        self.n_step = n_step

        self.data_loader = data_loader

        self.memory = ReplayMemory(ReplayMemorySize)

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 50000
        self.steps_done = 0

        self.optimize_steps = 0

        if experiment_dir is not None:
            self.PATH = str(Path(experiment_dir))
        else:

            self.PATH = os.path.join(
                Path(os.path.abspath(os.path.dirname(__file__))).parent,
                "Results",
                self.DATASET_NAME,
                f"{self.model_kind}_{self.data_train.data_kind}"
                f"_w{self.window_size}"
                f"_b{self.BATCH_SIZE}"
                f"_g{self.GAMMA}"
                f"_r{self.ReplayMemorySize}"
                f"_u{self.TARGET_UPDATE}"
                f"_s{self.n_step}",
            )

        os.makedirs(self.PATH, exist_ok=True)
        self.model_dir = os.path.join(self.PATH, "model.pkl")

    def _update_target_network(self):
        """Synchronize the target value network."""
        if getattr(self, "shared_encoder", False):
            self.target_decoder.load_state_dict(
                self.policy_decoder.state_dict()
            )
        else:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                self.policy_net.eval()
                action = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                return action

        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

    @staticmethod
    def _set_eval_temporarily(mod: Optional[torch.nn.Module]):
        if isinstance(mod, torch.nn.Module):
            was_training = mod.training
            mod.eval()
            return was_training
        return None

    @staticmethod
    def _restore_train_flag(
        mod: Optional[torch.nn.Module],
        was_training: Optional[bool],
    ):
        if isinstance(mod, torch.nn.Module) and was_training is not None:
            mod.train(was_training)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )

        if non_final_mask.any():
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            )
        else:
            non_final_next_states = None

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)

        if non_final_next_states is not None:

            enc = getattr(self, "encoder", None)

            enc_flag = self._set_eval_temporarily(enc)
            pol_flag = self._set_eval_temporarily(self.policy_net)
            tar_flag = self._set_eval_temporarily(self.target_net)

            try:
                with torch.no_grad():

                    next_actions = self.policy_net(non_final_next_states).max(1)[1]

                    next_q = (
                        self.target_net(non_final_next_states)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze(1)
                    )

                next_state_values[non_final_mask] = next_q.detach()

            finally:
                self._restore_train_flag(self.target_net, tar_flag)
                self._restore_train_flag(self.policy_net, pol_flag)
                self._restore_train_flag(enc, enc_flag)

        expected_state_action_values = (
            reward_batch
            + self.GAMMA * next_state_values
        )

        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
        )

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        self.optimize_steps += 1
        if self.optimize_steps % self.TARGET_UPDATE == 0:
            self._update_target_network()

    def train(self, num_episodes=100):
        print("Training", self.model_kind, "(DDQN) ...")
        for _ in range(num_episodes):
            self.data_train.reset()
            state = torch.tensor(
                [self.data_train.get_current_state()],
                dtype=torch.float,
                device=device,
            )

            for _t in count():
                action = self.select_action(state)
                done, reward, next_state = self.data_train.step(action.item())

                reward_value = float(reward)

                reward = torch.tensor([reward_value], dtype=torch.float, device=device)

                if next_state is not None:
                    next_state = torch.tensor(
                        [next_state],
                        dtype=torch.float,
                        device=device,
                    )

                self.memory.push(state, action, next_state, reward)

                if not done:
                    state = torch.tensor(
                        [self.data_train.get_current_state()],
                        dtype=torch.float,
                        device=device,
                    )

                self.optimize_model()

                if done:
                    break

        self.save_model(self.policy_net.state_dict())
        print("Complete")

    def save_model(self, model_state_dict):
        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        torch.save(model_state_dict, self.model_dir)

    def test(self, initial_investment=500000, test_type="test") -> Evaluation:
        """Evaluate the train, validation, or test segment."""
        if test_type == "train":
            data = self.data_train
            evaluation_dates = (
                self.data_loader.data_train_with_date.index
            )

        elif test_type == "val":
            if self.data_val is None:
                raise ValueError(
                    "Validation data is unavailable. "
                    "Use selection mode with val_start configured."
                )

            data = self.data_val
            evaluation_dates = (
                self.data_loader.data_val_with_date.index
            )

        elif test_type == "test":
            data = self.data_test
            evaluation_dates = (
                self.data_loader.data_test_with_date.index
            )

        else:
            raise ValueError(
                "test_type must be 'train', 'val', or 'test'."
            )

        self.test_net.load_state_dict(torch.load(self.model_dir, map_location=device))
        self.test_net.to(device)

        action_list = []
        data.__iter__()

        for batch in data:
            try:
                action_batch = self.test_net(batch).max(1)[1]
                action_list += list(action_batch.detach().cpu().numpy())
            except Exception:
                action_list += [1]

        data.make_investment(action_list)

        return Evaluation(
            data.data,
            data.action_name,
            initial_investment,
            self.transaction_cost,
            dates=evaluation_dates,
        )
