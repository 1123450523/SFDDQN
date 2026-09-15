from __future__ import annotations

import random
from itertools import count
from pathlib import Path

import torch
from RL.BaseTrain_DDQN import BaseTrain as BaseTrainDDQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseTrainMixed(BaseTrainDDQN):

    def __init__(
        self,
        data_loader,
        data_train,
        data_test,
        dataset_name,
        model_kind,
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
        experiment_dir: str | Path | None = None,
    ):
        super().__init__(
            data_loader=data_loader,
            data_train=data_train,
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

        self.train_envs = [self.data_train]
        self.train_env_names = [str(dataset_name)]

    def set_train_envs(self, train_envs, env_names=None):
        if train_envs is None or len(train_envs) == 0:
            raise ValueError("train_envs must contain at least one environment.")

        self.train_envs = list(train_envs)

        if env_names is None:
            self.train_env_names = [
                f"source_{i}" for i in range(len(self.train_envs))
            ]
        else:
            if len(env_names) != len(self.train_envs):
                raise ValueError("env_names must match the number of train_envs.")
            self.train_env_names = [str(x) for x in env_names]

    def set_target_env(self, target_env, target_name=None):
        name = str(target_name) if target_name is not None else "target"
        self.set_train_envs([target_env], [name])
        self.EPS_START = 0.2
        self.EPS_END = 0.05
        self.EPS_DECAY = 10000
        self.steps_done = 0

    def _build_balanced_schedule(self, num_episodes: int) -> list[int]:
        if num_episodes <= 0:
            return []

        n_envs = len(self.train_envs)
        if n_envs <= 0:
            raise RuntimeError("No training environments are configured.")

        schedule: list[int] = []
        base_cycle = list(range(n_envs))

        while len(schedule) < num_episodes:
            cycle = base_cycle.copy()
            random.shuffle(cycle)
            schedule.extend(cycle)

        return schedule[:num_episodes]

    def train_per_env(self, episodes_per_env=60):
        episodes_per_env = int(episodes_per_env)
        if episodes_per_env <= 0:
            raise ValueError("episodes_per_env must be positive.")

        n_envs = len(self.train_envs)
        if n_envs <= 0:
            raise RuntimeError("No training environments are configured.")

        total_episodes = episodes_per_env * n_envs
        print(
            f"Balanced training: {episodes_per_env} episodes/environment x "
            f"{n_envs} environments = {total_episodes} total episodes"
        )
        return self.train(num_episodes=total_episodes)

    def train(self, num_episodes=100):
        num_episodes = int(num_episodes)
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")

        print("Training", self.model_kind, "...")

        schedule = self._build_balanced_schedule(num_episodes)
        episode_counts = {name: 0 for name in self.train_env_names}

        for episode_idx, env_idx in enumerate(schedule, start=1):
            active_env = self.train_envs[env_idx]
            active_name = self.train_env_names[env_idx]
            episode_counts[active_name] += 1

            active_env.reset()
            state_np = active_env.get_current_state()
            if state_np is None:
                print(
                    f"Warning: skipped episode {episode_idx}/{num_episodes} "
                    f"for {active_name} because the initial state is None."
                )
                continue

            state = torch.tensor(
                [state_np],
                dtype=torch.float,
                device=device,
            )

            for _t in count():
                action = self.select_action(state)
                done, reward, next_state = active_env.step(action.item())

                reward_value = float(reward)
                reward_tensor = torch.tensor(
                    [reward_value],
                    dtype=torch.float,
                    device=device,
                )

                if next_state is not None:
                    next_state_tensor = torch.tensor(
                        [next_state],
                        dtype=torch.float,
                        device=device,
                    )
                else:
                    next_state_tensor = None

                self.memory.push(
                    state,
                    action,
                    next_state_tensor,
                    reward_tensor,
                )

                if not done:
                    next_state_for_action = active_env.get_current_state()
                    if next_state_for_action is None:
                        break

                    state = torch.tensor(
                        [next_state_for_action],
                        dtype=torch.float,
                        device=device,
                    )

                self.optimize_model()

                if done:
                    break

        self.save_model(self.policy_net.state_dict())

        counts_text = ", ".join(
            f"{name}={count}" for name, count in episode_counts.items()
        )
        print(f"Complete. Episode counts: {counts_text}")

        return episode_counts
