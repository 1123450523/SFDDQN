import math

import torch


class TradingEnv:
    """Three-position trading environment with an n-step forward reward."""

    def __init__(
        self,
        data,
        action_name,
        device,
        gamma=0.9,
        n_step=5,
        batch_size=32,
        start_index_reward=0,
        transaction_cost=0.0001,
    ):
        self.data = data
        self.states = []
        self.current_state_index = -1
        self.pos = 0

        self.batch_size = batch_size
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.close_price = list(data.close)
        self.action_name = action_name

        self.code_to_action = {0: "buy", 1: "None", 2: "sell"}
        self.code_to_signal = {0: +1, 1: 0, 2: -1}
        self.op_to_name = {+1: "buy", 0: "None", -1: "sell"}

        self.start_index_reward = start_index_reward
        self.trading_cost_ratio = float(transaction_cost)

    @staticmethod
    def _signal_to_operation(pos: int, signal: int) -> int:
        if pos == 0:
            return signal
        if pos == 1:
            return -1 if signal == -1 else 0
        if pos == -1:
            return +1 if signal == +1 else 0
        raise ValueError(f"invalid pos={pos}")

    @staticmethod
    def _apply_operation_to_pos(pos: int, op: int) -> int:
        if pos == 0:
            if op == +1:
                return +1
            if op == -1:
                return -1
            return 0
        if pos == 1:
            return 0 if op == -1 else 1
        if pos == -1:
            return 0 if op == +1 else -1
        raise ValueError(f"invalid pos={pos}")

    def get_current_state(self):
        self.current_state_index += 1
        if self.current_state_index == len(self.states):
            return None
        return self.states[self.current_state_index]

    def step(self, action):
        """Return s(t+1) while the reward uses the configured forward horizon."""
        last_reward_state_index = len(self.states) - self.n_step - 1

        if self.current_state_index > last_reward_state_index:
            raise RuntimeError(
                "step() called after the last state with a complete "
                "n-step reward horizon."
            )

        signal = int(self.code_to_signal[int(action)])
        op = self._signal_to_operation(self.pos, signal)
        self.pos = self._apply_operation_to_pos(self.pos, op)
        reward = float(self.get_reward(op))

        done = self.current_state_index >= last_reward_state_index
        if done:
            next_state = None
        else:
            next_state = self.states[self.current_state_index + 1]

        return done, reward, next_state

    def get_reward(self, op: int) -> float:
        reward_index_first = self.current_state_index + self.start_index_reward
        reward_index_last = (
            self.current_state_index + self.start_index_reward + self.n_step
            if self.current_state_index + self.n_step < len(self.states)
            else len(self.close_price) - 1
        )

        p1 = float(self.close_price[reward_index_first])
        p2 = float(self.close_price[reward_index_last])

        ret = (p2 - p1) / max(p1, 1e-12)
        gross = self.pos * ret
        fee = self.trading_cost_ratio if op != 0 else 0.0
        return (gross - fee) * 100.0

    def calculate_reward_for_one_step(self, action, index, rewards):
        index += self.start_index_reward
        a = int(action)
        signal = int(self.code_to_signal[a])
        op = self._signal_to_operation(self.pos, signal)
        next_pos = self._apply_operation_to_pos(self.pos, op)

        if next_pos == 1:
            diff = self.close_price[index + 1] - self.close_price[index]
        elif next_pos == -1:
            diff = self.close_price[index] - self.close_price[index + 1]
        else:
            diff = 0.0

        if op != 0:
            diff -= abs(self.close_price[index]) * self.trading_cost_ratio

        rewards.append(diff)

    def reset(self):
        self.current_state_index = -1
        self.pos = 0

    def __iter__(self):
        self.index_batch = 0
        self.num_batch = math.ceil(len(self.states) / self.batch_size)
        return self

    def __next__(self):
        if self.index_batch < self.num_batch:
            batch = [
                torch.tensor([s], dtype=torch.float, device=self.device)
                for s in self.states[
                    self.index_batch * self.batch_size : (self.index_batch + 1) * self.batch_size
                ]
            ]
            self.index_batch += 1
            return torch.cat(batch)
        raise StopIteration

    def get_total_reward(self, action_list):
        total_reward = 0.0
        self.reset()

        for a in action_list:
            self.current_state_index += 1
            if self.current_state_index + self.n_step >= len(self.states):
                break

            signal = int(self.code_to_signal[int(a)])
            op = self._signal_to_operation(self.pos, signal)
            self.pos = self._apply_operation_to_pos(self.pos, op)
            total_reward += float(self.get_reward(op))

        return total_reward

    def make_investment(self, action_list):
        """Write executed actions, raw signals, and positions to the data frame."""
        exec_col = self.action_name
        sig_col = f"{self.action_name}_signal"

        self.data[exec_col] = "None"
        self.data[sig_col] = "None"
        if "position" not in self.data.columns:
            self.data["position"] = 0

        pos = 0
        i = self.start_index_reward + 1

        for a in action_list:
            a = int(a)
            sig_name = self.code_to_action[a]
            self.data[sig_col][i] = sig_name

            signal = int(self.code_to_signal[a])
            op = self._signal_to_operation(pos, signal)
            self.data[exec_col][i] = self.op_to_name[op]

            pos = self._apply_operation_to_pos(pos, op)
            self.data["position"][i] = pos
            i += 1
