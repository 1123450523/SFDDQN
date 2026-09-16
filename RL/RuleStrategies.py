from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

# ============================================================
# 适配当前框架的 raw action 编码（与 RL/TradingEnv.step 一致）
#   0 -> buy
#   1 -> None
#   2 -> sell
#
# 当前框架的 make_investment() 从 index = window_size 开始写入动作。
# 因此，这里的固定策略会为每个“可执行日 i”生成一个动作：
#   i = window_size, ..., len(df)-1
# 并且只允许使用 i 之前（含 i-1）的历史数据来形成信号。
# 这样既能和当前框架对齐，也避免未来信息。
# ============================================================
ACTION_BUY = 0
ACTION_HOLD = 1
ACTION_SELL = 2


def signal_to_action(sig: int) -> int:
    """
    把交易信号 {-1, 0, +1} 映射到当前框架 raw action {2, 1, 0}
      +1 -> buy  -> 0
       0 -> None -> 1
      -1 -> sell -> 2
    """
    s = int(sig)
    if s == +1:
        return ACTION_BUY
    if s == 0:
        return ACTION_HOLD
    if s == -1:
        return ACTION_SELL
    raise ValueError(f"invalid signal {sig}, expected one of -1/0/+1")


@dataclass
class BaseRule:
    name: str

    def prepare(self, df: pd.DataFrame) -> None:
        """
        df: 当前框架中的价格数据（需包含 close 列，按时间升序）
        """
        self.df = df.reset_index(drop=True).copy()
        if "close" not in self.df.columns:
            raise ValueError(f"[{self.name}] 输入数据缺少 close 列。")
        self.close = self.df["close"].astype(float)

    def signal_from_history_end_at(self, t: int) -> int:
        """
        使用截至原始索引 t 的历史数据形成信号 {-1,0,+1}。
        子类必须实现。
        """
        raise NotImplementedError

    def action_from_history_end_at(self, t: int) -> int:
        return signal_to_action(self.signal_from_history_end_at(t))

    def generate_action_list(self, df: pd.DataFrame, window_size: int) -> List[int]:
        """
        为当前框架生成 action_list。

        当前框架中 make_investment() 从 i = window_size 开始写入动作，
        因此第一个执行日是 i = window_size，对应只能使用 [0, ..., window_size-1]
        的历史信息形成信号，即 t = i - 1 = window_size - 1。

        最终动作数量为：
            len(df) - window_size
        恰好与当前 make_investment() 的写入区间一致：
            i = window_size, ..., len(df)-1
        """
        self.prepare(df)

        n = len(self.df)
        if n <= window_size:
            raise ValueError(
                f"[{self.name}] 数据长度不足：len(df)={n}, window_size={window_size}"
            )

        actions: List[int] = []
        # 执行日 i = window_size ... n-1
        # 可用历史截止日 t = i-1 = window_size-1 ... n-2
        for hist_end_t in range(window_size - 1, n - 1):
            actions.append(self.action_from_history_end_at(hist_end_t))
        return actions


class BuyAndHold(BaseRule):
    def __init__(self):
        super().__init__(name="B&H")

    def signal_from_history_end_at(self, t: int) -> int:
        return +1


class SellAndHold(BaseRule):
    def __init__(self):
        super().__init__(name="S&H")

    def signal_from_history_end_at(self, t: int) -> int:
        return -1


class MeanReversionSMA5_10(BaseRule):
    """
    均值回归：SMA5 <= SMA10 -> 做多；SMA5 > SMA10 -> 做空；相等 -> 观望
    """
    def __init__(self):
        super().__init__(name="MR_SMA5_10")

    def prepare(self, df: pd.DataFrame) -> None:
        super().prepare(df)
        self.sma5 = self.close.rolling(5, min_periods=5).mean()
        self.sma10 = self.close.rolling(10, min_periods=10).mean()

    def signal_from_history_end_at(self, t: int) -> int:
        if np.isnan(self.sma5.iloc[t]) or np.isnan(self.sma10.iloc[t]):
            return 0

        eps = 1e-6
        if self.sma5.iloc[t] <= self.sma10.iloc[t] * (1 - eps):
            return +1
        if self.sma5.iloc[t] > self.sma10.iloc[t] * (1 + eps):
            return -1
        return 0


class TrendFollowSMA5_10(BaseRule):
    """
    趋势跟随：SMA5 > SMA10 -> 做多；SMA5 < SMA10 -> 做空；相等 -> 观望
    """
    def __init__(self):
        super().__init__(name="TF_SMA5_10")

    def prepare(self, df: pd.DataFrame) -> None:
        super().prepare(df)
        self.sma5 = self.close.rolling(5, min_periods=5).mean()
        self.sma10 = self.close.rolling(10, min_periods=10).mean()

    def signal_from_history_end_at(self, t: int) -> int:
        if np.isnan(self.sma5.iloc[t]) or np.isnan(self.sma10.iloc[t]):
            return 0

        eps = 1e-6
        if self.sma5.iloc[t] > self.sma10.iloc[t] * (1 + eps):
            return +1
        if self.sma5.iloc[t] < self.sma10.iloc[t] * (1 - eps):
            return -1
        return 0


RULES = [
    BuyAndHold(),
    SellAndHold(),
    MeanReversionSMA5_10(),
    TrendFollowSMA5_10(),
]
