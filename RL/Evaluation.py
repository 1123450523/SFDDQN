from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    CR: float
    AR: float
    Sharpe: float
    MDD: float
    Volatility: float
    Sortino: float
    Calmar: float


class Evaluation:
    """Evaluate long/neutral/short trading results and standard performance metrics."""

    def __init__(
        self,
        data: pd.DataFrame,
        action_label: str,
        initial_investment: float = 500000,
        trading_cost_ratio: float = 0.0001,
        dates: Optional[Sequence] = None,
    ):
        if action_label not in data.columns:
            raise ValueError(
                f"action column '{action_label}' not found in Data.columns"
            )

        if "close" not in data.columns:
            raise ValueError("close column not found in Data.columns")

        self.data = data.reset_index(drop=True).copy()

        self.initial_investment = float(initial_investment)
        self.action_label = str(action_label)
        self.trading_cost_ratio = float(trading_cost_ratio)


        self.dates: Optional[pd.DatetimeIndex] = None

        if dates is not None:
            parsed_dates = pd.to_datetime(
                pd.Index(dates),
                errors="coerce",
            )

            parsed_dates = parsed_dates[~pd.isna(parsed_dates)]

            if len(parsed_dates) >= 2:
                self.dates = pd.DatetimeIndex(parsed_dates)

        self._portfolio_cache: Optional[List[float]] = None


    def get_daily_portfolio_value(self) -> List[float]:
        """Replay portfolio value from executed buy, sell, and hold operations."""

        if self._portfolio_cache is not None:
            return self._portfolio_cache

        pv: List[float] = [self.initial_investment]

        equity = float(self.initial_investment)

        pos = 0
        prev_px: Optional[float] = None

        for i in range(len(self.data)):
            act = str(
                self.data[self.action_label].iloc[i]
            ).strip()

            px = float(self.data["close"].iloc[i])
            px_safe = max(px, 1e-12)


            if pos != 0:
                if prev_px is None:
                    prev_px = px_safe

                ratio = (
                    px_safe
                    / max(float(prev_px), 1e-12)
                )

                if pos == 1:
                    equity *= ratio
                else:
                    multiplier = 2.0 - ratio
                    equity *= max(multiplier, 0.0)

                prev_px = px_safe


            if act == "buy":
                if pos == 0:
                    equity *= (1.0 - self.trading_cost_ratio)
                    pos = 1
                    prev_px = px_safe

                elif pos == -1:
                    equity *= (1.0 - self.trading_cost_ratio)
                    pos = 0
                    prev_px = None

            elif act == "sell":
                if pos == 0:
                    equity *= (1.0 - self.trading_cost_ratio)
                    pos = -1
                    prev_px = px_safe

                elif pos == 1:
                    equity *= (1.0 - self.trading_cost_ratio)
                    pos = 0
                    prev_px = None

            pv.append(float(equity))

        self._portfolio_cache = pv
        return pv


    def compute_metrics(
        self,
        annualization_days: int = 365,
    ) -> Metrics:
        """Compute CR, AR, Sharpe, MDD, volatility, Sortino, and Calmar."""

        pv = np.asarray(
            self.get_daily_portfolio_value(),
            dtype=float,
        )

        if pv.size < 2:
            return Metrics(
                CR=0.0,
                AR=0.0,
                Sharpe=0.0,
                MDD=0.0,
                Volatility=0.0,
                Sortino=0.0,
                Calmar=0.0,
            )

        w0 = float(pv[0])
        wt = float(pv[-1])

        if (
            w0 <= 0
            or not np.isfinite(w0)
            or not np.isfinite(wt)
        ):
            return Metrics(
                CR=0.0,
                AR=0.0,
                Sharpe=0.0,
                MDD=0.0,
                Volatility=0.0,
                Sortino=0.0,
                Calmar=0.0,
            )


        cr = (
            (wt - w0)
            / max(w0, 1e-12)
        ) * 100.0


        daily_ret = (
            np.diff(pv)
            / np.maximum(pv[:-1], 1e-12)
        )


        cur = (
            wt / max(w0, 1e-12)
        ) - 1.0

        if self.dates is None or len(self.dates) < 2:
            raise ValueError(
                "AR calculation requires the actual date sequence. "
                "Please pass dates to Evaluation(..., dates=...)."
            )

        start_date = self.dates[0]
        end_date = self.dates[-1]

        days_held = int(
            (end_date - start_date).days
        )
        days_held = max(days_held, 1)

        base = 1.0 + cur

        if base > 0.0:
            ar = (
                math.pow(
                    base,
                    float(annualization_days)
                    / float(days_held),
                )
                - 1.0
            ) * 100.0
        else:
            ar = -100.0


        trading_days = 252

        mu = (
            float(np.mean(daily_ret))
            if daily_ret.size
            else 0.0
        )

        sigma = (
            float(np.std(daily_ret))
            if daily_ret.size
            else 0.0
        )

        if sigma > 1e-12:
            sharpe = (
                mu * math.sqrt(float(trading_days))
                / sigma
            )
        else:
            sharpe = 0.0


        peak = np.maximum.accumulate(pv)

        dd = (
            (pv - peak)
            / np.maximum(peak, 1e-12)
        )

        mdd = (
            float(-np.min(dd) * 100.0)
            if dd.size
            else 0.0
        )


        volatility = (
            sigma
            * math.sqrt(float(trading_days))
            * 100.0
        )


        if daily_ret.size:
            downside = np.minimum(daily_ret, 0.0)
            downside_deviation = float(
                np.sqrt(np.mean(np.square(downside)))
            )
        else:
            downside_deviation = 0.0

        if downside_deviation > 1e-12:
            sortino = (
                mu
                * math.sqrt(float(trading_days))
                / downside_deviation
            )
        else:
            sortino = 0.0


        if mdd > 1e-12:
            calmar = ar / mdd
        else:
            calmar = 0.0

        return Metrics(
            CR=round(float(cr), 4),
            AR=round(float(ar), 4),
            Sharpe=round(float(sharpe), 4),
            MDD=round(float(mdd), 4),
            Volatility=round(float(volatility), 4),
            Sortino=round(float(sortino), 4),
            Calmar=round(float(calmar), 4),
        )


    def validate_portfolio_calculation(self) -> bool:
        pv = self.get_daily_portfolio_value()

        if any(
            (x is None)
            or (
                isinstance(x, float)
                and np.isnan(x)
            )
            for x in pv
        ):
            return False

        if any(x < 0 for x in pv):
            return False

        if len(pv) != len(self.data) + 1:
            return False

        return True
