import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from sklearn.preprocessing import StandardScaler


class YahooFinanceDataLoader:
    """Load Yahoo-style OHLCV data and create train/validation/test splits."""

    def __init__(
        self,
        dataset_name,
        split_point,
        begin_date=None,
        end_date=None,
        load_from_file=False,
        val_start=None,
    ):
        warnings.filterwarnings("ignore")

        self.DATA_NAME = dataset_name
        self.DATA_PATH = os.path.join(
            Path(os.path.abspath(os.path.dirname(__file__))).parent,
            f"Data/{dataset_name}",
        ) + "/"
        self.DATA_FILE = dataset_name + ".csv"

        self.split_point = split_point
        self.val_start = val_start
        self.begin_date = begin_date
        self.end_date = end_date

        if not load_from_file:
            self.data = self.load_data()
            self.data.to_csv(
                f"{self.DATA_PATH}data_processed.csv",
                index=True,
            )
        else:
            self.data = pd.read_csv(
                f"{self.DATA_PATH}data_processed.csv"
            )

            if "Date" in self.data.columns:
                self.data["Date"] = pd.to_datetime(self.data["Date"])
                self.data.set_index("Date", inplace=True)
            elif "Unnamed: 0" in self.data.columns:
                self.data["Unnamed: 0"] = pd.to_datetime(
                    self.data["Unnamed: 0"]
                )
                self.data.set_index("Unnamed: 0", inplace=True)
                self.data.index.name = "Date"
            else:
                raise ValueError(
                    "data_processed.csv 缺少 Date 列（或 Unnamed: 0 索引列）"
                )

            norm_cols = [
                "open_norm",
                "high_norm",
                "low_norm",
                "close_norm",
                "volume_norm",
            ]
            existing_norm_cols = [
                c for c in norm_cols if c in self.data.columns
            ]
            if existing_norm_cols:
                self.data.drop(
                    columns=existing_norm_cols,
                    inplace=True,
                )

        self.data = self._filter_by_date(self.data)
        raw_train, raw_val, raw_test = self._split_train_val_test(self.data)

        (
            self.data_train,
            self.data_val,
            self.data_test,
        ) = self.normalize_train_val_test(
            raw_train,
            raw_val,
            raw_test,
        )

        self.data_train_with_date = self.data_train.copy()
        self.data_val_with_date = (
            self.data_val.copy() if self.data_val is not None else None
        )
        self.data_test_with_date = self.data_test.copy()

        parts = [self.data_train]
        if self.data_val is not None:
            parts.append(self.data_val)
        parts.append(self.data_test)
        self.data = pd.concat(parts, axis=0)

        self._plot_split_paper_style_auto()

        self.data_train.reset_index(drop=True, inplace=True)
        if self.data_val is not None:
            self.data_val.reset_index(drop=True, inplace=True)
        self.data_test.reset_index(drop=True, inplace=True)

    def load_data(self):
        df = pd.read_csv(f"{self.DATA_PATH}{self.DATA_FILE}")
        df = df.dropna().copy()

        cols = {c.lower(): c for c in df.columns}

        def pick(*names):
            for name in names:
                if name in df.columns:
                    return name
                if name.lower() in cols:
                    return cols[name.lower()]
            return None

        col_date = pick("Date")
        col_open = pick("Open")
        col_high = pick("High")
        col_low = pick("Low")
        col_close = pick("Close")
        col_volume = pick("Volume")

        if col_date is None:
            raise ValueError("CSV 缺少 Date 列")
        if any(
            col is None
            for col in [col_open, col_high, col_low, col_close]
        ):
            raise ValueError("CSV 缺少 open/high/low/close 必要列")

        df[col_date] = pd.to_datetime(df[col_date])
        df.set_index(col_date, inplace=True)
        df.index.name = "Date"

        out = pd.DataFrame(index=df.index)
        out["open"] = df[col_open].astype(float)
        out["high"] = df[col_high].astype(float)
        out["low"] = df[col_low].astype(float)
        out["close"] = df[col_close].astype(float)

        if col_volume is not None:
            out["volume"] = df[col_volume].astype(float)

        return out

    def _filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.begin_date is not None:
            df = df[df.index >= pd.to_datetime(self.begin_date)]
        if self.end_date is not None:
            df = df[df.index <= pd.to_datetime(self.end_date)]

        return df

    def _split_train_val_test(self, df: pd.DataFrame):
        if isinstance(self.split_point, str):
            test_start = pd.to_datetime(self.split_point)

            if self.val_start is None:
                train_df = df[df.index < test_start].copy()
                val_df = None
                test_df = df[df.index >= test_start].copy()
            else:
                val_start = pd.to_datetime(self.val_start)
                if val_start >= test_start:
                    raise ValueError(
                        "val_start must be earlier than split_point/test_start."
                    )

                train_df = df[df.index < val_start].copy()
                val_df = df[
                    (df.index >= val_start) & (df.index < test_start)
                ].copy()
                test_df = df[df.index >= test_start].copy()

        elif isinstance(self.split_point, int):
            if self.val_start is None:
                train_df = df[: self.split_point].copy()
                val_df = None
                test_df = df[self.split_point :].copy()
            elif isinstance(self.val_start, int):
                if self.val_start >= self.split_point:
                    raise ValueError(
                        "Integer val_start must be smaller than split_point."
                    )

                train_df = df[: self.val_start].copy()
                val_df = df[self.val_start : self.split_point].copy()
                test_df = df[self.split_point :].copy()
            else:
                raise ValueError(
                    "When split_point is int, val_start must be int or None."
                )
        else:
            raise ValueError(
                "split_point should be either int or date string."
            )

        if len(train_df) == 0:
            raise ValueError(
                f"[{self.DATA_NAME}] 训练集为空，请检查日期切分设置。"
            )
        if self.val_start is not None and (
            val_df is None or len(val_df) == 0
        ):
            raise ValueError(
                f"[{self.DATA_NAME}] 验证集为空，请检查 val_start/split_point。"
            )
        if len(test_df) == 0:
            raise ValueError(
                f"[{self.DATA_NAME}] 测试集为空，请检查 split_point/end_date。"
            )

        return train_df, val_df, test_df

    def normalize_train_val_test(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
    ):
        """Fit Z-score scalers on training data only and transform all splits."""
        train_df = train_df.copy()
        val_df = val_df.copy() if val_df is not None else None
        test_df = test_df.copy()

        cols = [
            ("open", "open_norm"),
            ("high", "high_norm"),
            ("low", "low_norm"),
            ("close", "close_norm"),
        ]
        if "volume" in train_df.columns:
            cols.append(("volume", "volume_norm"))

        self.scalers = {}

        for col_in, col_out in cols:
            scaler = StandardScaler()
            train_df[col_out] = scaler.fit_transform(train_df[[col_in]])
            self.scalers[col_in] = scaler

            if val_df is not None:
                val_df[col_out] = scaler.transform(val_df[[col_in]])
            test_df[col_out] = scaler.transform(test_df[[col_in]])

        return train_df, val_df, test_df

    def normalize_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        train_df, _, test_df = self.normalize_train_val_test(
            train_df,
            None,
            test_df,
        )
        return train_df, test_df

    def plot_data(self):
        self._plot_split_paper_style_auto()

    @staticmethod
    def _compute_uniform_ticks(
        start: pd.Timestamp,
        end: pd.Timestamp,
        n: int = 7,
    ):
        n = max(3, int(n))
        step = (end - start) / (n - 1)
        return [start + i * step for i in range(n)]

    def _plot_split_paper_style_auto(self):
        try:
            train = self.data_train_with_date
            val = self.data_val_with_date
            test = self.data_test_with_date

            if len(train) == 0:
                warnings.warn(
                    f"[{self.DATA_NAME}] 训练集为空，跳过 split 图绘制。"
                )
                return

            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": [
                    "Times New Roman",
                    "Times",
                    "STIXGeneral",
                    "DejaVu Serif",
                ],
                "mathtext.fontset": "stix",
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            })

            fig = plt.figure(figsize=(8.2, 3.0), dpi=600)
            ax = fig.add_subplot(111)

            ax.plot(
                train.index,
                train["close"].values,
                lw=1.6,
                label="Training Set",
            )

            if val is not None and len(val) > 0:
                ax.plot(
                    val.index,
                    val["close"].values,
                    lw=1.6,
                    label="Validation Set",
                )
                ax.axvline(
                    pd.to_datetime(self.val_start),
                    lw=1.0,
                    linestyle=(0, (4, 4)),
                )

            ax.plot(
                test.index,
                test["close"].values,
                lw=1.6,
                label="Test Set",
            )

            if isinstance(self.split_point, str):
                ax.axvline(
                    pd.to_datetime(self.split_point),
                    lw=1.0,
                    linestyle=(0, (4, 4)),
                )

            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.set_title(
                f"{self.DATA_NAME} "
                + (
                    "Train/Validation/Test Split"
                    if val is not None
                    else "Train/Test Split"
                )
            )

            ax.grid(
                True,
                which="major",
                alpha=0.18,
                linestyle="--",
                linewidth=0.5,
            )
            ax.tick_params(axis="x", rotation=20, pad=0.6)
            ax.tick_params(axis="y", pad=3)

            global_start = min(train.index.min(), test.index.min())
            global_end = max(train.index.max(), test.index.max())

            start = pd.Timestamp(
                year=global_start.year,
                month=1,
                day=1,
            )
            end = pd.Timestamp(
                year=global_end.year,
                month=12,
                day=31,
            )

            ticks = self._compute_uniform_ticks(start, end, n=7)
            tick_pos = [mdates.date2num(t) for t in ticks]
            ax.xaxis.set_major_locator(mticker.FixedLocator(tick_pos))
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y-%m-%d")
            )

            span_days = max(1, (global_end - global_start).days)
            pad = pd.Timedelta(days=max(30, int(span_days * 0.05)))
            ax.set_xlim(global_start - pad, global_end + pad)
            ax.legend(loc="upper left", frameon=True)

            out_dir = Path(self.DATA_PATH)
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = (
                "train_val_test_split"
                if val is not None
                else "train_test_split"
            )

            fig.savefig(
                out_dir / f"{self.DATA_NAME}_{suffix}.jpg",
                bbox_inches="tight",
                dpi=600,
            )
            plt.close(fig)

        except Exception as e:
            warnings.warn(
                f"[{self.DATA_NAME}] split 图绘制失败：{e}"
            )

    def _plot_train_test_split_paper_style_auto(self):
        self._plot_split_paper_style_auto()
