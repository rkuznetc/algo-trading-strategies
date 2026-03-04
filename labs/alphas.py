import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from alphapackage.operators import *
from alphapackage.data_utils import DataHolder
from alphapackage.core import BaseAlpha, BasePortfolio


class PortfolioEqual(BasePortfolio):
    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        names = list(self.alphas.keys())
        
        if len(names) == 0:
            return pd.Series(dtype=float)
        
        w = np.repeat(1.0 / len(names), len(names))
        return pd.Series(w, index=names)


class PortfolioSharpe(BasePortfolio):
    """
        Weights:        ~ SharpeRatio for last N months (default = 3 months)
        Cold start:     First N months - equal weights
        Rebalance:      Every N months
    """

    def __init__(self, alphas: dict[str, BaseAlpha],
                 freq: str = "QS",
                 lookback_months: int = 3):
        super().__init__(alphas, frequency=freq)
        self.lookback_months = lookback_months

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        alpha_names = list(self.alphas.keys())
        n = len(alpha_names)

        start_date = date - pd.DateOffset(months=self.lookback_months)

        sharpe_vals = []
        enough_data = True

        for name in alpha_names:
            alpha = self.alphas[name]

            pnl_slice = alpha.pnl[
                (alpha.pnl.index > start_date) &
                (alpha.pnl.index < date)
            ]

            # require at least 1 week of data
            if pnl_slice.shape[0] < 5:
                enough_data = False

            if pnl_slice.std() == 0 or pnl_slice.empty:
                sharpe = 0.0
            else:
                sharpe = pnl_slice.mean() / pnl_slice.std()

            sharpe_vals.append(sharpe)

        if not enough_data:
            return pd.Series(1.0 / n, index=alpha_names)

        sharpe_s = pd.Series(sharpe_vals, index=alpha_names)

        if sharpe_s.abs().sum() == 0:
            return pd.Series(1.0 / n, index=alpha_names)

        weights = sharpe_s / sharpe_s.abs().sum()

        return weights


class PortfolioPNL(BasePortfolio):
    """
        Weights:        ~ cumulative pnl for last 1 year or less period
        Cold Start:     first N months ()
        Rebalance:      every N months (default = 3 months)
    """

    def __init__(self, alphas: dict[str, BaseAlpha], freq: str = "QS"):
        super().__init__(alphas, frequency=freq)

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:

        alpha_names = list(self.alphas.keys())
        n = len(alpha_names)

        # cold start
        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / n, index=alpha_names)

        # 
        start_date = date - pd.DateOffset(years=1)
        first_date = dh.data['close'].index[0]

        if start_date < first_date:
            start_date = first_date

        totals = []

        for name in alpha_names:
            alpha = self.alphas[name]

            pnl_slice = alpha.pnl[
                (alpha.pnl.index > start_date) &
                (alpha.pnl.index < date)
            ]

            total = pnl_slice.sum() if pnl_slice.size > 0 else 0.0
            totals.append(total)

        totals_s = pd.Series(totals, index=alpha_names)

        if totals_s.abs().sum() == 0:
            return pd.Series(1.0 / n, index=alpha_names)

        weights = totals_s / totals_s.abs().sum()

        return weights

class PortfolioCorr(BasePortfolio):
    """
        Weights:        ~ sum(corr(other_alpha)) for last 1 month
        Cold Start:     first 1 month
        Rebalance:      every 1 month
    """

    def __init__(self, alphas: dict[str, BaseAlpha], freq: str = "MS"):
        super().__init__(alphas, frequency=freq)

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:

        alpha_names = list(self.alphas.keys())
        n = len(alpha_names)

        # cold start
        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / n, index=alpha_names)

        months_to_include = pos
        start_date = date - pd.DateOffset(months=months_to_include)
        first_date = dh.data['close'].index[0]

        if start_date < first_date:
            start_date = first_date

        pnl_df = []

        for name in alpha_names:
            alpha = self.alphas[name]

            pnl_slice = alpha.pnl[
                (alpha.pnl.index > start_date) &
                (alpha.pnl.index < date)
            ]

            pnl_df.append(pnl_slice.rename(name))

        df = pd.concat(pnl_df, axis=1)

        if df.shape[0] == 0: # or df.shape[1] < 2:
            return pd.Series(1.0 / n, index=alpha_names)

        corr = df.corr().fillna(0.0)
        np.fill_diagonal(corr.values, 0.0)
        corr_sum = corr.sum(axis=1)

        if corr_sum.abs().sum() == 0:
            return pd.Series(1.0 / n, index=alpha_names)

        weights = corr_sum / corr_sum.abs().sum()

        return weights




### Self-made alphas
class Alpha1(BaseAlpha):
    def get_weights(self, dh : DataHolder):
        raw = dh.data['open'].div(dh.data['close'])
        self.w = decay(raw, 5)
        self.w = neutralize(self.w)
        self.w = normalize(self.w)


class Alpha2(BaseAlpha):
    def get_weights(self, dh):
        raw = dh.data['high'].div(dh.data['close'])
        self.w = decay(raw, 5)
        self.w = normalize(neutralize(self.w))


class Alpha3(BaseAlpha):
    def get_weights(self, dh):
        raw = dh.data['close'].shift(8).div(dh.data['close'].shift(1))
        self.w = decay(raw, 5)
        self.w = normalize(neutralize(self.w))


class Alpha4(BaseAlpha):
    def get_weights(self, dh):
        raw = (-1 * ((dh.data['low'] - dh.data['close']) * (dh.data['open'].pow(1)))) / ((dh.data['low'] - dh.data['high']) * (dh.data['close'].pow(1)))
        self.w = decay(raw, 5)
        self.w = normalize(neutralize(self.w))


class Alpha5(BaseAlpha):
    def get_weights(self, dh):
        self.w = dh.data['volume'] / ts_mean(dh.data['volume'], 5)
        self.w = ts_rank(self.w, 50)
        self.w = decay(self.w, 4)
        self.w = normalize(neutralize(self.w))


class Alpha6(BaseAlpha):
    def get_weights(self, dh):
        self.w = ts_rank(dh.data['open'], 50) / ts_rank(dh.data['close'], 50)
        self.w = decay(self.w, 4)
        self.w = normalize(neutralize(self.w))


class Alpha7(BaseAlpha):
    def get_weights(self, dh):
        self.w = ts_rank(dh.data['high'], 5) / ts_rank(dh.data['close'], 5)
        self.w = decay(self.w, 6)
        self.w = normalize(neutralize(self.w))
        self.w = truncate(self.w, 0.001)
        self.w = normalize(neutralize(self.w))


class Alpha8(BaseAlpha):
    def get_weights(self, dh):
        self.w = 1 - dh.data['close'] / ts_mean(dh.data['close'], 120)
        # self.w = decay(self.w, 2)
        self.w = normalize(neutralize(self.w))


class Alpha9(BaseAlpha):
    def get_weights(self, dh):
        self.w = - stddev(dh.data['high'], 10).rank(axis=1) * ts_correlation(dh.data['high'], dh.data['volume'], 15)
        # self.w = CutOutliers(self.w, 10)
        self.w = normalize(neutralize(self.w))