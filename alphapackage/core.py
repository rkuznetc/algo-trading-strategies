from abc import ABC, abstractmethod
from typing import Literal, Union, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .data_utils import DataHolder
from .operators import *


class BaseAlpha(ABC):
    """
        Abstract class for alpha strategies

        Args:
            - dh: object DataHolder, which keep OHLCVR data (OHLCV + returns)

        Methods:
            - get_weights: executes formula and apply operations
            - run_weights: calculates statistics (pnl, turndown, cumpnl)
    """
    def __init__(self):
        self.w : pd.DataFrame = None
        self.pnl : pd.Series = None
        self.turnover : pd.Series = None
        self.cumpnl : pd.Series = None

    @abstractmethod
    def get_weights(self, dh : DataHolder):
        pass

    def run_weights(self, dh : DataHolder):
        self.pnl = (self.w.shift(1) * dh.data['return']).sum(axis=1)
        self.turnover = self.w.diff().abs().sum(axis=1)
        self.cumpnl = self.pnl.cumsum()

    def reset(self):
        self.w = None
        self.pnl = None
        self.turnover = None
        self.cumpnl = None


class BasePortfolio(BaseAlpha, ABC):

    def __init__(self, alphas: dict[str, BaseAlpha], frequency: str):
        super().__init__()
        self.alphas = alphas
        self.n = len(alphas)
        self.freq = frequency

        self.alpha_weights_rebalance = None
        self.alpha_weights = None
        self.rebalance_dates = None
        self.w = None

    @abstractmethod
    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        pass

    def _compute_rebalance_dates(self, all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
            frequency: 'QS', 'Q', 'MS', 'W', 'Y', etc.
        """

        calendar_points = pd.date_range(
            start=all_dates[0],
            end=all_dates[-1],
            freq=self.freq
        )

        rebalance_dates = pd.DatetimeIndex(
            [all_dates.asof(d) for d in calendar_points]
        ).dropna().unique()

        if rebalance_dates[0] != all_dates[0]:
            rebalance_dates = pd.DatetimeIndex([all_dates[0]]).append(rebalance_dates).unique()

        return rebalance_dates

    def get_weights(self, dh: 'DataHolder'):

        all_dates = dh.data['close'].index

        # ensure alphas ready
        for alpha in self.alphas.values():
            if alpha.w is None:
                alpha.get_weights(dh)
            if alpha.pnl is None:
                alpha.run_weights(dh)

        # rebalance dates
        reb_dates = self._compute_rebalance_dates(all_dates)
        self.rebalance_dates = reb_dates

        alpha_names = list(self.alphas.keys())
        reb_df = pd.DataFrame(index=reb_dates, columns=alpha_names, dtype=float)

        # compute weights only on rebalance dates
        for d in reb_dates:
            weights = self.generate_alpha_weights(d, dh)
            reb_df.loc[d] = weights.reindex(alpha_names).fillna(0.0)

        self.alpha_weights_rebalance = reb_df

        # forward-fill to daily
        weights_daily = reb_df.reindex(all_dates, method='ffill').fillna(0.0)
        self.alpha_weights = weights_daily

        # aggregate positions
        all_tickers = sorted({
            t for alpha in self.alphas.values()
            for t in alpha.w.columns
        })

        portfolio_positions = pd.DataFrame(
            0.0, index=all_dates, columns=all_tickers
        )

        for name, alpha in self.alphas.items():
            alpha_pos = alpha.w.reindex(index=all_dates, columns=all_tickers).fillna(0.0)
            w_series = weights_daily[name]
            portfolio_positions += alpha_pos.mul(w_series, axis=0)


        portfolio_positions = neutralize(portfolio_positions)
        portfolio_positions = normalize(portfolio_positions)
        self.w = portfolio_positions


class AlphaStats:
    """
        Args:
            - alpha (dataframe)

        Methods:
            - ...
    """
    
    def __init__(self, alpha : BaseAlpha):
        self.alpha = alpha

    def sharpes(self) -> pd.Series:
        sharpes = self.alpha.pnl.groupby(self.alpha.pnl.index.year).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252)
        )
        return sharpes

    def drawdown(self, type: Literal['all', 'max']) -> float | pd.Series:
        if type not in ('all', 'max'): raise TypeError('Type must be all/max')

        drawdown = self.alpha.cumpnl.cummax() - self.alpha.cumpnl
        return drawdown if type == 'all' else drawdown.max()

    def turnover(self, type: Literal['all', 'avg']) -> float | pd.Series:
        if type not in ('all', 'avg'): raise TypeError('Type must be all/avg')

        turnover = self.alpha.w.diff().abs().sum(axis=1)
        return turnover if type == 'all' else turnover.mean()

    def corr_with(self, other : BaseAlpha) -> float:
        return self.alpha.pnl.corr(other.pnl)
    
    def plot_cumpnl(self):
        self.alpha.cumpnl.plot(figsize=(12, 8))
        plt.title(f"Суммарная доходность для {self.alpha.__class__.__name__}")
        plt.xlabel("Дата")
        plt.ylabel("CumPNL")
        plt.grid(True)
        plt.show()


    def yearly_stats(self) -> dict:
        all_years = self.alpha.pnl.index.year.unique()
        yearly_stats = {}

        for year in all_years:
            pnl_yr : pd.Series = self.alpha.pnl.loc[str(year)]

            cumpnl_yr : pd.Series = self.alpha.cumpnl.loc[str(year)]
            sharpe_yr : float = self.alpha.pnl.loc[str(year)].mean() / self.alpha.pnl.loc[str(year)].std() * np.sqrt(252)
            drawdown_yr : float = (cumpnl_yr.cummax() - cumpnl_yr).max()
            turnover_yr : float = self.alpha.turnover.loc[str(year)].mean()
            
            yearly_stats[year] = {
                'drawdown' : drawdown_yr,
                'sharpe': sharpe_yr,
                'turnover': turnover_yr,
                'sum_pnl': pnl_yr.sum()
            }
        return yearly_stats
    
    # TODO:
    def avg_stats(self) -> dict:
        turnover_av : float = self.alpha.turnover.mean()

        all_years = self.alpha.pnl.index.year.unique()
        sharpes = np.array([])
        for year in all_years:
            sharpes = np.append(sharpes, self.alpha.pnl.loc[str(year)].mean() / self.alpha.pnl.loc[str(year)].std() * np.sqrt(252))

        return {
            'average_turnover' : turnover_av,
            'average_Sharpe' : sharpes.mean(),
            'sum_pnl' : self.alpha.cumpnl[-1]
        }
    
    # TODO: turn into summary()
    def print_yearly_stats(self) -> None:
        yearly_stats = self.yearly_stats()
        for year, stats in yearly_stats.items():
            print(f"Максимальная просадка за {year} год: {stats['drawdown']}")
            print(f"Коэффициент Шарпа за {year} год: {stats['sharpe']}")
            print(f"Средний оборот за {year} год: {stats['turnover']}")
            print(f"Суммарная доходность за {year} год: {stats['sum_pnl']}")
            print('-' * 30)
    def print_average_stats(self) -> None:
        stats = self.avg_stats()
        print('*' * 40)
        for stat, value in stats.items():
            print(f"{stat} = {value}")
        print('*' * 40)
