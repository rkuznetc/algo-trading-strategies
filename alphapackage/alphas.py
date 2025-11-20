import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
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
            - plot_cumpnl: display cumulative PNL plot
            - get_yearly_stats: returns dict(dict), primary key is year and value is dict with yearly stats
            - print_yearly_stats: pretty print yearly stats 
            - get_average_stats: returns dict, keys are stats name (average_turnover, average_sharpe)
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

    def get_yearly_stats(self) -> dict:
        all_years = self.pnl.index.year.unique()
        yearly_stats = {}

        for year in all_years:
            pnl_yr : pd.Series = self.pnl.loc[str(year)]

            cumpnl_yr : pd.Series = self.cumpnl.loc[str(year)]
            sharpe_yr : float = self.pnl.loc[str(year)].mean() / self.pnl.loc[str(year)].std() * np.sqrt(252)
            drawdown_yr : float = (cumpnl_yr.cummax() - cumpnl_yr).max()
            turnover_yr : float = self.turnover.loc[str(year)].mean()
            
            yearly_stats[year] = {
                'drawdown' : drawdown_yr,
                'sharpe': sharpe_yr,
                'turnover': turnover_yr,
                'sum_pnl': pnl_yr.sum()
            }
        return yearly_stats
    
    def get_average_stats(self) -> dict:
        # drawdown_av : float = (self.cumpnl.cummax() - self.cumpnl).max()
        turnover_av : float = self.turnover.mean()

        all_years = self.pnl.index.year.unique()
        sharpes = np.array([])
        for year in all_years:
            sharpes = np.append(sharpes, self.pnl.loc[str(year)].mean() / self.pnl.loc[str(year)].std() * np.sqrt(252))

        return {
            # 'average_drawdown' : drawdown_av,
            'average_turnover' : turnover_av,
            'average_Sharpe' : sharpes.mean(),
            'sum_pnl' : self.cumpnl[-1]
        }
    
    def plot_cumpnl(self) -> None:
        self.cumpnl.plot(figsize=(12, 8))
        plt.title(f"Суммарная доходность для {self.__class__.__name__}")
        plt.xlabel("Дата")
        plt.ylabel("CumPNL")
        plt.grid(True)
        plt.show()

    def print_yearly_stats(self) -> None:
        yearly_stats = self.get_yearly_stats()
        for year, stats in yearly_stats.items():
            print(f"Максимальная просадка за {year} год: {stats['drawdown']}")
            print(f"Коэффициент Шарпа за {year} год: {stats['sharpe']}")
            print(f"Средний оборот за {year} год: {stats['turnover']}")
            print(f"Суммарная доходность за {year} год: {stats['sum_pnl']}")
            print('-' * 30)

    def print_average_stats(self) -> None:
        stats = self.get_average_stats()
        print('*' * 40)
        for stat, value in stats.items():
            print(f"{stat} = {value}")
        print('*' * 40)

