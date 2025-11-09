import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def neutralize(matrix : pd.DataFrame) -> pd.DataFrame:
    return matrix.sub(matrix.mean(axis=1), axis=0)


def normalize(matrix : pd.DataFrame) -> pd.DataFrame:
    return matrix.div(matrix.abs().sum(axis=1), axis=0)


def calculate_returns(close_df : pd.DataFrame) -> pd.DataFrame:
    return close_df.pct_change()


def calculate_holding_pnl(alpha : pd.DataFrame, returns : pd.DataFrame) -> pd.Series:
    return (alpha.shift(1) * returns).sum(axis=1)


def calculate_turnover(alpha : pd.DataFrame) -> pd.DataFrame:
    return alpha.diff().abs().sum(axis=1)


def calculate_Sharpe_ratio(pnl_vector : pd.Series, T = 252) -> float:
    mean = pnl_vector.mean()
    std = pnl_vector.std()
    return mean / std * np.sqrt(T)


def calculate_cumpnl(pnl : pd.Series) -> pd.Series:
    return pnl.cumsum()


def calculate_drawdown(cum_pnl : pd.Series) -> float:
    running_max = cum_pnl.cummax()
    drawdown = running_max - cum_pnl
    return drawdown.max()


def truncate(df : pd.DataFrame, threshold) -> pd.DataFrame:
    # truncate
    truncated_df = df.copy()
    mask = truncated_df.abs() > threshold
    truncated_df[mask] = np.sign(truncated_df[mask]) * threshold
    result = truncated_df.copy()

    # norm-neut (positive values)
    pos_mask = truncated_df > 0
    pos_sum = truncated_df.where(pos_mask).sum(axis=1)
    
    pos_denom = pos_sum * 2
    pos_denom = pos_denom.replace(0., np.nan)
    result[pos_mask] = result[pos_mask].div(pos_denom, axis=0)

    # norm-neut (negative values)
    neg_mask = truncated_df < 0
    neg_sum = truncated_df.where(neg_mask).sum(axis=1)
    
    neg_denom = -2 * neg_sum
    neg_denom = neg_denom.replace(0., np.nan)
    result[neg_mask] = result[neg_mask].div(neg_denom, axis=0)

    return result


def rank_df(df : pd.DataFrame) -> pd.DataFrame:
    ranked_df = df.rank(axis=1, method="first") - 1
    denom = ranked_df.count(axis=1).replace(0, np.nan)
    return ranked_df.div(denom, axis=0)


def calculate_volatility(returns : pd.DataFrame, T : int = 60) -> pd.Series: 
    return returns.iloc[returns.shape[0] - T:].std(axis=0)


def CutOutliers(df : pd.DataFrame, n : int) -> pd.DataFrame: 
    ranks = df.rank(axis=1)
    columns_num = df.shape[1]
    mask = (ranks <= n) | (ranks > columns_num - n)
    return df.where(~mask, 0)


def CutMiddle(df : pd.DataFrame, n : int) -> pd.DataFrame:
    columns_num = df.shape[1]
    if n >= columns_num:
        return df * 0
    
    ranks = df.rank(axis=1, method='first')
    low = (columns_num - n) // 2 + 1
    high = low + n - 1
    mask = (ranks >= low) & (ranks <= high)
    return df.where(~mask, 0)


def calculate_alphas_corr(pnl1 : pd.Series, pnl2 : pd.Series) -> float:
    return pnl1.corr(pnl2)


def decay(alpha : pd.DataFrame, n : int) -> pd.DataFrame:
    weights = np.arange(1, n + 1) / n
    
    result = alpha.rolling(window=n, min_periods=1).apply(
        lambda x: np.dot(x, weights[-len(x):]), 
        raw=True
    )
    
    return normalize(result)


class AlphaStats:
    """
        Args:
            - close (dataframe с close)
            - alpha (dataframe с альфами)

        Methods:
            - plot_cumpnl (график доходности за всю историю)

            - print_Sharpe_ratios (коэффициенты Шарпа за каждый год_
            - print_turnovers (средний оборот за каждый год)
            - print_sum_pnls (суммарная доходность за каждый год)
            - print_drawdowns (максимальные просадки за каждый год)
    """
    def __init__(self, close, alpha):
        self.close = close
        self.alpha = alpha
        
        self.returns = calculate_returns(self.close)
        self.holding_pnl = calculate_holding_pnl(self.alpha, self.returns)
        self.cumpnl = calculate_cumpnl(self.holding_pnl)

    def plot_cumpnl(self):
        _, ax = plt.subplots(figsize=(10, 6))
        self.cumpnl.plot(ax=ax)
        
        ax.set_title("График накопленной доходности за всю историю")
        ax.set_xlabel("T")
        ax.set_ylabel("cumpnl")
        ax.grid(True)
        
        return ax

    def print_Sharpe_ratios(self):
        all_years = self.close.index.year.unique()

        for year in all_years:
            sharpe_yr = calculate_Sharpe_ratio(self.holding_pnl.loc[str(year)])
            print(f"Коэффициент Шарпа за {year} год: {sharpe_yr}")
        print("\n")

    def print_turnovers(self):
        all_years = self.close.index.year.unique()

        for year in all_years:
            turnovers_yr = calculate_turnover(self.alpha.loc[str(year)])
            print(f"Средний оборот за {year} год: {turnovers_yr.mean()}")
        print("\n")

    def print_sum_pnls(self):
        all_years = self.close.index.year.unique()

        for year in all_years:
            print(f"Суммарная доходность за {year} год: {self.holding_pnl.loc[str(year)].sum()}")
        print("\n")

    def print_drawdowns(self):
        all_years = self.close.index.year.unique()

        for year in all_years:
            drawdown_yr = calculate_drawdown(self.cumpnl.loc[str(year)])
            print(f"Максимальная просадка за {year} год: {drawdown_yr}")
        print("\n")