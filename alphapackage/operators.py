import numpy as np
import pandas as pd

def neutralize(df : pd.DataFrame) -> pd.DataFrame:
    """
        Neutralizes dataframe by row (sum of elements in row equals 0)
    """
    return df.sub(df.mean(axis=1), axis=0)


def normalize(df : pd.DataFrame) -> pd.DataFrame:
    """
        Normalizes dataframe by row (sum of abs(elements) in row equals 1)
    """
    return df.div(df.abs().sum(axis=1), axis=0)


def decay(df : pd.DataFrame, n : int) -> pd.DataFrame:
    """
        Decays dataframe by replacing element in a row with linear combination of previous n elements in a column using the following formula:
                a(i,j) = c1 * a(i,j) + c2 * a(i-1,j) + ... + cn * a(i-n+1,j)
                
                c_{i} = 1 - (i-1)/n
    """
    weights = np.arange(1, n + 1) / n
    
    result = df.rolling(window=n, min_periods=1).apply(
        lambda x: np.dot(x, weights[-len(x):]), 
        raw=True
    )
    
    return normalize(result)


def truncate(df : pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
        Returns truncted dataframe. Steps, which are followed:
            - clipping every value using argument "threshold"
            - normalize and neutralize positive value by row
            - normalize and neutralize negative value by row
    """
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


def CutOutliers(df : pd.DataFrame, n : int) -> pd.DataFrame:
    """
        Returns dataframe, where N outlier values are nullyfied 
    """
    ranks = df.rank(axis=1)
    columns_num = df.shape[1]
    mask = (ranks <= n) | (ranks > columns_num - n)
    return df.where(~mask, 0)


def CutMiddle(df : pd.DataFrame, n : int) -> pd.DataFrame:
    """
        Returns dataframe, where N middle values are nullyfied 
    """
    columns_num = df.shape[1]
    if n >= columns_num:
        return df * 0
    
    ranks = df.rank(axis=1, method='first')
    low = (columns_num - n) // 2 + 1
    high = low + n - 1
    mask = (ranks >= low) & (ranks <= high)
    return df.where(~mask, 0)


def ts_rank(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """
        Returns time-series rank of dataframe values within rolling window
    """
    return df.rolling(win).rank() / win


def ts_scale(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """
        Returns time-series scaled dataframe values within rolling window (min-max normalization)
    """
    return (df - df.rolling(win).min()) / (df.rolling(win).max() - df.rolling(win).min())


def ts_zscore(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """
        Returns time-series z-score of dataframe values within rolling window
    """
    return (df - df.rolling(win).mean()) / df.rolling(win).std()


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
        Returns cross-sectional rank of dataframe values by row (normalized to [0, 1])
    """
    _df = df.rank(axis=1)
    return _df.div(_df.max(axis=1), axis=0)


def cs_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
        Returns cross-sectional scaled dataframe values by row (min-max normalization)
    """
    nom = df.sub(df.min(axis=1), axis=0)
    denom = df.max(axis=1) - df.min(axis=1)
    return nom.div(denom, axis=0)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
        Returns cross-sectional z-score of dataframe values by row
    """
    nom = df.sub(df.mean(axis=1), axis=0)
    denom = df.std(axis=1)
    return nom.div(denom, axis=0)


def ts_mean(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """
        Returns time-series rolling mean of dataframe values within window
    """
    return df.rolling(win).mean()


def ts_correlation(df1: pd.DataFrame, df2: pd.DataFrame, win: int) -> pd.DataFrame:
    """
        Returns time-series rolling correlation between two dataframes for the past win rows
    """
    return df1.rolling(win).corr(df2)


def ts_covariance(df1: pd.DataFrame, df2: pd.DataFrame, win: int) -> pd.DataFrame:
    """
        Returns time-series rolling covariance between two dataframes for the past win rows
    """
    return df1.rolling(win).cov(df2)


def ts_max(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """
        Returns time-series rolling max of dataframe values within window
    """
    return df.rolling(win).max()


def ts_min(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """
        Returns time-series rolling min of dataframe values within window
    """
    return df.rolling(win).min()