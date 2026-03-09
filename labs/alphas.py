from typing import Dict

import pandas as pd
import numpy as np
import cvxpy as cp

from alphapackage.operators import *
from alphapackage.data_utils import DataHolder
from alphapackage.core import BaseAlpha, BasePortfolio


# Heuristic portfolios
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


# Markowitz potfolios
# EPS_REG = 1e-8

def _build_R_C(alphas: Dict[str, BaseAlpha], date: pd.Timestamp, dh: 'DataHolder',
               lookback_months: int = 12, use_full_history: bool = False):
    """
        Returns (R, C, alpha_names) inside time-series window (start_date, date)
            - R - vector of expected returns
            - C - alpha correlation matrix
            - alpha_names - list with alpha names

        Default time-series window = 12 months
        Optional = full history
    """
    alpha_names = list(alphas.keys())
    first_date = dh.data['close'].index[0]

    if use_full_history:
        start_date = first_date
    else:
        start_date = date - pd.DateOffset(months=lookback_months)
        if start_date < first_date:
            start_date = first_date

    pnl_list = []
    for name in alpha_names:
        pnl = alphas[name].pnl[(alphas[name].pnl.index > start_date) & (alphas[name].pnl.index < date)].rename(name)
        pnl_list.append(pnl)

    if len(pnl_list) == 0:
        return None, None, alpha_names

    df = pd.concat(pnl_list, axis=1)
    if df.shape[0] == 0:
        return None, None, alpha_names

    R = df.mean(axis=0).to_numpy()        # vector (k,)
    C = df.corr().fillna(0.0).to_numpy()  # correlation matrix (k,k)
    # C = C + EPS_REG * np.eye(C.shape[0])
    return R, C, alpha_names


class MarkowitzUnconstrained(BasePortfolio):
    def __init__(self, alphas: Dict[str, BaseAlpha], freq: str = "QS",
                 lookback_months: int = 12, use_full_history: bool = False):
        super().__init__(alphas=alphas, frequency=freq)
        self.lookback_months = int(lookback_months)
        self.use_full_history = bool(use_full_history)

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        names = list(self.alphas.keys())
        k = len(names)

        # first rebalance -> equal
        if self.rebalance_dates is None:
            return pd.Series(1.0 / k, index=names)
        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / k, index=names)

        R, C, alpha_names = _build_R_C(self.alphas, date, dh, self.lookback_months, self.use_full_history)
        if R is None or C is None:
            return pd.Series(1.0 / k, index=names)

        x = cp.Variable(k)
        objective = cp.Minimize(cp.quad_form(x, C))
        constraints = [R.T @ x == 1]
        prob = cp.Problem(objective, constraints)
        # prob.solve(solver=cp.SCS, verbose=False)
        prob.solve()
        if x.value is None:
            return pd.Series(1.0 / k, index=names)
        return pd.Series(np.array(x.value).reshape(-1), index=alpha_names)


class MarkowitzNonNegative(BasePortfolio):
    def __init__(self, alphas: Dict[str, BaseAlpha], freq: str = "QS",
                 lookback_months: int = 12, use_full_history: bool = False):
        super().__init__(alphas=alphas, frequency=freq)
        self.lookback_months = int(lookback_months)
        self.use_full_history = bool(use_full_history)

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        names = list(self.alphas.keys())
        k = len(names)

        if self.rebalance_dates is None:
            return pd.Series(1.0 / k, index=names)
        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / k, index=names)

        R, C, alpha_names = _build_R_C(self.alphas, date, dh, self.lookback_months, self.use_full_history)
        if R is None or C is None:
            return pd.Series(1.0 / k, index=names)

        x = cp.Variable(k)
        objective = cp.Minimize(cp.quad_form(x, C))
        constraints = [R.T @ x == 1, x >= 0]
        prob = cp.Problem(objective, constraints)
        # prob.solve(solver=cp.SCS, verbose=False)
        prob.solve()
        if x.value is None:
            return pd.Series(1.0 / k, index=names)
        return pd.Series(np.array(x.value).reshape(-1), index=alpha_names)


class MarkowitzMinLower(BasePortfolio):
    def __init__(self, alphas: Dict[str, BaseAlpha], freq: str = "QS",
                 lookback_months: int = 12, use_full_history: bool = False):
        super().__init__(alphas=alphas, frequency=freq)
        self.lookback_months = int(lookback_months)
        self.use_full_history = bool(use_full_history)

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        names = list(self.alphas.keys())
        k = len(names)

        if self.rebalance_dates is None:
            return pd.Series(1.0 / k, index=names)
        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / k, index=names)

        R, C, alpha_names = _build_R_C(self.alphas, date, dh, lookback_months=self.lookback_months,
                                       use_full_history=self.use_full_history)
        if R is None or C is None:
            return pd.Series(1.0 / k, index=names)

        # build matrix U
        v_base = np.full(k, 1.0 / (2.0 * k), dtype=float)
        V = np.zeros((k, k), dtype=float)  # rows == v_i
        for i in range(k):
            vi = v_base.copy()
            vi[i] = (k + 1.0) / (2.0 * k)
            V[i, :] = vi
            
        denom = V @ R  # denom[i] = v_i^T R
        if np.any(np.isclose(denom, 0.0, atol=1e-12)):
            return pd.Series(1.0 / k, index=names)

        U = (V.T / denom).T

        try:
            invU = np.linalg.inv(U)
        except np.linalg.LinAlgError:
            return pd.Series(1.0 / k, index=names)

        # OPTIMIZATION
        # invU * e_j == col j of invU
        # invU_cols = invU  # invU[:, j] is vector for constraint j

        x = cp.Variable(k)
        objective = cp.Minimize(cp.quad_form(x, C))

        constraints = [R.T @ x == 1]
        for j in range(k):
            vec = invU[:, j]
            vec = vec.astype(float)
            constraints.append(vec @ x >= 0)

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if x.value is None:
            return pd.Series(1.0 / k, index=names)

        return pd.Series(np.array(x.value).reshape(-1), index=alpha_names)


# Risk portfolios
EPS_REG = 1e-8

class PortfolioRiskBudgetEqual(BasePortfolio):
    def __init__(self, alphas: Dict[str, BaseAlpha],
                 freq: str = "QS",
                 lookback_months: int = 12,
                 use_full_history: bool = False):
        super().__init__(alphas=alphas, frequency=freq)
        self.lookback_months = int(lookback_months)
        self.use_full_history = bool(use_full_history)
        self._solver = cp.SCS

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:

        names = list(self.alphas.keys())
        k = len(names)

        if self.rebalance_dates is None:
            return pd.Series(1.0 / k, index=names)

        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / k, index=names)

        _, C, alpha_names = _build_R_C(
            self.alphas,
            date,
            dh,
            lookback_months=self.lookback_months,
            use_full_history=self.use_full_history
        )

        if C is None:
            return pd.Series(1.0 / k, index=names)

        C = C + EPS_REG * np.eye(k)
        b = np.full(k, 1.0 / k)
        x = cp.Variable(k)

        lb = 1e-12
        constraints = [x >= lb, cp.sum(x) == 1]

        objective = cp.Minimize(
            0.5 * cp.quad_form(x, C)
            - cp.sum(cp.multiply(b, cp.log(x)))
        )

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=self._solver, verbose=False)
        except Exception:
            return pd.Series(1.0 / k, index=names)

        if x.value is None or np.any(np.isnan(x.value)):
            return pd.Series(1.0 / k, index=names)

        return pd.Series(np.array(x.value).reshape(-1), index=alpha_names)


class PortfolioRiskBudgetSharpe(BasePortfolio):
    def __init__(self, alphas, freq: str = "QS", lookback_months: int = 3, use_full_history: bool = False):
        super().__init__(alphas=alphas, frequency=freq)
        self.lookback_months = int(lookback_months)
        self.use_full_history = bool(use_full_history)
        self._solver = cp.SCS

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        names = list(self.alphas.keys()); k = len(names)
        if self.rebalance_dates is None:
            return pd.Series(1.0 / k, index=names)
        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / k, index=names)

        # C using last 3 months window
        R, C, alpha_names = _build_R_C(self.alphas, date, dh, lookback_months=self.lookback_months, use_full_history=False)
        if C is None:
            return pd.Series(1.0 / k, index=names)
        C = C + EPS_REG * np.eye(k)

        first_date = dh.data['close'].index[0]
        start_date = date - pd.DateOffset(months=self.lookback_months)
        if start_date < first_date:
            start_date = first_date

        sharpe_vals = []
        for name in alpha_names:
            pnl = self.alphas[name].pnl
            s = pnl[(pnl.index > start_date) & (pnl.index < date)].dropna()
            if s.size == 0 or s.std() == 0:
                sharpe_vals.append(0.0)
            else:
                sharpe_vals.append((s.mean() / s.std()))
        sharpe_arr = np.array(sharpe_vals, dtype=float)

        b_raw = np.clip(sharpe_arr, a_min=0.0, a_max=None)
        if b_raw.sum() == 0:
            return pd.Series(1.0 / k, index=names)
        b = b_raw / b_raw.sum()

        x = cp.Variable(k)
        lb = 1e-12
        constraints = [x >= lb, cp.sum(x) == 1]
        objective = cp.Minimize(0.5 * cp.quad_form(x, C) - cp.sum(cp.multiply(b, cp.log(x))))
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=self._solver, verbose=False)
        except Exception:
            return pd.Series(1.0 / k, index=names)

        if x.value is None or np.any(np.isnan(x.value)):
            return pd.Series(1.0 / k, index=names)
        xval = np.maximum(np.array(x.value).reshape(-1), lb)
        return pd.Series(xval, index=alpha_names)


class PortfolioRiskBudgetPNL(BasePortfolio):
    def __init__(self, alphas, freq: str = "QS", lookback_months: int = 3, use_full_history: bool = False):
        super().__init__(alphas=alphas, frequency=freq)
        self.lookback_months = int(lookback_months)
        self.use_full_history = bool(use_full_history)
        self._solver = cp.SCS

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        names = list(self.alphas.keys()); k = len(names)
        if self.rebalance_dates is None:
            return pd.Series(1.0 / k, index=names)
        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / k, index=names)

        R, C, alpha_names = _build_R_C(self.alphas, date, dh, lookback_months=self.lookback_months, use_full_history=False)
        if C is None:
            return pd.Series(1.0 / k, index=names)
        C = C + EPS_REG * np.eye(k)

        first_date = dh.data['close'].index[0]
        start_date = date - pd.DateOffset(months=self.lookback_months)
        if start_date < first_date:
            start_date = first_date

        total_vals = []
        for name in alpha_names:
            pnl = self.alphas[name].pnl
            s = pnl[(pnl.index > start_date) & (pnl.index < date)].dropna()
            total_vals.append(s.sum() if s.size > 0 else 0.0)
        total_arr = np.array(total_vals, dtype=float)

        b_raw = np.clip(total_arr, a_min=0.0, a_max=None)
        if b_raw.sum() == 0:
            return pd.Series(1.0 / k, index=names)
        b = b_raw / b_raw.sum()

        x = cp.Variable(k)
        lb = 1e-12
        constraints = [x >= lb, cp.sum(x) == 1]
        objective = cp.Minimize(0.5 * cp.quad_form(x, C) - cp.sum(cp.multiply(b, cp.log(x))))
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=self._solver, verbose=False)
        except Exception:
            return pd.Series(1.0 / k, index=names)

        if x.value is None or np.any(np.isnan(x.value)):
            return pd.Series(1.0 / k, index=names)
        xval = np.maximum(np.array(x.value).reshape(-1), lb)
        return pd.Series(xval, index=alpha_names)


class PortfolioRiskBudgetStd(BasePortfolio):
    def __init__(self, alphas, freq: str = "QS", lookback_months: int = 3, use_full_history: bool = False):
        super().__init__(alphas=alphas, frequency=freq)
        self.lookback_months = int(lookback_months)
        self.use_full_history = bool(use_full_history)
        self._solver = cp.SCS

    def generate_alpha_weights(self, date: pd.Timestamp, dh: 'DataHolder') -> pd.Series:
        names = list(self.alphas.keys()); k = len(names)
        if self.rebalance_dates is None:
            return pd.Series(1.0 / k, index=names)
        pos = int(self.rebalance_dates.get_loc(date))
        if pos == 0:
            return pd.Series(1.0 / k, index=names)

        R, C, alpha_names = _build_R_C(self.alphas, date, dh, lookback_months=self.lookback_months, use_full_history=False)
        if C is None:
            return pd.Series(1.0 / k, index=names)
        C = C + EPS_REG * np.eye(k)

        first_date = dh.data['close'].index[0]
        start_date = date - pd.DateOffset(months=self.lookback_months)
        if start_date < first_date:
            start_date = first_date

        std_vals = []
        for name in alpha_names:
            pnl = self.alphas[name].pnl
            s = pnl[(pnl.index > start_date) & (pnl.index < date)].dropna()
            std_vals.append(s.std() if s.size > 0 else 0.0)
        std_arr = np.array(std_vals, dtype=float)

        b_raw = np.clip(std_arr, a_min=0.0, a_max=None)  
        if b_raw.sum() == 0:
            return pd.Series(1.0 / k, index=names)
        b = b_raw / b_raw.sum()

        x = cp.Variable(k)
        lb = 1e-12
        constraints = [x >= lb, cp.sum(x) == 1]
        objective = cp.Minimize(0.5 * cp.quad_form(x, C) - cp.sum(cp.multiply(b, cp.log(x))))
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=self._solver, verbose=False)
        except Exception:
            return pd.Series(1.0 / k, index=names)

        if x.value is None or np.any(np.isnan(x.value)):
            return pd.Series(1.0 / k, index=names)
        xval = np.maximum(np.array(x.value).reshape(-1), lb)
        return pd.Series(xval, index=alpha_names)


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