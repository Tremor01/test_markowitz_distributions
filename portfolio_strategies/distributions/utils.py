import numpy as np
from cvxpy import Variable, psd_wrap, quad_form
from pandas import DataFrame, Series
from sklearn.covariance import LedoitWolf


def get_ledoit_wolf_risk(data: DataFrame, weights: Variable):
    returns = calculate_pct_returns(data)
    risk = calculate_ledoit_wolf_covariance_matrix(returns)
    goal_func = quad_form(weights, risk)
    return goal_func

def get_corr_risk(data: DataFrame, weights: Variable):
    returns = calculate_pct_returns(data)
    risk = calculate_correlation_matrix(returns)
    return weights.T @ risk @ weights

def get_std_risk(data: DataFrame, weights: Variable):
    returns = calculate_log_returns(data)
    risk = calculate_simple_covariance_matrix(returns)
    return weights.T @ risk @ weights


def calculate_simple_covariance_matrix(returns: DataFrame) -> np.ndarray:
    return (returns.cov() * 365).values

def calculate_correlation_matrix(returns: DataFrame) -> np.ndarray:
    return returns.corr().values

def calculate_ledoit_wolf_covariance_matrix(returns: DataFrame) -> np.ndarray:
    lw = LedoitWolf()
    lw.fit(returns)
    cov_matrix = lw.covariance_ * 365
    return cov_matrix


def calculate_log_returns(data: DataFrame) -> DataFrame:
    return np.log(data / data.shift(1)).dropna()

def calculate_pct_returns(data: DataFrame) -> DataFrame:
    return data.pct_change().dropna()

def get_ema_returns(prices: DataFrame, span: float = 10) -> Series:
    returns = calculate_pct_returns(prices)
    expected_returns = returns.ewm(span=span).mean().iloc[-1]
    return expected_returns

def get_expected_median_returns(prices: DataFrame) -> Series:
    returns = calculate_log_returns(prices)
    expected_returns = returns.median() * 365
    return expected_returns

def get_expected_returns(prices: DataFrame):
    returns = calculate_pct_returns(prices)
    expected_returns = returns.mean() * 365
    return expected_returns

def calculate_exp_returns(data: DataFrame) -> Series:
    returns = calculate_pct_returns(data)
    weights = _get_weights(len(returns))
    expected_returns = (returns.T @ weights) * 365
    return Series(expected_returns, index=data.columns, name='Expected Return')

def _get_weights(n: int, halflife: int = 2) -> np.ndarray:
    decay = 0.5 ** (1 / halflife)
    weights = decay ** np.arange(n)[::-1]
    weights = weights / weights.sum()
    return weights