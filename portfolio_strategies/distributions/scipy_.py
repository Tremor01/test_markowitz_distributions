from functools import partial
import numpy as np
from pandas import DataFrame, Series
from scipy.optimize import minimize
from .utils import calculate_simple_covariance_matrix, calculate_log_returns


def min_risk(
        prices: DataFrame,
        min_weights: float = 0.0,
        target: float | None = None,
) -> dict[str, float]:
    try:
        if target is None:
            target = get_minimum_risk(prices, min_weights)
        weights = optimize(prices, target, partial(maximize_profit, min_weights=min_weights))
    except:
        if target is None:
            target = get_minimum_risk(prices, min_weights=0.0)
        weights = optimize(prices, target, partial(maximize_profit, min_weights=0.0))
    return weights


def max_profit(
        prices: DataFrame,
        min_weights: float = 0.0,
        target: float | None = None
) -> dict[str, float]:
    try:
        if target is None:
            target = get_maximum_risk(prices, min_weights) * 0.7
        weights = optimize(prices, target, partial(maximize_profit, min_weights=min_weights))
    except:
        if target is None:
            target = get_maximum_risk(prices, min_weights=0.0) * 0.7
        weights = optimize(prices, target, partial(maximize_profit, min_weights=0.0))
    return weights


def sharp(
        prices: DataFrame,
        min_weights: float = 0.0,
        risk_free_rate: float = 0.0
) -> dict[str, float]:
    try:
        weights = optimize(prices, risk_free_rate, partial(sharp_ratio, min_weights=min_weights))
    except:
        weights = optimize(prices, risk_free_rate, partial(sharp_ratio, min_weights=0))
    return weights


def optimize(prices: DataFrame, target: float, type_optimization) -> dict[str, float]:
    log_returns = calculate_log_returns(prices)
    cov_matrix = calculate_simple_covariance_matrix(log_returns)
    expected_returns = log_returns.mean() * 365
    weights = type_optimization(cov_matrix, expected_returns, target)
    allocation = {ticker: weight for ticker, weight in zip(prices.keys(), weights)}
    return allocation


def minimize_risk(
        cov_matrix: DataFrame,
        expected_returns: Series,
        target_return: float,
        smearing_coefficient: float = 0.0,
        min_weights: float = 0.0,
) -> np.ndarray:
    num_assets = len(expected_returns)
    constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: np.dot(weights, expected_returns) - target_return}
        )
    bounds = [(min_weights, 1) for _ in range(num_assets)]
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    def l2_regularisation(weights):
        return (weights ** 2).sum() * smearing_coefficient
    def main_function(weights):
        return portfolio_volatility(weights) + l2_regularisation(weights)
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(main_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def maximize_profit(
        cov_matrix: DataFrame,
        expected_returns: DataFrame,
        target_risk: float,
        smearing_coefficient: float = 0.0,
        min_weights: float = 0.0,
) -> np.ndarray:
    num_assets = len(expected_returns)
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) - target_risk}
    )
    bounds = [(min_weights, 1) for _ in range(num_assets)]
    def negative_return(weights):
        return -np.dot(weights, expected_returns)
    def l2_regularisation(weights):
        return (weights ** 2).sum() * smearing_coefficient
    def main_function(weights):
        return negative_return(weights) + l2_regularisation(weights)
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(main_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def sharp_ratio(
        cov_matrix: DataFrame,
        expected_returns: DataFrame,
        risk_free_rate: float = 0.0,
        smearing_coefficient: float = 0.0,
        min_weights: float = 0.0,
) -> np.ndarray:
    num_assets = len(expected_returns)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(min_weights, 1) for _ in range(num_assets)]
    def negative_sharpe_ratio(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio
    def l2_regularisation(weights):
        return (weights ** 2).sum() * smearing_coefficient
    def main_function(weights):
        return negative_sharpe_ratio(weights) + l2_regularisation(weights)
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(main_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def get_minimum_risk(data: DataFrame, min_weights: float) -> float:
    log_returns = calculate_log_returns(data)
    cov_matrix = calculate_simple_covariance_matrix(log_returns)
    num_assets = len(cov_matrix)
    initial_weights = np.ones(num_assets) / num_assets

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(min_weights, 1) for _ in range(num_assets)]

    result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return portfolio_volatility(result.x)


def get_maximum_risk(data: DataFrame, min_weights: float) -> float:
    log_returns = calculate_log_returns(data)
    cov_matrix = calculate_simple_covariance_matrix(log_returns)
    num_assets = len(cov_matrix)
    initial_weights = np.ones(num_assets) / num_assets

    def portfolio_volatility(weights):
        return -np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(min_weights, 1) for _ in range(num_assets)]

    result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return -portfolio_volatility(result.x)

