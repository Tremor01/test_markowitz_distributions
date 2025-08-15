from functools import partial

import numpy as np
import cvxpy as cp

from enum import StrEnum

import pandas as pd
from cvxpy import psd_wrap
from pandas import DataFrame, Series

from .utils import (
    calculate_ledoit_wolf_covariance_matrix,
    calculate_returns, calculate_log_returns, \
    calculate_simple_covariance_matrix
)


class RiskType(StrEnum):
    STD = "std"
    DRAWDOWN = "drawdown"
    LEDOIT_WOLF = "ledoit_wolf"

def get_expected_median_returns(prices: DataFrame) -> pd.Series:
    log_returns = calculate_log_returns(prices)
    expected_returns = log_returns.median() * 10
    return expected_returns

def get_expected_returns(prices: DataFrame):
    returns = calculate_returns(prices)
    expected_returns = returns.mean() * 365
    return expected_returns


def get_std_risk(data: DataFrame, weights: cp.Variable):
    log_returns = calculate_log_returns(data)
    risk = calculate_simple_covariance_matrix(log_returns)
    goal_func = weights.T @ risk @ weights
    return goal_func


def get_ledoit_wolf_risk(data: DataFrame, weights: cp.Variable, psd: bool = False):
    returns = calculate_returns(data)
    risk = calculate_ledoit_wolf_covariance_matrix(returns)

    if psd: risk = cp.psd_wrap(risk)

    goal_func = cp.quad_form(weights, risk)
    return goal_func


RISK_FUNC = {
    RiskType.STD: get_std_risk,
    RiskType.LEDOIT_WOLF: get_ledoit_wolf_risk
}


def min_risk(prices: DataFrame, min_weights: float = 0.0) -> dict[str, float]:
    weights = optimize(prices, partial(minimize_risk, min_weights=min_weights))
    if weights is None:
        weights = optimize(prices, partial(minimize_risk, min_weights=min_weights, psd=True))
    return weights


def max_profit(prices: DataFrame, min_weights: float = 0.0) -> dict[str, float]:
    weights = optimize(prices, partial(maximize_profit, min_weights=min_weights))
    if weights is None:
        weights = optimize(prices, partial(maximize_profit, min_weights=0))
    return weights


def sharp(prices: DataFrame, min_weights: float = 0.0, risk_free_rate: float = 0.0, short: bool = False) -> dict[str, float]:
    weights = optimize(prices, partial(sharp_ratio, min_weights=min_weights, is_short=short, risk_free_rate=risk_free_rate))
    if weights is None:
        weights = optimize(prices, partial(sharp_ratio, min_weights=min_weights, is_short=short, risk_free_rate=risk_free_rate, psd=True))
    if weights is None:
        weights = optimize(prices, partial(sharp_ratio, min_weights=0, is_short=short, risk_free_rate=risk_free_rate))
    if weights is None:
        weights = optimize(prices, partial(sharp_ratio, min_weights=0, is_short=short, risk_free_rate=risk_free_rate, psd=True))
    return weights

def alpha_sharp(prices: DataFrame, min_weights: float = 0.0, risk_free_rate: float = 0.0, a: float = 0.5) -> dict[str, float]:
    min_weights_long = min_weights if a == 0 else min_weights / a
    min_weights_short = min_weights if (1 - a) == 0 else min_weights / (1 - a)
    weights_long = optimize(prices, partial(sharp_ratio_only_long, min_weights=min_weights_long, risk_free_rate=risk_free_rate))
    weights_short = optimize(prices, partial(sharp_ratio_only_short, min_weights=min_weights_short, risk_free_rate=risk_free_rate))
    if weights_short is None or weights_long is None:
        return {}
    weights = {}
    for k in weights_long:
        weights[k] = weights_long[k] * a
    for k in weights_short:
        if k not in weights:
            weights[k] = 0
        weights[k] += weights_short[k] * (1 - a)
    return weights

def alpha_sharp_brute_force(prices: DataFrame, min_weights: float = 0.0, risk_free_rate: float = 0.0) -> dict[str, float]:
    min_risk_ = float('inf')
    weights = {}
    returns = calculate_returns(prices)
    risk = calculate_ledoit_wolf_covariance_matrix(returns)
    col = returns.columns
    for a in range(0, 101, 10):
        temp_weights = alpha_sharp(prices, min_weights, risk_free_rate, a)
        if len(temp_weights) == 0:
            continue
        array_weights = [temp_weights[key] for key in col]
        np_array_weights = np.array(array_weights)
        temp_risk = np_array_weights.T @ risk @ np_array_weights
        if temp_risk < min_risk_:
            min_risk_ = temp_risk
            weights = temp_weights
    return  weights




def optimize(prices: DataFrame, type_optimization) -> dict[str, float] | None:
    weights = type_optimization(prices)
    if weights is None: return None

    allocation = {ticker: weight for ticker, weight in zip(prices.keys(), weights)}
    for coin, w in allocation.items():
        try:
            allocation[coin] = w[0]
        except: pass

    return allocation


def minimize_risk(
        data: DataFrame,
        risk_type: RiskType = RiskType.LEDOIT_WOLF,
        min_weights: float = 0.0,
        smearing_coefficient: float = 0.0,
        is_short: bool = False,
        leverage: float = 1.0,
        psd: bool = False
) -> np.ndarray | None:
    expected_returns = get_expected_returns(data)
    num_assets = len(expected_returns)

    weights = cp.Variable(num_assets)
    portfolio_risk = RISK_FUNC[risk_type](data, weights, psd)
    k = cp.Variable()
    objective = cp.Minimize(portfolio_risk + cp.sum_squares(weights) * smearing_coefficient)

    constraints = [
        cp.sum(cp.pos(weights)) + cp.sum(cp.neg(weights)) <= k * leverage,
        k >= 0,
    ]
    if not is_short:
        constraints += [weights >= min_weights, ]

    problem = cp.Problem(objective, constraints)  # type: ignore
    for solver in cp.installed_solvers():
        try:
            problem.solve(solver=solver)
        except Exception as e:
            pass
        if weights.value is not None: break

    if weights.value is None: return None

    weights = np.array(weights.value, ndmin=2).T
    return weights #type: ignore


def maximize_profit(
        data: DataFrame,
        risk_type: RiskType = RiskType.LEDOIT_WOLF,
        min_weights: float = 0.0,
        smearing_coefficient: float = 0.0,
        is_short: bool = False,
        leverage: float = 1.0,
        psd: bool = False
) -> np.ndarray | None:
    expected_returns = get_expected_returns(data)
    num_assets = len(expected_returns)
    expected_returns_np = expected_returns.values
    weights = cp.Variable(num_assets)
    portfolio_return = expected_returns_np @ weights

    k = cp.Variable()
    objective = cp.Minimize(-portfolio_return + cp.sum_squares(weights) * smearing_coefficient)
    constraints = [
        cp.sum(cp.pos(weights)) + cp.sum(cp.neg(weights)) <= k * leverage,
        k >= 0,
    ]

    if not is_short:
        constraints += [weights >= min_weights, k == 1,]

    problem = cp.Problem(objective, constraints)  # type: ignore
    for solver in cp.installed_solvers():
        try:
            problem.solve(solver=solver)
            if weights.value: break
        except Exception as e:
            pass

    if weights.value is None: return None

    weights = np.array(weights.value, ndmin=2).T
    return weights #type: ignore


def sharp_ratio(
        data: DataFrame,
        risk_type: RiskType = RiskType.LEDOIT_WOLF,
        min_weights: float = 0.0,
        risk_free_rate: float = 0.0,
        smearing_coefficient: float = 0.0,
        is_short: bool = False,
        leverage: float = 1.0,
        psd: bool = False
) -> np.ndarray | None:
    expected_returns = get_expected_median_returns(data)
    num_assets = len(expected_returns)
    expected_returns_np = expected_returns.values
    weights = cp.Variable(num_assets)
    portfolio_risk = RISK_FUNC[risk_type](data, weights, psd)

    k = cp.Variable((1, 1))

    objective = cp.Minimize(portfolio_risk + cp.sum_squares(weights) * smearing_coefficient)
    constraints = [
        expected_returns_np @ weights - risk_free_rate * k == 1,  # type: ignore
        cp.sum(cp.pos(weights)) + cp.sum(cp.neg(weights)) <= k * leverage,
        cp.abs(weights) <= k * leverage,
        k >= 0,
    ]
    if not is_short:
        constraints += [weights >= min_weights * k]
    # else:
    #     constraints += [cp.abs(weights) >= min_weights * k]

    problem = cp.Problem(objective, constraints)
    for solver in cp.installed_solvers():
        try:
            problem.solve(solver=solver)
            if weights.value: break
        except Exception as e:
            pass

    if weights.value is None: return None

    weights = np.array(weights.value / k.value, ndmin=2).T  # type: ignore
    return weights # type: ignore

def sharp_ratio_only_long(
        data: DataFrame,
        risk_type: RiskType = RiskType.LEDOIT_WOLF,
        min_weights: float = 0.0,
        risk_free_rate: float = 0.0,
        leverage: float = 1.0,
        psd: bool = False
) -> np.ndarray | None:
    expected_returns = get_expected_returns(data)
    cols = data.columns
    for col in cols:
        i = data.columns.get_loc(col)
        if expected_returns[i] < 0:
            data.drop(col, axis=1)
    expected_returns = get_expected_returns(data)
    num_assets = len(expected_returns)
    expected_returns_np = expected_returns.values
    weights = cp.Variable(num_assets)
    portfolio_risk = RISK_FUNC[risk_type](data, weights, psd)
    k = cp.Variable((1, 1))
    objective = cp.Minimize(cp.abs(portfolio_risk))
    constraints = [
        expected_returns_np @ weights - risk_free_rate * k == 1,  # type: ignore
        cp.sum(weights) <= k * leverage,
        k >= 0,
        weights >= min_weights*k,
    ]
    problem = cp.Problem(objective, constraints)
    for solver in cp.installed_solvers():
        try:
            problem.solve(solver=solver)
            if weights.value: break
        except Exception as e:
            pass
    if weights.value is None: return None
    weights = np.array(weights.value / k.value, ndmin=2).T  # type: ignore
    return weights # type: ignore

def sharp_ratio_only_short(
        data: DataFrame,
        risk_type: RiskType = RiskType.LEDOIT_WOLF,
        min_weights: float = 0.0,
        risk_free_rate: float = 0.0,
        leverage: float = 1.0,
        psd: bool = False
) -> np.ndarray | None:
    expected_returns = get_expected_returns(data)
    cols = data.columns
    for col in cols:
        i = data.columns.get_loc(col)
        if expected_returns[i] > 0:
            data.drop(col, axis=1)
    expected_returns = get_expected_returns(data)
    num_assets = len(expected_returns.values)
    expected_returns_np = expected_returns.values
    weights = cp.Variable(num_assets)
    portfolio_risk = RISK_FUNC[risk_type](data, weights, psd)
    k = cp.Variable((1, 1))
    objective = cp.Minimize(portfolio_risk)
    constraints = [
        expected_returns_np @ weights - risk_free_rate * k == 1,  # type: ignore
        cp.sum(weights) >= -k * leverage,
        k >= 0,
        weights <= -min_weights*k,
    ]
    problem = cp.Problem(objective, constraints)
    for solver in cp.installed_solvers():
        try:
            problem.solve(solver=solver)
            if weights.value: break
        except Exception as e:
            pass
    if weights.value is None: return None
    weights = np.array(weights.value / k.value, ndmin=2).T  # type: ignore
    return weights # type: ignore