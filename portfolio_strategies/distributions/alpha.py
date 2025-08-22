from typing import Any

import time
import numpy as np
import cvxpy as cp
from functools import partial

from cvxpy import Variable
from cvxpy.constraints import Inequality
from pandas import DataFrame

from .constants import *
from .utils import (
    calculate_pct_returns, get_ema_returns, calculate_correlation_matrix
)


def alpha_sharp_brute_force(
        prices: DataFrame,
        min_weights: float = 0.0,
        max_weights: dict[str, float] | None = None,
        risk_free_rate: float = 0.0
) -> dict[str, float]:
    max_sharp = -float('inf')
    weights = dict()

    alpha = 0; best_wa = 0

    col = calculate_pct_returns(prices).columns
    returns_pct = calculate_pct_returns(prices)
    risk = calculate_correlation_matrix(returns_pct)
    # for walpha in range(5, 101, 5):
    walpha = 10
    returns = get_ema_returns(prices, alpha=walpha / 100).values
    for a in range(0, 101, 5):
        temp_weights = alpha_sharp(
            prices, min_weights, max_weights, risk_free_rate, a / 100
        )
        if len(temp_weights) == 0:
            continue
        array_weights = [temp_weights[key] for key in col]
        np_array_weights = np.array(array_weights)
        temp_sharp = (returns @ np_array_weights - risk_free_rate) / (np_array_weights.T @ risk @ np_array_weights)
        if temp_sharp > max_sharp:
            max_sharp = temp_sharp
            weights = temp_weights
            alpha = a; best_wa = walpha / 100
    return weights, alpha, best_wa


def optimize(prices: DataFrame, type_optimization) -> dict[str, float] | None:
    weights = type_optimization(prices)
    if weights is None: return None

    allocation = {ticker: weight for ticker, weight in zip(prices.keys(), weights)}
    for coin, w in allocation.items():
        try:
            allocation[coin] = w[0]
        except: pass

    return allocation


def alpha_sharp(
        prices: DataFrame,
        min_weights: float = 0.0,
        max_weights: dict[str, float] | None = None,
        risk_free_rate: float = 0.0,
        a: float = 0.5
) -> dict[str, float]:
    min_weights_long = min_weights if a == 0 else min_weights / a
    min_weights_short = min_weights if (1 - a) == 0 else min_weights / (1 - a)
    weights_long = optimize(prices, partial(sharp_ratio_only_long, min_weights=min_weights_long, max_weights=max_weights, risk_free_rate=risk_free_rate))
    weights_short = optimize(prices, partial(sharp_ratio_only_short, min_weights=min_weights_short, max_weights=max_weights, risk_free_rate=risk_free_rate))
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


def sharp_ratio_only_long(
        data: DataFrame,
        risk_type: RiskType = RiskType.LEDOIT_WOLF,
        returns_type: ReturnsType = ReturnsType.EMA,
        min_weights: float = 0.0,
        max_weights: dict[str, float] | None = None,
        risk_free_rate: float = 0.0,
        leverage: float = 1.0,
) -> np.ndarray | None:
    expected_returns = RETURNS_FUNC[returns_type](data)
    num_assets = len(expected_returns)
    expected_returns_np = expected_returns.values
    weights = cp.Variable(num_assets)
    portfolio_risk = RISK_FUNC[risk_type](data, weights)
    k = cp.Variable((1, 1))
    objective = cp.Minimize(portfolio_risk)
    constraints = [
        expected_returns_np @ weights - risk_free_rate * k == 1,  # type: ignore
        cp.sum(weights) <= k * leverage,
        k >= 0,
        weights >= min_weights*k,
    ]
    if max_weights is not None:
        constraints += get_max_weights_constraints(data, max_weights, weights, k)

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
        returns_type: ReturnsType = ReturnsType.EMA,
        min_weights: float = 0.0,
        max_weights: dict[str, float] | None = None,
        risk_free_rate: float = 0.0,
        leverage: float = 1.0,
        psd: bool = False
) -> np.ndarray | None:
    expected_returns = RETURNS_FUNC[returns_type](data)
    num_assets = len(expected_returns.values)
    expected_returns_np = expected_returns.values
    weights = cp.Variable(num_assets)
    portfolio_risk = RISK_FUNC[risk_type](data, weights)
    k = cp.Variable((1, 1))
    objective = cp.Minimize(portfolio_risk)
    constraints = [
        expected_returns_np @ weights - risk_free_rate * k == 1,  # type: ignore
        cp.sum(weights) >= -k * leverage,
        k >= 0,
        weights <= -min_weights*k,
    ]
    if max_weights is not None:
        constraints += get_max_weights_constraints(data, max_weights, weights, k, is_short=True)

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


def get_max_weights_constraints(
        data: DataFrame,
        max_weights: dict[str, float],
        weights: Variable,
        k: Variable,
        is_short: bool = False,
) -> list[Inequality | Any]:
    constraints = list()

    columns = data.columns.tolist()
    for coin, max_w in max_weights.items():
        if coin in columns:
            i = columns.index(coin)
            if is_short:
                constraints.append(weights[i] >= -max_w * k)
            else:
                constraints.append(weights[i] <=  max_w * k)

    return constraints
