from typing import Any

from riskfolio import Portfolio, assets_constraints
from pandas import DataFrame


def max_profit(train: DataFrame, min_weights: float = 0.0) -> dict[str, float]:
    try:
        port = get_base_portfolio(train, min_weights)
        w = port.optimization(model='Classic', rm='MV', obj='MaxRet')
        if w is None: min_weights = 0

        port = get_base_portfolio(train, min_weights)
        w = port.optimization(model='Classic', rm='MV', obj='MaxRet')
        return w.to_dict()['weights']
    except:
        port = get_base_portfolio(train, min_weight=0.0)
        w = port.optimization(model='Classic', rm='MV', obj='MaxRet')
        return w.to_dict()['weights']


def sharp(
        train: DataFrame,
        min_weights: float = 0.0,
        max_weights: dict[str, float] | None = None,
        risk_free_rate: float = 0.0,
        short: bool = False
) -> dict[str, float]:
    try:
        port = get_base_portfolio(train, min_weights, max_weights, short)
        w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=risk_free_rate)
        if w is None: min_weights = 0

        port = get_base_portfolio(train, min_weights, max_weights, short)
        w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=risk_free_rate)
        return w.to_dict()['weights']
    except:
        port = get_base_portfolio(train, min_weight=0.0)
        w = port.optimization(model='Classic', rm='MV', obj='MaxRet')
        return w.to_dict()['weights']


def min_risk(train: DataFrame, min_weights: float = 0.0) -> dict[str, float]:
    try:
        port = get_base_portfolio(train, min_weights)
        w = port.optimization(model='Classic', rm='MV', obj='MinRisk')
        if w is None: min_weights = 0

        port = get_base_portfolio(train, min_weights)
        w = port.optimization(model='Classic', rm='MV', obj='MinRisk')
        return w.to_dict()['weights']
    except :
        port = get_base_portfolio(train, min_weight=0.0)
        w = port.optimization(model='Classic', rm='MV', obj='MaxRet')
        return w.to_dict()['weights']


def get_base_portfolio(
        train: DataFrame,
        min_weight: float = 0.0,
        max_weights: dict[str, float] | None = None,
        short: bool = False,
        uppersht: float = 0,
        budgetsht: float = 1,
) -> Portfolio:
    assets = train.columns
    Y = train[assets].pct_change().dropna()

    constraints = get_min_weight_constraint(min_weight)
    if max_weights is not None:
        constraints += get_max_weights_constraints(max_weights)

    constraints = DataFrame(constraints)
    asset_classes = DataFrame({'Assets': assets})
    A, B = assets_constraints(constraints, asset_classes)
    port = Portfolio(returns=Y, ainequality=A, binequality=B, sht=short, uppersht=uppersht, budgetsht=budgetsht)

    port.assets_stats(method_mu='hist', method_cov='ledoit')
    return port


def get_min_weight_constraint(min_weight: float) -> list[dict[str, Any]]:
    rows = list()
    rows.append({
        'Disabled': False,
        'Type': 'All Assets',
        'Set': '',
        'Position': '',
        'Sign': '>=',
        'Weight': min_weight,
        'Type Relative': '',
        'Relative Set': '',
        'Relative': '',
        'Factor': ''
    })
    return rows


def get_max_weights_constraints(max_weights: dict[str, float]) -> list[dict[str, Any]]:
    rows = list()
    for asset, max_w in max_weights.items():
        rows.append({
            'Disabled': False,
            'Type': 'Asset',
            'Set': '',
            'Position': asset,
            'Sign': '<=',
            'Weight': max_w,
            'Type Relative': '',
            'Relative Set': '',
            'Relative': '',
            'Factor': ''
        })
    return rows
