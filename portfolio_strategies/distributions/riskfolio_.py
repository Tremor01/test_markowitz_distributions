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
        port = get_base_portfolio(train, min_weights=0.0)
        w = port.optimization(model='Classic', rm='MV', obj='MaxRet')
        return w.to_dict()['weights']


def sharp(
        train: DataFrame, min_weights: float = 0.0, risk_free_rate: float = 0.0, short: bool = False
) -> dict[str, float]:
    try:
        port = get_base_portfolio(train, min_weights, short)
        w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=risk_free_rate)
        if w is None: min_weights = 0

        port = get_base_portfolio(train, min_weights, short)
        w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=risk_free_rate)
        return w.to_dict()['weights']
    except:
        port = get_base_portfolio(train, min_weights=0.0)
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
        port = get_base_portfolio(train, min_weights=0.0)
        w = port.optimization(model='Classic', rm='MV', obj='MaxRet')
        return w.to_dict()['weights']


def get_base_portfolio(
        train: DataFrame,
        min_weights: float,
        short: bool = False,
        uppersht: float = 0,
        budgetsht: float = 1,
) -> Portfolio:
    assets = train.columns
    Y = train[assets].pct_change().dropna()

    if min_weights != 0:
        constraints = get_constraints(min_weights)
        asset_classes = DataFrame({'Assets': assets})
        A, B = assets_constraints(constraints, asset_classes)
        port = Portfolio(returns=Y, ainequality=A, binequality=B, sht=short, uppersht=uppersht, budgetsht=budgetsht)
    else:
        port = Portfolio(returns=Y, sht=short, uppersht=uppersht, budgetsht=budgetsht)

    port.assets_stats(method_mu='hist', method_cov='ledoit')
    return port


def get_constraints(min_weight: float):
    constraints = DataFrame({
        'Disabled': [False],
        'Type': ['All Assets'],
        'Set': [''],
        'Position': [''],
        'Sign': ['>='],
        'Weight': [min_weight],
        'Type Relative': [''],
        'Relative Set': [''],
        'Relative': [''],
        'Factor': ['']
    })
    return constraints
