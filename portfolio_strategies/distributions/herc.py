from numpy import random
from riskfolio import HCPortfolio
from pandas import DataFrame


def sharp(train: DataFrame, min_weights: float = 0.0, risk_free_rate: float = 0.0) -> dict[str, float]:
    try:
        port = get_base_portfolio(train, min_weights)
        w = port.optimization(model="HERC", obj="Sharpe", rm='CVaR', rf=risk_free_rate, linkage='ward', l=1, k=12)
    except:
        port = get_base_portfolio(train, min_weights=0.0)
        w = port.optimization(model="HERC", obj="Sharpe", rm='CVaR', rf=risk_free_rate, linkage='ward', l=1, k=12)
    return w.to_dict()['weights']


def min_risk(train: DataFrame, min_weights: float = 0.0) -> dict[str, float]:
    try:
        port = get_base_portfolio(train, min_weights)
        w = port.optimization(model="HERC", obj="MinRisk", rm='CVaR', linkage='ward', l=3)
    except:
        port = get_base_portfolio(train, min_weights=0.0)
        w = port.optimization(model="HERC", obj="MinRisk", rm='CVaR', linkage='ward', l=3)
    return w.to_dict()['weights']


def max_profit(train: DataFrame, min_weights: float = 0.0) -> dict[str, float]:
    try:
        port = get_base_portfolio(train, min_weights)
        w = port.optimization(model="HERC", obj="Utility", rm='CVaR', linkage='ward', l=0.5, k=6)
    except:
        port = get_base_portfolio(train, min_weights=0.0)
        w = port.optimization(model="HERC", obj="Utility", rm='CVaR', linkage='ward', l=0.5, k=6)
    return w.to_dict()['weights']


def get_base_portfolio(train: DataFrame, min_weights: float) -> HCPortfolio:
    assets = train.columns
    Y = train[assets].pct_change().dropna()
    Y += random.normal(0, 1e-8, size=Y.shape)
    port = HCPortfolio(returns=Y, w_min=min_weights)
    return port
