from enum import StrEnum
from .utils import (
    get_std_risk, get_ledoit_wolf_risk, get_corr_risk,
    get_expected_returns, calculate_exp_returns, get_ema_returns
)


class RiskType(StrEnum):
    STD = "std"
    DRAWDOWN = "drawdown"
    LEDOIT_WOLF = "ledoit_wolf"
    CORRELATION = "correlation"

RISK_FUNC = {
    RiskType.STD: get_std_risk,
    RiskType.LEDOIT_WOLF: get_ledoit_wolf_risk,
    RiskType.CORRELATION: get_corr_risk,
}


class ReturnsType(StrEnum):
    PTC = "ptc"
    EMA = "ema"
    EXP = "exp"

RETURNS_FUNC = {
    ReturnsType.PTC: get_expected_returns,
    ReturnsType.EMA: get_ema_returns,
    ReturnsType.EXP: calculate_exp_returns,
}


__all__ = [
    'RiskType',
    'ReturnsType',
    'RISK_FUNC',
    'RETURNS_FUNC'
]