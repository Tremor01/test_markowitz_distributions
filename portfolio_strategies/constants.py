from enum import Enum


class Metrics(str, Enum):
    max_draw_down_percent = 'max_draw_down_percent'
    max_draw_down_absolut = 'max_draw_down_absolut'

    total_returns_absolut = 'total_returns_absolut'
    total_returns_percent = 'total_returns_percent'


MIN_INVEST = 5.0
VOLUME_PERCENT = 10.0
START_CAPITAL = 10_000.0
FEE_RATE = 0.001


MAX_WEIGHTS = {
    'ZEC': 25,
    'XTZ': 25,
    'ALGO': 25,
    'ICX': 25,
    'ONT': 25,
    'KAVA': 25,
    'HBAR': 25,
    'VET': 25,
    'ENJ': 25,
    'CHZ': 25,
    'FET': 25,
    'STX': 25,
    'ARPA': 25,
    'ONE': 25,
    'DASH': 25,
    'IOTA': 25,
    'NKN': 25,
    'CELR': 25,
    'DENT': 25,
    'MTL': 25,
    'FUN': 25,
    'DUSK': 25,
    'WAN': 25,
    'WIN': 25,
    'RLC': 25,
    'IOTX': 25,
    'ANKR': 25,
    'BAND': 25,
    'COS': 25,
    'HOT': 25,
    'TFUEL': 25,
    'ZIL': 25,
    'ZRX': 25,
    'RVN': 25,
    'ONG': 25,
}


__all__ = [
    'MIN_INVEST',
    'VOLUME_PERCENT',
    'START_CAPITAL',
    'FEE_RATE',
    'Metrics'
]