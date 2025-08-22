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
    'ZEC': 15,
    'XTZ': 15,
    'ALGO': 15,
    'ICX': 15,
    'ONT': 15,
    'KAVA': 15,
    'HBAR': 15,
    'VET': 15,
    'ENJ': 15,
    'CHZ': 15,
    'FET': 15,
    'STX': 15,
    'ARPA': 15,
    'ONE': 15,
    'DASH': 15,
    'IOTA': 15,
    'NKN': 15,
    'CELR': 15,
    'DENT': 15,
    'MTL': 15,
    'FUN': 15,
    'DUSK': 15,
    'WAN': 15,
    'WIN': 15,
    'RLC': 15,
    'IOTX': 15,
    'ANKR': 15,
    'BAND': 15,
    'COS': 15,
    'HOT': 15,
    'TFUEL': 15,
    'ZIL': 15,
    'ZRX': 15,
    'RVN': 15,
    'ONG': 15,
}


__all__ = [
    'MIN_INVEST',
    'VOLUME_PERCENT',
    'START_CAPITAL',
    'FEE_RATE',
    'Metrics'
]