from enum import Enum

MIN_INVEST = 5.0
VOLUME_PERCENT = 10.0
START_CAPITAL = 10_000.0
FEE_RATE = 0.001


class Metrics(str, Enum):
    max_draw_down_percent = 'max_draw_down_percent'
    max_draw_down_absolut = 'max_draw_down_absolut'

    total_returns_absolut = 'total_returns_absolut'
    total_returns_percent = 'total_returns_percent'


__all__ = [
    'MIN_INVEST',
    'VOLUME_PERCENT',
    'START_CAPITAL',
    'FEE_RATE',
    'Metrics'
]