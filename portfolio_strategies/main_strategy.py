from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Callable
from funcs import *
from .constants import *


class Strategy:

    def __init__(
        self,
        strategy: Callable[[pd.DataFrame], dict[str, float]] | None = None,
        fee_rate: float = FEE_RATE
    ):
        self._capital = self._peak_capital = START_CAPITAL
        self._fee_rate = fee_rate if strategy is not None else 0.0

        self._quantities = defaultdict(float)
        self._weights = defaultdict(float)

        self.entry_prices = defaultdict(float)
        self.metrics = defaultdict(list)
        self.date_metrics = list()

        self.weights_history = list()
        self.rebalancing_dates = list()

        self.strategy = strategy

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def capital(self) -> float:
        return self._capital

    @property
    def peak_capital(self) -> float:
        return self._peak_capital

    @property
    def quantities(self) -> dict[str, float]:
        return self._quantities

    @property
    def weights(self) -> dict[str, float]:
        return self._weights

    def weights_in_money(self) -> dict[str, float]:
        return {coin: self._capital * weight for coin, weight in self._weights}

    def training(self, prices: pd.DataFrame, volumes: pd.DataFrame):
        if self.strategy is not None:
            self._weights.clear()
            self._quantities.clear()

        if self.strategy is not None:
            self._weights = partial(self.strategy, min_weights=MIN_INVEST / self._capital)(prices)
            if self._weights is None: self._weights = dict()
            # self._weights = self.strategy(prices)

        self._fix_weights(prices, volumes)

        if self.strategy is not None or (self.strategy is None and len(self._quantities) == 0):
            self._set_quantity(prices)

        self._save_statistics(prices)

    def _fix_weights(self, prices: pd.DataFrame, volumes: pd.DataFrame):
        self._normalize_weights()

        self._reset_less_min_w()
        
        self._normalize_weights()
        self._only_ten_percent_volume(prices, volumes)

    def _normalize_weights(self):
        sum_ = sum([abs(weight) for _, weight in self._weights.items()])
        if sum_ == 0: return

        for coin, weight in self._weights.items():
            self._weights[coin] /= sum_

    def _reset_less_min_w(self):
        for coin, weight in self._weights.items():
            if abs(weight * self._capital) < MIN_INVEST: self._weights[coin] = 0

    def _keep_last(self, train_prices: pd.DataFrame):
        if len(self.weights_history) > 0:
            for coin in self.weights_history[-1]:
                if train_prices[coin].isna().sum() > 0:
                    self._weights.clear()
                    return

            self._weights = deepcopy(self.weights_history[-1])

    def _save_statistics(self, prices: pd.DataFrame):
        self.weights_history.append(deepcopy(self._weights))
        self.rebalancing_dates.append(prices.index[-1])

    def _only_ten_percent_volume(self, prices: pd.DataFrame, volumes: pd.DataFrame):
        date = prices.index[-1]
        for coin, weight in self._weights.items():
            daily_volume = volumes.loc[date, coin]
            if abs(weight * self._capital) > VOLUME_PERCENT * daily_volume / 100.0:
                self._weights[coin] = (VOLUME_PERCENT * daily_volume) / self._capital

    def _set_quantity(self, prices: pd.DataFrame):
        purchase_prices = prices.iloc[-1]
        for coin, weight in self._weights.items():
            amount_to_buy = self._capital * weight
            quantity = amount_to_buy / purchase_prices[coin]

            self._quantities[coin] = quantity
            self.entry_prices[coin] = purchase_prices[coin]
    
    def predict(self, test_prices: pd.DataFrame):
        self.date_metrics.append(test_prices.index[-1])

        mdd_per, mdd_abs = get_max_drawdown(test_prices, self)
        self.metrics[Metrics.max_draw_down_percent].append(mdd_per)
        self.metrics[Metrics.max_draw_down_absolut].append(mdd_abs)

        if self.strategy is None:
            self._calculate_rate(test_prices)
        else:
            self._calculate_total_return(test_prices)

    def _calculate_total_return(self, prices: pd.DataFrame):
        total_return_per, total_return_abs = get_total_return(prices, self, self._fee_rate)
        self._capital = total_return_abs
        self.metrics[Metrics.total_returns_absolut].append(total_return_abs)
        self.metrics[Metrics.total_returns_percent].append(total_return_per)

    def _calculate_rate(self,  prices: pd.DataFrame):
        total_return_per, total_return_abs = get_portfolio_rate(prices, self)
        self.metrics[Metrics.total_returns_absolut].append(total_return_abs)
        self.metrics[Metrics.total_returns_percent].append(total_return_per)


__all__ = [
    'Metrics',
    'Strategy',
    'FEE_RATE',
    'START_CAPITAL'
]
