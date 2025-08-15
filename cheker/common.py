from portfolio_strategies import Strategy
from pandas import DataFrame
from portfolio_strategies.constants import Metrics


class StrategyChecker:

    def __init__(self, prices: DataFrame, start_capital: float):
        self._prices = prices
        self._start_capital = start_capital
        self._fee_rate = 0.001
    
    def get_report(self, strategy: Strategy) -> dict[str, float]:
        return self._check_weights(strategy) | self._check_pair_returns(strategy)
    
    def _check_weights(self, strategy: Strategy) -> dict[str, float]:
        weights = strategy.weights_history[-1]
        sum_      = sum([abs(w) for _, w in weights.items()])
        try:
            cur_min = min([w for _, w in weights.items()])
        except ValueError:
            cur_min = 0.0
        sum_plus  = sum([w for _, w in weights.items() if w > 0])
        sum_mines = sum([w for _, w in weights.items() if w < 0])
        return {
            'date': strategy.date_metrics[-1],
            'sum_weights': sum_,
            'sum_positive': sum_plus,
            'sum_negative': sum_mines,
            'min_weight': cur_min,
            'min_weight_in_cash': cur_min * strategy.capital,
        }

    def _check_pair_returns(self, strategy: Strategy) -> dict[str, float]:
        returns = strategy.metrics[Metrics.total_returns_absolut]
        if len(returns) == 1:
            left, right = self._start_capital, returns[-1]
        else:
            left, right = returns[-2], returns[-1]

        end_date = strategy.date_metrics[-1]
        start_date = strategy.rebalancing_dates[-1]
        weight = strategy.weights_history[-1]

        profit = 0.0
        for coin, w in weight.items():
            price_end, price_start = self._prices.loc[end_date, coin], self._prices.loc[start_date, coin]
            quantity = (w * left) / price_start
            profit += self._get_returns(quantity, price_start, price_end)

        month_ret = (right - left) / left
        real_ret = profit / left
        return {
            'returns_percent': month_ret,
            'check_percent': real_ret,
            'abs_delta_returns': abs(real_ret - month_ret)
        }

    def _get_returns(self, quantity: float, price_start: float, price_end: float) -> float:
        fee = (1.0 - self._fee_rate)
        fee_sht = (1.0 + self._fee_rate)

        result = 0.0
        if quantity > 0:
            entry_quantity = quantity * fee
            cost_entry = entry_quantity * price_start
            cost_exit = entry_quantity * price_end * fee
            result += cost_exit - cost_entry
        else:
            qty_abs = abs(quantity)
            cost_entry = qty_abs * price_start * fee
            cost_exit = qty_abs * fee_sht * price_end
            result += cost_entry - cost_exit

        return result

    def _check_all_returns(self, strategy: Strategy) -> bool:
        capital = float(self._start_capital)

        count = 0
        for i in range(len(strategy.date_metrics)):
            sell_date = strategy.date_metrics[i]
            buy_date  = strategy.rebalancing_dates[i]
            weights   = strategy.weights_history[i]

            profit = 0.0
            for coin in weights:
                price_start, price_end  = self._prices.loc[buy_date,  coin], self._prices.loc[sell_date, coin]
                quantity = weights[coin] * capital / price_start
                profit += self._get_returns(quantity, price_start, price_end)

            capital += profit
            count += (strategy.metrics[Metrics.total_returns_absolut][i] == capital)

        return count == len(strategy.metrics[Metrics.total_returns_absolut])
