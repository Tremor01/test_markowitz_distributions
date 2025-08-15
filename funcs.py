import numpy as np
import pandas as pd

from portfolio_strategies.constants import START_CAPITAL


def get_rebalanced_return(returns: list[float], start_capital: float)-> tuple[list[float], list[float]]:
    percents, abs_values = [], []
    cumulative_percent, cumulative_abs = 1.0, start_capital

    for r in returns:
        cumulative_percent *= (1 + r / 100); cumulative_abs *= (1 + r / 100)
        percents.append((cumulative_percent - 1) * 100); abs_values.append(cumulative_abs)

    return percents, abs_values


def get_portfolio_value(prices: pd.DataFrame, strategy) -> pd.Series:
    """
    Возвращает стоимость портфеля во времени.
    """
    portfolio_value = pd.Series(index=prices.index, dtype=float)

    for date in prices.index:
        value = strategy.capital
        for coin, qty in strategy.quantities.items():
            if qty > 0:
                value += qty * (prices.loc[date, coin] - strategy.entry_prices[coin])
            else:
                entry_price = strategy.entry_prices[coin]
                value += (entry_price - prices.loc[date, coin]) * abs(qty)
        portfolio_value.loc[date] = value

    return portfolio_value


def get_max_drawdown(prices: pd.DataFrame, strategy) -> tuple[float, float]:
    """
    Максимальная относительная просадка (%).
    """
    portfolio_value = get_portfolio_value(prices, strategy)
    rolling_max = portfolio_value.cummax()
    drawdowns = (portfolio_value - rolling_max) / rolling_max * 100
    abs_drawdowns = portfolio_value - rolling_max
    return drawdowns.min(), abs_drawdowns.min()


def get_max_drawdown_from_start(prices: pd.DataFrame, strategy) -> tuple[float, float]:
    """
    Максимальная просадка с начала периода (%).
    """
    portfolio_value = get_portfolio_value(prices, strategy)

    drawdowns = (portfolio_value.iloc[0] - portfolio_value.min()) / portfolio_value.iloc[0] * 100
    abs_drawdowns = portfolio_value.iloc[0] - portfolio_value.min()
    return drawdowns, abs_drawdowns


def get_total_return(prices: pd.DataFrame, strategy, fee_rate: float = 0.0) -> tuple[float, float]:
    """
    Общая доходность портфеля на конец периода (%).
    """
    end_date, start_date = prices.index[-1], prices.index[0]

    end_capital = 0.0
    for coin in strategy.weights:
        quantity = strategy.quantities[coin]
        price_end, price_start = prices.loc[end_date, coin], prices.loc[start_date, coin]

        if quantity > 0:
            entry_quantity = quantity * (1 - fee_rate)
            cost_entry = entry_quantity * price_start
            cost_exit  = entry_quantity * price_end * (1 - fee_rate)
            end_capital += cost_exit - cost_entry
        else:
            qty_abs = abs(quantity)
            cost_entry = qty_abs * price_start * (1 - fee_rate)
            cost_exit = qty_abs * (1 + fee_rate) * price_end
            end_capital += cost_entry - cost_exit

    total_return_abs = end_capital + strategy.capital
    total_return_percent = total_return_abs / START_CAPITAL * 100

    return total_return_percent, total_return_abs


def get_peak_return_by_coins(prices: pd.DataFrame, strategy, fee_rate: float = 0.0) -> tuple[float, float]:
    """
    Максимальная доходность от начала до пика каждой монеты (%).
    """
    quantities = strategy.quantities

    max_price = prices.max()
    min_price = prices.min()

    peak = 0.0
    for coin in strategy.weights:
        if quantities[coin] >= 0:
            amount = max_price[coin] * quantities[coin]
        else:
            amount = min_price[coin] * quantities[coin]

        peak += amount * (1 - fee_rate)

    return (peak - strategy.capital) / strategy.capital * 100, peak - strategy.capital


def get_volatility(
    prices: pd.DataFrame,
    strategy,
    target_period_in_days: float = 365,
    data_period: float = 1 / 365
) -> tuple[float, float]:
    """
    Волатильность портфеля (%).
    """

    period = 365 // target_period_in_days
    portfolio_value = get_portfolio_value(prices, strategy)

    returns_per = portfolio_value.pct_change().dropna()
    sigma_sd = returns_per.std()
    average_annual_volatility = sigma_sd / np.sqrt(data_period)
    period_volatility = average_annual_volatility * np.sqrt(1 / period)

    sigma_abs_sd = portfolio_value.std()
    average_annual_volatility_abs = sigma_abs_sd / np.sqrt(data_period)
    period_volatility_abs = average_annual_volatility_abs * np.sqrt(1 / period)

    return period_volatility * 100, period_volatility_abs


def get_total_return_trailing_stop(
        prices: pd.DataFrame,
        strategy,
        trailing_stop: float = 0.02,
        fee_rate: float = 0.0
) -> tuple[float, float]:
    """
    Общая доходность портфеля за весь период (%).
    """
    end_date = prices.index[-1]

    end_value = 0
    for coin in strategy.weights:
        price_end = prices.loc[end_date, coin]
        temp_price = strategy.weights_in_money()[coin] / strategy.quantities[coin]
        for date in prices.index:
            price_date = prices.loc[date, coin]
            if (price_date - temp_price) * strategy.quantities[coin] / temp_price * (1 - fee_rate) < -trailing_stop:
                price_end = price_date
                break
            temp_price = max(temp_price, price_date)
        coin_end_usd = price_end * strategy.quantities[coin] * (1 - fee_rate)
        end_value += coin_end_usd

    total_return_percent = (end_value - strategy.capital) / strategy.capital * 100
    total_return_abs = end_value - strategy.capital
    return total_return_percent, total_return_abs


def get_portfolio_rate(prices: pd.DataFrame, strategy):
    end_date = prices.index[-1]
    end_capital = 0.0
    for coin in strategy.weights:
        price_end = prices.loc[end_date, coin]
        coin_end_usd = price_end * strategy.quantities[coin]
        end_capital += coin_end_usd

    change_percent = end_capital / strategy.capital * 100
    return change_percent, end_capital


def get_portfolio_max_rate(prices: pd.DataFrame, strategy):
    end_capital = 0.0
    for coin in strategy.weights:
        price_max = max(prices[coin])
        coin_end_usd = price_max * strategy.quantities[coin]
        end_capital += coin_end_usd

    change_percent = end_capital / strategy.peak_capital * 100
    return change_percent, end_capital