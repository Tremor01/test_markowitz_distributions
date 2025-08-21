import random
from typing import Any
import  pandas as pd
from checkers import ReportBuilder
from data import get_volumes, get_prices
from plotting import plot_metrics
from portfolio_strategies import ConvexMarkowitzSharpBruteForceRF, SHARP_SHORT, StrategyBTC, Strategy
from portfolio_strategies.constants import START_CAPITAL, Metrics
from collections import defaultdict
from copy import deepcopy
import time


import warnings
warnings.filterwarnings("ignore")


PRICES = get_prices()
VOLUMES = get_volumes()


def free_tests() -> tuple[Any, Any, int]:
    left = random.randint(869, 1900)
    prices  =  PRICES.iloc[left: 1965]
    volumes = VOLUMES.iloc[left: 1965]
    valid_coins = [
        coin for coin in prices.columns
        if prices[coin].isna().sum() == 0 and prices[coin].isna().sum() == 0
    ]
    return prices[valid_coins], volumes[valid_coins], left


def main():
    train, test = 30, 7

    statistics = defaultdict(lambda: defaultdict(list))

    for _ in range(10):
        prices, volumes, rand = free_tests()
        rf_strat = ConvexMarkowitzSharpBruteForceRF(0)
        s = [([rf_strat()] + deepcopy(SHARP_SHORT) + [StrategyBTC()], train, test),]

        for strategies, train_period, step in s:
            file_name = f'alpha_brute_force_{rand}'
            t = time.time()
            res = simulate(
                deepcopy(strategies), file_name, train_period, step,
                prices, volumes, f'report_{file_name}.html', plot=False
            )
            for strategy in res:
                statistics[strategy]['drowdawn_%'].append(res[strategy]['drowdawn_%'])
                statistics[strategy]['drowdawn_$'].append(res[strategy]['drowdawn_$'])
                statistics[strategy]['returns_%'].append(res[strategy]['returns_%'])
                statistics[strategy]['returns_$'].append(res[strategy]['returns_$'])
            print(time.time() - t)

    with open('random_stats.txt', 'w', encoding='utf-8') as f:
        f.write(str(statistics))


def simulate(
        strategies: list[Strategy],
        file_name: str,
        train_period: int,
        step: int,
        prices:  pd.DataFrame,
        volumes: pd.DataFrame,
        report_name: str | None = None,
        plot: bool = True
):
    if report_name is None:
        report_name = f'Report{strategies[0].name}{train_period}_{step}_{START_CAPITAL}.html'

    left = 0; right = train_period
    for _ in range(len(prices) // step - 1):
        train_prices = prices.iloc[left: right]
        train_volumes = volumes.iloc[left: right]
        test_prices = prices.iloc[right - 1: right + step]

        if train_prices.empty or train_volumes.empty or test_prices.empty:
            right += step; left += step
            continue

        valid_coins = filter_coins(train_prices, test_prices, volumes)

        valid_trains = train_prices[valid_coins]
        valid_volumes = train_volumes[valid_coins]
        valid_test = test_prices[valid_coins]

        for strategy in strategies:
            if valid_trains.empty or valid_test.empty: break
            strategy.training(valid_trains, valid_volumes)
            strategy.predict(valid_test)

        right += step; left += step

    builder = ReportBuilder(prices, volumes, strategies)
    builder.build_report(report_name)

    if plot: plot_metrics(strategies, file_name)

    return {
        strategy.name: {
            'drowdawn_%': min(strategy.metrics[Metrics.max_draw_down_percent]),
            'drowdawn_$': min(strategy.metrics[Metrics.max_draw_down_absolut]),
            'returns_%':  max(strategy.metrics[Metrics.total_returns_percent]),
            'returns_$':  max(strategy.metrics[Metrics.total_returns_absolut]),
        } for strategy in strategies
    }


def filter_coins(train_prices: pd.DataFrame, test_prices: pd.DataFrame, volumes: pd.DataFrame):
    # Монета существует минимум train_period + test_period
    valid_coins = [
        coin for coin in train_prices.columns
        if train_prices[coin].isna().sum() == 0 and test_prices[coin].isna().sum() == 0
    ]

    # В день покупки у монеты объём >= 10^4
    buy_date = test_prices.index[0]

    filtered_coin = list()
    for coin in valid_coins:
        if volumes.loc[buy_date, coin] >= 1e4: filtered_coin.append(coin)

    return filtered_coin


main()
