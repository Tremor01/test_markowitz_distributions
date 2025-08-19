import time
from copy import deepcopy
import pandas as pd
from plotting import plot_metrics

from portfolio_strategies.constants import START_CAPITAL
from portfolio_strategies import StrategyBTC, SHARP_SHORT, Strategy, ConvexMarkowitzSharpAlpha, ConvexMarkowitzSharpBruteForceRF
from data import get_prices, get_volumes
from checkers import ReportBuilder

import warnings
warnings.filterwarnings("ignore")


PRICES = get_prices()
VOLUMES = get_volumes()


def free_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices  =  PRICES.iloc[869: 1965]
    volumes = VOLUMES.iloc[869: 1965]
    valid_coins = [
        coin for coin in prices.columns
        if prices[coin].isna().sum() == 0 and prices[coin].isna().sum() == 0
    ]
    return prices[valid_coins], volumes[valid_coins]


def main():
    train, test = 30, 7

    prices, volumes = free_tests()
    # alpha_strat = ConvexMarkowitzSharpAlpha(1)
    # s = [([alpha_strat()] + deepcopy(SHARP_SHORT) + [StrategyBTC()], train, test),]
    # for strategies, train_period, step in s:
    #     file_name = strategies[0].name + f'{train_period}_{step}_{START_CAPITAL}'
    #     t = time.time()
    #     simulate(deepcopy(strategies), file_name, train_period, step, prices, volumes)
    #     print(time.time() - t)

    # for a in range(0, 101, 10):
    #     a /= 100
    #     alpha_strat = ConvexMarkowitzSharpAlpha(a)
    #     s = [
    #         ([alpha_strat()] + deepcopy(SHARP_SHORT) + [StrategyBTC()], train, test),
    #     ]
    #
    #     for strategies, train_period, step in s:
    #         file_name = strategies[0].name + f'{train_period}_{step}_{START_CAPITAL}'
    #         t = time.time()
    #         simulate(
    #             deepcopy(strategies), file_name, train_period, step,
    #             prices, volumes, f'report_alpha_{a * 100}_{int(100 - a * 100)}.html', plot=True
    #         )
    #         print(time.time() - t)

    for rf in range(0, 6):
        rf_strat = ConvexMarkowitzSharpBruteForceRF(rf)
        s = [
            ([rf_strat()] + deepcopy(SHARP_SHORT) + [StrategyBTC()], train, test),
        ]

        for strategies, train_period, step in s:
            file_name = strategies[0].name + f'{train_period}_{step}_{START_CAPITAL}'
            t = time.time()
            simulate(
                deepcopy(strategies), file_name, train_period, step,
                prices, volumes, f'report_rf_{rf}.html', plot=True
            )
            print(time.time() - t)


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

