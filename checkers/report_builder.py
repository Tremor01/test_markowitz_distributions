import math
from pathlib import Path

from pandas import DataFrame
from portfolio_strategies import Strategy
from portfolio_strategies.constants import Metrics, FEE_RATE, START_CAPITAL, VOLUME_PERCENT, MIN_INVEST
import os

from shared import get_unique_filepath

current_file = Path(__file__)
parent_dir = current_file.parent.parent

plots_dir = os.path.join(parent_dir, "portfolio_plots")
os.makedirs(plots_dir, exist_ok=True)


class ReportBuilder:

    def __init__(
        self,
        prices: DataFrame,
        volumes: DataFrame,
        strategies: list[Strategy],
        start_capital: float = START_CAPITAL,
        fee_rate: float = FEE_RATE,
        volume_percent: float = VOLUME_PERCENT,
        min_invest: float = MIN_INVEST,
    ):
        self._prices = prices
        self._volumes = volumes
        self._strategies = strategies
        self._start_capital = start_capital
        self._fee_rate = fee_rate
        self._volume_percent = volume_percent
        self._min_invest = min_invest
        self._report = dict()

    def build_report(self, file_name: str = 'report.html'):
        self._report = {strategy.name: dict() for strategy in self._strategies}

        for strategy in self._strategies:
            self._check_weights(strategy)
            self._check_returns(strategy)
            self._check_between_returns(strategy)
            self._check_dataframe(strategy, self._prices, 'prices')
            self._check_dataframe(strategy, self._volumes, 'volumes')

        self._save_report(os.path.join(plots_dir, file_name))

    def _check_weights(self, strategy: Strategy):
        min_w = volume_percent = normalize = 0
        self._report[strategy.name]['stats_weights'] = list()
        self._report[strategy.name]['list_weights'] = list()

        chk_sum_of_weights = chk_min_weights = daily_volume = 0
        for date in strategy.rebalancing_dates:
            fmin_w, fvolume_percent, fnormalize = self._check_weights_change(strategy.data_for_report['weights'][date])
            min_w += fmin_w; volume_percent += fvolume_percent; normalize += fnormalize
            
            weights = strategy.data_for_report['weights'][date]['final']
            capital = strategy.data_for_report['capital'][date]

            try:
                cur_min = min([abs(w) for _, w in weights.items() if abs(w) > 0])
            except ValueError:
                cur_min = 0.0

            for coin, weight in weights.items():
                if abs(weight * capital) > self._volume_percent * self._volumes.loc[date, coin] / 100:
                    daily_volume += 1

            sum_ = sum([abs(w) for _, w in weights.items()])
            sum_plus  = sum([w for _, w in weights.items() if w > 0])
            sum_mines = sum([w for _, w in weights.items() if w < 0])

            chk_sum_of_weights += (sum_ - 1 > 1e-6)
            chk_min_weights += (cur_min * capital < self._min_invest - 1e-3 and cur_min != 0)

            self._report[strategy.name]['stats_weights'].append({
                'date': date,
                'sum': self._round(sum_),
                'sum_pos': self._round(sum_plus),
                'sum_neg': self._round(sum_mines),
                'min_weight': self._round(cur_min),
                'capital': self._round(capital),
                '$ min_weight': self._round(cur_min * capital),
                '> daily_volume': self._round(daily_volume),
                'alpha': strategy.data_for_report['alpha'][date],
                'walpha': strategy.data_for_report['walpha'][date]
            })

            weights = dict(sorted(weights.items(), key=lambda p: abs(p[1]), reverse=True))
            str_weights = ', '.join([f"{coin}: {w * 100:.5f}" for coin, w in weights.items() if f'{w * 100:.5f}' != '0.00000'])
            self._report[strategy.name]['list_weights'].append({'date': date, 'weights': str_weights})

        self._report[strategy.name]['correct_weights'] = {
            'sum': chk_sum_of_weights,
            'min_w': chk_min_weights,
            'daily_volume': daily_volume,
        }
        self._report[strategy.name]['change_weights'] = {
            'min_w': min_w, 
            'volume_percent': volume_percent, 
            'normalize': normalize
        }

    def _round(self, x: float, precision: int = 5):
        return round(x, precision)

    def _check_weights_change(self, weights: dict[str, dict[str, float]]) -> tuple[bool, bool, bool]:
        fnormalize = weights['clear'] != weights['normalize']
        fmin_w = weights['normalize'] != weights['min_weights']
        fvolume_percent = weights['min_weights'] != weights['volumes_percent']
        return fmin_w, fvolume_percent, fnormalize

    def _check_dataframe(self, strategy: Strategy, dataframe: DataFrame, key: str):
        cnt_incorrect = 0
        for date in strategy.rebalancing_dates:
            weights = strategy.data_for_report['weights'][date]['final']
            for coin in weights:
                cnt_incorrect += (
                    dataframe.loc[date, coin] is None or
                    dataframe.loc[date, coin] <= 0
                )
        self._report[strategy.name][key] = {
            'is_correct': cnt_incorrect == 0,
            'cnt_incorrect': cnt_incorrect
        }

    def _check_between_returns(self, strategy: Strategy):
        returns = strategy.metrics[Metrics.total_returns_absolut]
        
        pairs_returns = list()
        cnt_incorrect = 0
        for i in range(len(returns)):
            if i == 0:
                left, right = self._start_capital, returns[i]
            else:
                left, right = returns[i - 1], returns[i]

            end_date = strategy.date_metrics[i]
            start_date = strategy.rebalancing_dates[i]
            weight = strategy.weights_history[i]
    
            profit = 0.0
            for coin, w in weight.items():
                price_end, price_start = self._prices.loc[end_date, coin], self._prices.loc[start_date, coin]
                quantity = (w * left) / price_start
                profit += self._get_returns(quantity, price_start, price_end)
    
            month_ret = (right - left) / left
            real_ret = profit / left
            cnt_incorrect += abs(month_ret - real_ret) > 1e-10
            
            pairs_returns.append({
                'date': end_date,
                'returns_percent': month_ret * 100,
                'check_percent': real_ret * 100,
                'abs_delta_returns': abs(real_ret - month_ret)
            })
        
        self._report[strategy.name]['between_returns'] = {
            'is_correct': cnt_incorrect == 0,
            'cnt_incorrect': cnt_incorrect,
            'between_returns': pairs_returns,
        }
    
    def _check_returns(self, strategy: Strategy):
        capital = float(self._start_capital)
        returns = strategy.metrics[Metrics.total_returns_absolut]

        count = 0
        for i in range(len(strategy.date_metrics)):
            sell_date = strategy.date_metrics[i]
            buy_date  = strategy.rebalancing_dates[i]
            weights   = strategy.weights_history[i]

            profit = 0.0
            for coin in weights:
                price_start = self._prices.loc[buy_date,  coin]
                price_end   = self._prices.loc[sell_date, coin]
                quantity = weights[coin] * capital / price_start
                profit += self._get_returns(quantity, price_start, price_end)

            capital += profit
            count += (returns[i] == capital)
        
        self._report[strategy.name]['returns'] = {
            'is_correct': count == len(returns),
            'cnt_incorrect': len(returns) - count
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

    def _save_report(
        self,
        output_path: str,
        title: str = "📊 Portfolio Strategy Backtest Report"
    ):
        """
        Генерирует HTML-отчёт из данных ReportBuilder.

        :param output_path: Путь для сохранения HTML
        :param title: Заголовок отчёта
        """

        styling = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f9f9fc;
                color: #333;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: auto;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            h1 {
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.95em;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border: 1px solid #ddd;
                max-height: 60px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                vertical-align: top;
            }
            th {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            tr:nth-child(even) {
                background-color: #f2f7fb;
            }
            tr:hover {
                background-color: #e3f2fd;
            }
            .metric {
                background: #ecf4ff;
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                font-weight: 500;
            }
            .error {
                color: #e74c3c;
                font-weight: bold;
            }
            .success {
                color: #27ae60;
            }
            .section {
                margin-bottom: 40px;
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                color: #7f8c8d;
                font-size: 0.9em;
            }
            .table-container {
                max-height: 600px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            .table-container table {
                margin: 0;
                box-shadow: none;
            }
            .table-container thead th {
                position: sticky;
                top: 0;
                background-color: #3498db;
                color: white;
                z-index: 100;
            }
        </style>
        """

        # Начинаем собирать HTML
        html_parts = [f"<html><head><meta charset='utf-8'><title>{title}</title>{styling}</head><body>",
                      f'<div class="container"><h1>{title}</h1>']

        for strategy_name, metrics in self._report.items():
            html_parts.append(f'<div class="section"><h2>📈 Стратегия: {strategy_name}</h2>')

            # === Изменения весов (change_weights) ===
            if 'change_weights' in metrics:
                cw = metrics['change_weights']
                html_parts.append('<h3>🔄 Изменения весов</h3>')
                html_parts.append('<ul>')
                html_parts.append(
                    f"<li><b>Меняли минимальный вес:</b> {f'❌ Да {cw['min_w']} раз' if cw['min_w'] else '✅ Нет'}</li>")
                html_parts.append(
                    f"<li><b>Был ли вес больше 10% от дневного объёма:</b> {f'❌ Да {cw['volume_percent']} раз' if cw['volume_percent'] else '✅ Нет'}</li>")
                html_parts.append(
                    f"<li><b>Нормализовывали ли веса:</b> {f'❌ Да {cw['normalize']} раз' if cw['normalize'] else '✅ Нет'}</li>")
                html_parts.append('</ul>')

            cw = metrics['correct_weights']
            html_parts.append('<h3>🔍 Контрольные значения</h3>')
            html_parts.append(
                f"<li><b>Сумма весов больше единицы:</b> {f'❌ Да {cw['sum']} раз' if cw['sum'] else '✅ Нет'}</li>")
            html_parts.append(
                f"<li><b>Минимальный вес меньше {self._min_invest}$:</b> {f'❌ Да {cw['min_w']} раз' if cw['min_w'] else '✅ Нет'}</li>")
            html_parts.append(
                f"<li><b>Вес больше чем {self._volume_percent}% дневного объёма:</b> {f'❌ Да {cw['daily_volume']} раз' if cw['daily_volume'] else '✅ Нет'}</li>")

            for key in ['prices', 'volumes']:
                if key in metrics:
                    is_correct = metrics[key]['is_correct']
                    cnt = metrics[key]['cnt_incorrect']
                    status = "✅ Все значения корректны" if is_correct else f"❌ {cnt} некорректных значений"
                    color = "success" if is_correct else "error"
                    html_parts.append(
                        f'<div class="metric"><b>{key.title()}:</b> <span class="{color}">{status}</span></div>')

            if 'returns' in metrics:
                ret = metrics['returns']
                is_correct = ret['is_correct']
                status = "✅ Все значения совпадают" if is_correct else f"❌ {ret['cnt_incorrect']} несовпадений"
                color = "success" if is_correct else "error"
                html_parts.append(
                    f'<div class="metric"><b>Проверка капитала:</b> <span class="{color}">{status}</span></div>')

            if 'between_returns' in metrics:
                br = metrics['between_returns']
                is_correct = br['is_correct']
                status = "✅ Все месячные доходности совпадают" if is_correct else f"❌ {br['cnt_incorrect']} несовпадений"
                color = "success" if is_correct else "error"
                html_parts.append(
                    f'<div class="metric"><b>Месячная доходность:</b> <span class="{color}">{status}</span></div>')

            df_weights = DataFrame(metrics['stats_weights'])
            if not df_weights.empty:
                df_weights = df_weights.set_index('date')

                if 'list_weights' in metrics and metrics['list_weights']:
                    list_weights = metrics['list_weights']
                    expanded = []
                    for item in list_weights:
                        date = item['date']
                        weights = item['weights']
                        row = {'date': date, 'weights': weights}
                        expanded.append(row)

                    df_list_weights = DataFrame(expanded).set_index('date')
                    df_weights = df_weights.join(df_list_weights, how='left')

                if 'between_returns' in metrics and metrics['between_returns']:
                    br = metrics['between_returns']
                    df_br = DataFrame(br['between_returns'])
                    if len(df_br) > 0:
                        df_br = df_br.set_index('date')
                        br_columns = [col for col in df_br.columns if col != 'date']
                        df_weights = df_weights.join(df_br[br_columns], how='left')

                # Обёртываем таблицу в контейнер с фиксированной шапкой
                html_parts.append('<div class="table-container">')
                df_weights = df_weights.reset_index()
                html_parts.append(df_weights.to_html(classes='table'))
                html_parts.append('</div>')  # конец контейнера таблицы

            html_parts.append('</div>')  # конец секции стратегии

        # === Футер ===
        html_parts.append('<div class="footer">Сгенерировано автоматически | Portfolio Backtest Report</div>')
        html_parts.append('</div></body></html>')

        full_html = ''.join(html_parts)
        with open(get_unique_filepath(output_path), "w", encoding="utf-8") as f:
            f.write(full_html)