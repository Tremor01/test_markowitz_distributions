import json
import os
from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from portfolio_strategies import StrategyBTC, MIN_RISK, MAX_RET, SHARP, SHARP_SHORT
from portfolio_strategies.main_strategy import Metrics


current_dir = os.path.dirname(__file__)
plots_dir = os.path.join(current_dir, "portfolio_plots")
os.makedirs(plots_dir, exist_ok=True)


def plot_metrics(strategies, file_name: str, reports: dict = None):
    metric_names = [metric for metric in Metrics]

    # Определяем начальные specs: по одному subplot на метрику + 1 для таблицы весов + 1 для таблицы отчетов
    specs = [[{"type": "table"}]]
    specs.extend([[{"type": "xy"}] for _ in range(len(metric_names))])
    specs.append([{"type": "table"}])  # Предпоследний subplot - таблица весов
    specs.append([{"type": "table"}])  # Последний subplot - таблица отчетов

    # Создаем фигуру с графиками и таблицами
    fig = make_subplots(
        rows=len(specs),
        cols=1,
        specs=specs,
        vertical_spacing=0.01,
        subplot_titles=["Статистика стратегий"] + metric_names + ["Веса активов", "Отчеты стратегий"],
    )

    dop_statistics = get_statsitsic_of_strategies(strategies)
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Стратегия</b>"] + [f"<b>{col}</b>" for col in dop_statistics.columns],
                font=dict(size=12, family="Arial Black"),
                fill_color="lightgrey",
                align="center"
            ),
            cells=dict(
                values=[dop_statistics.index] + [dop_statistics[col] for col in dop_statistics.columns],
                font=dict(size=11, family="Arial"),
                fill_color=[
                    ["lightgrey"] * len(dop_statistics),  # Заголовки строк
                    *[["white"] * len(dop_statistics) for _ in dop_statistics.columns]  # Данные
                ],
                align="center",
                height=30,
            ),
        ),
        row=1, col=1
    )

    trace_indices_by_row = {}
    traces_meta = []  # список словарей {index, name, row}

    current_row = 2
    for i, metric in enumerate(Metrics):
        row = current_row + i
        trace_indices_by_row[row] = []
        for strategy in strategies:
            trace_index = len(fig.data)  # индекс трейса до добавления
            traces_meta.append({'index': trace_index, 'name': strategy.name, 'row': row})
            trace_indices_by_row[row].append(trace_index)

            fig.add_trace(
                go.Scatter(
                    x=strategy.date_metrics, y=strategy.metrics[metric],
                    name=strategy.name, mode="lines+markers",
                    line=dict(dash="dot", color=strategy.color),
                    marker=dict(color=strategy.color),
                    hovertemplate="<b>%{fullData.name}</b><br>Month: %{x}<br>Value: %{y:.2f}",
                    legendgroup=strategy.name, showlegend=(i == 0)
                ), row=row, col=1,
            )

    # Добавляем кнопки для каждого subplot
    updatemenus = []
    for row, trace_indices in trace_indices_by_row.items():
        buttons = []
        for div in [1, 10, 100]:
            buttons.append(dict(
                label=f"/{div}",
                method="restyle",
                args=[
                    {"y": [
                        (list(map(lambda v: v / div, fig.data[idx].y)) if div != 1 else fig.data[idx].y)
                        for idx in trace_indices
                    ]},
                    trace_indices
                ]
            ))
        updatemenus.append(dict(
            buttons=buttons,
            direction="left",
            x=1.05,
            y=1 - (row - 1) * (1 / len(Metrics)),  # Разместим сбоку напротив каждого графика
            xanchor="left",
            yanchor="top"
        ))

    # --- Таблица весов ---
    table_data = []
    for strategy in strategies:
        for idx, weights_dict in enumerate(strategy.weights_history):
            non_zero_weights = {
                coin: f"{weight:.2%}"
                for coin, weight in weights_dict.items()
                if abs(weight) > 1e-4
            }

            assets_str = f"{strategy.rebalancing_dates[idx]}<br>" + "<br>".join([
                f"{coin}: {weight}" for coin, weight in non_zero_weights.items()
            ])

            table_data.append({"name": strategy.name, "color": strategy.color, "assets": assets_str})

    df_weights = pd.DataFrame(table_data)

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Стратегия</b>", "<b>Активы (вес)</b>"],
                font=dict(size=12, family="Arial Black"),
                fill_color="lightgrey",
            ),
            cells=dict(
                values=[df_weights["name"], df_weights["assets"]],
                font=dict(size=11, family="Arial", weight="bold"),
                fill_color=[
                    df_weights["color"].tolist(),
                    ["white"] * len(df_weights)
                ],
                line_color="darkslategray",
                align="left",
                height=30
            ),
        ),
        row=len(specs) - 1, col=1  # Предпоследняя строка
    )

    # --- Таблица отчетов ---
    if reports is not None:
        report_data = []
        for strategy in strategies:
            if strategy.name in reports:
                strategy_reports = reports[strategy.name]  # Это список словарей

                # Обрабатываем каждый отчет в списке
                reports_str = []
                for report in strategy_reports:
                    # Форматируем значения каждого отчета
                    formatted_report = {
                        k: f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
                        for k, v in report.items()
                    }
                    report_str = "<br>".join([f"{k}: {v}" for k, v in formatted_report.items()])
                    reports_str.append(report_str)

                # Объединяем все отчеты для стратегии с разделителем
                combined_reports = "<br><br>".join(reports_str)

                report_data.append({
                    "name": strategy.name,
                    "color": strategy.color,
                    "report": combined_reports
                })

        if report_data:  # Добавляем таблицу только если есть данные
            df_reports = pd.DataFrame(report_data)

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["<b>Стратегия</b>", "<b>Отчет</b>"],
                        font=dict(size=12, family="Arial Black"),
                        fill_color="lightgrey",
                    ),
                    cells=dict(
                        values=[df_reports["name"], df_reports["report"]],
                        font=dict(size=11, family="Arial", weight="bold"),
                        fill_color=[
                            df_reports["color"].tolist(),
                            ["white"] * len(df_reports)
                        ],
                        line_color="darkslategray",
                        align="left",
                        height=30
                    ),
                ),
                row=len(specs), col=1  # Последняя строка
            )

    # --- Обновляем layout ---
    fig.update_layout(
        title_text=f"Performance Metrics <br>",
        height=3500,
        showlegend=False,
        hovermode="x unified",
        margin=dict(t=100, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0.5,
            xanchor="center",
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(255,255,255,0)"
        ),
        updatemenus=updatemenus
    )

    filename = f"{file_name}.html"
    filepath = os.path.join(plots_dir, filename)

    trace_meta_json = json.dumps(traces_meta)
    metric_names_json = json.dumps(metric_names)
    strategy_names_json = json.dumps([s.name for s in strategies])

    inner_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    full_html = get_html_of_legend_sacle(strategies, inner_html, trace_meta_json, metric_names_json,
                                         strategy_names_json)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_html)

    return fig


def get_statsitsic_of_strategies(strategies):
    stats_list = []
    for strategy in strategies:
        returns = pd.Series(strategy.metrics[Metrics.total_returns_absolut])

        returns_pct = returns.pct_change(fill_method=None).dropna(how="all").mul(100) # Процентные изменения
        drawdowns = pd.Series(strategy.metrics[Metrics.max_draw_down_percent])

        strategy_stats = {
            'strategy': strategy.name,

            'avg_return': f'{returns_pct.mean():.2f}',
            'max_return': f'{returns_pct.max():.2f}',
            'min_return': f'{returns_pct.min():.2f}',
            'return_std': f'{returns_pct.std(ddof=1):.2f}',

            'avg_drawdown': f'{drawdowns.mean():.2f}',
            'max_drawdown': f'{drawdowns.min():.2f}',  # Most negative = max drawdown
            'drawdown_std': f'{drawdowns.std(ddof=1):.2f}',

            'max_return / max_drawdown': f'{abs(returns_pct.max() / drawdowns.min()):.2f}',
            'avg_return / avg_drawdown': f'{abs(returns_pct.mean() / drawdowns.mean()):.2f}',

            # Ratios
            'sharpe_ratio': f'{returns_pct.mean() / returns_pct.std() if returns_pct.std() > 0 else np.nan:.2f}',
            'sortino_ratio': f'{(returns_pct.mean() /
                              returns_pct[returns_pct < 0].std()) if (returns_pct < 0).any() else np.nan:.2f}',
            'calmar_ratio': f'{(returns_pct.mean() /
                             abs(drawdowns.min())) if drawdowns.min() < 0 else np.nan:.2f}',
            'median_return': f'{returns_pct.median():.2f}',
        }
        stats_list.append(strategy_stats)

    df_stats = pd.DataFrame(stats_list).set_index('strategy')
    return df_stats


def get_html_of_legend(strategies, inner_html):
    return f"""
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script> 
            <style>
                .fixed-legend {{
                    position: fixed;
                    top: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 90%;
                    z-index: 9999;
                    background: white;
                    padding: 10px;
                    border: 1px solid #ccc;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    overflow-x: auto;
                }}
                .legend-item {{
                    cursor: pointer;
                    display: inline-block;
                    margin-right: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="fixed-legend" id="custom-legend">
                <strong>Legend:</strong><br>
                {"".join(f'''
                    <span class="legend-item" onclick="toggleTrace('{strategy.name}')">
                        <span id="icon-{strategy.name}" style="color:{strategy.color}">●</span> {strategy.name}
                    </span>
                ''' for strategy in strategies)}
            </div>

            <div style="margin-top: 100px;">
                {inner_html}
            </div>

            <script>
                const strategyVisibility = {{
                    {"".join(f"'{strategy.name}': true," for strategy in strategies)}
                }};

                function toggleTrace(name) {{
                    const vis = !strategyVisibility[name];
                    strategyVisibility[name] = vis;

                    const icon = document.getElementById("icon-" + name);
                    icon.innerHTML = vis ? "●" : "○";

                    const gd = document.querySelector('.js-plotly-plot');
                    const traces = gd.data;
                    const indices = [];

                    for (let i = 0; i < traces.length; i++) {{
                        if (traces[i].name === name) {{
                            indices.push(i);
                        }}
                    }}

                    Plotly.restyle(gd, {{ visible: vis }}, indices);
                }}
            </script>
        </body>
        </html>
        """


def get_html_of_legend_sacle(strategies, inner_html, trace_meta_json, metric_names_json, strategy_names_json):
    return f"""
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script> 
            <style>
                .fixed-legend {{
                    position: fixed;
                    top: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 90%;
                    z-index: 9999;
                    background: white;
                    padding: 10px;
                    border: 1px solid #ccc;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    overflow-x: auto;
                }}
                .legend-item {{
                    cursor: pointer;
                    display: inline-block;
                    margin-right: 15px;
                }}
                .control-row {{
                    margin-top: 8px;
                }}
                .control-row label {{ margin-right: 8px; }}
            </style>
        </head>
        <body>
            <div class="fixed-legend" id="custom-legend">
                <strong>Legend:</strong><br>
                {"".join(f'''
                    <span class="legend-item" onclick="toggleTrace('{strategy.name}')">
                        <span id="icon-{strategy.name}" style="color:{strategy.color}">●</span> {strategy.name}
                    </span>
                ''' for strategy in strategies)}
                <div class="control-row" id="global-controls"></div>
            </div>

            <div style="margin-top: 140px;">
                {inner_html}
            </div>

            <script>
                const strategyVisibility = {{
                    {"".join(f"'{strategy.name}': true," for strategy in strategies)}
                }};
                function toggleTrace(name) {{
                    const vis = !strategyVisibility[name];
                    strategyVisibility[name] = vis;
                    const icon = document.getElementById("icon-" + name);
                    icon.innerHTML = vis ? "●" : "○";
                    const gd = document.querySelector('.js-plotly-plot');
                    const traces = gd.data;
                    const indices = [];
                    for (let i = 0; i < traces.length; i++) {{
                        if (traces[i].name === name) {{
                            indices.push(i);
                        }}
                    }}
                    Plotly.restyle(gd, {{ visible: vis }}, indices);
                }}

                // Метаданные, переданные из Python
                const traces_meta = {trace_meta_json};
                const metrics = {metric_names_json};
                const strategy_names = {strategy_names_json};

                // Контрольные переменные
                let selectedTraceIndex = null;
                const originalYs = {{}};

                function ensureOriginalYs(gd) {{
                    for (let i = 0; i < gd.data.length; i++) {{
                        if (!(i in originalYs)) {{
                            // копируем в обычный массив
                            originalYs[i] = Array.from(gd.data[i].y || []);
                        }}
                    }}
                }}

                function applyDivisorToIndices(indices, divisor) {{
                    const gd = document.querySelector('.js-plotly-plot');
                    if (!gd) return;
                    if (!indices || indices.length === 0) {{
                        alert('No traces selected for operation');
                        return;
                    }}
                    ensureOriginalYs(gd);
                    const newYs = indices.map(idx => originalYs[idx].map(v => (v === null ? null : v / divisor)));
                    Plotly.restyle(gd, {{ 'y': newYs }}, indices);
                }}

                function resetIndices(indices) {{
                    const gd = document.querySelector('.js-plotly-plot');
                    if (!gd) return;
                    ensureOriginalYs(gd);
                    const resetYs = indices.map(i => originalYs[i]);
                    Plotly.restyle(gd, {{ 'y': resetYs }}, indices);
                }}

                document.addEventListener('DOMContentLoaded', function() {{
                    const gd = document.querySelector('.js-plotly-plot');
                    if (!gd) return;

                    // Строим панель управления
                    const controls = document.getElementById('global-controls');
                    controls.innerHTML = `
                        <label>Metric:
                            <select id="metric-select">
                                ${{metrics.map((m,i) => `<option value="${{i+1}}">${{m}}</option>`).join('')}}
                            </select>
                        </label>
                        <label>Strategy:
                            <select id="strategy-select">
                                ${{strategy_names.map(n => `<option value="${{n}}">${{n}}</option>`).join('')}}
                            </select>
                        </label>
                        <label>Divisor:
                            <input id="div-input" type="number" min="0.000001" step="0.1" value="1" style="width:90px" />
                        </label>
                        <button id="apply-metric">Apply to metric+strategy</button>
                        <button id="apply-selected">Apply to selected trace</button>
                        <button id="reset-all">Reset all</button>
                        <div id="selected-trace" style="margin-top:6px">Selected: none</div>
                    `;

                    // Клик по точке — выбираем трейc
                    gd.on('plotly_click', function(evt) {{
                        selectedTraceIndex = evt.points[0].curveNumber;
                        document.getElementById('selected-trace').innerText = 'Selected: ' + gd.data[selectedTraceIndex].name + ' (trace ' + selectedTraceIndex + ')';
                    }});

                    // Кнопки
                    document.getElementById('apply-metric').addEventListener('click', function() {{
                        const row = parseInt(document.getElementById('metric-select').value, 10);
                        const strategy = document.getElementById('strategy-select').value;
                        const divisor = parseFloat(document.getElementById('div-input').value) || 1;
                        const indices = traces_meta.filter(t => t.row === row && t.name === strategy).map(t => t.index);
                        applyDivisorToIndices(indices, divisor);
                    }});

                    document.getElementById('apply-selected').addEventListener('click', function() {{
                        const divisor = parseFloat(document.getElementById('div-input').value) || 1;
                        if (selectedTraceIndex === null) {{ alert('Select a trace on the plot first (click a point).'); return; }}
                        applyDivisorToIndices([selectedTraceIndex], divisor);
                    }});

                    document.getElementById('reset-all').addEventListener('click', function() {{
                        const all_indices = traces_meta.map(t => t.index);
                        resetIndices(all_indices);
                    }});
                }});
            </script>
        </body>
        </html>
        """


def plot_stat_columns(data, colors, sim, cur_tickers):
    fig = make_subplots(
        rows=len(data), cols=1,
        vertical_spacing=0.03,
        subplot_titles = [row_data['metric_name'] for row_data in data]
    )

    for row, row_data in enumerate(data, 1):
        for i, name in enumerate(row_data['strategy_names']):
            y_values = []
            for group in row_data['values']:
                if i < len(group):
                    y_values.append(group[i])
                else:
                    y_values.append(0)

            fig.add_trace(go.Bar(
                x=row_data['train_periods'],
                y=y_values,
                name=f"{name}",
                marker_color=colors[name],
                legendgroup=name,
                showlegend=(row == 1)
            ), row=row, col=1)

    fig.update_layout(
        height=3000,
        width=1000,
        title_text=f"Statistics {' '.join(cur_tickers)}",
        barmode='group',
        bargap=0.3,
        bargroupgap=0,
        legend_title="Легенда",
        legend=dict(
            orientation="h",  # Горизонтальная легенда
            yanchor="bottom",
            y=-0.3,  # Размещаем под графиком
            xanchor="center",
            x=0.5
        )
    )

    filename = f"std_{sim}.html"
    filepath = os.path.join(plots_dir, filename)

    fig.write_html(filepath)


def plot_coin_prices_interactive(df_prices, file_name="coin_prices.html"):
    coins = df_prices.columns.tolist()
    n_coins = len(coins)

    cols = 2
    rows = (n_coins + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,
        vertical_spacing=0.15 / rows,
        horizontal_spacing=0.05,
        subplot_titles=coins
    )

    for i, coin in enumerate(coins):
        row = (i // cols) + 1
        col = (i % cols) + 1

        fig.add_trace(
            go.Scatter(
                x=df_prices.index,
                y=df_prices[coin],
                mode='lines',
                name=coin,
                hovertemplate=f"<b>{coin}</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>"
            ),
            row=row, col=col
        )

    # Обновляем layout
    fig.update_layout(
        title_text="Cryptocurrency Prices Over Time",
        height=300 * rows,
        showlegend=False,
        hovermode="x",
        margin=dict(t=80, l=50, r=50, b=50),
        template="plotly_white"
    )

    for i in range(1, rows * cols + 1):
        fig.update_xaxes(matches=None, row=((i-1)//cols)+1, col=((i-1)%cols)+1)

    # Сохраняем
    filepath = os.path.join(plots_dir, file_name)
    fig.write_html(filepath, include_plotlyjs='cdn')

    return fig


colors = {
    s.name: s.color for s in deepcopy([StrategyBTC()] + MIN_RISK + MAX_RET + SHARP + SHARP_SHORT)
}

def plot_month_statistics(statistics, file_name):
    # Создаем subplot для каждой метрики
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[metric for metric in statistics],
        vertical_spacing=0.1
    )

    # Для каждой метрики добавляем trace
    for i, metric in enumerate(statistics.keys()):
        for j, strategy_name in enumerate(statistics[metric].keys()):
            data = statistics[metric][strategy_name]
            fig.add_trace(
                go.Scatter(
                    legendgroup=strategy_name,
                    x=list(range(len(data))),
                    y=data,
                    name=strategy_name,
                    line=dict(color=colors[strategy_name]),
                    showlegend=(i == 0)  # Показывать легенду только для первого графика
                ),
                row=i + 1, col=1
            )

    # Настраиваем layout
    fig.update_layout(
        height=1200,
        width=1000,
        title_text="Strategy Metrics Over Time",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Настраиваем оси для каждого subplot
    for i, metric in enumerate(statistics.keys()):
        fig.update_yaxes(title_text=metric, row=i + 1, col=1)
        fig.update_xaxes(title_text="Month", row=i + 1, col=1)

    filepath = os.path.join(plots_dir, file_name)
    fig.write_html(filepath, include_plotlyjs='cdn')

    return fig