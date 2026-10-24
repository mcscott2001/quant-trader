"""STEP 7 - Backtest.

Replays history day by day as if we had really traded:
  * the signal is calculated at the close on the rebalance day
  * we trade `execution_lag_days` later (default: the next day's close), so we
    never use a price we could not actually have traded at
  * between rebalances, positions drift with the market (winners grow)
  * every trade pays `transaction_cost_bps` (commission + slippage)
  * any cash left over earns `cash_rate`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series          # daily net returns
    gross_returns: pd.Series    # before costs
    equity: pd.Series           # account value over time
    turnover: pd.Series         # fraction of portfolio traded on each trade day
    weights: pd.DataFrame       # end-of-day holdings (as weights)


def run_backtest(name: str, targets: pd.DataFrame, close: pd.DataFrame, cfg: dict,
                 end: pd.Timestamp | None = None) -> BacktestResult:
    b = cfg["backtest"]
    lag = int(b.get("execution_lag_days", 1))
    cost = b["transaction_cost_bps"] / 1e4
    cash_daily = (1 + b.get("cash_rate", 0.0)) ** (1 / 252) - 1

    tickers = list(targets.columns)
    px = close[tickers]
    rets = px.pct_change(fill_method=None).fillna(0.0)
    dates = px.index[px.index >= targets.index[0]]
    if end is not None:
        dates = dates[dates <= end]
    rets = rets.loc[dates].to_numpy()

    # when does each rebalance actually get executed?
    pos = {d: i for i, d in enumerate(dates)}
    exec_at = {}
    for d in targets.index:
        if d in pos and pos[d] + lag < len(dates):
            exec_at[pos[d] + lag] = targets.loc[d].to_numpy()

    w = np.zeros(len(tickers))
    net, gross, turn, hist = [], [], {}, []
    for i in range(len(dates)):
        r = rets[i]
        cash_w = 1.0 - w.sum()
        g = float(w @ r + cash_w * cash_daily)
        # let positions drift with today's returns
        if i > 0:
            w = w * (1 + r) / (1 + g)
        c = 0.0
        if i in exec_at:
            new = exec_at[i]
            t = float(np.abs(new - w).sum())
            c = t * cost
            turn[dates[i]] = t
            w = new.copy()
        gross.append(g)
        net.append((1 + g) * (1 - c) - 1)
        hist.append(w.copy())

    idx = pd.DatetimeIndex(dates)
    net = pd.Series(net, index=idx, name=name)
    # drop the warm-up days before the first trade
    first = min(exec_at) if exec_at else 0
    net = net.iloc[first:]
    eq = b["initial_capital"] * (1 + net).cumprod()
    return BacktestResult(
        name=name,
        returns=net,
        gross_returns=pd.Series(gross, index=idx).iloc[first:],
        equity=eq,
        turnover=pd.Series(turn),
        weights=pd.DataFrame(hist, index=idx, columns=tickers).iloc[first:],
    )


def benchmark_result(close: pd.Series, start: pd.Timestamp, end: pd.Timestamp, cfg: dict) -> BacktestResult:
    s = close.loc[start:end]
    r = s.pct_change(fill_method=None).fillna(0.0)
    r.iloc[0] = 0.0
    eq = cfg["backtest"]["initial_capital"] * (1 + r).cumprod()
    return BacktestResult(name="benchmark", returns=r, gross_returns=r, equity=eq,
                          turnover=pd.Series(dtype=float), weights=pd.DataFrame())
