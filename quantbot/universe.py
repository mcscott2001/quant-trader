"""STEP 1 - Universe selection.

Decides, for every single day, which stocks are "tradeable". A stock is in the
universe on a given day only if, using information available up to that day:
  * its price is above a minimum (avoids penny stocks)
  * it trades enough money per day (so our orders don't move the price)
  * it has enough history to calculate the features

The result is a True/False table: rows = dates, columns = tickers.

Note: the candidate list is today's big companies. Back-testing only on
today's winners flatters results a little ("survivorship bias"); treat the
backtest as an optimistic estimate.
"""
from __future__ import annotations

import pandas as pd


def universe_mask(close: pd.DataFrame, volume: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    u = cfg["universe"]
    stocks = [t for t in u["candidates"] if t in close.columns]
    close, volume = close[stocks], volume[stocks]

    dollar_volume = (close * volume).rolling(60, min_periods=40).median()
    history = close.notna().cumsum()

    mask = (close > u["min_price"]) & (dollar_volume > u["min_dollar_volume"]) & \
           (history >= u["min_history_days"])
    return mask.fillna(False)


def describe(mask: pd.DataFrame) -> str:
    counts = mask.sum(axis=1)
    counts = counts[counts > 0]
    if counts.empty:
        return "No stocks passed the filters!"
    return (f"{mask.shape[1]} candidates; on a typical day {int(counts.median())} pass the filters "
            f"(min {int(counts.min())}, max {int(counts.max())}). "
            f"First usable date: {counts.index[0].date()}")
