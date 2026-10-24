"""STEP 2 - Feature engineering.

Turns raw prices into "clues" (features) that might predict future returns.
Every feature on date t uses ONLY data up to and including date t.

Momentum clues (winners tend to keep winning for a while):
  mom_12_1   return over the last 12 months, skipping the most recent month
  mom_6_1    same, 6 months
  mom_3_1    same, 3 months
  high_52w   how close the price is to its 52-week high
  trend_200  price vs its 200-day average
  sma50_200  50-day average vs 200-day average ("golden cross" style)

Mean-reversion clues (sharp short-term moves tend to partly reverse):
  ret_5d     last week's return
  ret_1m     last month's return
  zscore_20  how many standard deviations the price is from its 20-day average
  rsi_14     the classic Relative Strength Index (0-100)
  dist_low_20 distance above the 20-day low

Risk / activity clues:
  vol_20, vol_60   recent volatility (how bumpy the price is)
  volume_ratio     this week's trading volume vs normal

Market clues (same for every stock; helps the model adapt to the market mood):
  mkt_ret_20, mkt_vol_20
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_FEATURES = ["mom_12_1", "mom_6_1", "mom_3_1", "high_52w", "trend_200", "sma50_200"]
REVERSION_FEATURES = ["ret_5d", "ret_1m", "zscore_20", "rsi_14", "dist_low_20"]
RISK_FEATURES = ["vol_20", "vol_60", "volume_ratio"]
MARKET_FEATURES = ["mkt_ret_20", "mkt_vol_20"]
STOCK_FEATURES = MOMENTUM_FEATURES + REVERSION_FEATURES + RISK_FEATURES
ALL_FEATURES = STOCK_FEATURES + MARKET_FEATURES


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def raw_features(close: pd.DataFrame, volume: pd.DataFrame, bench: pd.Series) -> dict[str, pd.DataFrame]:
    """Each feature as a wide table (dates x tickers)."""
    ret1 = close.pct_change(fill_method=None)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    f: dict[str, pd.DataFrame] = {}
    # momentum
    f["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    f["mom_6_1"] = close.shift(21) / close.shift(126) - 1
    f["mom_3_1"] = close.shift(21) / close.shift(63) - 1
    f["high_52w"] = close / close.rolling(252, min_periods=200).max()
    f["trend_200"] = close / sma200 - 1
    f["sma50_200"] = sma50 / sma200 - 1
    # mean reversion
    f["ret_5d"] = close / close.shift(5) - 1
    f["ret_1m"] = close / close.shift(21) - 1
    f["zscore_20"] = (close - sma20) / close.rolling(20).std()
    f["rsi_14"] = _rsi(close, 14)
    f["dist_low_20"] = close / close.rolling(20).min() - 1
    # risk / activity
    f["vol_20"] = ret1.rolling(20).std() * np.sqrt(252)
    f["vol_60"] = ret1.rolling(60).std() * np.sqrt(252)
    f["volume_ratio"] = volume.rolling(5).mean() / volume.rolling(60).mean()
    # market mood (broadcast the benchmark to every column)
    bret = bench.pct_change(fill_method=None)
    mkt_ret = bench / bench.shift(20) - 1
    mkt_vol = bret.rolling(20).std() * np.sqrt(252)
    f["mkt_ret_20"] = pd.DataFrame({c: mkt_ret for c in close.columns}, index=close.index)
    f["mkt_vol_20"] = pd.DataFrame({c: mkt_vol for c in close.columns}, index=close.index)
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in f.items()}


def to_long(wide: dict[str, pd.DataFrame], mask: pd.DataFrame) -> pd.DataFrame:
    """Stack wide tables into one long table: one row per (date, ticker), one column per feature.
    Only rows where the stock is in the universe are kept."""
    dates, tickers = mask.index, mask.columns
    keep = mask.to_numpy().ravel()
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])[keep]
    cols = {name: df.reindex(index=dates, columns=tickers).to_numpy().ravel()[keep]
            for name, df in wide.items()}
    return pd.DataFrame(cols, index=idx)


def build_features(close: pd.DataFrame, volume: pd.DataFrame, bench: pd.Series,
                   mask: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Returns (ml_features_long, raw_features_wide).

    ml_features_long: features ready for machine learning. Stock features are
    converted to cross-sectional ranks (0 = lowest that day, 1 = highest) when
    rank_normalise is on, so the model compares stocks with each other rather
    than with their own past - this is much more stable over time."""
    stocks = list(mask.columns)
    raw = raw_features(close[stocks], volume[stocks], bench)
    ml = {}
    for name, df in raw.items():
        df = df.where(mask)
        if cfg["features"].get("rank_normalise", True) and name in STOCK_FEATURES:
            df = df.rank(axis=1, pct=True)
        ml[name] = df
    return to_long(ml, mask), raw


def build_target(close: pd.DataFrame, mask: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
    """What we try to predict.
    label   = 1 if the stock's return over the next `horizon` days beats the
              median stock in the universe that day, else 0
    fwd_ret = that forward return minus the universe average (for evaluation)"""
    stocks = list(mask.columns)
    fwd = close[stocks].shift(-horizon) / close[stocks] - 1
    fwd = fwd.where(mask)
    excess = fwd.sub(fwd.mean(axis=1), axis=0)
    label = (fwd.gt(fwd.median(axis=1), axis=0)).astype(float).where(fwd.notna())
    lab = to_long({"label": label}, mask)["label"]
    exc = to_long({"fwd": excess}, mask)["fwd"]
    return lab, exc
