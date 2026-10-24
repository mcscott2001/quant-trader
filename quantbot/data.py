"""Getting daily price and volume history.

Two sources are supported (choose in config.yaml -> data.source):
  * yfinance - free Yahoo Finance data, no account needed
  * demo     - realistic-looking made-up prices, for testing the install offline
(Trading 212's API does not provide price history, so it is only used for trading.)

Whatever the source, you get back a dictionary with two tables:
  prices["close"]  - adjusted closing prices   (rows = dates, columns = tickers)
  prices["volume"] - shares traded each day    (same shape)
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .utils import ROOT, get_logger

log = get_logger()


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------
def load_prices(cfg: dict, source: str | None = None, start: str | None = None,
                refresh: bool = False) -> dict[str, pd.DataFrame]:
    source = source or cfg["data"]["source"]
    start = start or cfg["data"]["start_date"]
    tickers = list(dict.fromkeys(cfg["universe"]["candidates"] + [cfg["data"]["benchmark"]]))

    cache = ROOT / cfg["data"]["cache_dir"]
    cache.mkdir(exist_ok=True)
    close_file = cache / f"{source}_close.csv"
    vol_file = cache / f"{source}_volume.csv"

    if not refresh and close_file.exists() and vol_file.exists():
        age_h = (time.time() - close_file.stat().st_mtime) / 3600
        cached_close = pd.read_csv(close_file, index_col=0, parse_dates=True)
        same_tickers = set(tickers).issubset(cached_close.columns)
        same_start = cached_close.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=7)
        if age_h < cfg["data"].get("cache_max_age_hours", 12) and same_tickers and same_start:
            log.info(f"Using saved {source} prices from {close_file.name} ({age_h:.1f} hours old)")
            volume = pd.read_csv(vol_file, index_col=0, parse_dates=True)
            return _tidy({"close": cached_close[tickers], "volume": volume[tickers]})

    log.info(f"Downloading daily prices for {len(tickers)} tickers from '{source}' since {start} ...")
    if source == "yfinance":
        prices = _from_yfinance(tickers, start)
    elif source == "demo":
        prices = make_demo_data(tickers, start, seed=cfg["models"].get("random_state", 42))
    else:
        raise ValueError(f"Unknown data source '{source}'. Use yfinance or demo.")

    prices = _tidy(prices)
    prices["close"].to_csv(close_file)
    prices["volume"].to_csv(vol_file)
    missing = [t for t in tickers if t not in prices["close"].columns or prices["close"][t].notna().sum() == 0]
    if missing:
        log.warning(f"No data for: {', '.join(missing)} (they will be skipped)")
    log.info(f"Got {prices['close'].shape[0]} trading days, "
             f"{prices['close'].index.min().date()} to {prices['close'].index.max().date()}")
    return prices


def _tidy(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = prices["close"].sort_index()
    close = close[~close.index.duplicated(keep="last")]
    close = close.dropna(how="all")
    volume = prices["volume"].reindex(index=close.index, columns=close.columns)
    close.index.name = volume.index.name = "date"
    return {"close": close.astype(float), "volume": volume.astype(float)}


# ----------------------------------------------------------------------------
# Yahoo Finance
# ----------------------------------------------------------------------------
def _from_yfinance(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Check your internet connection, "
                           "or try again later (Yahoo sometimes limits requests).")
    if isinstance(raw.columns, pd.MultiIndex):
        close, volume = raw["Close"], raw["Volume"]
    else:  # only one ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = close.index
    return {"close": close, "volume": volume}


# ----------------------------------------------------------------------------
# Demo data (fake but realistic, with momentum and mean-reversion baked in)
# ----------------------------------------------------------------------------
def make_demo_data(tickers: list[str], start: str, end: str | None = None,
                   seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end = end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    dates = pd.bdate_range(start, end)
    n_days, n = len(dates), len(tickers)

    mkt = rng.normal(0.0004, 0.011, n_days)               # market's daily return
    beta = rng.uniform(0.7, 1.3, n)
    # slowly-changing "hidden quality" -> creates momentum (trends persist)
    drift = np.zeros((n_days, n))
    d = rng.normal(0, 0.0004, n)
    for i in range(n_days):
        d = 0.996 * d + rng.normal(0, 0.00003, n)
        drift[i] = d
    shocks = rng.normal(0, 0.015, (n_days, n))
    # short-term overreaction -> part of the last 5 days' moves reverses (mean reversion)
    past5 = pd.DataFrame(shocks).rolling(5, min_periods=1).mean().shift(1).fillna(0).to_numpy()
    rets = beta * mkt[:, None] + drift + shocks - 0.08 * past5
    close = 100 * rng.uniform(0.3, 2.0, n) * np.exp(np.cumsum(np.log1p(rets), axis=0))
    volume = rng.lognormal(np.log(5e6), 0.4, (n_days, n)) * (1 + 20 * np.abs(shocks))

    close = pd.DataFrame(close, index=dates, columns=tickers)
    volume = pd.DataFrame(volume, index=dates, columns=tickers)
    bench = tickers[-1]
    close[bench] = 300 * np.exp(np.cumsum(np.log1p(mkt)))
    volume[bench] = 8e7
    return {"close": close, "volume": volume}


# ----------------------------------------------------------------------------
# Exchange rate (for converting US dollar prices into your account currency)
# ----------------------------------------------------------------------------
def usd_to(currency: str) -> float:
    """How many units of `currency` one US dollar buys, from Yahoo Finance (latest close)."""
    if currency.upper() == "USD":
        return 1.0
    import yfinance as yf

    pair = f"{currency.upper()}USD=X"            # e.g. GBPUSD=X = dollars per pound
    raw = yf.download(pair, period="10d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Could not download the {pair} exchange rate from Yahoo Finance.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rate = float(close.dropna().iloc[-1])
    return 1.0 / rate
