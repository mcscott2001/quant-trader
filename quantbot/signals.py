"""STEP 6 - Create signals (turn scores into a target portfolio).

Three strategies are built so you can compare them fairly:

  momentum        Buy the stocks with the strongest 12-month return
                  (skipping the last month). Classic "momentum factor".
  mean_reversion  Buy the stocks that have fallen furthest below their
                  20-day average (most "oversold") - optionally only those
                  still in a long-term uptrend, so we buy dips, not collapses.
  ml_combined     Buy the stocks the machine-learning model rates highest.
                  The model sees BOTH momentum and mean-reversion clues and
                  learns how to blend them.

On every rebalance day each strategy picks its top N stocks and assigns
weights (how much of the money goes into each).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_LABELS = {
    "ml_combined": "ML combined",
    "momentum": "Momentum factor",
    "mean_reversion": "Mean reversion",
}


def strategy_scores(raw: dict[str, pd.DataFrame], ml_pred: pd.Series | None,
                    cfg: dict) -> dict[str, pd.DataFrame]:
    """A score table (dates x tickers) per strategy. Higher score = more attractive."""
    scores = {
        "momentum": raw["mom_12_1"].copy(),
        "mean_reversion": -raw["zscore_20"],
    }
    if cfg["portfolio"].get("mean_reversion_trend_filter", True):
        scores["mean_reversion"] = scores["mean_reversion"].where(raw["trend_200"] > 0)
    if ml_pred is not None:
        scores["ml_combined"] = ml_pred.unstack("ticker").reindex(columns=raw["mom_12_1"].columns)
    return scores


def rebalance_dates(dates: pd.DatetimeIndex, every: int) -> pd.DatetimeIndex:
    return dates[::every]


def _cap(w: pd.Series, cap: float) -> pd.Series:
    """Limit each weight to `cap`, re-spreading the excess over the others."""
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def _pick(s: pd.Series, n: int, current: list[str], buffer: float) -> pd.Index:
    """Top-n names, but keep a name we already hold while it is still ranked
    within the top n*buffer. This "hold buffer" cuts pointless trading."""
    ranked = s.sort_values(ascending=False)
    keep_zone = set(ranked.index[: int(round(n * buffer))])
    kept = [t for t in ranked.index if t in current and t in keep_zone][:n]
    fill = [t for t in ranked.index if t not in kept][: n - len(kept)]
    return pd.Index(kept + fill)


def target_weights_for_day(score: pd.Series, tradeable: pd.Series, vol: pd.Series | None,
                           cfg: dict, current: list[str] | None = None) -> pd.Series:
    """Target portfolio for one day. Returns weights indexed by ticker (sum = 1 for long-only).
    `current` = tickers held now (long side), used by the hold buffer."""
    p = cfg["portfolio"]
    s = score[tradeable.reindex(score.index).fillna(False).astype(bool)].dropna()
    n = p["top_n"]
    buffer = float(p.get("hold_buffer", 1.0))
    current = current or []
    if len(s) < n:
        return pd.Series(dtype=float)

    def side(names: pd.Index) -> pd.Series:
        if p.get("weighting", "equal") == "inverse_vol" and vol is not None:
            iv = 1 / vol.reindex(names).replace(0, np.nan)
            iv = iv.fillna(iv.median() if iv.notna().any() else 1.0)
            w = iv / iv.sum()
        else:
            w = pd.Series(1.0 / len(names), index=names)
        return _cap(w, p["max_weight"])

    longs = side(_pick(s, n, current, buffer))
    if not p.get("long_short", False):
        return longs
    shorts = side(_pick(-s.drop(longs.index, errors="ignore"), n, [], 1.0))
    return pd.concat([0.5 * longs, -0.5 * shorts])   # dollar-neutral: 50% long, 50% short


def build_target_weights(score: pd.DataFrame, mask: pd.DataFrame, vol: pd.DataFrame,
                         cfg: dict, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Target weights on each rebalance day (rows = rebalance dates, columns = tickers)."""
    dates = score.dropna(how="all").index
    if start is not None:
        dates = dates[dates >= start]
    out, held = {}, []
    for d in rebalance_dates(dates, cfg["portfolio"]["rebalance_every_days"]):
        w = target_weights_for_day(score.loc[d], mask.loc[d], vol.loc[d] if vol is not None else None,
                                   cfg, current=held)
        held = list(w[w > 0].index)
        out[d] = w
    return pd.DataFrame(out).T.reindex(columns=score.columns).fillna(0.0).sort_index()
