"""STEP 8 - Analyse portfolio performance.

Plain-English meaning of each number:
  CAGR           average yearly growth rate ("compound annual growth rate")
  Volatility     how bumpy the ride is (yearly standard deviation of returns)
  Sharpe         return per unit of bumpiness. ~0.5 ok, ~1 good, >2 rare/suspicious
  Sortino        like Sharpe but only punishes downside bumps
  Max drawdown   worst fall from a peak to a low (e.g. -25% = lost a quarter at worst)
  Calmar         CAGR divided by max drawdown (higher = better recovery profile)
  Beta           how much it moves with the market (1 = same as market)
  Alpha          yearly return NOT explained by the market
  Turnover       how much of the portfolio is traded per year (2.0 = 200%)
  Cost drag      how much yearly return was lost to trading costs
"""
from __future__ import annotations

import io
import base64

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestResult  # noqa: E402

# colours (fixed order, colour-blind checked) - one per strategy, grey for the benchmark
COLORS = {"ml_combined": "#2a78d6", "momentum": "#eb6834", "mean_reversion": "#1baf7a",
          "benchmark": "#8a8984"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def metrics(res: BacktestResult, bench: pd.Series | None, rf: float = 0.0) -> dict:
    r = res.returns
    if len(r) < 20:
        return {}
    years = len(r) / 252
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    rf_d = (1 + rf) ** (1 / 252) - 1
    ex = r - rf_d
    sharpe = ex.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    down = r[r < 0].std() * np.sqrt(252)
    sortino = (ex.mean() * 252) / down if down > 0 else np.nan
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    monthly = (1 + r).resample("ME").prod() - 1
    out = {
        "Total return": eq.iloc[-1] - 1,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Positive months": (monthly > 0).mean(),
        "Best month": monthly.max(),
        "Worst month": monthly.min(),
    }
    if bench is not None and res.name != "benchmark":
        b = bench.reindex(r.index).fillna(0.0)
        cov = np.cov(r, b)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
        out["Beta"] = beta
        out["Alpha (yearly)"] = (r.mean() - beta * b.mean()) * 252
        out["Correlation"] = r.corr(b)
    if len(res.turnover):
        out["Turnover (yearly)"] = res.turnover.sum() / years
        g_eq = (1 + res.gross_returns.reindex(r.index).fillna(0)).cumprod()
        out["Cost drag (yearly)"] = g_eq.iloc[-1] ** (1 / years) - 1 - cagr
    return out


def yearly_returns(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({k: (1 + v.returns).groupby(v.returns.index.year).prod() - 1
                         for k, v in results.items()})


# ----------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------
def _style(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_equity(results: dict[str, BacktestResult], labels: dict[str, str],
                 holdout_start: pd.Timestamp | None) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    _style(ax, "Growth of $100,000 (log scale)")
    for k, res in results.items():
        ax.plot(res.equity.index, res.equity.values, color=COLORS.get(k, INK2), linewidth=2,
                label=labels.get(k, k), linestyle="--" if k == "benchmark" else "-")
    ax.set_yscale("log")
    lo = min(r.equity.min() for r in results.values())
    hi = max(r.equity.max() for r in results.values())
    nice = [m * 10 ** e for e in range(3, 9) for m in (1, 1.5, 2, 3, 5, 7.5)]
    ticks = [t for t in nice if lo * 0.9 <= t <= hi * 1.1]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"${v/1e6:,.1f}m" if v >= 1e6 else f"${v/1000:,.0f}k"))
    if holdout_start is not None:
        ax.axvspan(holdout_start, max(r.equity.index[-1] for r in results.values()),
                   color="#2a78d6", alpha=0.06, lw=0)
        ax.text(holdout_start, ax.get_ylim()[1], "  holdout (unseen)", fontsize=8.5,
                color=INK2, va="top")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    ax.margins(x=0.02)
    return _to_b64(fig)


def chart_drawdown(results: dict[str, BacktestResult], labels: dict[str, str]) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=SURFACE)
    _style(ax, "Drawdown - how far below its previous peak")
    for k, res in results.items():
        eq = res.equity
        ax.plot(eq.index, (eq / eq.cummax() - 1) * 100, color=COLORS.get(k, INK2), linewidth=1.5,
                label=labels.get(k, k), linestyle="--" if k == "benchmark" else "-")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), labelcolor=INK2, ncol=4)
    return _to_b64(fig)


def chart_yearly(yearly: pd.DataFrame, labels: dict[str, str]) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    _style(ax, "Return by calendar year")
    n = yearly.shape[1]
    width = 0.8 / n
    x = np.arange(len(yearly))
    for i, k in enumerate(yearly.columns):
        ax.bar(x + (i - (n - 1) / 2) * width, yearly[k] * 100, width * 0.9,
               color=COLORS.get(k, INK2), label=labels.get(k, k))
    ax.set_xticks(x, [str(y) for y in yearly.index])
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), labelcolor=INK2, ncol=4)
    return _to_b64(fig)


def chart_deciles(dec: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=SURFACE)
    _style(ax, "ML model: next-week excess return by prediction decile (1 = lowest, 10 = highest)")
    ax.bar(dec.index, dec.values * 100, color=COLORS["ml_combined"], width=0.7)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(dec.index)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    return _to_b64(fig)


def chart_importance(imp: pd.Series) -> str:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(imp) + 1), facecolor=SURFACE)
    _style(ax, "Which clues the model relies on (drop in AUC when the clue is scrambled)")
    ax.barh(imp.index, imp.values, color=COLORS["ml_combined"], height=0.6)
    ax.grid(axis="y", visible=False)
    return _to_b64(fig)
