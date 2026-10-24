"""STEP 5 - Model evaluation.

Compares the models using ONLY the development folds (never the holdout),
picks the winner, and explains what it learned.

Measures (higher is better for all of them):
  AUC       0.50 = coin flip. Anything reliably above ~0.51-0.52 is useful in
            stock picking, because markets are very noisy.
  IC        Information Coefficient - see models.daily_rank_ic.
  IC t-stat How consistent the IC is. Above ~2 suggests it's not just luck.
  Spread    Weekly return of top-20% picks minus bottom-20% picks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .models import daily_rank_ic


def summarise(scores: pd.DataFrame, preds: dict[str, pd.Series], fwd: pd.Series,
              holdout_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for m, p in preds.items():
        d = p.index.get_level_values("date")
        for period, sel in (("development", d < holdout_start), ("holdout", d >= holdout_start)):
            ic = daily_rank_ic(p[sel], fwd.reindex(p.index)[sel])
            sc = scores[(scores.model == m) & (scores.holdout == (period == "holdout"))]
            rows.append({
                "model": m, "period": period,
                "folds": len(sc),
                "auc": sc["auc"].mean(),
                "accuracy": sc["accuracy"].mean(),
                "ic": ic.mean(),
                "ic_tstat": ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) > 2 else np.nan,
                "ic_positive_days": (ic > 0).mean(),
                "weekly_spread": sc["spread"].mean(),
            })
    return pd.DataFrame(rows)


def choose_best(summary: pd.DataFrame) -> str:
    dev = summary[summary.period == "development"].sort_values("ic", ascending=False)
    return str(dev.iloc[0]["model"])


def decile_returns(pred: pd.Series, fwd: pd.Series) -> pd.Series:
    """Average next-week excess return for each tenth of predictions (1 = lowest, 10 = highest).
    A good model shows a staircase going up from left to right."""
    df = pd.DataFrame({"p": pred, "f": fwd.reindex(pred.index)}).dropna()
    pct = df.groupby(level="date")["p"].rank(pct=True)
    dec = np.ceil(pct * 10).clip(1, 10).astype(int)
    return df["f"].groupby(dec.to_numpy()).mean()


def feature_importance(model, X: pd.DataFrame, y: pd.Series, max_rows: int = 20000,
                       random_state: int = 42) -> pd.Series:
    """How much worse does the model get if we scramble one feature? Bigger drop = more important."""
    ok = y.notna()
    Xs, ys = X[ok], y[ok]
    if len(Xs) > max_rows:
        Xs = Xs.sample(max_rows, random_state=random_state)
        ys = ys.loc[Xs.index]
    r = permutation_importance(model, Xs, ys, scoring="roc_auc", n_repeats=3,
                               random_state=random_state, n_jobs=1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
