"""STEP 4 - Machine learning models.

Three models of increasing complexity are trained inside the walk-forward
folds, so every prediction is made on data the model has never seen:

  logistic          - a simple, robust straight-line model (the baseline)
  random_forest     - many decision trees voting together
  gradient_boosting - trees built one after another, each fixing the last one's mistakes

Each model outputs a probability: "how likely is this stock to beat the
median stock over the next week?"  0.5 = no idea; higher = more confident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cv import Fold
from .utils import get_logger

log = get_logger()

MODEL_LABELS = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}


def make_model(name: str, random_state: int = 42):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=2000))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=200, max_depth=6,
                                                    min_samples_leaf=300, max_features="sqrt",
                                                    n_jobs=-1, random_state=random_state))
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_depth=3,
                                              min_samples_leaf=400, l2_regularization=1.0,
                                              random_state=random_state)
    raise ValueError(f"Unknown model '{name}'")


# ----------------------------------------------------------------------------
# evaluation helpers (used here and in evaluation.py)
# ----------------------------------------------------------------------------
def daily_rank_ic(pred: pd.Series, fwd: pd.Series) -> pd.Series:
    """Information Coefficient: each day, the rank correlation between what the
    model predicted and what actually happened. 0 = useless, 0.02-0.05 is
    typical for a decent stock-picking signal, above 0.1 is suspicious."""
    df = pd.DataFrame({"p": pred, "f": fwd}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    r = df.groupby(level="date").rank()
    r["pf"] = r["p"] * r["f"]
    g = r.groupby(level="date")
    n = g.size()
    cov = g["pf"].mean() - g["p"].mean() * g["f"].mean()
    ic = cov / (g["p"].std(ddof=0) * g["f"].std(ddof=0))
    return ic[n >= 5].dropna()


def top_bottom_spread(pred: pd.Series, fwd: pd.Series, q: float = 0.2) -> float:
    """Average forward return of the top 20% predictions minus the bottom 20%."""
    df = pd.DataFrame({"p": pred, "f": fwd}).dropna()
    if df.empty:
        return np.nan
    pct = df.groupby(level="date")["p"].rank(pct=True)
    top = df["f"][pct >= 1 - q].groupby(level="date").mean()
    bot = df["f"][pct <= q].groupby(level="date").mean()
    return float((top - bot).mean())


def fold_scores(y: pd.Series, p: pd.Series, fwd: pd.Series) -> dict:
    ok = y.notna() & p.notna()
    yy, pp = y[ok], p[ok]
    out = {"rows": int(ok.sum())}
    out["auc"] = roc_auc_score(yy, pp) if yy.nunique() == 2 else np.nan
    out["accuracy"] = accuracy_score(yy, pp > 0.5) if len(yy) else np.nan
    ic = daily_rank_ic(p, fwd)
    out["ic"] = float(ic.mean()) if len(ic) else np.nan
    out["spread"] = top_bottom_spread(p, fwd)
    return out


# ----------------------------------------------------------------------------
# walk-forward training
# ----------------------------------------------------------------------------
def walk_forward(X: pd.DataFrame, y: pd.Series, fwd: pd.Series, folds: list[Fold],
                 model_names: list[str], random_state: int = 42
                 ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Train every model on every fold. Returns:
       predictions: {model_name: out-of-sample probability for every (date, ticker)}
       scores:      one row per (model, fold) with AUC, accuracy, IC, spread"""
    dates = X.index.get_level_values("date")
    preds = {m: [] for m in model_names}
    rows = []
    for fold in folds:
        tr = dates.isin(fold.train_dates) & y.notna().to_numpy()
        te = dates.isin(fold.test_dates)
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        log.info(f"  {fold}  ({tr.sum():,} training rows)")
        for m in model_names:
            model = make_model(m, random_state)
            model.fit(Xtr, ytr)
            p = pd.Series(model.predict_proba(Xte)[:, 1], index=Xte.index, name=m)
            preds[m].append(p)
            s = fold_scores(y[te], p, fwd[te])
            rows.append({"model": m, "fold": fold.number, "holdout": fold.holdout,
                         "test_start": fold.test_dates[0].date(), "test_end": fold.test_dates[-1].date(),
                         **s})
            log.info(f"      {MODEL_LABELS[m]:<20} AUC {s['auc']:.3f} | IC {s['ic']:+.4f} | "
                     f"top-bottom weekly spread {s['spread']*100:+.2f}%")
    preds = {m: pd.concat(v).sort_index() for m, v in preds.items()}
    return preds, pd.DataFrame(rows)


def fit_final(name: str, X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Train the chosen model on ALL labelled history - this is the model used for live trading."""
    ok = y.notna()
    model = make_model(name, random_state)
    model.fit(X[ok], y[ok])
    return model
