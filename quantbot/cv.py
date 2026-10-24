"""STEP 3 - Time series cross validation.

Normal cross validation shuffles data randomly. With markets that is cheating:
the model would "see the future". Instead we use WALK-FORWARD validation:

   train on years 1-3  -> test on the next 6 months
   train on years 1-3.5 -> test on the next 6 months
   ... and so on, always training only on the past.

Two extra protections:
  * PURGE: our target looks `horizon` days ahead, so the last `horizon` days
    before each test block are removed from training (their answers overlap
    the test period).
  * HOLDOUT: the final ~year is never used to choose between models. It is a
    "final exam" that tells you how the chosen model does on truly unseen data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    number: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    holdout: bool

    def __str__(self) -> str:
        tag = "HOLDOUT" if self.holdout else "dev"
        return (f"fold {self.number:>2} [{tag:>7}] train {self.train_dates[0].date()} -> "
                f"{self.train_dates[-1].date()} ({len(self.train_dates)} days) | test "
                f"{self.test_dates[0].date()} -> {self.test_dates[-1].date()}")


def walk_forward_folds(dates: pd.DatetimeIndex, cfg: dict) -> list[Fold]:
    c = cfg["cv"]
    purge = cfg["target"]["horizon_days"]
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    n = len(dates)
    min_train, block, holdout = c["min_train_days"], c["test_block_days"], c["holdout_days"]
    if n < min_train + holdout + block // 2:
        raise ValueError(f"Not enough history: have {n} days, need about "
                         f"{min_train + holdout + block // 2}. Use an earlier start_date "
                         f"or reduce cv.min_train_days / cv.holdout_days.")
    holdout_start = n - holdout

    starts = list(range(min_train, holdout_start, block)) + list(range(holdout_start, n, block))
    folds = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else n
        e = min(e, holdout_start) if s < holdout_start else e
        if e - s < 5:
            continue
        folds.append(Fold(number=len(folds) + 1,
                          train_dates=dates[: s - purge],
                          test_dates=dates[s:e],
                          holdout=s >= holdout_start))
    return folds
