"""
RUN THE FULL 8-STEP RESEARCH PIPELINE
=====================================
    python run_research.py            # uses the data source in config.yaml
    python run_research.py --demo     # uses made-up data (tests your install, works offline)
    python run_research.py --refresh  # force a fresh download of prices

What it does:
  1. Universe selection      - which stocks are tradeable each day
  2. Feature engineering     - momentum and mean-reversion clues
  3. Time series CV          - walk-forward folds (train on past, test on future)
  4. ML modelling            - trains 3 models inside every fold
  5. Model evaluation        - compares them and picks the best
  6. Create signals          - top-N portfolios for momentum, mean reversion and ML
  7. Backtest                - replays history with costs
  8. Performance analysis    - metrics, charts and an HTML report in reports/

It also saves the chosen model (trained on all history) to models/model.joblib,
which run_live.py uses for paper/live trading.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import joblib
import pandas as pd

from quantbot import backtest as bt
from quantbot import cv, evaluation, features, models, performance, report, signals, universe
from quantbot.data import load_prices
from quantbot.utils import ROOT, banner, get_logger, load_config

STEPS = 8


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the 8-step research pipeline")
    ap.add_argument("--demo", action="store_true", help="use made-up demo data")
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--config", default=None, help="path to a different config file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.demo:
        cfg["data"]["source"] = "demo"
    log = get_logger(logfile="research.log")
    t0 = time.time()
    source = cfg["data"]["source"]
    bench_t = cfg["data"]["benchmark"]
    H = cfg["target"]["horizon_days"]

    # ------------------------------------------------------------------ 1
    banner(log, 1, STEPS, "Universe selection")
    prices = load_prices(cfg, refresh=args.refresh)
    close, volume = prices["close"], prices["volume"]
    if bench_t not in close.columns:
        raise SystemExit(f"Benchmark {bench_t} has no data - cannot continue.")
    mask = universe.universe_mask(close, volume, cfg)
    universe_text = universe.describe(mask)
    log.info(universe_text)

    # ------------------------------------------------------------------ 2
    banner(log, 2, STEPS, "Feature engineering")
    X, raw = features.build_features(close, volume, close[bench_t], mask, cfg)
    y, fwd = features.build_target(close, mask, H)
    X = X[features.ALL_FEATURES]
    log.info(f"{X.shape[0]:,} rows (stock-days) x {X.shape[1]} features")
    log.info(f"Target: beat the median stock over the next {H} days "
             f"(balance: {y.mean()*100:.1f}% are 1s)")

    # ------------------------------------------------------------------ 3
    banner(log, 3, STEPS, "Time series cross validation (walk-forward)")
    all_dates = X.index.get_level_values("date").unique().sort_values()
    folds = cv.walk_forward_folds(all_dates, cfg)
    holdout_start = next(f.test_dates[0] for f in folds if f.holdout)
    log.info(f"{len(folds)} folds; holdout starts {holdout_start.date()}")

    # ------------------------------------------------------------------ 4
    banner(log, 4, STEPS, "Machine learning models")
    model_names = cfg["models"]["candidates"]
    rs = cfg["models"].get("random_state", 42)
    preds, scores = models.walk_forward(X, y, fwd, folds, model_names, rs)

    # ------------------------------------------------------------------ 5
    banner(log, 5, STEPS, "Model evaluation")
    summary = evaluation.summarise(scores, preds, fwd, holdout_start)
    best = evaluation.choose_best(summary)
    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        log.info("\n" + summary.to_string(index=False))
    log.info(f"Best model on development folds: {models.MODEL_LABELS[best]}")
    dec = evaluation.decile_returns(preds[best], fwd)
    last_dev = [f for f in folds if not f.holdout][-1]
    d = X.index.get_level_values("date")
    imp_model = models.make_model(best, rs)
    tr = d.isin(last_dev.train_dates) & y.notna().to_numpy()
    imp_model.fit(X[tr], y[tr])
    te = d.isin(last_dev.test_dates)
    importance = evaluation.feature_importance(imp_model, X[te], y[te], random_state=rs)
    log.info("Top features: " + ", ".join(f"{k} ({v:+.4f})" for k, v in importance.head(5).items()))

    # ------------------------------------------------------------------ 6
    banner(log, 6, STEPS, "Create signals")
    oos_start = folds[0].test_dates[0]
    score_tables = signals.strategy_scores(raw, preds[best], cfg)
    vol = raw["vol_60"]
    targets = {k: signals.build_target_weights(s, mask, vol, cfg, start=oos_start)
               for k, s in score_tables.items()}
    for k, t in targets.items():
        log.info(f"  {signals.STRATEGY_LABELS[k]:<16} {len(t)} rebalances, "
                 f"avg {int((t.abs() > 0).sum(axis=1).mean())} positions")

    # ------------------------------------------------------------------ 7
    banner(log, 7, STEPS, "Backtest")
    end = close.index[-1]
    results = {k: bt.run_backtest(k, t, close, cfg, end=end) for k, t in targets.items()}
    common_start = max(r.returns.index[0] for r in results.values())
    for k, r in results.items():   # align so every strategy is judged over the same days
        r.returns = r.returns.loc[common_start:]
        r.gross_returns = r.gross_returns.loc[common_start:]
        r.equity = cfg["backtest"]["initial_capital"] * (1 + r.returns).cumprod()
    results["benchmark"] = bt.benchmark_result(close[bench_t], common_start, end, cfg)
    for k, r in results.items():
        log.info(f"  {k:<16} final value ${r.equity.iloc[-1]:,.0f}")

    # ------------------------------------------------------------------ 8
    banner(log, 8, STEPS, "Portfolio performance analysis")
    labels = {**signals.STRATEGY_LABELS, "benchmark": f"Benchmark ({bench_t})"}
    order = ["ml_combined", "momentum", "mean_reversion", "benchmark"]
    results = {k: results[k] for k in order if k in results}
    bret = results["benchmark"].returns
    rf = cfg["backtest"].get("cash_rate", 0.0)
    met = {k: performance.metrics(r, bret, rf) for k, r in results.items()}
    hold = {}
    for k, r in results.items():
        sub = bt.BacktestResult(k, r.returns.loc[holdout_start:], r.gross_returns.loc[holdout_start:],
                                r.equity.loc[holdout_start:], r.turnover.loc[holdout_start:]
                                if len(r.turnover) else r.turnover, pd.DataFrame())
        hold[k] = performance.metrics(sub, bret.loc[holdout_start:], rf)
    mt = pd.DataFrame(met)
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        log.info("\n" + mt.to_string())
    yearly = performance.yearly_returns(results)
    charts = {
        "equity": performance.chart_equity(results, labels, holdout_start),
        "drawdown": performance.chart_drawdown(results, labels),
        "yearly": performance.chart_yearly(yearly, labels),
        "deciles": performance.chart_deciles(dec),
        "importance": performance.chart_importance(importance),
    }

    # latest picks (what each strategy would hold today)
    latest = mask.index[-1]
    latest_picks = {}
    for k, s in score_tables.items():
        if latest in s.index:
            held = list(targets[k].columns[targets[k].iloc[-1] > 0]) if len(targets[k]) else []
            latest_picks[k] = signals.target_weights_for_day(s.loc[latest], mask.loc[latest],
                                                             vol.loc[latest], cfg, current=held)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    html = report.build_report({
        "labels": labels, "metrics": met, "holdout_metrics": hold, "live_strategy": cfg["live"]["strategy"],
        "eval_summary": summary, "model_labels": models.MODEL_LABELS, "best_model": best,
        "folds": folds, "latest_picks": {k: latest_picks[k] for k in order if k in latest_picks},
        "latest_date": latest.date(), "yearly": yearly, "charts": charts, "source": source,
        "n_stocks": mask.shape[1], "bt_start": common_start.date(), "bt_end": end.date(),
        "cost_bps": cfg["backtest"]["transaction_cost_bps"], "rebalance": cfg["portfolio"]["rebalance_every_days"],
        "top_n": cfg["portfolio"]["top_n"], "long_short": cfg["portfolio"]["long_short"],
        "weighting": cfg["portfolio"]["weighting"], "max_weight": cfg["portfolio"]["max_weight"],
        "lag": cfg["backtest"]["execution_lag_days"], "horizon": H, "universe_text": universe_text,
        "features": features.ALL_FEATURES, "demo": source == "demo",
    })
    out = ROOT / "reports" / f"report_{stamp}.html"
    out.write_text(html, encoding="utf-8")
    (ROOT / "reports" / "latest_report.html").write_text(html, encoding="utf-8")
    mt.to_csv(ROOT / "reports" / f"metrics_{stamp}.csv")
    scores.to_csv(ROOT / "reports" / f"fold_scores_{stamp}.csv", index=False)

    # final model on all history, for live trading
    log.info(f"Training final {models.MODEL_LABELS[best]} on all history for live trading ...")
    final = models.fit_final(best, X, y, rs)
    joblib.dump({"model": final, "model_name": best, "features": features.ALL_FEATURES,
                 "trained_until": str(y.dropna().index.get_level_values('date').max().date()),
                 "created": datetime.now().isoformat(timespec="seconds"), "data_source": source},
                ROOT / "models" / "model.joblib")
    log.info("Saved model to models/model.joblib")
    log.info(f"Report: {out}")
    log.info(f"Finished in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
