"""
TRADE THE STRATEGY THROUGH TRADING 212
======================================
    python run_live.py              # DRY RUN: shows the orders it WOULD place, sends nothing
    python run_live.py --execute    # actually sends the orders to Trading 212

Safety features:
  * Dry run unless you add --execute
  * config.yaml live.mode: demo trades the Practice account (pretend money).
    mode: live only sends orders if live.allow_live_account is also true
  * Refuses to --execute outside US market hours (orders would sit in a queue)
  * Only touches stocks in the strategy's candidate list; anything else
    in the account is left alone
  * Keeps a cash buffer, can be capped with live.max_capital, skips tiny trades

Run run_research.py first - it creates models/model.joblib.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import joblib
import pandas as pd

from quantbot import broker, features, signals, universe
from quantbot.data import load_prices, usd_to
from quantbot.utils import ROOT, get_logger, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebalance the Trading 212 account to the strategy's target")
    ap.add_argument("--execute", action="store_true", help="really send orders (default: dry run)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger(logfile="live.log")
    live = cfg["live"]
    strategy, mode = live["strategy"], live["mode"]
    log.info("=" * 64)
    log.info(f"Live run started - strategy '{strategy}' - Trading 212 {mode.upper()} - "
             f"{'EXECUTE' if args.execute else 'DRY RUN'}")

    if args.execute and mode == "live" and not live.get("allow_live_account", False):
        raise SystemExit("live.mode is 'live' (REAL MONEY) but live.allow_live_account is false in "
                         "config.yaml, so no orders were sent. (This is a safety feature. A dry run, "
                         "without --execute, is allowed.)")
    if args.execute and live.get("only_when_market_open", True) and not broker.us_market_open():
        raise SystemExit("The US stock market is closed now (open Mon-Fri 9:30am-4pm New York time, "
                         "usually 2:30pm-9pm UK time). Orders would queue until the next open, so nothing "
                         "was sent. Run it while the market is open, or use a dry run.")

    # ---- 1. the model
    bundle = None
    if strategy == "ml_combined":
        path = ROOT / "models" / "model.joblib"
        if not path.exists():
            raise SystemExit("No model found. Run:  python run_research.py   first.")
        bundle = joblib.load(path)
        age = (datetime.now() - datetime.fromisoformat(bundle["created"])).days
        log.info(f"Model: {bundle['model_name']} trained on data up to {bundle['trained_until']} ({age} days ago)")
        if age > live.get("max_model_age_days", 45):
            log.warning("Model is old - consider re-running run_research.py (or check the monthly schedule).")

    # ---- 2. Trading 212: which of our stocks can be traded there?
    client = broker.connect(cfg)
    managed = cfg["universe"]["candidates"]
    tmap = broker.instrument_map(client, managed)
    missing = [s for s in managed if s not in tmap]
    log.info(f"Trading 212 offers {len(tmap)} of the {len(managed)} candidate stocks"
             + (f" (not available: {', '.join(missing)})" if missing else ""))

    # ---- 3. latest prices -> features -> scores
    start = (datetime.now() - timedelta(days=640)).strftime("%Y-%m-%d")
    prices = load_prices(cfg, source=live["data_source"], start=start, refresh=True)
    close, volume = prices["close"], prices["volume"]
    bench = cfg["data"]["benchmark"]
    mask = universe.universe_mask(close, volume, cfg)
    mask.loc[:, ~mask.columns.isin(list(tmap))] = False  # only stocks Trading 212 offers
    X, raw = features.build_features(close, volume, close[bench], mask, cfg)
    day = mask.index[-1]
    if (datetime.now() - day.to_pydatetime()).days > 5:
        log.warning(f"Latest price is from {day.date()} - data may be stale.")
    log.info(f"Signal date: {day.date()} ({int(mask.loc[day].sum())} stocks in universe)")

    ml_pred = None
    if bundle is not None:
        Xd = X.xs(day, level="date", drop_level=False)[bundle["features"]]
        ml_pred = pd.Series(bundle["model"].predict_proba(Xd)[:, 1], index=Xd.index)
    scores = signals.strategy_scores(raw, ml_pred, cfg)[strategy]

    # ---- 4. the account
    acct = broker.account_info(client, tmap, fallback_rate=lambda: usd_to(
        client.account_summary().get("currency", "GBP")))
    log.info(f"Trading 212 {'PRACTICE (pretend money)' if acct.is_demo else 'LIVE - REAL MONEY'} account "
             f"{acct.account_id} | value {acct.total_value:,.2f} {acct.currency} | "
             f"cash {acct.cash_available:,.2f} | {len(acct.positions)} positions | "
             f"1 USD = {acct.usd_rate:.4f} {acct.currency}")

    held = [t for t, p in acct.positions.items() if p["quantity"] > 0 and t in managed]
    target = signals.target_weights_for_day(scores.loc[day], mask.loc[day], raw["vol_60"].loc[day],
                                            cfg, current=held)
    if target.empty:
        raise SystemExit("Not enough stocks with a valid signal today - no trades.")
    if (target < 0).any():
        raise SystemExit("portfolio.long_short is true, but Trading 212 Invest/ISA accounts cannot "
                         "short-sell. Set long_short: false in config.yaml.")
    log.info("Target portfolio: " + ", ".join(f"{t} {w*100:.1f}%" for t, w in
                                              target.sort_values(ascending=False).items()))

    plan = broker.plan_orders(target, close.loc[day], acct, cfg, managed, tmap)
    trades = plan[plan.action != "HOLD"] if len(plan) else plan
    with pd.option_context("display.width", 160, "display.float_format", "{:,.4f}".format):
        log.info("Order plan:\n" + (trades.to_string(index=False) if len(trades) else "  nothing to trade"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    plan.to_csv(ROOT / "logs" / f"order_plan_{stamp}.csv", index=False)

    if not args.execute:
        log.info("DRY RUN - no orders sent. Add --execute to trade.")
        return
    if trades.empty:
        log.info("Already on target - nothing to do.")
        return
    fills = broker.execute(client, trades, acct, cfg)
    fills.to_csv(ROOT / "logs" / f"fills_{stamp}.csv", index=False)
    bad = fills[~fills.status.isin(["FILLED"])] if len(fills) else fills
    if len(bad):
        log.warning(f"{len(bad)} orders are not confirmed as filled - check the Trading 212 app "
                    f"and logs/fills_{stamp}.csv.")
    log.info("Live run finished.")


if __name__ == "__main__":
    try:
        main()
    except broker.BrokerError as e:
        get_logger().error(str(e))
        raise SystemExit(1)
