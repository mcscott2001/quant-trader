"""Talking to Trading 212 through its official Public API (beta).

Docs: https://docs.trading212.com
  * demo (Practice account, pretend money): https://demo.trading212.com/api/v0
  * live (Invest or Stocks ISA, real money): https://live.trading212.com/api/v0
  * login = HTTP Basic auth: API key as the username, API secret as the password
  * sell = a market order with a NEGATIVE quantity
  * orders are paid in your main account currency (e.g. GBP); Trading 212
    converts to US dollars for US shares and charges a currency conversion fee
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .utils import ROOT, get_logger

log = get_logger()

BASE_URLS = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}
DONE_STATUSES = {"FILLED", "CANCELLED", "REJECTED", "REPLACED"}


class BrokerError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


# ----------------------------------------------------------------------------
# credentials (.env file)
# ----------------------------------------------------------------------------
def load_env(path=None) -> None:
    """Read KEY=VALUE lines from the .env file into environment variables."""
    path = path or ROOT / ".env"
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def credentials() -> tuple[str, str]:
    """API key and secret from .env. The secret may be blank for an older,
    single-code Trading 212 key."""
    load_env()
    key = os.environ.get("T212_API_KEY", "").strip()
    secret = os.environ.get("T212_API_SECRET", "").strip()
    if secret.startswith("paste"):
        secret = ""
    if not key or key.startswith("paste"):
        raise SystemExit("No Trading 212 API key found. Copy .env.example to .env and paste your "
                         "T212_API_KEY and T212_API_SECRET into it (see the setup booklet, Part 2).")
    return key, secret


# ----------------------------------------------------------------------------
# low-level client
# ----------------------------------------------------------------------------
class T212Client:
    """auth = "basic"  -> key + secret (current Trading 212 keys)
       auth = "legacy" -> the key alone in the Authorization header (older single-code keys)"""

    def __init__(self, mode: str, key: str, secret: str, timeout: float = 20, auth: str = "basic"):
        if mode not in BASE_URLS:
            raise SystemExit(f"live.mode must be 'demo' or 'live', not '{mode}'.")
        self.mode = mode
        self.auth = auth
        self.base = BASE_URLS[mode]
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/json"})
        if auth == "basic":
            self.s.auth = (key, secret)
        else:
            self.s.headers["Authorization"] = key

    def _request(self, method: str, path: str, body: dict | None = None,
                 params: dict | None = None, safe_to_retry: bool = True):
        url = self.base + path
        for attempt in range(6):
            try:
                r = self.s.request(method, url, json=body, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                # an order that timed out may still have been placed - never resend it blindly
                if safe_to_retry and attempt < 2:
                    time.sleep(3)
                    continue
                raise BrokerError(f"Network problem talking to Trading 212 ({e}).") from e
            if r.status_code == 429:      # rate limited: wait until the limit resets, then retry
                reset = r.headers.get("x-ratelimit-reset")
                wait = 5.0
                if reset:
                    try:
                        wait = float(reset) - time.time()
                    except ValueError:
                        pass
                wait = min(max(wait, 1.0), 65.0)
                log.info(f"  (Trading 212 rate limit reached - waiting {wait:.0f}s)")
                time.sleep(wait)
                continue
            if r.status_code == 401:
                raise BrokerError(f"Trading 212 rejected the API key (401). Check .env, and that the key "
                                  f"was made in the {'Practice' if self.mode == 'demo' else 'real-money'} "
                                  f"account (live.mode is '{self.mode}').", 401, r.text)
            if r.status_code == 403:
                raise BrokerError(f"The API key is missing a permission needed for {path} (403). Make a new "
                                  "key with account, metadata, portfolio, orders read and orders execute "
                                  "permissions ticked.", 403, r.text)
            if r.status_code >= 400:
                raise BrokerError(f"{method} {path} failed ({r.status_code}): {r.text[:300]}",
                                  r.status_code, r.text)
            if not r.content:
                return None
            return r.json()
        raise BrokerError("Trading 212 kept rate-limiting the request. Try again in a few minutes.")

    # --- endpoints
    def account_summary(self) -> dict:
        return self._request("GET", "/equity/account/summary")

    def positions(self) -> list[dict]:
        return self._request("GET", "/equity/positions") or []

    def instruments(self) -> list[dict]:
        return self._request("GET", "/equity/metadata/instruments") or []

    def active_orders(self) -> list[dict]:
        return self._request("GET", "/equity/orders") or []

    def order_history(self, limit: int = 50) -> list[dict]:
        data = self._request("GET", "/equity/history/orders", params={"limit": limit}) or {}
        return data.get("items", [])

    def market_order(self, ticker: str, quantity: float) -> dict:
        """Positive quantity = buy, negative = sell. Not retried on network errors (could duplicate)."""
        body = {"ticker": ticker, "quantity": quantity, "extendedHours": False}
        return self._request("POST", "/equity/orders/market", body=body, safe_to_retry=False)


def connect(cfg: dict) -> T212Client:
    """Log in, trying key+secret first and the older key-only style second."""
    key, secret = credentials()
    mode = cfg["live"]["mode"]
    last = None
    for auth in (["basic", "legacy"] if secret else ["legacy"]):
        client = T212Client(mode, key, secret, auth=auth)
        try:
            client.account_summary()
            if auth == "legacy":
                log.info("(Logged in with an older key-only Trading 212 API key.)")
            return client
        except BrokerError as e:
            if e.status != 401:
                raise
            last = e
        time.sleep(1.1)
    raise last


def probe(mode: str, key: str, secret: str) -> str | None:
    """Read-only test: does this key log in to `mode`? Returns the auth style that worked, or None."""
    for auth in (["basic", "legacy"] if secret else ["legacy"]):
        try:
            T212Client(mode, key, secret, auth=auth).account_summary()
            return auth
        except BrokerError as e:
            if e.status in (403, 429):      # logged in (but no permission / too fast) = key is valid here
                return auth
            if e.status != 401:
                return None
        time.sleep(1.1)
    return None


# ----------------------------------------------------------------------------
# instrument names: our "AAPL" <-> Trading 212's "AAPL_US_EQ"
# ----------------------------------------------------------------------------
def instrument_map(client: T212Client, symbols: list[str], max_age_hours: float = 24) -> dict[str, str]:
    cache = ROOT / "data" / f"t212_instruments_{client.mode}.json"
    items = None
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < max_age_hours:
        items = json.loads(cache.read_text(encoding="utf-8"))
    if items is None:
        items = client.instruments()
        cache.write_text(json.dumps(items), encoding="utf-8")
    us = [i for i in items if i.get("currencyCode") == "USD" and i.get("type") in ("STOCK", "ETF")]
    by_ticker = {i["ticker"]: i for i in us}
    out = {}
    for sym in symbols:
        guess = f"{sym.replace('.', '_').replace('-', '_')}_US_EQ"
        if guess in by_ticker:
            out[sym] = guess
            continue
        match = [i for i in us if str(i.get("shortName", "")).upper() == sym.upper()]
        if match:
            out[sym] = match[0]["ticker"]
    return out


# ----------------------------------------------------------------------------
# account snapshot
# ----------------------------------------------------------------------------
@dataclass
class AccountInfo:
    mode: str
    account_id: str
    currency: str
    total_value: float
    cash_available: float
    usd_rate: float                         # 1 USD = this many units of account currency
    positions: dict[str, dict] = field(default_factory=dict)   # symbol -> details

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"


def account_info(client: T212Client, ticker_map: dict[str, str], fallback_rate=None) -> AccountInfo:
    summ = client.account_summary()
    time.sleep(1.1)
    raw_pos = client.positions()
    ccy = summ.get("currency", "GBP")
    back = {v: k for k, v in ticker_map.items()}

    rates, positions = [], {}
    for p in raw_pos:
        inst = p.get("instrument") or {}
        t = inst.get("ticker") or p.get("ticker", "")
        qty = float(p.get("quantity") or 0)
        px = float(p.get("currentPrice") or 0)
        wi = p.get("walletImpact") or {}
        val = float(wi.get("currentValue") or 0)
        if inst.get("currency") == "USD" and qty > 0 and px > 0 and val > 0:
            rates.append(val / (qty * px))
        sym = back.get(t, t)
        positions[sym] = {"ticker": t, "quantity": qty,
                          "available": float(p.get("quantityAvailableForTrading", qty) or 0),
                          "price_usd": px if inst.get("currency") == "USD" else None,
                          "value": val}

    if ccy == "USD":
        rate = 1.0
    elif rates:
        rate = float(pd.Series(rates).median())   # the rate Trading 212 itself is using
    elif fallback_rate is not None:
        rate = float(fallback_rate() if callable(fallback_rate) else fallback_rate)
    else:
        raise SystemExit(f"Could not work out the USD to {ccy} exchange rate.")
    cash = summ.get("cash") or {}
    return AccountInfo(client.mode, str(summ.get("id", "?")), ccy, float(summ.get("totalValue") or 0),
                       float(cash.get("availableToTrade") or 0), rate, positions)


# ----------------------------------------------------------------------------
# order planning
# ----------------------------------------------------------------------------
def _floor(x: float, decimals: int) -> float:
    f = 10 ** decimals
    return math.floor(x * f + 1e-9) / f


def plan_orders(target_w: pd.Series, prices_usd: pd.Series, acct: AccountInfo, cfg: dict,
                managed: list[str], ticker_map: dict[str, str]) -> pd.DataFrame:
    """How many shares to buy or sell to move from today's holdings to the target.
    Works in your account currency. Only stocks in `managed` are touched."""
    live = cfg["live"]
    capital = acct.total_value * (1 - live["cash_buffer"])
    cap = float(live.get("max_capital", 0) or 0)
    if cap > 0 and capital > cap:
        log.info(f"Using {cap:,.0f} {acct.currency} of the {capital:,.0f} available (live.max_capital)")
        capital = cap
    dec = int(live.get("quantity_decimals", 2))
    held = {s for s, p in acct.positions.items() if s in managed and p["quantity"] > 0}
    rows = []
    for sym in sorted(set(target_w.index[target_w > 0]) | held):
        w = float(target_w.get(sym, 0.0))
        pos = acct.positions.get(sym, {})
        cur = float(pos.get("quantity", 0.0))
        avail = float(pos.get("available", cur))
        px_usd = pos.get("price_usd") or float(prices_usd.get(sym, float("nan")))
        if sym not in ticker_map:
            log.warning(f"  {sym} is not available on Trading 212; skipped")
            continue
        if not px_usd or math.isnan(px_usd):
            log.warning(f"  no price for {sym}; skipped")
            continue
        px = px_usd * acct.usd_rate                     # price per share in account currency
        tgt = _floor(w * capital / px, dec)
        diff = tgt - cur
        full_exit = w == 0 and cur > 0
        if full_exit:
            diff = -avail
        elif diff < 0:
            diff = -min(_floor(-diff, dec), avail)      # never sell more than we can
        value = diff * px
        if abs(diff) < 1e-9 or (abs(value) < live["min_trade_value"] and not full_exit):
            action = "HOLD"
        else:
            action = "BUY" if diff > 0 else "SELL"
        rows.append({"symbol": sym, "ticker": ticker_map[sym], "weight": round(w, 4),
                     "price_usd": round(px_usd, 2), "current": cur, "target": tgt,
                     "change": diff if action != "HOLD" else 0.0,
                     f"value_{acct.currency}": round(value, 2) if action != "HOLD" else 0.0,
                     "action": action})
    return pd.DataFrame(rows)


def us_market_open(now: datetime | None = None) -> bool:
    """Rough check: Monday-Friday 9:30-16:00 New York time (ignores US holidays)."""
    ny = (now or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:
        return False
    minutes = ny.hour * 60 + ny.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


# ----------------------------------------------------------------------------
# order execution
# ----------------------------------------------------------------------------
def _place(client: T212Client, ticker: str, qty: float, decimals: int) -> tuple[dict | None, float, str]:
    """Place a market order, retrying with fewer decimal places if Trading 212
    says the quantity is too precise for that stock. Returns (order, qty_sent, error)."""
    sign = 1 if qty > 0 else -1
    tried = set()
    for d in range(decimals, -1, -1):
        q = sign * _floor(abs(qty), d)
        if q == 0 or q in tried:
            continue
        tried.add(q)
        try:
            return client.market_order(ticker, q), q, ""
        except BrokerError as e:
            if e.status == 400 and "precision" in e.body.lower():
                continue
            return None, q, str(e)
    return None, 0.0, "quantity rounds to zero at the precision Trading 212 allows"


def _wait(client: T212Client, ids: set[int], seconds: float) -> set[int]:
    """Wait until these orders are no longer active. Returns the ids still active."""
    t0 = time.time()
    pending = set(ids)
    while pending and time.time() - t0 < seconds:
        time.sleep(5)
        try:
            active = {int(o["id"]) for o in client.active_orders()}
        except BrokerError as e:
            log.warning(f"  could not check order status ({e})")
            continue
        pending &= active
    return pending


def execute(client: T212Client, plan: pd.DataFrame, acct: AccountInfo, cfg: dict) -> pd.DataFrame:
    """Sells first (to free up cash), wait for them, then buys - scaled down if cash is short."""
    live = cfg["live"]
    dec = int(live.get("quantity_decimals", 2))
    gap = float(live.get("order_spacing_seconds", 1.5))
    wait = float(live.get("wait_for_fills_seconds", 180))
    results = []

    def run(rows: pd.DataFrame, side: str) -> set[int]:
        ids = set()
        for _, r in rows.iterrows():
            order, q, err = _place(client, r.ticker, float(r.change), dec)
            if order is None:
                log.warning(f"  {side} {r.symbol}: NOT placed - {err}")
                results.append({"symbol": r.symbol, "side": side, "quantity": q, "order_id": None,
                                "status": "NOT PLACED", "detail": err[:200]})
            else:
                oid = int(order.get("id"))
                ids.add(oid)
                log.info(f"  sent {side} {abs(q)} {r.symbol} (order {oid}, {order.get('status')})")
                results.append({"symbol": r.symbol, "side": side, "quantity": q, "order_id": oid,
                                "status": order.get("status", ""), "detail": ""})
            time.sleep(gap)
        return ids

    sells = plan[plan.action == "SELL"]
    buys = plan[plan.action == "BUY"].copy()
    sell_ids = run(sells, "SELL")
    if sell_ids:
        log.info(f"Waiting up to {wait:.0f}s for {len(sell_ids)} sell orders to fill ...")
        still = _wait(client, sell_ids, wait)
        if still:
            log.warning(f"{len(still)} sell orders are still open; buys will use only the cash available now.")

    if len(buys):
        time.sleep(1.1)
        cash = float((client.account_summary().get("cash") or {}).get("availableToTrade") or 0)
        need = float(buys[f"value_{acct.currency}"].sum())
        usable = cash * 0.995                       # small margin for price moves and fees
        if need > usable and need > 0:
            scale = max(usable, 0) / need
            log.warning(f"Buys need {need:,.2f} {acct.currency} but {cash:,.2f} is available - "
                        f"scaling every buy to {scale:.0%}.")
            buys["change"] = [_floor(c * scale, dec) for c in buys["change"]]
            buys[f"value_{acct.currency}"] = buys[f"value_{acct.currency}"] * scale
            buys = buys[(buys["change"] > 0) & (buys[f"value_{acct.currency}"] >= 1)]
        buy_ids = run(buys, "BUY")
        if buy_ids:
            _wait(client, buy_ids, wait)

    # final status for each order from the history (best effort)
    try:
        time.sleep(1.1)
        hist = {int(h["order"]["id"]): h for h in client.order_history(50) if h.get("order")}
        active = {int(o["id"]) for o in client.active_orders()}
    except BrokerError as e:
        log.warning(f"Could not read the final order status ({e}). Check the Trading 212 app.")
        hist, active = {}, set()
    for r in results:
        oid = r["order_id"]
        if oid is None:
            continue
        if oid in active:
            r["status"] = "STILL OPEN"
        elif oid in hist:
            o, fill = hist[oid]["order"], hist[oid].get("fill") or {}
            r["status"] = o.get("status", r["status"])
            r["filled_quantity"] = o.get("filledQuantity")
            r["fill_price"] = fill.get("price")
        log.info(f"  {r['side']} {r['symbol']}: {r['status']}"
                 + (f", filled {r.get('filled_quantity')} @ {r.get('fill_price')}" if r.get("fill_price") else ""))
    return pd.DataFrame(results)