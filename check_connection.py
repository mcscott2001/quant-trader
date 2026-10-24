"""
CHECK THAT PYTHON CAN TALK TO TRADING 212
=========================================
    python check_connection.py

Uses the API key in .env and live.mode in config.yaml. Prints the account,
its value, your positions and whether each permission the robot needs works.
Sends no orders. If the key is rejected, it works out the most likely reason.
"""
import os
import time

from quantbot import broker
from quantbot.utils import ROOT, load_config


def mask(code: str) -> str:
    return f"{code[:4]}...{code[-3:]}" if len(code) > 10 else "(too short)"


def diagnose(mode: str) -> None:
    """Explain a 401 in plain English. Only makes read-only requests."""
    print("\nWorking out why ...")
    env = ROOT / ".env"
    if not env.exists():
        print("  - There is no .env file in the quant_trader folder (the key came from somewhere else).")
    if (ROOT / ".env.txt").exists():
        print("  - Found a file called .env.txt: Notepad added .txt. The robot only reads .env, so\n"
              "    rename it (see Troubleshooting).")
    if env.exists():
        text = env.read_bytes().decode("utf-8-sig", "ignore").splitlines()
        for name in ("T212_API_KEY", "T212_API_SECRET"):
            n = sum(1 for l in text if l.strip().startswith(name))
            if n != 1:
                print(f"  - {name} appears {n} times in .env (it should appear once).")

    key, secret = broker.credentials()
    print(f"  - Key read from .env:    {mask(key)}  ({len(key)} characters)")
    print(f"  - Secret read from .env: {mask(secret) if secret else '(blank)'}  ({len(secret)} characters)")
    if secret and key == secret:
        print("  - The key and secret are identical: one of them was pasted twice.")
    if any(c in key + secret for c in " \"'<>"):
        print("  - The key or secret contains a space, quote mark or < > character. Re-paste them.")

    other = "live" if mode == "demo" else "demo"
    print(f"  - Testing the same key against your {'REAL-MONEY' if other == 'live' else 'Practice'} "
          "account (read-only, no orders) ...")
    time.sleep(1.1)
    swapped = broker.probe(other, key, secret)
    time.sleep(1.1)
    flipped = broker.probe(mode, secret, key) if secret else None

    print("\nMost likely cause:")
    if swapped:
        if mode == "demo":
            print("  This key belongs to your REAL-MONEY account, not your Practice account.\n"
                  "  Fix: in the Trading 212 app, switch to Practice first, then Settings > API (Beta) >\n"
                  "  Generate API key, and paste that new key and secret into .env.\n"
                  "  (Keep the real-money key safe for later, or delete it in the app if you don't need it.)")
        else:
            print("  This key belongs to your Practice account, but config.yaml says mode: live.\n"
                  "  Fix: make a key while the app is on your real account, or set mode: demo.")
    elif flipped:
        print("  The key and secret are the wrong way round in .env. Swap the two codes.")
    else:
        print("  The key wasn't accepted by either account. Most often this is one of:\n"
              "   1. A character missed when copying. Re-copy both codes (the secret is shown only once,\n"
              "      so if you no longer have it, delete the key in the app and generate a new one).\n"
              "   2. An IP restriction on the key that doesn't include this computer's address. For the\n"
              "      Practice key, choose Unrestricted while testing from home.\n"
              "   3. The key was deleted, or was made before the account finished setting up.\n"
              "  If a brand-new, unrestricted key made in Practice still fails, Trading 212's API may not\n"
              "  be enabled for your Practice account yet; check the API (Beta) page in the app.")


def main() -> None:
    cfg = load_config()
    mode = cfg["live"]["mode"]
    print(f"Connecting to Trading 212 ({'PRACTICE / demo' if mode == 'demo' else 'LIVE - REAL MONEY'}) ...")
    try:
        client = broker.connect(cfg)
    except broker.BrokerError as e:
        print(f"\nFAILED: {e}")
        if e.status == 401:
            diagnose(mode)
        raise SystemExit(1)
    ok = True
    time.sleep(5.1)   # account summary allows one request every 5 seconds

    summ = client.account_summary()
    cash = summ.get("cash") or {}
    print(f"  [ok] account data   account {summ.get('id')}  value {summ.get('totalValue'):,} "
          f"{summ.get('currency')}  cash available {cash.get('availableToTrade'):,}")
    time.sleep(1.1)

    try:
        pos = client.positions()
        print(f"  [ok] portfolio      {len(pos)} positions")
        for p in pos[:20]:
            inst = p.get("instrument") or {}
            print(f"         {inst.get('ticker', '?'):<16} {float(p.get('quantity', 0)):>12,.4f} shares")
    except broker.BrokerError as e:
        ok = False
        print(f"  [!!] portfolio      {e}")
    time.sleep(1.1)

    try:
        tmap = broker.instrument_map(client, cfg["universe"]["candidates"])
        print(f"  [ok] metadata       {len(tmap)} of {len(cfg['universe']['candidates'])} candidate stocks "
              "found on Trading 212")
    except broker.BrokerError as e:
        ok = False
        print(f"  [!!] metadata       {e}")
    time.sleep(1.1)

    try:
        orders = client.active_orders()
        print(f"  [ok] orders (read)  {len(orders)} open orders")
    except broker.BrokerError as e:
        ok = False
        print(f"  [!!] orders (read)  {e}")

    print("  [--] orders (execute) is only tested by a real order: see Part 5 of the booklet.")
    print("  US market is " + ("OPEN" if broker.us_market_open() else "CLOSED") + " right now.")
    print("\nAll good - Python can talk to Trading 212." if ok else
          "\nSome checks failed - see the messages above and the Troubleshooting section.")


if __name__ == "__main__":
    os.chdir(ROOT)
    try:
        main()
    except broker.BrokerError as e:
        print(f"\nFAILED: {e}")
        raise SystemExit(1)