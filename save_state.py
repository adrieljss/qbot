"""
save_state.py
-------------
Fetches current coin holdings from Roostoo + live prices from Binance,
then writes state.json so the bot can resume without losing position tracking.

Run this BEFORE restarting bot.py whenever the bot was stopped unexpectedly.

Usage:
    python save_state.py

You will be prompted to confirm each position's entry price, stop, and TP
(or accept the defaults derived from the current price + regime params).
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from roostoo_api import RoostooClient
from binance.client import Client as BinanceClient
from signals import P, to_roostoo_pair

SGT = timezone(timedelta(hours=8), name="SGT")

def ask(prompt: str, default: str) -> str:
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else default

def main():
    print("\n=== MRVS State Saver ===\n")

    roostoo = RoostooClient.from_env()
    binance = BinanceClient("", "")

    # 1. Fetch wallet from Roostoo
    print("Fetching Roostoo balance...")
    resp = roostoo.balance()
    if not resp.success:
        print(f"ERROR: {resp.err_msg}")
        sys.exit(1)

    # 2. Find all non-USD coin holdings
    holdings = {
        asset: entry.free
        for asset, entry in resp.wallet.items()
        if asset not in ("USD", "USDT") and entry.free > 0
    }

    if not holdings:
        print("No coin holdings found. Wallet is all cash.")
        usd = resp.wallet.get("USD") or resp.wallet.get("USDT")
        capital = usd.free if usd else 0.0
        print(f"Free USD: ${capital:,.2f}")
        state = {
            "positions": {},
            "capital": capital,
            "peak_capital": capital,
            "total_trades": 0,
            "last_regime": "UNKNOWN",
            "plan_week_key": -1,
            "week_closed": False,
            "last_rebalance_date": None,
            "last_order_date": None,
            "weekly_returns": [],
            "week_start_cap": capital,
            "current_week_key": -1,
        }
        _write(state)
        return

    print(f"Found {len(holdings)} coin holding(s): {list(holdings.keys())}\n")

    # 3. Fetch live prices from Binance
    prices = {}
    for asset in holdings:
        sym = asset + "USDT"
        try:
            t = binance.get_symbol_ticker(symbol=sym)
            prices[asset] = float(t["price"])
            print(f"  {asset}: ${prices[asset]:,.4f}  (qty={holdings[asset]:.6f})")
        except Exception as e:
            print(f"  {asset}: price fetch failed ({e}) — will prompt manually")
            prices[asset] = 0.0

    # 4. Compute total portfolio value
    usd_entry = resp.wallet.get("USD") or resp.wallet.get("USDT")
    free_usd  = usd_entry.free if usd_entry else 0.0
    coin_mtm  = sum(holdings[a] * prices[a] for a in holdings)
    total_cap = free_usd + coin_mtm
    print(f"\n  Free USD : ${free_usd:,.2f}")
    print(f"  Coin MTM : ${coin_mtm:,.2f}")
    print(f"  Total    : ${total_cap:,.2f}")

    # 5. Ask regime once (applies stop/TP defaults to all positions)
    print("\nWhat regime were these positions entered in?")
    print("  1. SIDEWAYS  (stop -5%, TP +15%)")
    print("  2. BULL      (stop -7%, TP disabled)")
    print("  3. BEAR      (stop -4%, TP +8%)")
    regime_choice = ask("Regime [1/2/3]", "1")
    regime_map = {"1": "SIDEWAYS", "2": "BULL", "3": "BEAR"}
    regime = regime_map.get(regime_choice, "SIDEWAYS")

    if regime == "BULL":
        default_stop_pct = P["bull_position_stop"]
        default_tp_pct   = 0.0
    elif regime == "BEAR":
        default_stop_pct = P["bear_position_stop"]
        default_tp_pct   = P["bear_rr_ratio"] * P["bear_position_stop"]
    else:
        default_stop_pct = P["sw_position_stop"]
        default_tp_pct   = P["sw_rr_ratio"] * P["sw_position_stop"]

    # 6. Build position dicts
    positions = {}
    now_sgt   = datetime.now(SGT).strftime("%Y-%m-%d %H:%M SGT")

    for asset, qty in holdings.items():
        sym        = asset + "USDT"
        cur_price  = prices[asset]

        print(f"\n--- {sym} ---")
        print(f"  Current price : ${cur_price:,.4f}  qty={qty:.6f}")

        raw_entry = ask("Entry price", f"{cur_price:.4f}")
        entry_price = float(raw_entry)

        default_stop = round(entry_price * (1 - default_stop_pct), 4)
        default_tp   = round(entry_price * (1 + default_tp_pct), 4) if default_tp_pct > 0 else "inf"
        default_alloc = round(qty * entry_price, 2)

        raw_stop  = ask("Stop price",    str(default_stop))
        raw_tp    = ask("Take profit",   str(default_tp))
        raw_alloc = ask("Cost basis (USD alloc)", str(default_alloc))
        raw_time  = ask("Entry time (SGT)", now_sgt)

        stop_price  = float(raw_stop)
        take_profit = float("inf") if raw_tp in ("inf", "0", "") else float(raw_tp)
        alloc       = float(raw_alloc)

        positions[sym] = {
            "entry_price":   entry_price,
            "shares":        qty,
            "alloc":         alloc,
            "stop_price":    stop_price,
            "take_profit":   take_profit,
            "highest_price": max(entry_price, cur_price),
            "regime":        regime,
            "entry_time":    raw_time,
            "order_id":      None,
        }

        pnl     = qty * cur_price - alloc
        pnl_pct = pnl / alloc * 100
        print(f"  PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")

    # 7. Ask for existing state file values if any
    existing = {}
    if Path("state.json").exists():
        try:
            with open("state.json") as f:
                existing = json.load(f)
            print(f"\nExisting state.json found — will merge weekly_returns and trade count.")
        except Exception:
            pass

    iso = datetime.now(SGT).isocalendar()
    week_key = iso[0] * 100 + iso[1]

    state = {
        "positions":        positions,
        "capital":          total_cap,
        "peak_capital":     max(total_cap, existing.get("peak_capital", total_cap)),
        "total_trades":     existing.get("total_trades", len(positions)),
        "last_regime":      regime,
        "plan_week_key":    existing.get("plan_week_key", week_key),
        "week_closed":      existing.get("week_closed", False),
        "last_rebalance_date": existing.get("last_rebalance_date"),
        "last_order_date":  existing.get("last_order_date"),
        "weekly_returns":   existing.get("weekly_returns", []),
        "week_start_cap":   existing.get("week_start_cap", total_cap),
        "current_week_key": existing.get("current_week_key", week_key),
    }

    _write(state)


def _write(state: dict) -> None:
    with open("state.json", "w") as f:
        json.dump(state, f, indent=2)
    print(f"\n✓ state.json written with {len(state['positions'])} position(s)")
    print(f"  Capital : ${state['capital']:,.2f}")
    print(f"  Positions: {list(state['positions'].keys())}")
    print("\nYou can now restart bot.py safely.\n")


if __name__ == "__main__":
    main()