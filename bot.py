"""
bot.py  —  MRVS Regime-Switching Trading Bot
=============================================
Execution loop (every 60 s):
  1.  Balance sync from Roostoo
  2.  Fresh signals from Binance
  3.  Weekly plan refresh (Monday SGT, or first startup)
  4.  Regime change alert
  5.  BEAR_GATE emergency liquidation
  6.  Weekly performance tracking
  7.  Portfolio stop check
  8.  Per-position exit checks (stop / TP / trailing / RSI)
  9.  Daily micro-rebalance at 09:00 SGT — nudge each held position back
      toward its target weight if drift > 5 %.
 10. Enter unowned plan coins immediately at current price (1 order/cycle).
      No red-day wait — competition timeframe too short for timing filter.
 11. Daily guarantee at 20:00 SGT — token entry if no order placed today.
 10.  Position entries from weekly plan (1 order/cycle, red-day or fallback)
 11.  Friday 22:00 SGT — weekly close

Run:
    python bot.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from binance.client import Client as BinanceClient
from signals import SYMBOLS, P, MarketSnapshot, generate_signals, to_roostoo_pair
from telegram_notifier import TelegramNotifier

sys.path.insert(0, str(Path(__file__).parent))
from roostoo_api import (
    RoostooClient, Side, OrderType,
    RoostooHTTPError, RoostooParseError,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("bot")

# ── Constants ─────────────────────────────────────────────────────────────────
SGT              = timezone(timedelta(hours=8), name="SGT")
LOOP_INTERVAL_S  = 60
MIN_ORDER_GAP_S  = 61
MIN_ORDER_USD    = 10.0
ROTATION_HOUR    = 9
STATE_FILE       = "state.json"    # 09:00 SGT — daily rebalance window

# A position must have drifted more than this fraction from its target weight
# before the rebalance trims/tops it. Keeps orders small and strategy-neutral.
REBALANCE_DRIFT_THRESHOLD = 0.05   # 5 % drift triggers a trim/top-up


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_sgt() -> str:
    return datetime.now(SGT).strftime("%Y-%m-%d %H:%M SGT")


def _usd(v: float) -> str:
    if v is None or v != v:
        return "n/a"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.3f}M"
    return f"${v:,.2f}"


# ── Bot ───────────────────────────────────────────────────────────────────────

class TradingBot:
    """
    All live state in one object.

    Position dict keys:
        entry_price, shares, alloc, stop_price, take_profit,
        highest_price, regime, entry_time, order_id
    """

    SYMBOLS = SYMBOLS

    def __init__(self) -> None:
        self.roostoo = RoostooClient.from_env()
        self.binance = BinanceClient("", "")
        self.tg      = TelegramNotifier(bot_ref=self)

        self.initial_capital = float(os.getenv("INITIAL_CAPITAL", "1000000"))
        self.capital         = self.initial_capital
        self.peak_capital    = self.initial_capital

        self.positions:      dict[str, dict]  = {}
        self.last_prices:    dict[str, float] = {}
        self.last_snapshot:  Optional[MarketSnapshot] = None
        self.last_regime     = "UNKNOWN"

        # Weekly plan
        self._plan_week_key: int = -1           # isoyear*100+isoweek
        self._plan_anchor:   Optional[datetime] = None   # SGT datetime
        self._weekly_plan:   Optional[MarketSnapshot] = None
        self._week_closed:   bool = False

        # Daily rebalance guard
        self._last_rebalance_date: Optional[str] = None   # "YYYY-MM-DD" SGT

        # Daily order guarantee
        self._last_order_date: Optional[str] = None       # "YYYY-MM-DD" SGT

        # Exchange info
        self._amount_precision: dict[str, int] = {}

        # Rate limiter
        self._last_order_ts: float = 0.0

        # Performance
        self.total_trades        = 0
        self._weekly_returns:    deque = deque(maxlen=520)
        self._week_start_cap:    float = self.initial_capital
        self._current_week_key:  int   = -1

        self._running = False
        log.info(f"TradingBot init  capital=${self.initial_capital:,.0f}")

    # ── State persistence ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist positions and capital to state.json after every cycle."""
        try:
            data = {
                "positions":        self.positions,
                "capital":          self.capital,
                "peak_capital":     self.peak_capital,
                "total_trades":     self.total_trades,
                "last_regime":      self.last_regime,
                "plan_week_key":    self._plan_week_key,
                "week_closed":      self._week_closed,
                "last_rebalance_date": self._last_rebalance_date,
                "last_order_date":  self._last_order_date,
                "weekly_returns":   list(self._weekly_returns),
                "week_start_cap":   self._week_start_cap,
                "current_week_key": self._current_week_key,
            }
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.warning(f"State save failed: {exc}")

    def _load_state(self) -> bool:
        """Load positions and capital from state.json. Returns True if loaded."""
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            self.positions       = data.get("positions", {})
            self.capital         = data.get("capital", self.initial_capital)
            self.peak_capital    = data.get("peak_capital", self.initial_capital)
            self.total_trades    = data.get("total_trades", 0)
            self.last_regime     = data.get("last_regime", "UNKNOWN")
            self._plan_week_key  = data.get("plan_week_key", -1)
            self._week_closed    = data.get("week_closed", False)
            self._last_rebalance_date = data.get("last_rebalance_date")
            self._last_order_date     = data.get("last_order_date")
            self._week_start_cap      = data.get("week_start_cap", self.initial_capital)
            self._current_week_key    = data.get("current_week_key", -1)
            for r in data.get("weekly_returns", []):
                self._weekly_returns.append(r)
            log.info(
                f"State loaded: {len(self.positions)} positions  "
                f"capital={_usd(self.capital)}  "
                f"pos={list(self.positions.keys())}"
            )
            return True
        except FileNotFoundError:
            log.info("No state.json found — starting fresh")
            return False
        except Exception as exc:
            log.warning(f"State load failed: {exc} — starting fresh")
            return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.tg.start()
        self._running = True
        log.info("Trading loop started")
        await self._load_exchange_info()

        # Load persisted state first — restores positions, capital, etc.
        loaded = self._load_state()

        if loaded and self.positions:
            log.info(f"Resumed with {len(self.positions)} persisted positions: {list(self.positions.keys())}")
            await self.tg.send(
                f"🔄 <b>Bot restarted — state restored</b>\n"
                f"Positions: <code>{', '.join(self.positions.keys())}</code>\n"
                f"Capital: <code>{_usd(self.capital)}</code>\n"
                f"New plan on Monday SGT."
            )
        else:
            log.info("Startup: no persisted state — placing $10 seed position")
            await self.tg.send("🔄 <b>Bot started</b> — placing seed position. New plan on Monday SGT.")
            # Place a tiny $10 starter so rebalance/guarantee have something to work with
            try:
                snap0 = await asyncio.to_thread(generate_signals, self.binance, SYMBOLS, self.capital)
                if snap0.selected_symbols:
                    top_sym   = snap0.selected_symbols[0]
                    top_sig   = snap0.signals.get(top_sym)
                    top_price = top_sig.price if top_sig else 0.0
                    if top_sig and not math.isnan(top_price) and top_price > 0:
                        regime = snap0.regime.regime
                        if regime == "BULL":
                            stop_pct = P["bull_position_stop"]; tp_pct = 0.0
                        elif regime == "BEAR":
                            stop_pct = P["bear_position_stop"]
                            tp_pct   = P["bear_rr_ratio"] * stop_pct
                        else:
                            stop_pct = P["sw_position_stop"]
                            tp_pct   = P["sw_rr_ratio"] * stop_pct
                        await self._place_buy(
                            sym           = top_sym,
                            alloc         = 10.0,
                            price         = top_price,
                            stop_price    = top_price * (1 - stop_pct),
                            tp_price      = top_price * (1 + tp_pct) if tp_pct > 0 else float("inf"),
                            regime        = regime,
                            entry_trigger = "startup_seed",
                        )
                        self._weekly_plan   = snap0
                        self._plan_week_key = snap0.computed_at.isocalendar()[0] * 100 + snap0.computed_at.isocalendar()[1]
                        self._plan_anchor   = datetime.now(SGT)
            except Exception as exc:
                log.warning(f"Startup seed failed: {exc}")
        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    await self._cycle()
                except Exception as exc:
                    log.exception(f"Cycle error: {exc}")
                    await self.tg.notify_error(str(exc))
                await asyncio.sleep(max(0.0, LOOP_INTERVAL_S - (time.monotonic() - t0)))
        finally:
            await self.tg.stop()
            self.roostoo.close()
            log.info("Bot shut down")

    # ── Main cycle ────────────────────────────────────────────────────────────

    async def _cycle(self) -> None:
        now     = datetime.now(SGT)
        iso     = now.isocalendar()
        wk      = iso[0] * 100 + iso[1]   # unique week key across year boundaries
        weekday = now.weekday()            # 0=Mon … 6=Sun
        today   = now.strftime("%Y-%m-%d")

        log.info(f"──── {now.strftime('%Y-%m-%d %H:%M:%S SGT')}  "
                 f"pos={list(self.positions)} ────")

        # 1. Balance sync
        await self._refresh_balance()

        # 2. Fresh signals (prices needed for exit checks every cycle)
        snap = await asyncio.to_thread(
            generate_signals, self.binance, SYMBOLS, self.capital)
        self.last_snapshot = snap
        for sym, sig in snap.signals.items():
            if not math.isnan(sig.price):
                self.last_prices[sym] = sig.price

        # 3. Weekly plan — only created on Monday SGT.
        #    On first startup we never auto-create a plan regardless of day,
        #    to avoid accidentally closing live positions. Monday will reconcile.
        first_run  = self._plan_week_key == -1
        too_late   = weekday >= 4    # Fri(4) Sat(5) Sun(6) — don't create new plans
        is_friday  = weekday == 4    # Friday only — don't open new positions either
        new_monday = (weekday == 0) and (wk != self._plan_week_key)

        if first_run:
            # Just mark the current week so we don't spam logs, but don't trade
            if self._plan_week_key != wk:
                self._plan_week_key = wk
                log.info(f"Startup ({now.strftime('%A')} SGT) — holding existing positions, plan on Monday")
        elif new_monday:
            self._weekly_plan    = snap
            self._plan_week_key  = wk
            self._plan_anchor    = now
            self._week_closed    = False
            log.info(f"📅 Plan (Monday week {iso[1]}): {snap.regime.regime}  {snap.selected_symbols}")

            # Reconcile carried positions against new plan:
            # - Coins NOT in new plan → full close (plan_change)
            # - Coins IN new plan but wrong size → delta adjust
            # - Coins IN new plan at right size → leave untouched
            new_selected = set(snap.selected_symbols)

            # 1. Full close for dropped coins
            dropped = [s for s in list(self.positions.keys())
                       if s not in new_selected]
            if dropped:
                log.info(f"Plan change: closing dropped {dropped}")
                asyncio.create_task(self.tg.send(
                    f"🔄 <b>Plan change — week {iso[1]}</b>\n"
                    f"Closing: <code>{', '.join(dropped)}</code>\n"
                    f"Keeping/adjusting: <code>"
                    f"{', '.join(s for s in snap.selected_symbols if s in self.positions) or 'none'}"
                    f"</code>\n"
                    f"New entries: <code>"
                    f"{', '.join(s for s in snap.selected_symbols if s not in self.positions) or 'none'}"
                    f"</code>"
                ))
                for sym in dropped:
                    price = self.last_prices.get(sym, self.positions[sym]["entry_price"])
                    await self._place_sell(sym, trigger="plan_change", exit_price=price)
                    if self.positions:
                        await asyncio.sleep(MIN_ORDER_GAP_S)

            # 2. Delta-adjust carried positions still in the new plan
            for sym in [s for s in snap.selected_symbols if s in self.positions]:
                target_alloc = snap.capital_allocs.get(sym, 0.0)
                if target_alloc < MIN_ORDER_USD:
                    continue
                pos   = self.positions[sym]
                price = self.last_prices.get(sym, pos["entry_price"])
                cur_val  = pos["shares"] * price
                delta    = target_alloc - cur_val   # + = need more, - = have too much

                if delta > MIN_ORDER_USD:
                    log.info(f"Plan adjust: top-up {sym}  cur={_usd(cur_val)}  "
                             f"tgt={_usd(target_alloc)}  delta=+{_usd(delta)}")
                    regime = snap.regime.regime
                    if regime == "BULL":
                        stop_pct = P["bull_position_stop"]; tp_pct = 0.0
                    elif regime == "BEAR":
                        stop_pct = P["bear_position_stop"]
                        tp_pct   = P["bear_rr_ratio"] * stop_pct
                    else:
                        stop_pct = P["sw_position_stop"]
                        tp_pct   = P["sw_rr_ratio"] * stop_pct
                    await self._place_delta_buy(
                        sym=sym, delta_usd=delta, price=price,
                        stop_pct=stop_pct, tp_pct=tp_pct, regime=regime,
                        trigger="plan_adjust_topup")
                    if not self._can_order():
                        await asyncio.sleep(MIN_ORDER_GAP_S)

                elif delta < -MIN_ORDER_USD:
                    trim_usd = abs(delta)
                    log.info(f"Plan adjust: trim {sym}  cur={_usd(cur_val)}  "
                             f"tgt={_usd(target_alloc)}  delta=-{_usd(trim_usd)}")
                    await self._place_delta_sell(
                        sym=sym, trim_usd=trim_usd, price=price,
                        trigger="plan_adjust_trim")
                    if not self._can_order():
                        await asyncio.sleep(MIN_ORDER_GAP_S)

            asyncio.create_task(self.tg.send(
                f"📅 <b>New plan — Monday week {iso[1]}</b>\n"
                f"Regime: <code>{snap.regime.regime}</code>\n"
                f"Selected: <code>{', '.join(snap.selected_symbols) or 'none'}</code>"
            ))

        # 4. Regime change alert
        new_regime = snap.regime.regime
        if self.last_regime not in ("UNKNOWN",) and new_regime != self.last_regime:
            asyncio.create_task(
                self.tg.notify_regime_change(self.last_regime, new_regime))
        self.last_regime = new_regime

        # 5. BEAR_GATE — alert only, no forced liquidation
        #    Individual position stops will handle exits if the market keeps falling.
        if new_regime == "BEAR_GATE":
            if self.positions:
                log.warning("BEAR_GATE detected — alerting only, NOT liquidating (stops will handle exits)")
                asyncio.create_task(self.tg.send(
                    "🚫 <b>BEAR_GATE</b> — BTC −10% this week.\n"
                    "Monitoring existing positions. No new entries until regime clears."))

        # 6. Weekly performance tracking
        self._tick_weekly(now)

        # 7. Portfolio stop — alert only, does not liquidate
        await self._check_portfolio_stop()

        # 8. Per-position exit checks (stop / TP / trailing / RSI)
        await self._check_exits(snap)

        # 9. Daily micro-rebalance at 09:00 SGT
        if (now.hour == ROTATION_HOUR
                and today != self._last_rebalance_date
                and new_regime != "BEAR_GATE"
                and not self._week_closed
                and not is_friday):
            await self._daily_rebalance(snap, now)

        # 10. Enter unowned plan coins (1 order/cycle, not on Friday)
        if (self._weekly_plan is not None
                and new_regime != "BEAR_GATE"
                and not self._week_closed
                and not is_friday):
            await self._enter_positions(self._weekly_plan, now)

        # 11. Daily guarantee — 20:00 SGT fallback: if no order placed today
        no_order_today = self._last_order_date != today
        if (now.hour == 20
                and no_order_today
                and new_regime != "BEAR_GATE"
                and not self._week_closed
                and not is_friday
                and self._weekly_plan is not None
                and any(s not in self.positions
                        for s in self._weekly_plan.selected_symbols)):
            log.info("Daily guarantee: forcing entry at 20:00 SGT")
            asyncio.create_task(self.tg.send(
                f"⏰ <b>Daily guarantee entry</b> — {today}\n"
                f"No order placed yet. Entering best available coin now."))
            await self._enter_positions(self._weekly_plan, now, force=True)

        # 12. Friday 22:00 SGT — weekly close
        if weekday == 4 and now.hour >= 22 and not self._week_closed:
            await self._weekly_close()

        log.info(f"capital={_usd(self.capital)}  peak={_usd(self.peak_capital)}  "
                 f"pos={list(self.positions)}  regime={new_regime}")
        self._save_state()

    # ── Exchange info ─────────────────────────────────────────────────────────

    async def _load_exchange_info(self) -> None:
        try:
            info = await asyncio.to_thread(self.roostoo.exchange_info)
            for pair, tp in info.trade_pairs.items():
                self._amount_precision[pair] = tp.amount_precision
            log.info(f"Loaded precision for {len(self._amount_precision)} pairs: "
                     + "  ".join(f"{p}={v}" for p, v in sorted(self._amount_precision.items())))
        except Exception as exc:
            log.warning(f"exchange_info failed: {exc} — using 6dp fallback")

    def _round_qty(self, pair: str, qty: float) -> float:
        """Floor quantity to exchange-mandated decimal places."""
        dp = self._amount_precision.get(pair, 6)
        if dp == 0:
            return float(math.floor(qty))
        factor = 10 ** dp
        return math.floor(qty * factor) / factor

    # ── Startup cleanup ───────────────────────────────────────────────────────

    async def _cleanup_orphan_positions(self) -> None:
        """Sell any coin holdings not tracked in self.positions."""
        log.info("Startup: checking for orphans…")
        try:
            resp = await asyncio.to_thread(self.roostoo.balance)
        except Exception as exc:
            log.warning(f"Orphan check failed: {exc}"); return
        if not resp.success:
            log.warning(f"Orphan balance error: {resp.err_msg}"); return

        tracked = {sym.replace("USDT", "") for sym in self.positions}
        orphans = [
            (asset + "USDT", asset + "/USD", asset, e.free)
            for asset, e in resp.wallet.items()
            if asset not in ("USD", "USDT") and e.free > 0 and asset not in tracked
        ]
        if not orphans:
            log.info("Startup: clean slate ✓"); return

        log.warning(f"Orphans found: {[o[2] for o in orphans]}")
        await self.tg.send(
            f"🧹 <b>Startup cleanup</b>\n"
            f"Untracked: <code>{', '.join(o[2] for o in orphans)}</code>\nLiquidating…")

        prices: dict[str, float] = {}
        for bsym, _, _, _ in orphans:
            try:
                t = await asyncio.to_thread(self.binance.get_symbol_ticker, symbol=bsym)
                prices[bsym] = float(t["price"])
            except Exception:
                prices[bsym] = 0.0

        sold = skipped = 0
        for bsym, pair, asset, qty in orphans:
            val = qty * prices.get(bsym, 0.0)
            if val < MIN_ORDER_USD:
                skipped += 1; continue
            wait = MIN_ORDER_GAP_S - (time.time() - self._last_order_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                r = await asyncio.to_thread(
                    self.roostoo.place_order, pair=pair, side=Side.SELL,
                    order_type=OrderType.MARKET, quantity=self._round_qty(pair, qty))
            except Exception as exc:
                log.error(f"Orphan SELL error {asset}: {exc}"); continue
            if not r.success:
                log.error(f"Orphan SELL rejected {asset}: {r.err_msg}"); continue
            self._last_order_ts = time.time()
            self.total_trades  += 1
            sold               += 1
            det = r.order_detail
            fp  = det.filled_aver_price if det and det.filled_aver_price else prices.get(bsym, 0)
            fq  = det.filled_quantity   if det and det.filled_quantity   else qty
            log.info(f"Orphan sold {asset}  {_usd(fp)} × {fq:.4f} = {_usd(fp*fq)}")
            asyncio.create_task(self.tg.notify_trade(
                action="SELL", symbol=bsym, pair=pair, price=fp, quantity=fq,
                value_usd=fp*fq, trigger="orphan_cleanup", regime="STARTUP"))

        log.info(f"Orphan cleanup done: sold={sold} skipped(dust)={skipped}")
        if sold:
            await self.tg.send(
                f"✅ <b>Cleanup done</b> — sold <code>{sold}</code>, "
                f"skipped <code>{skipped}</code> dust.")

    # ── Balance sync ──────────────────────────────────────────────────────────

    async def _refresh_balance(self) -> None:
        try:
            resp = await asyncio.to_thread(self.roostoo.balance)
        except Exception as exc:
            log.warning(f"Balance failed: {exc}"); return
        if not resp.success:
            log.warning(f"Balance error: {resp.err_msg}"); return
        usd_e    = resp.wallet.get("USD") or resp.wallet.get("USDT")
        free_usd = usd_e.free if usd_e else 0.0

        # Capital = initial capital + unrealised PnL on all open positions
        # PnL per position = (current price - entry price) × shares
        # Using initial_capital as the fixed reference so % PnL is vs starting portfolio
        unrealised_pnl = sum(
            pos["shares"] * (self.last_prices.get(sym, pos["entry_price"]) - pos["entry_price"])
            for sym, pos in self.positions.items()
        )
        self.capital      = self.initial_capital + unrealised_pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        log.debug(f"Balance: free_usd={_usd(free_usd)}  unrealised_pnl={_usd(unrealised_pnl)}  "
                  f"capital={_usd(self.capital)}  ({unrealised_pnl/self.initial_capital:+.2%})")

    # ── Portfolio stop ────────────────────────────────────────────────────────

    async def _check_portfolio_stop(self) -> bool:
        """
        Alert only — never liquidates positions.
        The per-position hard stops and trailing stops protect individual positions.
        A portfolio-level forced liquidation would crystallise losses at the worst moment.
        """
        threshold = self.peak_capital * (1.0 - P["portfolio_stop"])
        if self.capital >= threshold:
            return True
        log.warning(f"Portfolio stop threshold breached: {_usd(self.capital)} < {_usd(threshold)} — ALERT ONLY, not liquidating")
        asyncio.create_task(self.tg.notify_portfolio_stop(self.capital, self.peak_capital))
        return True   # always return True — individual stops handle exits

    # ── Per-position exit checks ──────────────────────────────────────────────

    async def _check_exits(self, snap: MarketSnapshot) -> None:
        """
        Mirrors backtest run_hourly_loop exit logic.
        Exit priority: take_profit > position_stop > trailing_stop > early_profit
        """
        for sym in list(self.positions.keys()):
            pos   = self.positions[sym]
            price = self.last_prices.get(sym)
            if price is None or math.isnan(price):
                continue

            # Compute drop-from-peak BEFORE updating highest_price
            from_hi = (price - pos["highest_price"]) / pos["highest_price"]
            pos["highest_price"] = max(pos["highest_price"], price)

            regime   = pos.get("regime", "SIDEWAYS")
            entry_px = pos["entry_price"]
            pos_ret  = (price - entry_px) / entry_px

            if regime == "BULL":
                stop_pct  = P["bull_position_stop"]
                be_trig   = P["bull_breakeven_trigger"]
                trail     = P["bull_trailing_stop"]
                rsi_exit  = P["bull_rsi_early_exit"]
                ep_profit = P["bull_early_exit_profit"]
            elif regime == "BEAR":
                stop_pct  = P["bear_position_stop"]
                be_trig   = P["bear_breakeven_trigger"]
                trail     = P["bear_trailing_stop"]
                rsi_exit  = P["bear_rsi_overbought"]
                ep_profit = P["bear_early_exit_profit"]
            else:
                stop_pct  = P["sw_position_stop"]
                be_trig   = P["sw_breakeven_trigger"]
                trail     = P["sw_trailing_stop"]
                rsi_exit  = P["sw_rsi_early_exit"]
                ep_profit = P["sw_early_exit_profit"]

            trigger: Optional[str] = None
            if pos["take_profit"] != float("inf") and price >= pos["take_profit"]:
                trigger = "take_profit"
            elif price <= entry_px * (1.0 - stop_pct):
                trigger = "position_stop"
            elif pos["highest_price"] >= entry_px * (1.0 + be_trig) and from_hi <= -trail:
                trigger = "trailing_stop"
            elif regime != "BULL":
                sig = snap.signals.get(sym)
                if sig and sig.rsi > rsi_exit and pos_ret > ep_profit:
                    trigger = "early_profit"

            if trigger:
                await self._place_sell(sym, trigger=trigger, exit_price=price)

    # ── Daily micro-rebalance ─────────────────────────────────────────────────

    async def _daily_rebalance(self, snap: MarketSnapshot, now: datetime) -> None:
        """
        09:00 SGT each trading day: nudge each held position back toward its
        target weight if it has drifted by more than REBALANCE_DRIFT_THRESHOLD.

        Why this is strategy-neutral:
          - Same coins, same regime, same target weights as the weekly plan
          - Only the SIZE changes slightly, not which coins are held
          - Trims winners (sells a tiny slice) / tops up losers (buys a tiny slice)
          - Identical to standard variance-weight rebalancing in the literature

        Competition requirement: guarantees ≥ 1 real order per trading day
        without manufacturing artificial churn or changing strategy intent.

        Only ONE order fires per rebalance window (rate limit respected).
        The position with the largest absolute drift goes first.
        """
        self._last_rebalance_date = now.strftime("%Y-%m-%d")

        if not self.positions or self._weekly_plan is None:
            log.info("Rebalance: no positions to rebalance")
            return

        total_value = sum(
            pos["shares"] * self.last_prices.get(sym, pos["entry_price"])
            for sym, pos in self.positions.items()
        )
        if total_value <= 0:
            return

        # Compute current weight and target weight for each held position
        target_allocs = self._weekly_plan.capital_allocs
        target_total  = sum(target_allocs.get(s, 0.0) for s in self.positions)
        if target_total <= 0:
            return

        drifts: list[tuple[float, str, float, float]] = []  # (abs_drift, sym, cur_val, tgt_val)
        for sym, pos in self.positions.items():
            cur_val = pos["shares"] * self.last_prices.get(sym, pos["entry_price"])
            cur_wt  = cur_val / total_value
            tgt_wt  = target_allocs.get(sym, 0.0) / target_total
            drift   = cur_wt - tgt_wt   # positive = overweight, negative = underweight
            if abs(drift) >= REBALANCE_DRIFT_THRESHOLD:
                drifts.append((abs(drift), sym, cur_val, tgt_wt * total_value))

        if not drifts:
            log.info(f"Rebalance: all positions within {REBALANCE_DRIFT_THRESHOLD:.0%} of target — no action")
            # Still counts as the rebalance check for today; guarantee will cover order if needed
            return

        # Sort by absolute drift descending — fix the most off-target first
        drifts.sort(reverse=True)
        abs_drift, sym, cur_val, tgt_val = drifts[0]
        price = self.last_prices.get(sym, self.positions[sym]["entry_price"])
        pos   = self.positions[sym]

        delta_usd = tgt_val - cur_val   # negative = trim, positive = top-up
        regime    = pos.get("regime", "SIDEWAYS")

        log.info(f"Rebalance: {sym}  cur={_usd(cur_val)}  tgt={_usd(tgt_val)}  "
                 f"delta={_usd(delta_usd)}  drift={abs_drift:.1%}")

        if delta_usd < -MIN_ORDER_USD:
            # Overweight — sell a small slice directly (bypass _place_sell
            # which sells the entire position; we only want a partial trim)
            trim_usd = abs(delta_usd)
            pair     = to_roostoo_pair(sym)
            trim_qty = self._round_qty(pair, trim_usd / price)
            if trim_qty <= 0 or trim_qty >= pos["shares"]:
                log.info(f"Rebalance: {sym} trim qty {trim_qty} invalid — skipping")
                return
            if not self._can_order():
                log.info(f"Rebalance: rate-limit, skipping trim of {sym}")
                return
            log.info(f"Rebalance trim {sym}: qty={trim_qty}  ≈{_usd(trim_usd)}")
            try:
                resp = await asyncio.to_thread(
                    self.roostoo.place_order, pair=pair, side=Side.SELL,
                    order_type=OrderType.MARKET, quantity=trim_qty)
            except Exception as exc:
                log.error(f"Rebalance trim error {sym}: {exc}"); return
            if not resp.success:
                log.error(f"Rebalance trim rejected {sym}: {resp.err_msg}"); return
            self._last_order_ts   = time.time()
            self._last_order_date = now.strftime("%Y-%m-%d")
            self.total_trades    += 1
            det = resp.order_detail
            fp  = det.filled_aver_price if det and det.filled_aver_price else price
            fq  = det.filled_quantity   if det and det.filled_quantity   else trim_qty
            pos["shares"]  -= fq
            pos["alloc"]   -= fq * fp
            log.info(f"✓ Rebalance trim {sym}: sold {fq}  @ {_usd(fp)}  "
                     f"remaining shares={pos['shares']:.4f}")
            asyncio.create_task(self.tg.notify_trade(
                action="SELL", symbol=sym, pair=pair, price=fp, quantity=fq,
                value_usd=fq * fp, trigger="rebalance_trim",
                regime=pos.get("regime", "?")))
            asyncio.create_task(self.tg.send(
                f"⚖️ <b>Rebalance trim</b> — {sym}\n"
                f"Overweight <code>{abs_drift:.1%}</code>. "
                f"Trimmed <code>{_usd(fq * fp)}</code>."))

        elif delta_usd > MIN_ORDER_USD:
            # Underweight — buy a small top-up
            alloc    = min(delta_usd, self.capital * 0.10)   # cap at 10% of capital
            tp_price = pos["take_profit"]
            # Re-derive stop_pct from regime
            regime_at_entry = pos.get("regime", "SIDEWAYS")
            if regime_at_entry == "BULL":
                stop_pct = P["bull_position_stop"]
            elif regime_at_entry == "BEAR":
                stop_pct = P["bear_position_stop"]
            else:
                stop_pct = P["sw_position_stop"]
            placed = await self._place_buy(
                sym=sym, alloc=alloc, price=price,
                stop_price=price * (1.0 - stop_pct),
                tp_price=tp_price, regime=regime_at_entry,
                entry_trigger="rebalance_topup")
            if placed:
                asyncio.create_task(self.tg.send(
                    f"⚖️ <b>Rebalance top-up</b> — {sym}\n"
                    f"Underweight <code>{abs_drift:.1%}</code>. "
                    f"Adding <code>{_usd(alloc)}</code>."))
        else:
            log.info(f"Rebalance: {sym} delta {_usd(delta_usd)} below minimum — skipping")

    # ── Position entries ──────────────────────────────────────────────────────

    async def _enter_positions(self, snap: MarketSnapshot, now: datetime,
                               force: bool = False) -> None:
        """
        Enter unowned selected coins from the weekly plan, 1 order per cycle.
        Never enters on Fri/Sat/Sun (too_late guard) — checked here as a
        second safety layer in addition to the caller-side guard in _cycle.
        """
        # Hard guard — never open new positions on Fri/Sat/Sun
        if now.weekday() >= 4 and not force:
            return
        GUARANTEE_ALLOC = max(self.capital * 0.01, 20.0)

        regime   = snap.regime.regime
        plan_age = int((now - self._plan_anchor).total_seconds() // 86400) \
                   if self._plan_anchor else 0

        mtm_held  = sum(
            pos["shares"] * self.last_prices.get(sym, pos["entry_price"])
            for sym, pos in self.positions.items())
        free_cash = max(self.capital - mtm_held, 0.0)

        unfilled  = [s for s in snap.selected_symbols if s not in self.positions]
        # Also consider coins already held but undersized vs new target
        # (carried from last week — plan_change delta may not have fired yet)
        undersized = []
        for sym in snap.selected_symbols:
            if sym in self.positions and sym not in unfilled:
                target = snap.capital_allocs.get(sym, 0.0)
                cur    = self.positions[sym]["shares"] * self.last_prices.get(
                    sym, self.positions[sym]["entry_price"])
                if target - cur > MIN_ORDER_USD:
                    undersized.append(sym)

        if not unfilled and not undersized:
            return
        raw_total = sum(snap.capital_allocs.get(s, 0.0) for s in unfilled)

        for sym in unfilled + undersized:
            sig   = snap.signals.get(sym)
            price = self.last_prices.get(sym)
            if sig is None or price is None or math.isnan(price):
                continue

            is_topup = sym in undersized   # already held, just adding delta

            if force:
                alloc   = GUARANTEE_ALLOC
                trigger = "daily_guarantee"

            elif is_topup:
                # Carried position — just fill the gap to target, no timing gate
                target  = snap.capital_allocs.get(sym, 0.0)
                cur_val = self.positions[sym]["shares"] * price
                alloc   = target - cur_val
                trigger = "carry_topup"

            elif regime == "BULL":
                alloc   = snap.capital_allocs.get(sym, 0.0) * (free_cash / raw_total) \
                          if raw_total > 0 else 0.0
                alloc   = min(alloc, free_cash * 0.80)
                trigger = "breakout_open"

            elif sig.is_red_day:
                alloc   = snap.capital_allocs.get(sym, 0.0) * (free_cash / raw_total) \
                          if raw_total > 0 else 0.0
                alloc   = min(alloc, free_cash * 0.80)
                trigger = "red_day"

            elif plan_age >= 3:
                alloc   = snap.capital_allocs.get(sym, 0.0) * (free_cash / raw_total) \
                          if raw_total > 0 else 0.0
                alloc   = min(alloc, free_cash * 0.80)
                trigger = "fallback_thursday"

            else:
                log.debug(f"Waiting for red day: {sym} (plan day {plan_age})")
                continue

            if alloc < MIN_ORDER_USD:
                continue

            stop_price = price * (1.0 - sig.stop_pct)
            tp_price   = price * (1.0 + sig.tp_pct) if sig.tp_pct > 0 else float("inf")

            if is_topup:
                # Delta buy — merges into existing position
                placed = await self._place_delta_buy(
                    sym=sym, delta_usd=alloc, price=price,
                    stop_pct=sig.stop_pct, tp_pct=sig.tp_pct,
                    regime=regime, trigger=trigger)
            else:
                placed = await self._place_buy(
                    sym=sym, alloc=alloc, price=price,
                    stop_price=stop_price, tp_price=tp_price,
                    regime=regime, entry_trigger=trigger)
            if placed:
                return   # one order per cycle

    # ── Weekly close ──────────────────────────────────────────────────────────

    async def _weekly_close(self) -> None:
        """Friday 22:00 SGT: close all open positions."""
        self._week_closed = True
        if not self.positions:
            return
        log.info(f"Friday EOD: closing {list(self.positions)}")
        await self.tg.send(
            f"🗓 <b>Friday EOD — weekly close</b>\n"
            f"Closing: <code>{', '.join(self.positions)}</code>")
        for sym in list(self.positions.keys()):
            price = self.last_prices.get(sym, self.positions[sym]["entry_price"])
            await self._place_sell(sym, trigger="weekly_exit", exit_price=price)
            if self.positions:
                await asyncio.sleep(MIN_ORDER_GAP_S)

    # ── Order primitives ──────────────────────────────────────────────────────

    def _can_order(self) -> bool:
        return (time.time() - self._last_order_ts) >= MIN_ORDER_GAP_S

    async def _place_buy(self, *, sym: str, alloc: float, price: float,
                         stop_price: float, tp_price: float,
                         regime: str, entry_trigger: str) -> bool:
        if not self._can_order():
            log.info(f"Rate-limit: skip BUY {sym} "
                     f"({time.time()-self._last_order_ts:.0f}s since last order)")
            return False

        pair = to_roostoo_pair(sym)
        qty  = self._round_qty(pair, alloc / price)
        if qty <= 0:
            log.warning(f"Skip BUY {sym}: qty rounds to 0  alloc={_usd(alloc)} price={_usd(price)}")
            return False

        log.info(f"→ BUY {sym}  pair={pair}  qty={qty}  "
                 f"alloc={_usd(alloc)}  price={_usd(price)}  trigger={entry_trigger}")
        try:
            resp = await asyncio.to_thread(
                self.roostoo.place_order, pair=pair, side=Side.BUY,
                order_type=OrderType.MARKET, quantity=qty)
        except Exception as exc:
            log.error(f"BUY error {sym}: {exc}"); return False
        if not resp.success:
            log.error(f"BUY rejected {sym}: {resp.err_msg}"); return False

        self._last_order_ts   = time.time()
        self._last_order_date = datetime.now(SGT).strftime("%Y-%m-%d")
        self.total_trades    += 1

        det        = resp.order_detail
        fill_price = det.filled_aver_price if det and det.filled_aver_price else price
        fill_qty   = det.filled_quantity    if det and det.filled_quantity    else qty
        order_id   = det.order_id           if det else None

        if regime == "BULL":
            actual_stop = fill_price * (1.0 - P["bull_position_stop"])
            actual_tp   = float("inf")
        elif regime == "BEAR":
            actual_stop = fill_price * (1.0 - P["bear_position_stop"])
            actual_tp   = fill_price * (1.0 + P["bear_rr_ratio"] * P["bear_position_stop"])
        else:
            actual_stop = fill_price * (1.0 - P["sw_position_stop"])
            actual_tp   = fill_price * (1.0 + P["sw_rr_ratio"] * P["sw_position_stop"])

        self.positions[sym] = dict(
            entry_price   = fill_price,
            shares        = fill_qty,
            alloc         = alloc,
            stop_price    = actual_stop,
            take_profit   = actual_tp,
            highest_price = fill_price,
            regime        = regime,
            entry_time    = _now_sgt(),
            order_id      = order_id,
        )
        log.info(f"✓ BUY {sym}  fill={_usd(fill_price)}  qty={fill_qty}  "
                 f"stop={_usd(actual_stop)}  "
                 f"tp={'inf' if actual_tp == float('inf') else _usd(actual_tp)}")
        asyncio.create_task(self.tg.notify_trade(
            action="BUY", symbol=sym, pair=pair,
            price=fill_price, quantity=fill_qty, value_usd=alloc,
            trigger=entry_trigger, regime=regime))
        return True

    async def _place_delta_buy(self, *, sym: str, delta_usd: float, price: float,
                               stop_pct: float, tp_pct: float,
                               regime: str, trigger: str) -> bool:
        """
        Buy additional shares of an already-held position (top-up).
        Merges into the existing position dict rather than creating a new one.
        """
        if not self._can_order():
            log.info(f"Rate-limit: skip delta BUY {sym}")
            return False

        pair = to_roostoo_pair(sym)
        qty  = self._round_qty(pair, delta_usd / price)
        if qty <= 0:
            log.warning(f"Skip delta BUY {sym}: qty rounds to 0")
            return False

        log.info(f"→ DELTA BUY {sym}  qty={qty}  ≈{_usd(delta_usd)}  trigger={trigger}")
        try:
            resp = await asyncio.to_thread(
                self.roostoo.place_order, pair=pair, side=Side.BUY,
                order_type=OrderType.MARKET, quantity=qty)
        except Exception as exc:
            log.error(f"Delta BUY error {sym}: {exc}"); return False
        if not resp.success:
            log.error(f"Delta BUY rejected {sym}: {resp.err_msg}"); return False

        self._last_order_ts   = time.time()
        self._last_order_date = datetime.now(SGT).strftime("%Y-%m-%d")
        self.total_trades    += 1

        det        = resp.order_detail
        fill_price = det.filled_aver_price if det and det.filled_aver_price else price
        fill_qty   = det.filled_quantity    if det and det.filled_quantity    else qty

        if sym in self.positions:
            pos = self.positions[sym]
            # Weighted average entry price
            old_cost = pos["entry_price"] * pos["shares"]
            new_cost = fill_price * fill_qty
            pos["shares"]      += fill_qty
            pos["alloc"]       += delta_usd
            pos["entry_price"]  = (old_cost + new_cost) / pos["shares"]
            # Keep tightest stop (don't move stop up on top-up)
            new_stop = fill_price * (1 - stop_pct)
            pos["stop_price"]   = max(pos["stop_price"], new_stop)
            if tp_pct > 0:
                pos["take_profit"] = pos["entry_price"] * (1 + tp_pct)
        else:
            # Shouldn't happen but handle gracefully
            stop_price = fill_price * (1 - stop_pct)
            tp_price   = fill_price * (1 + tp_pct) if tp_pct > 0 else float("inf")
            self.positions[sym] = dict(
                entry_price=fill_price, shares=fill_qty, alloc=delta_usd,
                stop_price=stop_price, take_profit=tp_price,
                highest_price=fill_price, regime=regime,
                entry_time=_now_sgt(), order_id=None)

        log.info(f"✓ DELTA BUY {sym}  fill={_usd(fill_price)}  qty={fill_qty}  trigger={trigger}")
        asyncio.create_task(self.tg.notify_trade(
            action="BUY", symbol=sym, pair=pair,
            price=fill_price, quantity=fill_qty, value_usd=delta_usd,
            trigger=trigger, regime=regime))
        return True

    async def _place_delta_sell(self, *, sym: str, trim_usd: float, price: float,
                                trigger: str) -> bool:
        """
        Sell a partial slice of an existing position (trim).
        Reduces shares and alloc proportionally without closing the full position.
        """
        if sym not in self.positions:
            return False
        if not self._can_order():
            log.info(f"Rate-limit: skip delta SELL {sym}")
            return False

        pos      = self.positions[sym]
        pair     = to_roostoo_pair(sym)
        trim_qty = self._round_qty(pair, trim_usd / price)
        # Never sell more than we hold
        trim_qty = min(trim_qty, self._round_qty(pair, pos["shares"] * 0.99))
        if trim_qty <= 0:
            log.warning(f"Skip delta SELL {sym}: qty rounds to 0")
            return False

        log.info(f"→ DELTA SELL {sym}  qty={trim_qty}  ≈{_usd(trim_usd)}  trigger={trigger}")
        try:
            resp = await asyncio.to_thread(
                self.roostoo.place_order, pair=pair, side=Side.SELL,
                order_type=OrderType.MARKET, quantity=trim_qty)
        except Exception as exc:
            log.error(f"Delta SELL error {sym}: {exc}"); return False
        if not resp.success:
            log.error(f"Delta SELL rejected {sym}: {resp.err_msg}"); return False

        self._last_order_ts   = time.time()
        self._last_order_date = datetime.now(SGT).strftime("%Y-%m-%d")
        self.total_trades    += 1

        det        = resp.order_detail
        fill_price = det.filled_aver_price if det and det.filled_aver_price else price
        fill_qty   = det.filled_quantity    if det and det.filled_quantity    else trim_qty
        proceeds   = fill_qty * fill_price
        pnl        = fill_qty * (fill_price - pos["entry_price"])

        # Reduce position proportionally
        fraction        = fill_qty / pos["shares"]
        pos["shares"]  -= fill_qty
        pos["alloc"]   -= pos["alloc"] * fraction

        log.info(f"✓ DELTA SELL {sym}  fill={_usd(fill_price)}  qty={fill_qty}  "
                 f"pnl={_usd(pnl)}  remaining={pos['shares']:.4f}  trigger={trigger}")
        asyncio.create_task(self.tg.notify_trade(
            action="SELL", symbol=sym, pair=pair,
            price=fill_price, quantity=fill_qty, value_usd=proceeds,
            trigger=trigger, regime=pos.get("regime", "?"), pnl=pnl))
        return True

    async def _place_sell(self, sym: str, *, trigger: str, exit_price: float) -> None:
        if sym not in self.positions:
            return
        if not self._can_order():
            log.info(f"Rate-limit: skip SELL {sym} "
                     f"({time.time()-self._last_order_ts:.0f}s since last order)")
            return

        pos  = self.positions[sym]
        pair = to_roostoo_pair(sym)
        qty  = self._round_qty(pair, pos["shares"])

        log.info(f"→ SELL {sym}  pair={pair}  qty={qty}  "
                 f"price≈{_usd(exit_price)}  trigger={trigger}")
        try:
            resp = await asyncio.to_thread(
                self.roostoo.place_order, pair=pair, side=Side.SELL,
                order_type=OrderType.MARKET, quantity=qty)
        except Exception as exc:
            log.error(f"SELL error {sym}: {exc}"); return
        if not resp.success:
            log.error(f"SELL rejected {sym}: {resp.err_msg}"); return

        self._last_order_ts   = time.time()
        self._last_order_date = datetime.now(SGT).strftime("%Y-%m-%d")
        self.total_trades    += 1

        det        = resp.order_detail
        fill_price = det.filled_aver_price if det and det.filled_aver_price else exit_price
        fill_qty   = det.filled_quantity    if det and det.filled_quantity    else qty
        # Use actual commission from Roostoo if available; else estimate taker fee
        commission = (det.commission_charge_value
                      if det and det.commission_charge_value
                      else fill_qty * fill_price * P["taker_fee"])
        proceeds   = fill_qty * fill_price - commission
        pnl        = proceeds - pos["alloc"]

        del self.positions[sym]
        self.capital += pnl

        log.info(f"✓ SELL {sym}  fill={_usd(fill_price)}  fee={_usd(commission)}  "
                 f"pnl={_usd(pnl)}  trigger={trigger}")
        asyncio.create_task(self.tg.notify_trade(
            action="SELL", symbol=sym, pair=pair,
            price=fill_price, quantity=fill_qty, value_usd=proceeds,
            trigger=trigger, regime=pos.get("regime", "?"), pnl=pnl))

    # ── Performance ───────────────────────────────────────────────────────────

    def _tick_weekly(self, now: datetime) -> None:
        iso = now.isocalendar()
        wk  = iso[0] * 100 + iso[1]
        if self._current_week_key == -1:
            self._current_week_key = wk
            self._week_start_cap   = self.capital
            return
        if wk != self._current_week_key:
            if self._week_start_cap > 0:
                wr = (self.capital - self._week_start_cap) / self._week_start_cap
                self._weekly_returns.append(wr)
                log.info(f"Weekly return: {wr:+.3%}  n={len(self._weekly_returns)}")
            self._current_week_key = wk
            self._week_start_cap   = self.capital

    def metrics(self) -> dict:
        r = np.array(list(self._weekly_returns), dtype=float)
        if len(r) < 2:
            return dict(sharpe=np.nan, sortino=np.nan, calmar=np.nan,
                        max_drawdown=0.0, win_rate=0.0)
        avg   = float(r.mean())
        std   = float(r.std(ddof=1)) if len(r) > 1 else 1e-9
        neg   = r[r < 0]
        std_d = float(neg.std(ddof=1)) if len(neg) > 1 else 1e-9
        ann   = (1.0 + avg) ** 52 - 1.0
        mdd   = float((self.capital - self.peak_capital) / self.peak_capital) \
                if self.peak_capital else 0.0
        return dict(
            sharpe       = avg / std  * 52 ** 0.5,
            sortino      = avg / std_d * 52 ** 0.5,
            calmar       = ann / abs(mdd) if mdd != 0 else np.nan,
            max_drawdown = mdd,
            win_rate     = float((r > 0).mean()),
        )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    await TradingBot().run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted")