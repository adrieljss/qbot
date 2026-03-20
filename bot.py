"""
bot.py  —  MRVS Regime-Switching Trading Bot
=============================================
Main execution loop.  Every 60 seconds:
  1. Sync portfolio balance from Roostoo
  2. Fetch live signals from Binance (unauthenticated, daily bars)
  3. Detect regime changes  →  Telegram notification
  4. Check portfolio-level stop-loss
  5. Check per-position exits (hard stop, TP, trailing, RSI early-exit)
  6. Enter new positions for selected coins not currently held

Order rate-limit: max 1 order per 61 seconds (enforced via timestamp gap).

Run:
    python bot.py

Requires: .env file with ROOSTOO_API_KEY, ROOSTOO_SECRET, TELEGRAM_BOT_TOKEN,
          TELEGRAM_CHANNEL_ID, TELEGRAM_TOPIC_ID (optional), INITIAL_CAPITAL.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# ── Env must load before any other import that reads env vars ─────────────────
load_dotenv()

from binance.client import Client as BinanceClient
from signals import (
    SYMBOLS, P, MarketSnapshot,
    generate_signals, to_roostoo_pair,
)
from telegram_notifier import TelegramNotifier

# Roostoo client lives in roostoo_api/roostoo_api.py (provided by hackathon)
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

# ── Runtime constants ─────────────────────────────────────────────────────────
LOOP_INTERVAL_S = 60    # target cycle cadence
MIN_ORDER_GAP_S = 61    # enforced gap between any two orders (≥1 order/min)
MIN_ORDER_USD   = 10.0  # skip allocations below this threshold


class TradingBot:
    """
    All live state in one place.

    Positions dict structure:
        { "BTCUSDT": {
            entry_price:   float,   # average fill price
            shares:        float,   # units held
            alloc:         float,   # USD deployed at entry
            stop_price:    float,   # hard stop — sell if price ≤ this
            take_profit:   float,   # TP level; float("inf") = disabled
            highest_price: float,   # rolling peak since entry (for trailing)
            regime:        str,     # regime at entry
            entry_time:    str,     # ISO timestamp string
            order_id:      int|None,
        }, … }
    """

    SYMBOLS = SYMBOLS

    def __init__(self) -> None:
        # ── API clients ───────────────────────────────────────────────────
        self.roostoo  = RoostooClient.from_env()
        self.binance  = BinanceClient("", "")   # unauthenticated
        self.tg       = TelegramNotifier(bot_ref=self)

        # ── Capital ───────────────────────────────────────────────────────
        self.initial_capital = float(os.getenv("INITIAL_CAPITAL", "1000000"))
        self.capital         = self.initial_capital
        self.peak_capital    = self.initial_capital

        # ── State ─────────────────────────────────────────────────────────
        self.positions:   dict[str, dict]  = {}
        self.last_prices: dict[str, float] = {}
        self.last_snapshot: Optional[MarketSnapshot] = None
        self.last_regime    = "UNKNOWN"

        # ── Weekly plan (set once per week, Monday open) ───────────────────
        # Mirrors backtest: entry decisions are made once at week start and
        # held until Friday close or stop/TP.  Only exit monitoring runs
        # every 60s.  Reset each Monday (or on first startup).
        self._plan_isoweek: int = -1          # ISO week the plan was made for
        self._plan_monday:  Optional[datetime] = None   # exact Monday anchor
        self._weekly_plan:  Optional[MarketSnapshot] = None  # locked snapshot
        self._week_closed:  bool = False      # True after Friday EOD close fires

        # ── Exchange info (amount_precision per pair, loaded at startup) ──
        # Maps Roostoo pair string → int decimal places, e.g. "TRX/USD" → 0
        self._amount_precision: dict[str, int] = {}

        # ── Order rate-limiter ────────────────────────────────────────────
        self._last_order_ts: float = 0.0

        # ── Performance ───────────────────────────────────────────────────
        self.total_trades        = 0
        self._weekly_returns: deque = deque(maxlen=520)  # ≈10 years
        self._week_start_cap: float = self.initial_capital
        self._current_isoweek: int  = -1

        self._running = False
        log.info(f"TradingBot init  initial_capital=${self.initial_capital:,.0f}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.tg.start()
        self._running = True
        log.info("Trading loop started — press Ctrl+C to stop")

        # Load per-pair amount precision from Roostoo exchange info
        await self._load_exchange_info()

        # Sell any coin balances left over from a previous run
        await self._cleanup_orphan_positions()

        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    await self._cycle()
                except Exception as exc:
                    log.exception(f"Unhandled cycle error: {exc}")
                    await self.tg.notify_error(str(exc))
                sleep = max(0.0, LOOP_INTERVAL_S - (time.monotonic() - t0))
                await asyncio.sleep(sleep)
        finally:
            await self.tg.stop()
            self.roostoo.close()
            log.info("Bot shut down cleanly")

    # ── Exchange info ─────────────────────────────────────────────────────────

    async def _load_exchange_info(self) -> None:
        """
        Fetch Roostoo exchange info and cache amount_precision per pair.
        This tells us how many decimal places each coin's quantity must have,
        e.g. TRX/USD → 0 (whole numbers only), ETH/USD → 4, BTC/USD → 5.
        """
        try:
            info = await asyncio.to_thread(self.roostoo.exchange_info)
        except Exception as exc:
            log.warning(f"exchange_info fetch failed: {exc} — will use 6dp fallback")
            return

        for pair, tp in info.trade_pairs.items():
            self._amount_precision[pair] = tp.amount_precision

        log.info(
            f"Loaded amount_precision for {len(self._amount_precision)} pairs: "
            + "  ".join(
                f"{p}={v}" for p, v in
                sorted(self._amount_precision.items())
            )
        )

    def _round_qty(self, pair: str, qty: float) -> float:
        """
        Truncate (floor) quantity to the exchange-mandated decimal places.
        Flooring rather than rounding ensures we never exceed available funds.
        Falls back to 6 dp if the pair wasn't in exchange_info.
        """
        dp = self._amount_precision.get(pair, 6)
        if dp == 0:
            return float(math.floor(qty))
        factor = 10 ** dp
        return math.floor(qty * factor) / factor

    # ── Startup cleanup ───────────────────────────────────────────────────────

    async def _cleanup_orphan_positions(self) -> None:
        """
        On startup, check the Roostoo wallet for any coin holdings that are
        NOT tracked in self.positions (leftovers from a previous bot run or
        manual trades) and market-sell them to return to clean cash.

        Coins with a USD value below MIN_ORDER_USD are skipped (dust).
        The 1-order/minute rate limit is honoured between each sell.
        """
        log.info("Startup: checking for orphan positions…")

        try:
            resp = await asyncio.to_thread(self.roostoo.balance)
        except Exception as exc:
            log.warning(f"Orphan cleanup: balance fetch failed: {exc}")
            return
        if not resp.success:
            log.warning(f"Orphan cleanup: balance error: {resp.err_msg}")
            return

        # Build set of tracked coin bases (e.g. {"BTC", "ETH"})
        tracked_bases = {sym.replace("USDT", "") for sym in self.positions}

        orphans: list[tuple[str, float]] = []   # (binance_symbol, quantity)
        for asset, entry in resp.wallet.items():
            if asset in ("USD", "USDT"):
                continue
            qty = entry.free
            if qty <= 0:
                continue

            # Convert to a Binance symbol and check whether it's tracked
            binance_sym = asset + "USDT"
            if asset in tracked_bases:
                continue   # bot already knows about this position

            # Check it's actually in our universe (or tradeable on Roostoo)
            pair = asset + "/USD"
            orphans.append((binance_sym, pair, asset, qty))

        if not orphans:
            log.info("Startup: no orphan positions found — clean slate ✓")
            return

        log.warning(
            f"Startup: found {len(orphans)} orphan holding(s): "
            f"{[o[2] for o in orphans]} — liquidating…"
        )
        await self.tg.send(
            f"🧹 <b>Startup cleanup</b>\n"
            f"Found {len(orphans)} untracked holding(s): "
            f"<code>{', '.join(o[2] for o in orphans)}</code>\n"
            f"Liquidating before trading begins…"
        )

        # Fetch current prices from Binance so we can log USD value
        prices: dict[str, float] = {}
        for binance_sym, pair, asset, qty in orphans:
            try:
                ticker = await asyncio.to_thread(
                    self.binance.get_symbol_ticker, symbol=binance_sym
                )
                prices[binance_sym] = float(ticker["price"])
            except Exception as exc:
                log.warning(f"Orphan cleanup: price fetch failed for {binance_sym}: {exc}")
                prices[binance_sym] = 0.0

        sold, skipped = 0, 0
        for binance_sym, pair, asset, qty in orphans:
            price     = prices.get(binance_sym, 0.0)
            value_usd = qty * price

            if value_usd < MIN_ORDER_USD:
                log.info(
                    f"Orphan cleanup: skipping {asset} "
                    f"(qty={qty:.6f}  value≈${value_usd:.2f} — dust)"
                )
                skipped += 1
                continue

            # Honour rate limit even during startup cleanup
            wait = MIN_ORDER_GAP_S - (time.time() - self._last_order_ts)
            if wait > 0:
                log.info(f"Orphan cleanup: rate-limit wait {wait:.0f}s before selling {asset}")
                await asyncio.sleep(wait)

            log.info(
                f"Orphan cleanup: SELL {asset}  pair={pair}  "
                f"qty={qty:.6f}  ≈${value_usd:.2f}"
            )
            try:
                resp_sell = await asyncio.to_thread(
                    self.roostoo.place_order,
                    pair       = pair,
                    side       = Side.SELL,
                    order_type = OrderType.MARKET,
                    quantity   = self._round_qty(pair, qty),
                )
            except Exception as exc:
                log.error(f"Orphan cleanup: SELL failed for {asset}: {exc}")
                continue

            if not resp_sell.success:
                log.error(f"Orphan cleanup: SELL rejected for {asset}: {resp_sell.err_msg}")
                continue

            self._last_order_ts = time.time()
            self.total_trades  += 1
            sold               += 1

            det          = resp_sell.order_detail
            filled_price = det.filled_aver_price if det and det.filled_aver_price else price
            filled_qty   = det.filled_quantity    if det and det.filled_quantity    else qty
            proceeds     = filled_qty * filled_price

            log.info(
                f"Orphan cleanup: sold {asset}  "
                f"filled_price={_usd(filled_price)}  proceeds={_usd(proceeds)}"
            )
            asyncio.create_task(
                self.tg.notify_trade(
                    action    = "SELL",
                    symbol    = binance_sym,
                    pair      = pair,
                    price     = filled_price,
                    quantity  = filled_qty,
                    value_usd = proceeds,
                    trigger   = "orphan_cleanup",
                    regime    = "STARTUP",
                )
            )

        log.info(
            f"Orphan cleanup complete: sold={sold}  skipped(dust)={skipped}"
        )
        if sold:
            await self.tg.send(
                f"✅ <b>Cleanup complete</b>  "
                f"sold <code>{sold}</code> position(s), "
                f"skipped <code>{skipped}</code> dust holding(s).\n"
                f"Bot is now trading."
            )

    async def _cycle(self) -> None:
        now       = datetime.now(timezone.utc)
        isoweek   = now.isocalendar()[1]
        isoyear   = now.isocalendar()[0]
        weekday   = now.weekday()   # 0=Mon … 4=Fri
        is_monday = weekday == 0
        week_key  = isoyear * 100 + isoweek   # unique across year boundaries

        log.info(f"──── cycle {now.strftime('%H:%M:%S UTC')} "
                 f"week={isoweek}  pos={list(self.positions)} ────")

        # 1. Sync balance from Roostoo
        await self._refresh_balance()

        # 2. Always fetch fresh signals — needed for exit price updates
        snap = await asyncio.to_thread(
            generate_signals,
            self.binance,
            SYMBOLS,
            self.capital,
        )
        self.last_snapshot = snap
        for sym, sig in snap.signals.items():
            if not math.isnan(sig.price):
                self.last_prices[sym] = sig.price

        # 3. Lock in a new weekly plan on Monday (or very first run)
        first_run    = self._plan_isoweek == -1
        new_monday   = is_monday and week_key != self._plan_isoweek
        need_new_plan = first_run or new_monday
        if need_new_plan:
            self._weekly_plan   = snap
            self._plan_isoweek  = week_key
            self._plan_monday   = now
            self._week_closed   = False
            plan_reason = "startup" if first_run else f"Monday week {isoweek}"
            log.info(
                f"📅 New weekly plan ({plan_reason}): "
                f"regime={snap.regime.regime}  "
                f"selected={snap.selected_symbols}"
            )
            asyncio.create_task(self.tg.send(
                f"📅 <b>New weekly plan — {plan_reason}</b>\n"
                f"Regime: <code>{snap.regime.regime}</code>\n"
                f"Selected: <code>"
                f"{', '.join(snap.selected_symbols) or 'none (BEAR_GATE)'}"
                f"</code>"
            ))

        # 4. Regime change notification
        new_regime = snap.regime.regime
        if self.last_regime not in ("UNKNOWN",) and new_regime != self.last_regime:
            log.info(f"Regime change: {self.last_regime} → {new_regime}")
            asyncio.create_task(
                self.tg.notify_regime_change(self.last_regime, new_regime))
        self.last_regime = new_regime

        # 5. BEAR_GATE mid-week: liquidate all open positions immediately
        #    (mirrors backtest — sit out entirely when weekly crash gate fires)
        if new_regime == "BEAR_GATE" and self.positions:
            log.warning("BEAR_GATE detected — liquidating all open positions")
            asyncio.create_task(self.tg.send(
                "🚫 <b>BEAR_GATE</b> — BTC down >10 % this week\n"
                "Liquidating all positions and sitting out."
            ))
            for sym in list(self.positions.keys()):
                price = self.last_prices.get(sym, self.positions[sym]["entry_price"])
                await self._place_sell(sym, trigger="bear_gate", exit_price=price)
                if self.positions:
                    await asyncio.sleep(MIN_ORDER_GAP_S)

        # 6. Weekly performance tracking
        self._tick_weekly(now)

        # 7. Portfolio stop
        if not await self._check_portfolio_stop():
            return

        # 8. Exit checks every cycle (stop/TP/trailing use live prices)
        await self._check_exits(snap)

        # 9. Enter positions — use locked weekly plan, 1 order per cycle,
        #    scale allocations to currently free cash (not full capital)
        if (self._weekly_plan is not None
                and new_regime != "BEAR_GATE"
                and not self._week_closed):
            await self._enter_positions(self._weekly_plan, now)

        # 10. Friday EOD: close all remaining positions (weekly exit)
        if weekday == 4 and now.hour >= 22 and not self._week_closed:
            await self._weekly_close()

        log.info(
            f"capital={_usd(self.capital)}  "
            f"peak={_usd(self.peak_capital)}  "
            f"positions={list(self.positions)}  "
            f"regime={new_regime}"
        )

    # ── Weekly close ──────────────────────────────────────────────────────────

    async def _weekly_close(self) -> None:
        """
        Friday EOD: close all remaining positions (mirrors backtest weekly_exit).
        Guarded by _week_closed so it fires exactly once per week.
        """
        if not self.positions:
            self._week_closed = True
            return

        self._week_closed = True
        log.info(f"Friday EOD weekly close — liquidating {list(self.positions)}")
        await self.tg.send(
            f"🗓 <b>Friday EOD — weekly close</b>\n"
            f"Closing: <code>{', '.join(self.positions)}</code>"
        )
        for sym in list(self.positions.keys()):
            price = self.last_prices.get(sym, self.positions[sym]["entry_price"])
            await self._place_sell(sym, trigger="weekly_exit", exit_price=price)
            if self.positions:
                await asyncio.sleep(MIN_ORDER_GAP_S)

    # ── Balance sync ──────────────────────────────────────────────────────────

    async def _refresh_balance(self) -> None:
        try:
            resp = await asyncio.to_thread(self.roostoo.balance)
        except Exception as exc:
            log.warning(f"Balance fetch failed: {exc}")
            return
        if not resp.success:
            log.warning(f"Balance response error: {resp.err_msg}")
            return

        # Free USD/USDT in the wallet
        usd_entry = resp.wallet.get("USD") or resp.wallet.get("USDT")
        free_usd  = usd_entry.free if usd_entry else 0.0

        # Mark-to-market value of open positions
        mtm = 0.0
        for sym, pos in self.positions.items():
            price = self.last_prices.get(sym, pos["entry_price"])
            mtm  += pos["shares"] * price

        self.capital      = free_usd + mtm
        self.peak_capital = max(self.peak_capital, self.capital)
        log.debug(f"Balance: free_usd={free_usd:.2f}  mtm={mtm:.2f}  "
                  f"total={self.capital:.2f}")

    # ── Portfolio stop ────────────────────────────────────────────────────────

    async def _check_portfolio_stop(self) -> bool:
        """
        Returns False (and liquidates everything) if portfolio value has
        fallen more than portfolio_stop% from its peak.
        """
        threshold = self.peak_capital * (1.0 - P["portfolio_stop"])
        if self.capital >= threshold:
            return True

        log.warning(
            f"PORTFOLIO STOP: capital={self.capital:.0f}  "
            f"peak={self.peak_capital:.0f}  threshold={threshold:.0f}"
        )
        asyncio.create_task(
            self.tg.notify_portfolio_stop(self.capital, self.peak_capital))

        for sym in list(self.positions.keys()):
            price = self.last_prices.get(sym, self.positions[sym]["entry_price"])
            await self._place_sell(sym, trigger="portfolio_stop", exit_price=price)
            if self.positions:
                await asyncio.sleep(MIN_ORDER_GAP_S)

        return False

    # ── Per-position exit checks ──────────────────────────────────────────────

    async def _check_exits(self, snap: MarketSnapshot) -> None:
        """
        Mirrors the per-hour exit logic from backtest.run_hourly_loop().
        Runs every 60 s with the latest Binance price (from signals snapshot).

        Exit hierarchy (same as backtest):
          1. Take-profit       — price ≥ tp level
          2. Hard stop         — price ≤ entry × (1 − stop_pct)
          3. Trailing stop     — drop from rolling high ≥ trail% (only after
                                  price has risen enough to hit breakeven trigger)
          4. RSI early exit    — RSI > threshold AND pos_ret > min_profit
                                  AND not BULL regime
        """
        for sym in list(self.positions.keys()):
            pos   = self.positions[sym]
            price = self.last_prices.get(sym)
            if price is None or math.isnan(price):
                continue

            # Update rolling high AFTER reading from_hi
            # (must compute drop-from-peak before updating the peak)
            from_hi  = (price - pos["highest_price"]) / pos["highest_price"]
            pos["highest_price"] = max(pos["highest_price"], price)

            regime   = pos.get("regime", "SIDEWAYS")
            entry_px = pos["entry_price"]
            pos_ret  = (price - entry_px) / entry_px

            # Regime-specific params (same values as run_week)
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
            else:  # SIDEWAYS
                stop_pct  = P["sw_position_stop"]
                be_trig   = P["sw_breakeven_trigger"]
                trail     = P["sw_trailing_stop"]
                rsi_exit  = P["sw_rsi_early_exit"]
                ep_profit = P["sw_early_exit_profit"]

            trigger: Optional[str] = None

            # 1. Take-profit
            if pos["take_profit"] != float("inf") and price >= pos["take_profit"]:
                trigger = "take_profit"

            # 2. Hard stop
            elif price <= entry_px * (1.0 - stop_pct):
                trigger = "position_stop"

            # 3. Trailing stop
            elif (pos["highest_price"] >= entry_px * (1.0 + be_trig)
                  and from_hi <= -trail):
                trigger = "trailing_stop"

            # 4. RSI early-exit  (not in BULL — bull plays let winners run)
            elif regime != "BULL":
                sig     = snap.signals.get(sym)
                rsi_now = sig.rsi if sig is not None else 50.0
                if rsi_now > rsi_exit and pos_ret > ep_profit:
                    trigger = "early_profit"

            if trigger:
                await self._place_sell(sym, trigger=trigger, exit_price=price)

    # ── Position entry ────────────────────────────────────────────────────────

    async def _enter_positions(self, snap: MarketSnapshot, now: datetime) -> None:
        """
        Place at most ONE new buy order per cycle.

        Entry timing mirrors backtest:
          BULL     → enter any cycle (breakout_open at Monday open)
          SIDEWAYS → prefer red-day close Mon–Wed; fall back Thu onwards
          BEAR     → same as SIDEWAYS

        Allocations are re-scaled to free cash so we never over-deploy when
        some positions from the weekly plan are already held.
        """
        regime  = snap.regime.regime

        # Days elapsed since the weekly plan was made (0 = plan day)
        plan_age_days = 0
        if self._plan_monday is not None:
            plan_age_days = (now - self._plan_monday).days

        # Compute free cash: capital minus mark-to-market of open positions
        mtm_held = sum(
            pos["shares"] * self.last_prices.get(sym, pos["entry_price"])
            for sym, pos in self.positions.items()
        )
        free_cash = max(self.capital - mtm_held, 0.0)

        # How many slots are still unfilled this week?
        unfilled = [s for s in snap.selected_symbols if s not in self.positions]
        if not unfilled:
            return

        # Re-scale allocations proportionally to free cash
        raw_total = sum(snap.capital_allocs.get(s, 0.0) for s in unfilled)
        if raw_total <= 0:
            return

        for sym in unfilled:
            sig   = snap.signals.get(sym)
            price = self.last_prices.get(sym)
            if sig is None or price is None or math.isnan(price):
                continue

            # Entry timing gate
            if regime == "BULL":
                entry_trigger = "breakout_open"
            else:
                # Red-day preferred for first 3 days of plan;
                # fall back unconditionally from day 3 onwards
                if sig.is_red_day:
                    entry_trigger = "red_day"
                elif plan_age_days >= 3:
                    entry_trigger = "fallback_thursday"
                else:
                    log.debug(f"Waiting for red day on {sym} (plan day {plan_age_days})")
                    continue

            # Scale alloc to free cash
            raw_alloc = snap.capital_allocs.get(sym, 0.0)
            alloc     = raw_alloc * (free_cash / raw_total)
            alloc     = min(alloc, free_cash * 0.80)   # never >80% of free cash in one go

            if alloc < MIN_ORDER_USD:
                log.debug(f"Skip {sym}: scaled alloc ${alloc:.2f} below minimum")
                continue

            stop_pct  = sig.stop_pct
            tp_pct    = sig.tp_pct
            stop_price = price * (1.0 - stop_pct)
            tp_price   = price * (1.0 + tp_pct) if tp_pct > 0.0 else float("inf")

            placed = await self._place_buy(
                sym           = sym,
                alloc         = alloc,
                price         = price,
                stop_price    = stop_price,
                tp_price      = tp_price,
                regime        = regime,
                entry_trigger = entry_trigger,
            )
            if placed:
                return   # one order per cycle

    # ── Order placement ───────────────────────────────────────────────────────

    def _can_order(self) -> bool:
        return (time.time() - self._last_order_ts) >= MIN_ORDER_GAP_S

    async def _place_buy(
        self,
        *,
        sym:           str,
        alloc:         float,
        price:         float,
        stop_price:    float,
        tp_price:      float,
        regime:        str,
        entry_trigger: str,
    ) -> bool:
        """Place a market BUY. Returns True if the order was accepted."""
        if not self._can_order():
            log.info(
                f"Rate-limit: skip BUY {sym} "
                f"(last order {time.time()-self._last_order_ts:.0f}s ago)"
            )
            return False

        pair     = to_roostoo_pair(sym)
        quantity = self._round_qty(pair, alloc / price)

        if quantity <= 0:
            log.warning(f"Skip BUY {sym}: rounded quantity is 0 (alloc={_usd(alloc)}, price={_usd(price)})")
            return False

        log.info(
            f"→ BUY {sym}  pair={pair}  qty={quantity}  "
            f"alloc={_usd(alloc)}  price={_usd(price)}  trigger={entry_trigger}"
        )

        try:
            resp = await asyncio.to_thread(
                self.roostoo.place_order,
                pair       = pair,
                side       = Side.BUY,
                order_type = OrderType.MARKET,
                quantity   = quantity,
            )
        except (RoostooHTTPError, RoostooParseError, Exception) as exc:
            log.error(f"BUY order error {sym}: {exc}")
            return False

        if not resp.success:
            log.error(f"BUY rejected {sym}: {resp.err_msg}")
            return False

        self._last_order_ts = time.time()
        self.total_trades  += 1

        det          = resp.order_detail
        filled_price = det.filled_aver_price if det and det.filled_aver_price else price
        filled_qty   = det.filled_quantity    if det and det.filled_quantity    else quantity
        order_id     = det.order_id           if det else None

        if regime == "BULL":
            actual_stop = filled_price * (1.0 - P["bull_position_stop"])
            actual_tp   = float("inf")
        elif regime == "BEAR":
            actual_stop = filled_price * (1.0 - P["bear_position_stop"])
            actual_tp   = filled_price * (1.0 + P["bear_rr_ratio"] * P["bear_position_stop"])
        else:
            actual_stop = filled_price * (1.0 - P["sw_position_stop"])
            actual_tp   = filled_price * (1.0 + P["sw_rr_ratio"]   * P["sw_position_stop"])

        self.positions[sym] = dict(
            entry_price    = filled_price,
            shares         = filled_qty,
            alloc          = alloc,
            stop_price     = actual_stop,
            take_profit    = actual_tp,
            highest_price  = filled_price,
            regime         = regime,
            entry_time     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            order_id       = order_id,
        )

        log.info(
            f"✓ BUY filled {sym}  price={_usd(filled_price)}  "
            f"qty={filled_qty}  stop={_usd(actual_stop)}  "
            f"tp={'inf' if actual_tp == float('inf') else _usd(actual_tp)}"
        )

        asyncio.create_task(
            self.tg.notify_trade(
                action    = "BUY",
                symbol    = sym,
                pair      = pair,
                price     = filled_price,
                quantity  = filled_qty,
                value_usd = alloc,
                trigger   = entry_trigger,
                regime    = regime,
            )
        )
        return True

    async def _place_sell(
        self,
        sym:        str,
        *,
        trigger:    str,
        exit_price: float,
    ) -> None:
        if sym not in self.positions:
            return
        if not self._can_order():
            log.info(
                f"Rate-limit: skip SELL {sym} "
                f"(last order {time.time()-self._last_order_ts:.0f}s ago)"
            )
            return

        pos      = self.positions[sym]
        pair     = to_roostoo_pair(sym)
        quantity = self._round_qty(pair, pos["shares"])

        log.info(
            f"→ SELL {sym}  pair={pair}  qty={quantity:.6f}  "
            f"price={_usd(exit_price)}  trigger={trigger}"
        )

        try:
            resp = await asyncio.to_thread(
                self.roostoo.place_order,
                pair       = pair,
                side       = Side.SELL,
                order_type = OrderType.MARKET,
                quantity   = quantity,
            )
        except (RoostooHTTPError, RoostooParseError, Exception) as exc:
            log.error(f"SELL order error {sym}: {exc}")
            return

        if not resp.success:
            log.error(f"SELL rejected {sym}: {resp.err_msg}")
            return

        self._last_order_ts = time.time()
        self.total_trades  += 1

        det          = resp.order_detail
        filled_price = det.filled_aver_price if det and det.filled_aver_price else exit_price
        filled_qty   = det.filled_quantity    if det and det.filled_quantity    else quantity
        proceeds     = filled_qty * filled_price
        pnl          = proceeds - pos["alloc"]

        del self.positions[sym]
        # Capital will be reconciled properly on next _refresh_balance()
        # but we update optimistically so metrics are approximately correct
        self.capital += pnl

        log.info(
            f"✓ SELL filled {sym}  price={_usd(filled_price)}  "
            f"pnl={_usd(pnl)}  trigger={trigger}"
        )

        asyncio.create_task(
            self.tg.notify_trade(
                action    = "SELL",
                symbol    = sym,
                pair      = pair,
                price     = filled_price,
                quantity  = filled_qty,
                value_usd = proceeds,
                trigger   = trigger,
                regime    = pos.get("regime", "?"),
                pnl       = pnl,
            )
        )

    # ── Performance ───────────────────────────────────────────────────────────

    def _tick_weekly(self, now: datetime) -> None:
        """Record a weekly return snapshot when the ISO week rolls over."""
        iso       = now.isocalendar()
        week_key  = iso[0] * 100 + iso[1]   # e.g. 202601 — unique across years
        if self._current_isoweek == -1:
            self._current_isoweek  = week_key
            self._week_start_cap   = self.capital
            return
        if week_key != self._current_isoweek:
            if self._week_start_cap > 0:
                wr = (self.capital - self._week_start_cap) / self._week_start_cap
                self._weekly_returns.append(wr)
                log.info(f"Weekly return recorded: {wr:+.3%}  "
                         f"(n={len(self._weekly_returns)})")
            self._current_isoweek = week_key
            self._week_start_cap  = self.capital

    def metrics(self) -> dict:
        """
        Compute annualised Sharpe, Sortino, Calmar from weekly return history.
        Mirrors compute_metrics() in backtest.py.
        """
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
            sharpe       = avg / std  * (52 ** 0.5),
            sortino      = avg / std_d * (52 ** 0.5),
            calmar       = ann / abs(mdd) if mdd != 0 else np.nan,
            max_drawdown = mdd,
            win_rate     = float((r > 0).mean()),
        )


# ── Formatting util (local, so telegram_notifier stays independent) ───────────

def _usd(v: float) -> str:
    if v is None or v != v:
        return "n/a"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.3f}M"
    return f"${v:,.2f}"


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    bot = TradingBot()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted by user — goodbye")