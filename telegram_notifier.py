"""
telegram_notifier.py  —  MRVS Telegram integration
====================================================
Sends trade/regime/stop notifications to a Telegram channel topic.
Handles /status, /positions, /signals slash commands.

Env vars required:
    TELEGRAM_BOT_TOKEN    — BotFather token
    TELEGRAM_CHANNEL_ID   — numeric channel id, e.g. -1001234567890
    TELEGRAM_TOPIC_ID     — message thread / topic id (int, omit for no topic)

Usage (inside bot.py):
    notifier = TelegramNotifier(bot_ref=trading_bot)
    await notifier.start()          # begin polling
    await notifier.send("hello")    # push a message
    await notifier.stop()           # clean shutdown
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

log = logging.getLogger("telegram")

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from telegram.constants import ParseMode
    _TG_OK = True
except ImportError:
    _TG_OK = False
    log.warning("python-telegram-bot not installed — Telegram disabled")

if TYPE_CHECKING:
    from bot import TradingBot


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v: float, decimals: int = 2) -> str:
    if v is None or v != v:
        return "n/a"
    return f"{v:+.{decimals}%}"


def _usd(v: float) -> str:
    if v is None or v != v:
        return "n/a"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.3f}M"
    if abs(v) >= 1_000:
        return f"${v:,.2f}"
    return f"${v:.4f}"


def _f(v: float, dp: int = 3) -> str:
    if v is None or v != v:
        return "n/a"
    return f"{v:.{dp}f}"


REGIME_EMOJI = {"BULL": "🐂", "SIDEWAYS": "↔️", "BEAR": "🐻", "BEAR_GATE": "🚫"}


# ── Notifier class ────────────────────────────────────────────────────────────

class TelegramNotifier:

    def __init__(self, bot_ref: "TradingBot"):
        self._bot         = bot_ref
        self._token       = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._channel_id  = os.getenv("TELEGRAM_CHANNEL_ID", "")
        raw_topic         = os.getenv("TELEGRAM_TOPIC_ID", "")
        self._topic_id    = int(raw_topic) if raw_topic.lstrip("-").isdigit() else None
        self._app: Optional[Application] = None
        self._enabled     = False

        if not _TG_OK:
            return
        if not self._token:
            log.warning("TELEGRAM_BOT_TOKEN not set — Telegram disabled")
            return
        if not self._channel_id:
            log.warning("TELEGRAM_CHANNEL_ID not set — Telegram disabled")
            return

        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )
        self._app.add_handler(CommandHandler("status",    self._cmd_status))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("signals",   self._cmd_signals))
        self._enabled = True
        log.info(f"Telegram ready  channel={self._channel_id}  topic={self._topic_id}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self._enabled or self._app is None:
            return
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        log.info("Telegram polling started")

    async def stop(self) -> None:
        if not self._enabled or self._app is None:
            return
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    # ── Core send ─────────────────────────────────────────────────────────────

    async def send(self, text: str) -> None:
        """Push HTML-formatted text to the configured channel/topic."""
        if not self._enabled or not self._channel_id:
            log.info(f"[TG MOCK] {text[:120]}")
            return
        try:
            kwargs: dict = dict(
                chat_id    = self._channel_id,
                text       = text,
                parse_mode = ParseMode.HTML,
            )
            if self._topic_id:
                kwargs["message_thread_id"] = self._topic_id
            await self._app.bot.send_message(**kwargs)
        except Exception as exc:
            log.error(f"Telegram send error: {exc}")

    # ── Push notifications ────────────────────────────────────────────────────

    async def notify_trade(
        self,
        *,
        action:    str,
        symbol:    str,
        pair:      str,
        price:     float,
        quantity:  float,
        value_usd: float,
        trigger:   str,
        regime:    str,
        pnl:       Optional[float] = None,
    ) -> None:
        """Called immediately after every filled order."""
        emoji  = "🟢" if action == "BUY" else "🔴"
        now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines  = [
            f"{emoji} <b>{action} {symbol}</b>",
            f"  Pair    : <code>{pair}</code>",
            f"  Price   : <code>{_usd(price)}</code>",
            f"  Qty     : <code>{quantity:.6f}</code>",
            f"  Value   : <code>{_usd(value_usd)}</code>",
            f"  Trigger : <code>{trigger}</code>",
            f"  Regime  : <code>{REGIME_EMOJI.get(regime,'')} {regime}</code>",
        ]
        if pnl is not None:
            pnl_e = "📈" if pnl >= 0 else "📉"
            cost  = value_usd - pnl
            pct   = pnl / cost if cost != 0 else 0.0
            lines.append(
                f"  PnL     : <code>{_usd(pnl)} ({_pct(pct)})</code> {pnl_e}"
            )
        lines.append(f"<i>{now}</i>")
        await self.send("\n".join(lines))

    async def notify_regime_change(self, old: str, new: str) -> None:
        e_old = REGIME_EMOJI.get(old, "")
        e_new = REGIME_EMOJI.get(new, "")
        now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        await self.send(
            f"⚡ <b>Regime change</b>\n"
            f"  {e_old} <code>{old}</code>  →  {e_new} <code>{new}</code>\n"
            f"<i>{now}</i>"
        )

    async def notify_portfolio_stop(self, value: float, peak: float) -> None:
        dd  = (value - peak) / peak if peak else 0.0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        await self.send(
            f"🚨 <b>PORTFOLIO STOP TRIGGERED</b>\n"
            f"  Value : <code>{_usd(value)}</code>\n"
            f"  Peak  : <code>{_usd(peak)}</code>\n"
            f"  DD    : <code>{_pct(dd)}</code>\n"
            f"  All positions liquidated.\n"
            f"<i>{now}</i>"
        )

    async def notify_error(self, message: str) -> None:
        await self.send(f"⚠️ <b>Bot error</b>\n<code>{message[:300]}</code>")

    # ── Command handlers ──────────────────────────────────────────────────────

    async def _cmd_status(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /status  —  brief portfolio snapshot
        Capital, PnL, drawdown, Sharpe, Sortino, Calmar, win rate, regime.
        """
        b  = self._bot
        m  = b.metrics()
        re = b.last_regime
        snap = b.last_snapshot
        pnl_usd  = b.capital - b.initial_capital
        pnl_pct  = pnl_usd / b.initial_capital if b.initial_capital else 0.0
        dd_pct   = (b.capital - b.peak_capital) / b.peak_capital \
                   if b.peak_capital else 0.0
        last_upd = snap.computed_at.strftime("%H:%M:%S UTC") \
                   if snap else "not yet"

        lines = [
            f"📊 <b>MRVS Bot — Status</b>",
            "",
            f"Regime    : {REGIME_EMOJI.get(re,'')} <code>{re}</code>",
            f"Capital   : <code>{_usd(b.capital)}</code>",
            f"Total PnL : <code>{_usd(pnl_usd)} ({_pct(pnl_pct)})</code>",
            f"Peak      : <code>{_usd(b.peak_capital)}</code>",
            f"Drawdown  : <code>{_pct(dd_pct)}</code>",
            "",
            f"Sharpe    : <code>{_f(m.get('sharpe'))}</code>",
            f"Sortino   : <code>{_f(m.get('sortino'))}</code>",
            f"Calmar    : <code>{_f(m.get('calmar'))}</code>",
            f"Win rate  : <code>{_pct(m.get('win_rate', 0), 1)}</code>",
            "",
            f"Open pos  : <code>{len(b.positions)}</code>   "
            f"Trades: <code>{b.total_trades}</code>",
            f"<i>Signals updated: {last_upd}</i>",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def _cmd_positions(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /positions  —  detailed open-position breakdown
        Entry price, current price, PnL, stop, TP, regime, entry time.
        """
        b = self._bot
        if not b.positions:
            await update.message.reply_text(
                "No open positions.", parse_mode=ParseMode.HTML)
            return

        lines = ["📋 <b>Open Positions</b>", ""]
        for sym, pos in b.positions.items():
            cur   = b.last_prices.get(sym, pos["entry_price"])
            pnl_pct = (cur - pos["entry_price"]) / pos["entry_price"]
            pnl_usd = pos["shares"] * (cur - pos["entry_price"])
            val     = pos["shares"] * cur
            em      = "📈" if pnl_pct >= 0 else "📉"
            tp_str  = "disabled" if pos["take_profit"] == float("inf") \
                      else _usd(pos["take_profit"])
            lines += [
                f"{em} <b>{sym}</b>",
                f"  Entry   : <code>{_usd(pos['entry_price'])}</code>",
                f"  Current : <code>{_usd(cur)}</code>",
                f"  PnL     : <code>{_usd(pnl_usd)} ({_pct(pnl_pct)})</code>",
                f"  Value   : <code>{_usd(val)}</code>",
                f"  Stop    : <code>{_usd(pos['stop_price'])}</code>",
                f"  TP      : <code>{tp_str}</code>",
                f"  Regime  : <code>{pos.get('regime','?')}</code>",
                f"  Since   : <code>{pos.get('entry_time','?')}</code>",
                "",
            ]
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def _cmd_signals(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /signals  —  current regime + per-coin signal table
        RSI, momentum, relative-strength, breakout-str, entry signal.
        """
        b    = self._bot
        snap = b.last_snapshot
        if snap is None:
            await update.message.reply_text(
                "No signals computed yet — bot is still initialising.",
                parse_mode=ParseMode.HTML)
            return

        ri = snap.regime
        re = ri.regime
        lines = [
            f"📡 <b>Signals  {REGIME_EMOJI.get(re,'')} {re}</b>",
            (f"BTC  7d:<code>{_pct(ri.btc_ret_7d,1)}</code>  "
             f"20d:<code>{_pct(ri.btc_ret_20d,1)}</code>  "
             f"MA50:<code>{_pct(ri.btc_ma50_gap,1)}</code>"),
            "",
        ]
        for sym in b.SYMBOLS:
            sig = snap.signals.get(sym)
            if sig is None:
                continue
            sel  = "✅" if sig.is_selected else "  "
            rank = f"#{sig.rank}" if sig.is_selected else "   "
            red  = "🔴" if sig.is_red_day else "  "
            lines.append(
                f"{sel}{red}<b>{sym:<10}</b> {rank:<3} "
                f"P:<code>{_usd(sig.price)}</code>  "
                f"RSI:<code>{sig.rsi:>5.1f}</code>  "
                f"Mom:<code>{_pct(sig.momentum_7d,1)}</code>  "
                f"RS:<code>{_pct(sig.rel_strength,1)}</code>  "
                f"BK:<code>{_pct(sig.breakout_str,1)}</code>  "
                f"→<code>{sig.entry_signal}</code>"
            )
        lines += [
            "",
            f"<i>Updated: {snap.computed_at.strftime('%H:%M:%S UTC')}</i>",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)