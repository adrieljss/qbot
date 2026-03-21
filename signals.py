"""
signals.py  —  MRVS live signal engine
=======================================
Ports the exact signal, regime-classification, and coin-selection logic
from backtest.py to work with live Binance daily candles (fetched via
python-binance in unauthenticated mode).

Key design decisions that mirror the backtest exactly:
  - All indicators (RSI, momentum, variance) computed on DAILY bars
  - Regime re-classified on every call (no Monday-only lock in live trading)
  - Coin selection functions are verbatim translations of select_sideways /
    select_bull / select_bear from backtest.py
  - Variance-scaled allocations use the same formula as run_week()
  - Stop/TP prices are computed here and enforced by bot.py each cycle

Usage:
    from binance.client import Client
    from signals import generate_signals, SYMBOLS

    client = Client("", "")          # unauthenticated
    snap   = generate_signals(client)
    print(snap.regime.regime)        # BULL | SIDEWAYS | BEAR | BEAR_GATE
    for sym, sig in snap.signals.items():
        print(sym, sig.is_selected, sig.rsi, sig.momentum_7d)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from binance.client import Client as BinanceClient

log = logging.getLogger("signals")

# ── Universe ──────────────────────────────────────────────────────────────────

SYMBOLS: list[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "TRXUSDT", "SUIUSDT", "ADAUSDT", "LINKUSDT",
]


def to_roostoo_pair(binance_symbol: str) -> str:
    """'BTCUSDT'  ->  'BTC/USD'"""
    return binance_symbol.replace("USDT", "") + "/USD"


def to_binance_symbol(roostoo_pair: str) -> str:
    """'BTC/USD'  ->  'BTCUSDT'"""
    return roostoo_pair.replace("/USD", "USDT")


# ── Parameters (identical to backtest.py PARAMS) ─────────────────────────────

P: dict = dict(
    min_avg_volume       = 10_000_000,  

    # Shared indicators
    momentum_window      = 7,
    rsi_period           = 14,
    ma_short             = 20,   # used for 20-day BTC return in regime
    ma_long              = 50,   # 50-day MA for regime MA-gap check
    rv_window            = 5,

    # Regime thresholds
    regime_symbol        = "BTCUSDT",
    bull_return_thresh   =  0.08,
    bear_return_thresh   = -0.08,
    bear_ma_gap          =  0.05,
    bear_weekly_gate     = -0.10,

    # SIDEWAYS playbook
    sw_n_positions       = 3,
    sw_rsi_overbought    = 70,
    sw_rsi_early_exit    = 75,
    sw_early_exit_profit = 0.03,
    sw_position_stop     = 0.05,
    sw_breakeven_trigger = 0.05,
    sw_trailing_stop     = 0.04,
    sw_rr_ratio          = 1.0,     # TP = 1 × stop = +5 %
    sw_size_cap          = 1.5,
    sw_size_floor        = 0.25,
    sw_target_var        = 0.0025,

    # BULL playbook
    bull_n_positions     = 3,
    bull_breakout_days   = 10,
    bull_rsi_overbought  = 80,
    bull_position_stop   = 0.07,
    bull_breakeven_trigger = 0.08,
    bull_trailing_stop   = 0.05,
    bull_rr_ratio        = 0.0,    # TP OFF — let winners run
    bull_size_cap        = 1.5,
    bull_size_floor      = 0.5,
    bull_target_var      = 0.0025,
    bull_rsi_early_exit  = 85,
    bull_early_exit_profit = 0.10,

    # BEAR playbook
    bear_n_positions     = 2,
    bear_rs_window       = 14,
    bear_rsi_overbought  = 65,
    bear_position_stop   = 0.04,
    bear_breakeven_trigger = 0.04,
    bear_trailing_stop   = 0.03,
    bear_rr_ratio        = 1.0,    # TP = 1 × stop = +4 %
    bear_size_cap        = 1.0,
    bear_size_floor      = 0.25,
    bear_target_var      = 0.0025,
    bear_early_exit_profit = 0.03,

    # Portfolio stop
    portfolio_stop       = 0.12,

    # Fees (taker only — Roostoo MARKET orders)
    taker_fee            = 0.001,
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RegimeInfo:
    regime:      str    # BULL | SIDEWAYS | BEAR | BEAR_GATE
    btc_ret_7d:  float
    btc_ret_20d: float
    btc_ma50_gap: float  # fraction above/below 50d MA
    updated_at:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CoinSignal:
    symbol:      str
    pair:        str    # Roostoo pair, e.g. "BTC/USD"
    regime:      str

    # Selection
    is_selected: bool
    rank:        int    # 1 = best; 99 = not selected

    # Indicators
    momentum_7d:    float
    rsi:            float
    rel_strength:   float   # coin 14d ret − BTC 14d ret  (used in BEAR)
    breakout_str:   float   # fraction above N-day high   (used in BULL)
    is_red_day:     bool    # today's close < yesterday's close

    # Trade params (regime-specific)
    stop_pct:    float
    tp_pct:      float      # 0 = TP disabled (BULL)
    size_scale:  float      # variance-based scale factor

    # Market data
    price:       float
    vol_24h_usd: float

    entry_signal: str       # "breakout_open" | "red_day" | "none"
    updated_at:   datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MarketSnapshot:
    regime:           RegimeInfo
    signals:          dict[str, CoinSignal]   # BTCUSDT -> CoinSignal, …
    selected_symbols: list[str]               # ordered: rank 1 first
    capital_allocs:   dict[str, float]        # sym -> USD allocation
    computed_at:      datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))


# ── Indicators  (verbatim from backtest.py) ───────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d  = series.diff()
    ag = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    al = (-d).clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def _nday_return(closes: pd.Series, n_back: int) -> float:
    """Return over the last n_back days; uses iloc so no index dependency."""
    if len(closes) < n_back + 1:
        return np.nan
    p = closes.iloc[-(n_back + 1)]
    t = closes.iloc[-1]
    return np.nan if p == 0 else float((t - p) / p)


def _realized_variance(closes: pd.Series, window: int, target_var: float) -> float:
    """Sum of squared log-returns over the last `window` days."""
    if len(closes) < window + 1:
        return target_var
    lr = np.log(closes / closes.shift(1)).dropna().iloc[-window:]
    rv = float((lr ** 2).sum())
    return rv if rv > 0 else target_var


def _variance_scale(closes: pd.Series, target_var: float,
                    size_floor: float, size_cap: float) -> float:
    rv = _realized_variance(closes, P["rv_window"], target_var)
    return float(np.clip(target_var / rv, size_floor, size_cap))


# ── Binance data fetch ────────────────────────────────────────────────────────

def _fetch_daily(client: BinanceClient, symbol: str,
                 lookback: int = 70) -> pd.DataFrame:
    """
    Fetch the last `lookback` daily OHLCV candles from Binance (unauthenticated).
    Returns a DataFrame indexed by UTC midnight DatetimeIndex.
    Empty DataFrame on error.
    """
    try:
        raw = client.get_klines(
            symbol   = symbol,
            interval = BinanceClient.KLINE_INTERVAL_1DAY,
            limit    = lookback,
        )
    except Exception as exc:
        log.warning(f"Binance fetch failed {symbol}: {exc}")
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume", "quote_volume"]]


# ── Regime classifier  (mirrors backtest.classify_regime) ─────────────────────

def _classify_regime(btc_daily: pd.DataFrame) -> RegimeInfo:
    close  = btc_daily["close"]
    needed = P["ma_long"] + 1

    if len(close) < needed:
        return RegimeInfo("SIDEWAYS", np.nan, np.nan, np.nan)

    ret_7d   = _nday_return(close, 7)
    ret_20d  = _nday_return(close, P["ma_short"])
    ma50     = float(close.iloc[-P["ma_long"]:].mean())
    ma50_gap = (float(close.iloc[-1]) - ma50) / ma50

    # Hard gate — too violent to trade at all
    if not np.isnan(ret_7d) and ret_7d < P["bear_weekly_gate"]:
        return RegimeInfo("BEAR_GATE", ret_7d, ret_20d, ma50_gap)

    if (not np.isnan(ret_20d)
            and ret_20d > P["bull_return_thresh"]
            and ma50_gap > 0):
        return RegimeInfo("BULL", ret_7d, ret_20d, ma50_gap)

    if (not np.isnan(ret_20d)
            and (ret_20d < P["bear_return_thresh"]
                 or ma50_gap < -P["bear_ma_gap"])):
        return RegimeInfo("BEAR", ret_7d, ret_20d, ma50_gap)

    return RegimeInfo("SIDEWAYS", ret_7d, ret_20d, ma50_gap)


# ── Coin selectors  (mirrors backtest select_sideways/bull/bear) ──────────────

def _select_sideways(daily: dict[str, pd.DataFrame],
                     universe: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for sym in universe:
        df = daily.get(sym, pd.DataFrame())
        if df.empty or len(df) < P["momentum_window"] + 1:
            continue
        r = _nday_return(df["close"], P["momentum_window"])
        if not np.isnan(r):
            scores[sym] = r

    # Top 3× pool, then filter by RSI
    candidates = sorted(scores, key=scores.__getitem__, reverse=True)[
        : P["sw_n_positions"] * 3]
    valid: list[str] = []
    for sym in candidates:
        df = daily[sym]
        if len(df) < P["rsi_period"] + 1:
            continue
        rsi = float(_compute_rsi(df["close"], P["rsi_period"]).iloc[-1])
        if rsi <= P["sw_rsi_overbought"]:
            valid.append(sym)
        if len(valid) == P["sw_n_positions"]:
            break
    return valid


def _select_bull(daily: dict[str, pd.DataFrame],
                 universe: list[str]) -> list[str]:
    breakouts: dict[str, float] = {}
    for sym in universe:
        df = daily.get(sym, pd.DataFrame())
        if df.empty or len(df) < P["bull_breakout_days"] + 1:
            continue
        close_now  = float(df["close"].iloc[-1])
        prev_high  = float(df["close"].iloc[-(P["bull_breakout_days"] + 1):-1].max())
        rsi_now    = float(_compute_rsi(df["close"], P["rsi_period"]).iloc[-1])
        if close_now >= prev_high * 0.98 and rsi_now <= P["bull_rsi_overbought"]:
            breakouts[sym] = (close_now - prev_high) / prev_high
    ranked = sorted(breakouts, key=breakouts.__getitem__, reverse=True)
    return ranked[: P["bull_n_positions"]]


def _select_bear(daily: dict[str, pd.DataFrame],
                 universe: list[str]) -> list[str]:
    btc_sym = P["regime_symbol"]
    btc_df  = daily.get(btc_sym, pd.DataFrame())
    btc_ret = _nday_return(btc_df["close"], P["bear_rs_window"]) \
              if not btc_df.empty else np.nan

    rel_strengths: dict[str, float] = {}
    for sym in universe:
        if sym == btc_sym:
            continue
        df = daily.get(sym, pd.DataFrame())
        if df.empty or len(df) < P["bear_rs_window"] + P["rsi_period"] + 1:
            continue
        coin_ret = _nday_return(df["close"], P["bear_rs_window"])
        if np.isnan(coin_ret):
            continue
        rs  = coin_ret - btc_ret if not np.isnan(btc_ret) else coin_ret
        rsi = float(_compute_rsi(df["close"], P["rsi_period"]).iloc[-1])
        if rs > 0 and rsi <= P["bear_rsi_overbought"]:
            rel_strengths[sym] = rs
    ranked = sorted(rel_strengths, key=rel_strengths.__getitem__, reverse=True)
    return ranked[: P["bear_n_positions"]]


# ── Entry condition helper ────────────────────────────────────────────────────

def _is_red_day(df: pd.DataFrame) -> bool:
    """True when the most recent daily close < the prior daily close."""
    if len(df) < 2:
        return False
    return bool(df["close"].iloc[-1] < df["close"].iloc[-2])


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_signals(
    client:      BinanceClient,
    symbols:     list[str] = SYMBOLS,
    capital:     float = 1_000_000.0,
    lookback:    int   = 70,
) -> MarketSnapshot:
    """
    Fetch live Binance daily candles for all symbols, classify the regime,
    select coins, compute per-coin indicators, and return a MarketSnapshot.

    Called by bot.py every cycle (≈60 s).  All blocking network I/O is done
    here so the caller can wrap it in asyncio.to_thread().

    Parameters
    ----------
    client   : python-binance Client (unauthenticated is fine)
    symbols  : universe to scan
    capital  : current portfolio value in USD (used to size allocations)
    lookback : number of daily candles to request from Binance
    """
    # 1. Fetch BTC candles first — needed for regime
    btc_daily = _fetch_daily(client, "BTCUSDT", lookback)
    if btc_daily.empty:
        log.error("Cannot fetch BTC daily candles — defaulting to SIDEWAYS")

    regime_info = _classify_regime(btc_daily) if not btc_daily.empty \
                  else RegimeInfo("SIDEWAYS", np.nan, np.nan, np.nan)
    regime = regime_info.regime
    log.info(f"Regime: {regime}  "
             f"BTC 7d={regime_info.btc_ret_7d:.1%}  "
             f"20d={regime_info.btc_ret_20d:.1%}  "
             f"MA50gap={regime_info.btc_ma50_gap:.1%}")

    # 2. Fetch all other coins
    daily: dict[str, pd.DataFrame] = {"BTCUSDT": btc_daily}
    for sym in symbols:
        if sym == "BTCUSDT":
            continue
        daily[sym] = _fetch_daily(client, sym, lookback)

    # 3. Universe filter — minimum daily volume
    universe = [
        sym for sym in symbols
        if not daily.get(sym, pd.DataFrame()).empty
        and len(daily[sym]) >= 20
        and (
            daily[sym]["quote_volume"].tail(20).mean()
            if "quote_volume" in daily[sym].columns
            else (daily[sym]["close"] * daily[sym]["volume"]).tail(20).mean()
        ) >= P["min_avg_volume"]
    ]
    log.info(f"Universe ({len(universe)}): {universe}")

    # 4. Select coins per regime
    if regime == "BEAR_GATE":
        selected: list[str] = []
    elif regime == "BULL":
        selected = _select_bull(daily, universe)
    elif regime == "BEAR":
        selected = _select_bear(daily, universe)
    else:  # SIDEWAYS
        selected = _select_sideways(daily, universe)

    log.info(f"Selected ({len(selected)}): {selected}")

    # 5. Per-regime trade parameters
    if regime == "BULL":
        stop_pct  = P["bull_position_stop"]
        tp_pct    = P["bull_rr_ratio"] * stop_pct   # 0.0 — TP off
        t_var     = P["bull_target_var"]
        s_cap     = P["bull_size_cap"]
        s_floor   = P["bull_size_floor"]
        n_pos     = P["bull_n_positions"]
    elif regime == "BEAR":
        stop_pct  = P["bear_position_stop"]
        tp_pct    = P["bear_rr_ratio"] * stop_pct   # 0.08
        t_var     = P["bear_target_var"]
        s_cap     = P["bear_size_cap"]
        s_floor   = P["bear_size_floor"]
        n_pos     = P["bear_n_positions"]
    else:  # SIDEWAYS or BEAR_GATE
        stop_pct  = P["sw_position_stop"]
        tp_pct    = P["sw_rr_ratio"] * stop_pct     # 0.15
        t_var     = P["sw_target_var"]
        s_cap     = P["sw_size_cap"]
        s_floor   = P["sw_size_floor"]
        n_pos     = P["sw_n_positions"]

    # 6. Variance-scaled allocations (mirrors run_week allocs block)
    base = capital / max(len(selected), 1)
    raw_allocs: dict[str, float] = {}
    for sym in selected:
        df  = daily.get(sym, pd.DataFrame())
        sc  = _variance_scale(df["close"], t_var, s_floor, s_cap) \
              if not df.empty and len(df) > P["rv_window"] else 1.0
        raw_allocs[sym] = base * sc

    total = sum(raw_allocs.values())
    if total > capital * 0.99 and total > 0:
        scale_down = capital * 0.99 / total
        raw_allocs = {s: v * scale_down for s, v in raw_allocs.items()}

    # 7. Compute per-coin indicator values for every symbol in universe
    btc_ret_14d = _nday_return(btc_daily["close"], P["bear_rs_window"]) \
                  if not btc_daily.empty else np.nan

    signals: dict[str, CoinSignal] = {}
    for sym in symbols:
        df = daily.get(sym, pd.DataFrame())

        price       = float(df["close"].iloc[-1]) if not df.empty else np.nan
        vol_usd     = float(
            df["quote_volume"].tail(1).iloc[0] if not df.empty
            and "quote_volume" in df.columns else np.nan
        )
        mom7        = _nday_return(df["close"], P["momentum_window"]) \
                      if not df.empty else np.nan
        rsi_val     = float(_compute_rsi(df["close"], P["rsi_period"]).iloc[-1]) \
                      if not df.empty and len(df) >= P["rsi_period"] else 50.0
        ret_14d     = _nday_return(df["close"], P["bear_rs_window"]) \
                      if not df.empty else np.nan
        rel_str     = float(ret_14d - btc_ret_14d) \
                      if not (np.isnan(ret_14d) or np.isnan(btc_ret_14d)) else 0.0
        is_bk       = (not df.empty and len(df) >= P["bull_breakout_days"] + 1)
        if is_bk:
            close_now = float(df["close"].iloc[-1])
            prev_hi   = float(df["close"].iloc[-(P["bull_breakout_days"] + 1):-1].max())
            bk_str    = (close_now - prev_hi) / prev_hi if prev_hi > 0 else 0.0
        else:
            bk_str    = 0.0
        red_day     = _is_red_day(df) if not df.empty else False
        is_sel      = sym in selected
        rank        = selected.index(sym) + 1 if is_sel else 99
        sc          = _variance_scale(df["close"], t_var, s_floor, s_cap) \
                      if not df.empty and len(df) > P["rv_window"] else 1.0

        if regime == "BULL":
            entry_sig = "breakout_open" if is_sel else "none"
        elif is_sel:
            entry_sig = "red_day" if red_day else "fallback_thursday"
        else:
            entry_sig = "none"

        signals[sym] = CoinSignal(
            symbol       = sym,
            pair         = to_roostoo_pair(sym),
            regime       = regime,
            is_selected  = is_sel,
            rank         = rank,
            momentum_7d  = mom7,
            rsi          = rsi_val,
            rel_strength = rel_str,
            breakout_str = bk_str,
            is_red_day   = red_day,
            stop_pct     = stop_pct,
            tp_pct       = tp_pct,
            size_scale   = sc,
            price        = price,
            vol_24h_usd  = vol_usd,
            entry_signal = entry_sig,
        )

    return MarketSnapshot(
        regime           = regime_info,
        signals          = signals,
        selected_symbols = selected,
        capital_allocs   = raw_allocs,
    )