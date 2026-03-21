"""
backtest.py  --  MRVS Regime-Switching Strategy on 1-hour OHLCV data

Architecture
  Signals  : computed on DAILY bars (resampled from hourly)
  Stops/TP : checked against every HOURLY candle high/low (realistic)

Regime detection (every Monday, using BTC daily data):
  BULL      BTC 20-day return > +8%  AND BTC price > 50-day MA
  BEAR      BTC 20-day return < -8%  OR  BTC price < 50-day MA by >5%
  SIDEWAYS  everything else
  BEAR_GATE BTC 7-day return < -10%  (sit out entirely)

Regime playbooks:
  SIDEWAYS  Mean reversion on red days, RSI guard, 1:3 RR, tight stops
  BULL      Momentum breakout (N-day highs), relaxed RSI, wider stops,
            TP disabled so big runs aren't cut short
  BEAR      Relative-strength selection (coins holding up vs BTC),
            tighter stops, smaller sizes, faster TP

Usage
  python backtest.py
  python backtest.py --start 2025-01-01 --end 2025-11-01
  python backtest.py --capital 1000000 --no-plot
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings("ignore")


def ts(d):
    return pd.Timestamp(d).normalize()


# ── Parameters ────────────────────────────────────────────────────────────────

PARAMS = dict(
    data_dir        = "./data",
    start_date      = "2022-01-01",
    end_date        = "2024-12-31",
    initial_capital = 1_000_000,

    # Universe
    min_avg_volume  = 10_000_000,

    # Shared indicators (daily bars)
    momentum_window = 7,
    rsi_period      = 14,
    ma_short        = 20,
    ma_long         = 50,
    rv_window       = 5,

    # Regime thresholds
    regime_symbol       = "BTCUSDT",
    bull_return_thresh  =  0.08,
    bear_return_thresh  = -0.08,
    bear_ma_gap         =  0.05,
    bear_weekly_gate    = -0.10,

    # ── SIDEWAYS playbook ──────────────────────────────────────────
    sw_n_positions       = 3,
    sw_rsi_overbought    = 70,
    sw_rsi_early_exit    = 75,
    sw_early_exit_profit = 0.03,
    sw_position_stop     = 0.05,
    sw_breakeven_trigger = 0.05,
    sw_trailing_stop     = 0.04,
    sw_rr_ratio          = 3.0,    # TP at +15%
    sw_size_cap          = 1.5,
    sw_size_floor        = 0.25,
    sw_target_var        = 0.0025,

    # ── BULL playbook ──────────────────────────────────────────────
    bull_n_positions       = 3,
    bull_breakout_days     = 10,   # enter coins at N-day high
    bull_rsi_overbought    = 80,   # relaxed — overbought is fine in bull
    bull_position_stop     = 0.07, # wider — give momentum room
    bull_breakeven_trigger = 0.08,
    bull_trailing_stop     = 0.05,
    bull_rr_ratio          = 0.0,  # TP OFF — let winners run
    bull_size_cap          = 1.5,
    bull_size_floor        = 0.5,  # bolder sizing in bull
    bull_target_var        = 0.0025,

    # ── BEAR playbook ──────────────────────────────────────────────
    bear_n_positions       = 2,    # fewer, more selective
    bear_rs_window         = 14,   # relative-strength vs BTC window
    bear_rsi_overbought    = 65,   # tighter entry guard
    bear_position_stop     = 0.04, # tightest stop
    bear_breakeven_trigger = 0.04,
    bear_trailing_stop     = 0.03,
    bear_rr_ratio          = 2.0,  # TP at +8% — take profits quickly
    bear_size_cap          = 1.0,  # never above base in bear
    bear_size_floor        = 0.25,
    bear_target_var        = 0.0025,

    # Portfolio stop (all regimes)
    portfolio_stop  = 0.12,

    # Fees
    maker_fee        = 0.0005,
    taker_fee        = 0.001,
    use_limit_orders = True,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}. Run download_data.py first.")

    hourly_data = {}
    daily_data  = {}

    for fp in sorted(files):
        symbol   = os.path.basename(fp).replace(".csv", "")
        df       = pd.read_csv(fp)
        date_col = "datetime" if "datetime" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = (df.dropna(subset=[date_col])
               .rename(columns={date_col: "datetime"})
               .set_index("datetime")
               .sort_index())
        df.index = pd.DatetimeIndex(df.index)
        df = df[(df.index.year >= 2000) & (df.index.year <= 2035)]
        cols = [c for c in ["open","high","low","close","volume","quote_volume"]
                if c in df.columns]
        df = df[cols].astype(float)
        if len(df) < 50:
            continue

        hourly_data[symbol] = df
        agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        if "quote_volume" in df.columns:
            agg["quote_volume"] = "sum"
        daily_data[symbol] = df.resample("1D").agg(agg).dropna(subset=["close"])

    print(f"\nLoaded {len(hourly_data)} symbols from {data_dir}")
    print(f"  {'Symbol':<14} {'Hours':>7}  {'Days':>5}  {'From':<12} {'To'}")
    print("  " + "-" * 54)
    for sym in sorted(hourly_data):
        h = hourly_data[sym]; d = daily_data[sym]
        print(f"  {sym:<14} {len(h):>7,}  {len(d):>5}  "
              f"{str(h.index[0].date()):<12} {h.index[-1].date()}")
    print()
    return hourly_data, daily_data


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    d  = series.diff()
    ag = d.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    al = (-d).clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def realized_variance(log_rets, window):
    r = log_rets.iloc[-window:]
    return float((r**2).sum()) if len(r) >= 2 else PARAMS["target_var"]


def nday_return(closes, idx, window):
    if idx < window: return np.nan
    p, t = closes.iloc[idx - window], closes.iloc[idx]
    return np.nan if p == 0 else (t - p) / p


# ── Universe filter ───────────────────────────────────────────────────────────

_diagnosed = False

def get_universe(daily_data, cutoff_ts, p):
    global _diagnosed
    cutoff = ts(cutoff_ts)
    eligible, diag = [], []
    for sym, df in daily_data.items():
        hist = df.loc[:cutoff]
        if len(hist) < 20:
            diag.append((sym, 0, "< 20 days")); continue
        r   = hist.tail(20)
        vol = (r["quote_volume"].mean() if "quote_volume" in r.columns
               else (r["close"] * r["volume"]).mean())
        ok  = vol >= p["min_avg_volume"]
        diag.append((sym, vol, "PASS" if ok else "FAIL"))
        if ok: eligible.append(sym)
    if not _diagnosed:
        _diagnosed = True
        print(f"\n  [Universe at {cutoff.date()}  threshold=${p['min_avg_volume']:,.0f}]")
        print(f"  {'Symbol':<14} {'Avg USD Vol':>18}  Status")
        print("  " + "-" * 48)
        for sym, vol, status in sorted(diag, key=lambda x: -x[1]):
            print(f"  {sym:<14} ${vol:>17,.0f}  {status}")
        print(f"\n  Eligible: {eligible}\n")
    return eligible


# ── Regime classifier ─────────────────────────────────────────────────────────

def classify_regime(daily_data, monday_ts, p):
    """
    Returns (regime_str, info_dict).
    regime_str: "BULL" | "SIDEWAYS" | "BEAR" | "BEAR_GATE"
    """
    sym  = p.get("regime_symbol", "BTCUSDT")
    hist = daily_data.get(sym, pd.DataFrame()).loc[:ts(monday_ts)]
    needed = max(p["ma_long"] + 1, 21)
    empty_info = dict(ret_7d=np.nan, ret_20d=np.nan, ma50_gap=np.nan)

    if len(hist) < needed:
        return "SIDEWAYS", empty_info

    close    = hist["close"]
    n        = len(close)
    ret_7d   = nday_return(close, n-1, 7)
    ret_20d  = nday_return(close, n-1, p["ma_short"])
    ma50     = close.iloc[-p["ma_long"]:].mean()
    ma50_gap = (close.iloc[-1] - ma50) / ma50
    info     = dict(ret_7d=ret_7d, ret_20d=ret_20d, ma50_gap=ma50_gap)

    if not np.isnan(ret_7d) and ret_7d < p["bear_weekly_gate"]:
        return "BEAR_GATE", info

    if (not np.isnan(ret_20d)
            and ret_20d > p["bull_return_thresh"]
            and ma50_gap > 0):
        return "BULL", info

    if (not np.isnan(ret_20d)
            and (ret_20d < p["bear_return_thresh"]
                 or ma50_gap < -p["bear_ma_gap"])):
        return "BEAR", info

    return "SIDEWAYS", info


# ── Coin selectors ────────────────────────────────────────────────────────────

def select_sideways(daily_data, universe, m_ts, p):
    scores = {}
    for sym in universe:
        df = daily_data[sym]
        if m_ts not in df.index: continue
        idx = df.index.get_loc(m_ts)
        r   = nday_return(df["close"], idx, p["momentum_window"])
        if not np.isnan(r): scores[sym] = r
    candidates = sorted(scores, key=scores.get, reverse=True)[:p["sw_n_positions"]*3]
    valid = []
    for sym in candidates:
        hist = daily_data[sym].loc[:m_ts]
        if len(hist) < p["rsi_period"]+1: continue
        rsi = compute_rsi(hist["close"], p["rsi_period"]).iloc[-1]
        if rsi <= p["sw_rsi_overbought"]:
            valid.append(sym)
        if len(valid) == p["sw_n_positions"]: break
    return valid


def select_bull(daily_data, universe, m_ts, p):
    """Select coins hitting N-day breakout highs, sorted by breakout strength."""
    breakouts = {}
    for sym in universe:
        df = daily_data[sym]
        if m_ts not in df.index: continue
        hist = df.loc[:m_ts]
        if len(hist) < p["bull_breakout_days"] + 1: continue
        close_now  = hist["close"].iloc[-1]
        prev_high  = hist["close"].iloc[-(p["bull_breakout_days"]+1):-1].max()
        rsi_now    = compute_rsi(hist["close"], p["rsi_period"]).iloc[-1]
        if close_now >= prev_high * 0.98 and rsi_now <= p["bull_rsi_overbought"]:
            breakouts[sym] = (close_now - prev_high) / prev_high
    ranked = sorted(breakouts, key=breakouts.get, reverse=True)
    return ranked[:p["bull_n_positions"]]


def select_bear(daily_data, universe, m_ts, p):
    """Select coins with the best relative strength vs BTC."""
    btc_sym = p.get("regime_symbol", "BTCUSDT")
    btc_df  = daily_data.get(btc_sym)
    btc_ret = np.nan
    if btc_df is not None and m_ts in btc_df.index:
        idx     = btc_df.index.get_loc(m_ts)
        btc_ret = nday_return(btc_df["close"], idx, p["bear_rs_window"])

    rel_strengths = {}
    for sym in universe:
        if sym == btc_sym: continue
        df = daily_data[sym]
        if m_ts not in df.index: continue
        hist = df.loc[:m_ts]
        if len(hist) < p["bear_rs_window"] + p["rsi_period"] + 1: continue
        idx      = df.index.get_loc(m_ts)
        coin_ret = nday_return(df["close"], idx, p["bear_rs_window"])
        if np.isnan(coin_ret): continue
        rs  = coin_ret - btc_ret if not np.isnan(btc_ret) else coin_ret
        rsi = compute_rsi(hist["close"], p["rsi_period"]).iloc[-1]
        if rs > 0 and rsi <= p["bear_rsi_overbought"]:
            rel_strengths[sym] = rs
    ranked = sorted(rel_strengths, key=rel_strengths.get, reverse=True)
    return ranked[:p["bear_n_positions"]]


# ── Entry helpers ─────────────────────────────────────────────────────────────

def _make_pos(entry_price, alloc, entry_date, fee, tp_pct):
    ep  = entry_price * (1 + fee)
    sh  = alloc / ep
    tp  = entry_price * (1 + tp_pct) if tp_pct > 0 else float("inf")
    return dict(entry_price=entry_price, shares=sh, entry_date=entry_date,
                alloc=alloc, highest_price=entry_price, take_profit=tp)


def _buy_log(sym, date, price, shares, value, trigger, regime):
    return dict(symbol=sym, action="BUY", date=date, price=round(price,4),
                shares=round(shares,6), value=round(value,2),
                trigger=trigger, regime=regime)


def red_day_entry(daily_data, sym, alloc, week_days, fee, regime, tp_pct):
    daily = daily_data[sym]
    for ed in week_days[:3]:
        eday = ts(ed)
        if eday not in daily.index: continue
        idx = daily.index.get_loc(eday)
        if idx == 0: continue
        pc, tc = daily["close"].iloc[idx-1], daily["close"].iloc[idx]
        if tc < pc:
            pos = _make_pos(tc, alloc, eday, fee, tp_pct)
            log = _buy_log(sym, eday.date(), tc, pos["shares"], alloc,
                           "red_day", regime)
            return pos, log
    thu = ts(week_days[3] if len(week_days) > 3 else week_days[-1])
    if thu in daily.index:
        ep  = daily["open"].loc[thu]
        pos = _make_pos(ep, alloc, thu, fee, tp_pct)
        log = _buy_log(sym, thu.date(), ep, pos["shares"], alloc,
                       "fallback_thursday", regime)
        return pos, log
    return None, None


def breakout_entry(daily_data, sym, alloc, week_days, fee, regime, tp_pct):
    daily = daily_data[sym]
    eday  = ts(week_days[0])
    if eday not in daily.index:
        return None, None
    ep  = daily["open"].loc[eday]
    if np.isnan(ep) or ep <= 0:
        ep = daily["close"].loc[eday]
    pos = _make_pos(ep, alloc, eday, fee, tp_pct)
    log = _buy_log(sym, eday.date(), ep, pos["shares"], alloc,
                   "breakout_open", regime)
    return pos, log


# ── Hourly intraweek monitoring ───────────────────────────────────────────────

REBALANCE_HOUR            = 9     # 09:00 local — mirrors bot.py ROTATION_HOUR
REBALANCE_DRIFT_THRESHOLD = 0.05  # 5 % weight drift triggers a nudge


def run_hourly_loop(hourly_data, daily_data, positions, capital, peak,
                    monday, friday, fee, p,
                    position_stop, breakeven_trigger, trailing_stop,
                    rsi_early_exit, early_exit_profit, regime,
                    allocs):          # ← target USD allocations per symbol
    week_pnl  = {s: 0.0 for s in positions}
    remaining = dict(positions)
    trades    = []

    m_ts  = ts(monday)
    f_ts  = ts(friday)
    w_end = f_ts + pd.Timedelta(hours=23)

    all_hrs = set()
    for sym in remaining:
        if sym in hourly_data:
            h = hourly_data[sym]
            all_hrs.update(h.index[(h.index >= m_ts) & (h.index <= w_end)].tolist())
    all_hrs   = sorted(all_hrs)
    last_hour = all_hrs[-1] if all_hrs else None

    rsi_cache       = {}
    rebalanced_days = set()   # date objects already rebalanced this week

    for h_ts in all_hrs:
        if not remaining: break
        h_day  = ts(h_ts)
        is_eod = (h_ts.hour == 23) or (h_ts == last_hour)
        is_eow = (h_day >= f_ts) and (h_ts == last_hour)

        # ── Daily micro-rebalance at 09:00 ────────────────────────────────
        # Mirrors bot.py _daily_rebalance: nudge the most-drifted position
        # back toward its target weight.  Only ONE partial order per day.
        if (h_ts.hour == REBALANCE_HOUR
                and h_day.date() not in rebalanced_days
                and len(remaining) > 0):
            rebalanced_days.add(h_day.date())

            # Current mark-to-market value per position
            prices_now = {}
            for sym in remaining:
                h = hourly_data.get(sym)
                prices_now[sym] = (
                    float(h["close"].loc[h_ts])
                    if h is not None and h_ts in h.index
                    else remaining[sym]["entry_price"]
                )

            total_value = sum(remaining[sym]["shares"] * prices_now[sym]
                              for sym in remaining)
            target_total = sum(allocs.get(s, 0.0) for s in remaining)

            if total_value > 0 and target_total > 0:
                # Find position with largest absolute weight drift
                best_sym, best_drift, best_delta = None, 0.0, 0.0
                for sym in remaining:
                    cur_val = remaining[sym]["shares"] * prices_now[sym]
                    cur_wt  = cur_val / total_value
                    tgt_wt  = allocs.get(sym, 0.0) / target_total
                    drift   = cur_wt - tgt_wt
                    if abs(drift) > best_drift:
                        best_drift = abs(drift)
                        best_sym   = sym
                        best_delta = (tgt_wt - cur_wt) * total_value  # neg=trim, pos=topup

                if best_sym and best_drift >= REBALANCE_DRIFT_THRESHOLD:
                    sym   = best_sym
                    pos   = remaining[sym]
                    price = prices_now[sym]

                    if best_delta < 0:
                        # Overweight — partial sell (trim)
                        trim_usd = abs(best_delta)
                        trim_qty = trim_usd / price
                        if trim_qty < pos["shares"] and trim_usd > 1.0:
                            proceeds = trim_qty * price * (1 - fee)
                            pos["shares"] -= trim_qty
                            pos["alloc"]  -= trim_usd
                            week_pnl[sym]  = week_pnl.get(sym, 0.0) + (
                                proceeds - trim_usd)
                            trades.append(dict(
                                symbol=sym, action="SELL",
                                date=h_ts.date(), hour=str(h_ts),
                                price=round(price, 4),
                                shares=round(trim_qty, 6),
                                value=round(proceeds, 2),
                                trigger="rebalance_trim",
                                pnl=round(proceeds - trim_usd, 2),
                                regime=regime))
                    else:
                        # Underweight — partial buy (top-up), capped at 10% capital
                        topup_usd = min(best_delta, capital * 0.10)
                        if topup_usd > 1.0:
                            cost      = topup_usd * (1 + fee)
                            new_shares = topup_usd / price
                            pos["shares"] += new_shares
                            pos["alloc"]  += topup_usd
                            week_pnl[sym]  = week_pnl.get(sym, 0.0) - cost
                            trades.append(dict(
                                symbol=sym, action="BUY",
                                date=h_ts.date(), hour=str(h_ts),
                                price=round(price, 4),
                                shares=round(new_shares, 6),
                                value=round(topup_usd, 2),
                                trigger="rebalance_topup",
                                pnl=0.0,
                                regime=regime))

        # Portfolio stop — check at day close
        if is_eod:
            pv = capital - sum(pos["alloc"] for pos in remaining.values())
            for sym, pos in list(remaining.items()):
                h  = hourly_data.get(sym)
                px = (h["close"].loc[h_ts]
                      if h is not None and h_ts in h.index
                      else pos["entry_price"])
                pv += pos["shares"] * px
            if pv < peak * (1 - p["portfolio_stop"]):
                for sym, pos in list(remaining.items()):
                    h  = hourly_data.get(sym)
                    ep = (h["close"].loc[h_ts]
                          if h is not None and h_ts in h.index
                          else pos["entry_price"])
                    proceeds = pos["shares"] * ep * (1 - fee)
                    week_pnl[sym] = proceeds - pos["alloc"]
                    trades.append(dict(symbol=sym, action="SELL",
                                       date=h_ts.date(), hour=str(h_ts),
                                       price=round(ep,4), shares=round(pos["shares"],6),
                                       value=round(proceeds,2), trigger="portfolio_stop",
                                       pnl=round(week_pnl[sym],2), regime=regime))
                remaining = {}
                break

        for sym in list(remaining.keys()):
            pos = remaining[sym]
            if h_ts < pos["entry_date"]: continue
            if sym not in hourly_data: continue
            h = hourly_data[sym]
            if h_ts not in h.index: continue

            c_high = float(h["high"].loc[h_ts])
            c_low  = float(h["low"].loc[h_ts])
            c_cls  = float(h["close"].loc[h_ts])

            pos["highest_price"] = max(pos["highest_price"], c_high)
            pos_ret = (c_cls - pos["entry_price"]) / pos["entry_price"]
            rf_hi   = (c_cls - pos["highest_price"]) / pos["highest_price"]

            if sym not in rsi_cache or h_ts.hour == 0:
                dh = daily_data.get(sym, pd.DataFrame()).loc[:h_day]
                rsi_cache[sym] = (
                    float(compute_rsi(dh["close"], p["rsi_period"]).iloc[-1])
                    if len(dh) >= p["rsi_period"] else 50.0)
            rsi_now = rsi_cache[sym]

            exit_trigger = None
            exit_price   = c_cls

            if c_high >= pos["take_profit"]:
                exit_trigger = "take_profit"
                exit_price   = pos["take_profit"]
            elif c_low <= pos["entry_price"] * (1 - position_stop):
                exit_trigger = "position_stop"
                exit_price   = pos["entry_price"] * (1 - position_stop)
            elif (pos["highest_price"] >= pos["entry_price"] * (1 + breakeven_trigger)
                  and rf_hi <= -trailing_stop):
                exit_trigger = "trailing_stop"
            elif (is_eod and rsi_now > rsi_early_exit
                  and pos_ret > early_exit_profit
                  and h_day > pos["entry_date"]):
                exit_trigger = "early_profit"
            elif is_eow:
                exit_trigger = "weekly_exit"

            if exit_trigger:
                proceeds      = pos["shares"] * exit_price * (1 - fee)
                week_pnl[sym] = proceeds - pos["alloc"]
                trades.append(dict(symbol=sym, action="SELL",
                                   date=h_ts.date(), hour=str(h_ts),
                                   price=round(exit_price,4),
                                   shares=round(pos["shares"],6),
                                   value=round(proceeds,2),
                                   trigger=exit_trigger,
                                   pnl=round(week_pnl[sym],2),
                                   regime=regime))
                del remaining[sym]

    for sym, pos in remaining.items():
        h  = hourly_data.get(sym)
        ep = (h["close"].iloc[-1]
              if h is not None and len(h) > 0
              else pos["entry_price"])
        proceeds      = pos["shares"] * ep * (1 - fee)
        week_pnl[sym] = proceeds - pos["alloc"]
        trades.append(dict(symbol=sym, action="SELL",
                           date=f_ts.date(), hour="end",
                           price=round(ep,4), shares=round(pos["shares"],6),
                           value=round(proceeds,2), trigger="forced_exit",
                           pnl=round(week_pnl[sym],2), regime=regime))

    new_peak = max(peak, capital + sum(week_pnl.values()))
    return week_pnl, trades, new_peak


# ── Weekly execution ──────────────────────────────────────────────────────────

def run_week(hourly_data, daily_data, monday, friday, capital, peak, p):
    fee   = p["maker_fee"] if p["use_limit_orders"] else p["taker_fee"]
    m_ts  = ts(monday)
    f_ts  = ts(friday)

    regime, _ = classify_regime(daily_data, m_ts, p)
    if regime == "BEAR_GATE":
        return {}, capital, peak, [], regime

    universe  = get_universe(daily_data, m_ts, p)
    if not universe:
        return {}, capital, peak, [], regime

    week_days = pd.date_range(m_ts, f_ts, freq="B")

    # Per-regime parameters
    if regime == "BULL":
        coins      = select_bull(daily_data, universe, m_ts, p)
        stop       = p["bull_position_stop"]
        be_trig    = p["bull_breakeven_trigger"]
        trail      = p["bull_trailing_stop"]
        rsi_exit   = 85
        ep_profit  = 0.10
        tp_pct     = p["bull_rr_ratio"] * stop
        t_var      = p["bull_target_var"]
        size_cap   = p["bull_size_cap"]
        size_floor = p["bull_size_floor"]
        entry_fn   = breakout_entry
    elif regime == "BEAR":
        coins      = select_bear(daily_data, universe, m_ts, p)
        stop       = p["bear_position_stop"]
        be_trig    = p["bear_breakeven_trigger"]
        trail      = p["bear_trailing_stop"]
        rsi_exit   = p["bear_rsi_overbought"]
        ep_profit  = 0.03
        tp_pct     = p["bear_rr_ratio"] * stop
        t_var      = p["bear_target_var"]
        size_cap   = p["bear_size_cap"]
        size_floor = p["bear_size_floor"]
        entry_fn   = red_day_entry
    else:  # SIDEWAYS
        coins      = select_sideways(daily_data, universe, m_ts, p)
        stop       = p["sw_position_stop"]
        be_trig    = p["sw_breakeven_trigger"]
        trail      = p["sw_trailing_stop"]
        rsi_exit   = p["sw_rsi_early_exit"]
        ep_profit  = p["sw_early_exit_profit"]
        tp_pct     = p["sw_rr_ratio"] * stop
        t_var      = p["sw_target_var"]
        size_cap   = p["sw_size_cap"]
        size_floor = p["sw_size_floor"]
        entry_fn   = red_day_entry

    if not coins:
        return {}, capital, peak, [], regime

    # Variance-scaled allocations — mirrors bot.py signals.py exactly:
    # base uses max n_positions for regime (not actual selected count),
    # per-position cap 40%, total deployed cap 80%
    MAX_POS_PCT    = 0.40
    MAX_DEPLOY_PCT = 0.80
    n_pos          = (p["bull_n_positions"] if regime == "BULL"
                      else p["bear_n_positions"] if regime == "BEAR"
                      else p["sw_n_positions"])
    base        = capital / max(n_pos, 1)
    per_pos_cap = capital * MAX_POS_PCT
    allocs = {}
    for sym in coins:
        hist = daily_data[sym].loc[:m_ts]
        lr   = np.log(hist["close"] / hist["close"].shift(1)).dropna()
        rv   = (realized_variance(lr, p["rv_window"])
                if len(hist) > p["rv_window"] else t_var)
        rv   = rv if rv > 0 else t_var
        allocs[sym] = min(base * np.clip(t_var / rv, size_floor, size_cap),
                          per_pos_cap)

    total = sum(allocs.values())
    if total > capital * MAX_DEPLOY_PCT:
        allocs = {s: v * (capital * MAX_DEPLOY_PCT / total)
                  for s, v in allocs.items()}

    # Build positions
    positions = {}
    buy_logs  = []
    for sym, alloc in allocs.items():
        pos, log = entry_fn(daily_data, sym, alloc,
                            week_days, fee, regime, tp_pct)
        if pos is not None:
            log["week"] = monday.date()
            positions[sym] = pos
            buy_logs.append(log)

    if not positions:
        return {}, capital, peak, buy_logs, regime

    week_pnl, sell_logs, new_peak = run_hourly_loop(
        hourly_data, daily_data, positions, capital, peak,
        monday, friday, fee, p,
        stop, be_trig, trail, rsi_exit, ep_profit, regime,
        allocs)

    for log in sell_logs:
        log.setdefault("week", monday.date())

    end_capital = capital + sum(week_pnl.values())
    return week_pnl, end_capital, new_peak, buy_logs + sell_logs, regime


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(wr, cv, initial):
    r  = np.array(wr, dtype=float)
    cv = np.array(cv, dtype=float)
    avg   = float(r.mean()) if len(r) else 0.0
    std   = r.std(ddof=1) if len(r) > 1 else 1e-9
    neg   = r[r < 0]
    std_d = neg.std(ddof=1) if len(neg) > 1 else 1e-9
    peak  = np.maximum.accumulate(cv)
    dd    = (cv - peak) / peak
    mdd   = float(dd.min())
    ann   = (1 + avg) ** 52 - 1
    return dict(
        total_return   = (cv[-1] - initial) / initial,
        avg_weekly_ret = avg,
        ann_return     = ann,
        sharpe         = avg / std  * np.sqrt(52),
        sortino        = avg / std_d * np.sqrt(52),
        calmar         = ann / abs(mdd) if mdd != 0 else np.nan,
        max_drawdown   = mdd,
        win_rate       = float((r > 0).mean()) if len(r) else 0.0,
        n_weeks        = len(r),
    )


# ── Backtest loop ─────────────────────────────────────────────────────────────

def backtest(p):
    hourly_data, daily_data = load_data(p["data_dir"])
    mondays = pd.date_range(pd.Timestamp(p["start_date"]),
                            pd.Timestamp(p["end_date"]), freq="W-MON")
    capital = float(p["initial_capital"])
    peak    = capital
    all_trades, log = [], []
    regime_counts   = {"BULL": 0, "SIDEWAYS": 0, "BEAR": 0, "BEAR_GATE": 0}

    print(f"Running: {p['start_date']} -> {p['end_date']}")
    print(f"Capital: ${capital:,.0f}\n")

    for monday in mondays:
        friday = monday + timedelta(days=4)
        if friday > pd.Timestamp(p["end_date"]): break

        week_pnl, capital, peak, trades, regime = run_week(
            hourly_data, daily_data, monday, friday, capital, peak, p)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        wr = sum(week_pnl.values()) / (capital - sum(week_pnl.values()) or 1)

        log.append(dict(week=monday.date(),
                        pnl=round(sum(week_pnl.values()), 2),
                        weekly_return=wr,
                        capital=round(capital, 2),
                        coins_traded=list(week_pnl.keys()),
                        regime=regime))
        for t in trades: t.setdefault("week", monday.date())
        all_trades.extend(trades)

        print(f"  {monday.date()} -> {friday.date()} | "
              f"PnL: ${sum(week_pnl.values()):+,.0f} | "
              f"Capital: ${capital:,.0f} | "
              f"Coins: {list(week_pnl.keys())} [{regime}]")

    wdf = pd.DataFrame(log)
    tdf = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    m   = compute_metrics(wdf["weekly_return"].tolist(),
                          wdf["capital"].tolist(), p["initial_capital"])
    return wdf, tdf, m, regime_counts


# ── Competition-mode parameters (live bot settings) ───────────────────────────

COMP_PARAMS = dict(
    PARAMS,                       # inherit everything from PARAMS
    # Allocation caps matching the live bot
    max_deploy_pct   = 0.80,      # total capital deployed ≤ 80%
    max_position_pct = 0.40,      # single position ≤ 40% of capital
    # 1-day red-day fallback (was: 3 days / Thursday)
    red_day_fallback = 1,         # enter regardless after N days
    # No forced weekly close — stops/TP/trailing handle all exits
    force_weekly_close = False,
    # Taker fee only (MARKET orders on Roostoo)
    use_limit_orders = False,
)


# ── Competition entry function (1-day fallback, any day of week) ──────────────

def red_day_entry_comp(daily_data, sym, alloc, all_days, plan_date, fee, regime, tp_pct, fallback_days=1):
    """
    Competition variant of red_day_entry:
      - Checks for a red day on plan_date only (day 0)
      - Falls back immediately on day 1+ (no Thursday wait)
      - all_days: DatetimeIndex of all calendar days from plan_date onward
    """
    daily = daily_data[sym]
    day0  = ts(plan_date)

    # Day 0: check for red day
    if day0 in daily.index:
        idx = daily.index.get_loc(day0)
        if idx > 0:
            pc, tc = daily["close"].iloc[idx - 1], daily["close"].iloc[idx]
            if tc < pc:
                pos = _make_pos(tc, alloc, day0, fee, tp_pct)
                lg  = _buy_log(sym, day0.date(), tc, pos["shares"], alloc, "red_day", regime)
                return pos, lg

    # Day 1+: enter at next available daily open
    for d in all_days[1:fallback_days + 2]:
        eday = ts(d)
        if eday not in daily.index:
            continue
        ep = daily["open"].loc[eday]
        if np.isnan(ep) or ep <= 0:
            ep = daily["close"].loc[eday]
        pos = _make_pos(ep, alloc, eday, fee, tp_pct)
        lg  = _buy_log(sym, eday.date(), ep, pos["shares"], alloc, "fallback_entry", regime)
        return pos, lg

    return None, None


# ── Competition allocation with 80%/40% caps ──────────────────────────────────

def comp_allocs(coins, capital, daily_data, cutoff_ts, t_var, size_floor, size_cap, p):
    """
    Variance-scaled allocations with:
      - Base = capital / max_n_positions (not len(selected))
      - Per-position cap: 40% of capital
      - Total deployment cap: 80% of capital
    """
    n_pos    = max(len(coins), 1)
    base     = capital / n_pos
    pos_cap  = capital * p.get("max_position_pct", 0.40)
    raw      = {}
    for sym in coins:
        hist = daily_data[sym].loc[:ts(cutoff_ts)]
        lr   = np.log(hist["close"] / hist["close"].shift(1)).dropna()
        rv   = (realized_variance(lr, p["rv_window"])
                if len(hist) > p["rv_window"] else t_var)
        rv   = rv if rv > 0 else t_var
        sc   = float(np.clip(t_var / rv, size_floor, size_cap))
        raw[sym] = min(base * sc, pos_cap)

    total      = sum(raw.values())
    max_deploy = capital * p.get("max_deploy_pct", 0.80)
    if total > max_deploy and total > 0:
        f   = max_deploy / total
        raw = {s: v * f for s, v in raw.items()}
    return raw


# ── Competition backtest: continuous loop, no forced weekly close ──────────────

def backtest_comp(p):
    """
    Competition-mode backtest:
      - Plans made on Mondays (coin selection + regime), but positions carry over
      - No forced Friday close — stops/TP/trailing are the only exits
      - 1-day red-day fallback before entering regardless
      - 80%/40% allocation caps
      - Runs on a daily cadence using hourly data for intraday stop/TP checks
    """
    hourly_data, daily_data = load_data(p["data_dir"])
    fee = p["taker_fee"]   # MARKET orders only

    start = pd.Timestamp(p["start_date"])
    end   = pd.Timestamp(p["end_date"])

    # All calendar days in range
    all_days = pd.date_range(start, end, freq="D")
    mondays  = [d for d in all_days if d.weekday() == 0]

    capital  = float(p["initial_capital"])
    peak     = capital
    positions = {}          # {sym: pos_dict}  — carries across weeks
    plan_coins: list  = []  # locked coin list for current week
    plan_date = None        # when current plan was made
    plan_regime = "SIDEWAYS"

    all_trades, log = [], []
    regime_counts   = {"BULL": 0, "SIDEWAYS": 0, "BEAR": 0, "BEAR_GATE": 0}
    last_plan_monday = None

    print(f"Running (competition mode): {p['start_date']} -> {p['end_date']}")
    print(f"Capital: ${capital:,.0f}  fee={fee:.3%}  "
          f"max_deploy={p.get('max_deploy_pct',0.8):.0%}  "
          f"fallback={p.get('red_day_fallback',1)}d\n")

    # ── Outer loop: each calendar day ─────────────────────────────────────────
    for day in all_days:
        day_ts = ts(day)
        is_monday = day.weekday() == 0

        # ── Monday: new plan ─────────────────────────────────────────────────
        if is_monday or last_plan_monday is None:
            regime, _ = classify_regime(daily_data, day_ts, p)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            if regime == "BEAR_GATE":
                # Liquidate all at day open price
                for sym, pos in list(positions.items()):
                    h  = hourly_data.get(sym)
                    ep = (float(h.loc[h.index >= day_ts, "open"].iloc[0])
                          if h is not None and len(h.loc[h.index >= day_ts]) > 0
                          else pos["entry_price"])
                    proceeds = pos["shares"] * ep * (1 - fee)
                    pnl      = proceeds - pos["alloc"]
                    capital += pnl
                    all_trades.append(dict(
                        symbol=sym, action="SELL", date=day.date(),
                        price=round(ep, 4), shares=round(pos["shares"], 6),
                        value=round(proceeds, 2), trigger="bear_gate",
                        pnl=round(pnl, 2), regime=regime))
                positions.clear()
                plan_coins   = []
                plan_date    = day
                plan_regime  = regime
                last_plan_monday = day
                continue

            universe   = get_universe(daily_data, day_ts, p)
            if regime == "BULL":
                new_coins  = select_bull(daily_data, universe, day_ts, p)
                t_var, s_cap, s_fl = p["bull_target_var"], p["bull_size_cap"], p["bull_size_floor"]
                stop    = p["bull_position_stop"]
                be_trig = p["bull_breakeven_trigger"]
                trail   = p["bull_trailing_stop"]
                rsi_exit  = 85
                ep_profit = 0.10
                tp_pct    = p["bull_rr_ratio"] * stop
            elif regime == "BEAR":
                new_coins  = select_bear(daily_data, universe, day_ts, p)
                t_var, s_cap, s_fl = p["bear_target_var"], p["bear_size_cap"], p["bear_size_floor"]
                stop    = p["bear_position_stop"]
                be_trig = p["bear_breakeven_trigger"]
                trail   = p["bear_trailing_stop"]
                rsi_exit  = p["bear_rsi_overbought"]
                ep_profit = 0.03
                tp_pct    = p["bear_rr_ratio"] * stop
            else:
                new_coins  = select_sideways(daily_data, universe, day_ts, p)
                t_var, s_cap, s_fl = p["sw_target_var"], p["sw_size_cap"], p["sw_size_floor"]
                stop    = p["sw_position_stop"]
                be_trig = p["sw_breakeven_trigger"]
                trail   = p["sw_trailing_stop"]
                rsi_exit  = p["sw_rsi_early_exit"]
                ep_profit = p["sw_early_exit_profit"]
                tp_pct    = p["sw_rr_ratio"] * stop

            # Close positions no longer in the new plan
            for sym in list(positions.keys()):
                if sym not in new_coins:
                    h  = hourly_data.get(sym)
                    ep = (float(h.loc[h.index >= day_ts, "open"].iloc[0])
                          if h is not None and len(h.loc[h.index >= day_ts]) > 0
                          else positions[sym]["entry_price"])
                    proceeds = positions[sym]["shares"] * ep * (1 - fee)
                    pnl      = proceeds - positions[sym]["alloc"]
                    capital += pnl
                    all_trades.append(dict(
                        symbol=sym, action="SELL", date=day.date(),
                        price=round(ep, 4), shares=round(positions[sym]["shares"], 6),
                        value=round(proceeds, 2), trigger="plan_change",
                        pnl=round(pnl, 2), regime=regime))
                    del positions[sym]

            plan_coins   = new_coins
            plan_date    = day
            plan_regime  = regime
            last_plan_monday = day

            # Store regime params on each position for exit checks
            _stop, _be, _trail, _rsi_e, _ep_p, _tp = stop, be_trig, trail, rsi_exit, ep_profit, tp_pct

        # ── Entry: open new positions for unowned plan coins ─────────────────
        # Compute free cash (capital minus mark-to-market of held positions)
        if plan_coins and plan_regime != "BEAR_GATE":
            regime = plan_regime
            if regime == "BULL":
                t_var, s_cap, s_fl = p["bull_target_var"], p["bull_size_cap"], p["bull_size_floor"]
                stop    = p["bull_position_stop"]
                be_trig = p["bull_breakeven_trigger"]
                trail   = p["bull_trailing_stop"]
                rsi_exit  = 85; ep_profit = 0.10
                tp_pct    = p["bull_rr_ratio"] * stop
            elif regime == "BEAR":
                t_var, s_cap, s_fl = p["bear_target_var"], p["bear_size_cap"], p["bear_size_floor"]
                stop    = p["bear_position_stop"]
                be_trig = p["bear_breakeven_trigger"]
                trail   = p["bear_trailing_stop"]
                rsi_exit  = p["bear_rsi_overbought"]; ep_profit = 0.03
                tp_pct    = p["bear_rr_ratio"] * stop
            else:
                t_var, s_cap, s_fl = p["sw_target_var"], p["sw_size_cap"], p["sw_size_floor"]
                stop    = p["sw_position_stop"]
                be_trig = p["sw_breakeven_trigger"]
                trail   = p["sw_trailing_stop"]
                rsi_exit  = p["sw_rsi_early_exit"]; ep_profit = p["sw_early_exit_profit"]
                tp_pct    = p["sw_rr_ratio"] * stop

            # Mark-to-market free cash
            mtm = 0.0
            for sym, pos in positions.items():
                h  = hourly_data.get(sym)
                px = (float(h.loc[h.index >= day_ts, "open"].iloc[0])
                      if h is not None and len(h.loc[h.index >= day_ts]) > 0
                      else pos["entry_price"])
                mtm += pos["shares"] * px
            free_cash = max(capital - mtm, 0.0)

            unfilled  = [s for s in plan_coins if s not in positions]
            allocs    = comp_allocs(unfilled, free_cash, daily_data, day_ts,
                                    t_var, s_fl, s_cap, p) if unfilled else {}

            future_days = pd.date_range(day, end, freq="D")
            for sym in unfilled:
                alloc = allocs.get(sym, 0.0)
                if alloc < 10.0:
                    continue
                days_since_plan = (day - ts(plan_date)).days

                if regime == "BULL":
                    pos, lg = breakout_entry(daily_data, sym, alloc, [day], fee, regime, tp_pct)
                else:
                    pos, lg = red_day_entry_comp(
                        daily_data, sym, alloc, future_days, day,
                        fee, regime, tp_pct,
                        fallback_days=p.get("red_day_fallback", 1))

                if pos is not None:
                    pos["stop"]    = stop
                    pos["be_trig"] = be_trig
                    pos["trail"]   = trail
                    pos["rsi_exit"] = rsi_exit
                    pos["ep_profit"] = ep_profit
                    pos["regime"]  = regime
                    positions[sym] = pos
                    lg["date"]     = day.date()
                    all_trades.append(lg)

        # ── Hourly exit checks for all open positions ─────────────────────────
        next_day = day_ts + pd.Timedelta(days=1)
        all_hrs  = set()
        for sym in positions:
            if sym in hourly_data:
                h = hourly_data[sym]
                all_hrs.update(h.index[(h.index >= day_ts) & (h.index < next_day)].tolist())
        all_hrs  = sorted(all_hrs)
        last_hour = all_hrs[-1] if all_hrs else None

        for h_ts in all_hrs:
            if not positions:
                break
            h_day  = ts(h_ts)
            is_eod = (h_ts.hour == 23) or (h_ts == last_hour)

            # Portfolio stop at EOD
            if is_eod:
                pv = capital - sum(pos["alloc"] for pos in positions.values())
                for sym, pos in positions.items():
                    h  = hourly_data.get(sym)
                    px = (h["close"].loc[h_ts]
                          if h is not None and h_ts in h.index
                          else pos["entry_price"])
                    pv += pos["shares"] * px
                if pv < peak * (1 - p["portfolio_stop"]):
                    for sym, pos in list(positions.items()):
                        h  = hourly_data.get(sym)
                        ep = (h["close"].loc[h_ts]
                              if h is not None and h_ts in h.index
                              else pos["entry_price"])
                        proceeds  = pos["shares"] * ep * (1 - fee)
                        pnl       = proceeds - pos["alloc"]
                        capital  += pnl
                        all_trades.append(dict(
                            symbol=sym, action="SELL", date=h_ts.date(),
                            price=round(ep, 4), shares=round(pos["shares"], 6),
                            value=round(proceeds, 2), trigger="portfolio_stop",
                            pnl=round(pnl, 2), regime=pos.get("regime", "?")))
                    positions.clear()
                    break

            for sym in list(positions.keys()):
                pos = positions[sym]
                if h_ts < pos["entry_date"]:
                    continue
                if sym not in hourly_data:
                    continue
                h = hourly_data[sym]
                if h_ts not in h.index:
                    continue

                c_high = float(h["high"].loc[h_ts])
                c_low  = float(h["low"].loc[h_ts])
                c_cls  = float(h["close"].loc[h_ts])

                pos["highest_price"] = max(pos["highest_price"], c_high)
                pos_ret = (c_cls - pos["entry_price"]) / pos["entry_price"]
                rf_hi   = (c_cls - pos["highest_price"]) / pos["highest_price"]

                _stop    = pos.get("stop",    p["sw_position_stop"])
                _be      = pos.get("be_trig", p["sw_breakeven_trigger"])
                _trail   = pos.get("trail",   p["sw_trailing_stop"])
                _rsi_e   = pos.get("rsi_exit",  p["sw_rsi_early_exit"])
                _ep_p    = pos.get("ep_profit", p["sw_early_exit_profit"])
                _tp      = pos["take_profit"]

                # RSI (refresh at start of each day)
                rsi_now = 50.0
                if is_eod or h_ts.hour == 0:
                    dh = daily_data.get(sym, pd.DataFrame()).loc[:h_day]
                    if len(dh) >= p["rsi_period"]:
                        rsi_now = float(compute_rsi(dh["close"], p["rsi_period"]).iloc[-1])

                exit_trigger = None
                exit_price   = c_cls

                if c_high >= _tp:
                    exit_trigger = "take_profit"
                    exit_price   = _tp
                elif c_low <= pos["entry_price"] * (1 - _stop):
                    exit_trigger = "position_stop"
                    exit_price   = pos["entry_price"] * (1 - _stop)
                elif (pos["highest_price"] >= pos["entry_price"] * (1 + _be)
                      and rf_hi <= -_trail):
                    exit_trigger = "trailing_stop"
                elif (is_eod and rsi_now > _rsi_e
                      and pos_ret > _ep_p
                      and h_day > pos["entry_date"]):
                    exit_trigger = "early_profit"

                if exit_trigger:
                    proceeds  = pos["shares"] * exit_price * (1 - fee)
                    pnl       = proceeds - pos["alloc"]
                    capital  += pnl
                    peak      = max(peak, capital)
                    all_trades.append(dict(
                        symbol=sym, action="SELL", date=h_ts.date(),
                        price=round(exit_price, 4), shares=round(pos["shares"], 6),
                        value=round(proceeds, 2), trigger=exit_trigger,
                        pnl=round(pnl, 2), regime=pos.get("regime", "?")))
                    del positions[sym]

        peak = max(peak, capital)
        log.append(dict(date=day.date(), capital=round(capital, 2),
                        n_positions=len(positions),
                        coins=list(positions.keys()),
                        regime=plan_regime))

    # Close any remaining positions at end
    last_day = ts(end)
    for sym, pos in list(positions.items()):
        h  = hourly_data.get(sym)
        ep = (h["close"].iloc[-1]
              if h is not None and len(h) > 0
              else pos["entry_price"])
        proceeds  = pos["shares"] * ep * (1 - fee)
        pnl       = proceeds - pos["alloc"]
        capital  += pnl
        all_trades.append(dict(
            symbol=sym, action="SELL", date=end.date(),
            price=round(ep, 4), shares=round(pos["shares"], 6),
            value=round(proceeds, 2), trigger="end_of_backtest",
            pnl=round(pnl, 2), regime=pos.get("regime", "?")))

    ddf = pd.DataFrame(log)
    tdf = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    # Compute weekly returns from daily capital series for metrics
    ddf["date"]    = pd.to_datetime(ddf["date"])
    ddf["isoweek"] = ddf["date"].dt.isocalendar().week
    ddf["isoyear"] = ddf["date"].dt.isocalendar().year
    ddf["week_key"] = ddf["isoyear"] * 100 + ddf["isoweek"]
    weekly_caps = ddf.groupby("week_key")["capital"].last().values
    init        = float(p["initial_capital"])
    weekly_rets = [((weekly_caps[i] - weekly_caps[i-1]) / weekly_caps[i-1])
                   for i in range(1, len(weekly_caps))]

    m = compute_metrics(weekly_rets, list(weekly_caps), init)
    m["total_return"] = (capital - init) / init

    print(f"  Final capital: ${capital:,.2f}  ({m['total_return']:+.2%})")
    return ddf, tdf, m, regime_counts




def print_report(m, p, wdf=None, tdf=None, regime_counts=None):
    print("\n" + "=" * 58)
    print("  MRVS Regime-Switching Strategy — Report")
    print("=" * 58)
    print(f"  Period    : {p['start_date']} -> {p['end_date']}")
    print(f"  Capital   : ${p['initial_capital']:>14,.0f}")
    print("-" * 58)
    print(f"  Total return   : {m['total_return']:>13.2%}")
    print(f"  Ann. return    : {m['ann_return']:>13.2%}")
    print(f"  Avg weekly ret : {m['avg_weekly_ret']:>13.4%}")
    print(f"  Weeks          : {m['n_weeks']:>13d}")
    print(f"  Win rate       : {m['win_rate']:>13.2%}")
    print("-" * 58)
    print(f"  Sharpe         : {m['sharpe']:>13.3f}")
    print(f"  Sortino        : {m['sortino']:>13.3f}")
    print(f"  Calmar         : {m['calmar']:>13.3f}")
    print(f"  Max drawdown   : {m['max_drawdown']:>13.2%}")

    if regime_counts:
        print("-" * 58)
        print("  Regime weeks:")
        for r, cnt in regime_counts.items():
            print(f"    {r:<12} {cnt:>4d}")

    if wdf is not None and "regime" in wdf.columns:
        print("-" * 58)
        print("  Per-regime avg weekly return:")
        for regime, grp in wdf.groupby("regime"):
            avg = grp["weekly_return"].mean()
            wr  = (grp["weekly_return"] > 0).mean()
            print(f"    {regime:<12}  avg={avg:+.3%}  win={wr:.0%}  n={len(grp)}")

    if tdf is not None and not tdf.empty and "trigger" in tdf.columns:
        sells = tdf[tdf["action"] == "SELL"]; n = len(sells)
        if n > 0:
            print("-" * 58)
            print(f"  Exit breakdown ({n} sells):")
            for trig, cnt in sells["trigger"].value_counts().items():
                print(f"    {trig:<24} {cnt:>4d}  ({cnt/n:.0%})")

    print("=" * 58)
    comp = 0.4*m["sortino"] + 0.3*m["sharpe"] + 0.3*m["calmar"]
    print(f"\n  Composite = 0.4×{m['sortino']:.3f} + "
          f"0.3×{m['sharpe']:.3f} + 0.3×{m['calmar']:.3f}")
    print(f"           = {comp:.3f}")
    print("=" * 58)


def plot_results(wdf, p):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("\nmatplotlib not installed — pip install matplotlib"); return

    regime_colors = {"BULL": "#1D9E75", "SIDEWAYS": "#378ADD",
                     "BEAR": "#E24B4A",  "BEAR_GATE": "#888780"}

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("MRVS Regime-Switching Backtest", fontsize=14, fontweight="bold")
    dates = pd.to_datetime(wdf["week"])

    for i in range(len(wdf) - 1):
        c  = regime_colors.get(wdf["regime"].iloc[i], "gray")
        axes[0].plot([dates.iloc[i], dates.iloc[i+1]],
                     [wdf["capital"].iloc[i], wdf["capital"].iloc[i+1]],
                     color=c, linewidth=1.8)
    axes[0].axhline(p["initial_capital"], color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title("Portfolio Value  (green=BULL  blue=SIDEWAYS  red=BEAR  gray=GATE)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    bar_colors = [regime_colors.get(r, "gray") for r in wdf["regime"]]
    axes[1].bar(dates, wdf["weekly_return"] * 100, color=bar_colors, width=5, alpha=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Weekly Return (%)")
    axes[1].set_title("Weekly Returns by Regime")

    cap  = wdf["capital"].values
    pk   = np.maximum.accumulate(cap)
    dd   = (cap - pk) / pk * 100
    axes[2].fill_between(dates, dd, 0, color="#E24B4A", alpha=0.35)
    axes[2].plot(dates, dd, color="#E24B4A", linewidth=1)
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].set_title("Drawdown from Peak")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = "backtest_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved -> {out_path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MRVS Regime-Switching backtester (1h candles)")
    parser.add_argument("--data",    default=PARAMS["data_dir"])
    parser.add_argument("--start",   default=PARAMS["start_date"])
    parser.add_argument("--end",     default=PARAMS["end_date"])
    parser.add_argument("--capital", default=PARAMS["initial_capital"], type=float)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--mode",    default="both",
                        choices=["original", "competition", "both"],
                        help="original=weekly-close strategy  "
                             "competition=no-close+1d-fallback+80pct-cap  "
                             "both=run both and compare (default)")
    args = parser.parse_args()

    p_orig = dict(PARAMS)
    p_orig["data_dir"]        = args.data
    p_orig["start_date"]      = args.start
    p_orig["end_date"]        = args.end
    p_orig["initial_capital"] = args.capital

    p_comp = dict(COMP_PARAMS)
    p_comp["data_dir"]        = args.data
    p_comp["start_date"]      = args.start
    p_comp["end_date"]        = args.end
    p_comp["initial_capital"] = args.capital

    results = {}

    if args.mode in ("original", "both"):
        print("\n" + "█" * 58)
        print("  MODE: ORIGINAL  (weekly close, 3-day fallback, no caps)")
        print("█" * 58)
        wdf, tdf, m, rc = backtest(p_orig)
        print_report(m, p_orig, wdf, tdf, rc)
        wdf.to_csv("weekly_results_original.csv", index=False)
        if not tdf.empty:
            tdf.to_csv("trade_log_original.csv", index=False)
        results["original"] = (m, wdf)

    if args.mode in ("competition", "both"):
        print("\n" + "█" * 58)
        print("  MODE: COMPETITION  (no forced close, 1-day fallback, 80% cap)")
        print("█" * 58)
        ddf, tdf2, m2, rc2 = backtest_comp(p_comp)
        print_report(m2, p_comp, tdf=tdf2, regime_counts=rc2)
        ddf.to_csv("daily_results_competition.csv", index=False)
        if not tdf2.empty:
            tdf2.to_csv("trade_log_competition.csv", index=False)
        results["competition"] = (m2, ddf)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    if args.mode == "both" and len(results) == 2:
        mo, mc = results["original"][0], results["competition"][0]
        print("\n" + "=" * 68)
        print("  SIDE-BY-SIDE COMPARISON")
        print("=" * 68)
        print(f"  {'Metric':<22} {'Original':>16} {'Competition':>16}  {'Delta':>10}")
        print("  " + "-" * 66)
        rows = [
            ("Total return",   f"{mo['total_return']:>+15.2%}", f"{mc['total_return']:>+15.2%}",
             f"{mc['total_return']-mo['total_return']:>+9.2%}"),
            ("Sharpe",         f"{mo['sharpe']:>16.3f}",        f"{mc['sharpe']:>16.3f}",
             f"{mc['sharpe']-mo['sharpe']:>+9.3f}"),
            ("Sortino",        f"{mo['sortino']:>16.3f}",       f"{mc['sortino']:>16.3f}",
             f"{mc['sortino']-mo['sortino']:>+9.3f}"),
            ("Calmar",         f"{mo['calmar']:>16.3f}",        f"{mc['calmar']:>16.3f}",
             f"{mc['calmar']-mo['calmar']:>+9.3f}"),
            ("Max drawdown",   f"{mo['max_drawdown']:>+15.2%}", f"{mc['max_drawdown']:>+15.2%}",
             f"{mc['max_drawdown']-mo['max_drawdown']:>+9.2%}"),
            ("Win rate",       f"{mo['win_rate']:>15.1%}",      f"{mc['win_rate']:>15.1%}",
             f"{mc['win_rate']-mo['win_rate']:>+9.1%}"),
        ]
        for label, v_o, v_c, delta in rows:
            print(f"  {label:<22} {v_o}  {v_c}  {delta}")

        comp_o = 0.4*mo["sortino"] + 0.3*mo["sharpe"] + 0.3*mo["calmar"]
        comp_c = 0.4*mc["sortino"] + 0.3*mc["sharpe"] + 0.3*mc["calmar"]
        print("  " + "-" * 66)
        print(f"  {'Composite score':<22} {comp_o:>16.3f}  {comp_c:>16.3f}  {comp_c-comp_o:>+9.3f}")
        print("=" * 68)
        winner = "Competition" if comp_c > comp_o else "Original"
        print(f"\n  → {winner} mode scores higher on hackathon composite metric.\n")

    if not args.no_plot and args.mode in ("original", "both"):
        wdf = results["original"][1]
        plot_results(wdf, p_orig)


if __name__ == "__main__":
    main()