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

def run_hourly_loop(hourly_data, daily_data, positions, capital, peak,
                    monday, friday, fee, p,
                    position_stop, breakeven_trigger, trailing_stop,
                    rsi_early_exit, early_exit_profit, regime):
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

    rsi_cache = {}

    for h_ts in all_hrs:
        if not remaining: break
        h_day  = ts(h_ts)
        is_eod = (h_ts.hour == 23) or (h_ts == last_hour)
        is_eow = (h_day >= f_ts) and (h_ts == last_hour)

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

    # Variance-scaled allocations
    base   = capital / max(len(coins), 1)
    allocs = {}
    for sym in coins:
        hist = daily_data[sym].loc[:m_ts]
        lr   = np.log(hist["close"] / hist["close"].shift(1)).dropna()
        rv   = (realized_variance(lr, p["rv_window"])
                if len(hist) > p["rv_window"] else t_var)
        rv   = rv if rv > 0 else t_var
        allocs[sym] = base * np.clip(t_var / rv, size_floor, size_cap)

    total = sum(allocs.values())
    if total > capital * 0.99:
        allocs = {s: v * (capital * 0.99 / total) for s, v in allocs.items()}

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
        stop, be_trig, trail, rsi_exit, ep_profit, regime)

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


# ── Reporting ─────────────────────────────────────────────────────────────────

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
    args = parser.parse_args()

    p = dict(PARAMS)
    p["data_dir"]        = args.data
    p["start_date"]      = args.start
    p["end_date"]        = args.end
    p["initial_capital"] = args.capital

    wdf, tdf, m, regime_counts = backtest(p)
    print_report(m, p, wdf, tdf, regime_counts)

    wdf.to_csv("weekly_results.csv", index=False)
    if not tdf.empty:
        tdf.to_csv("trade_log.csv", index=False)
        print(f"\n  weekly_results.csv  ({len(wdf)} weeks)")
        print(f"  trade_log.csv       ({len(tdf)} trades)")

    if not args.no_plot:
        plot_results(wdf, p)


if __name__ == "__main__":
    main()