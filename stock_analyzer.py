import json
import math
import random
from datetime import datetime, timedelta

# ─── Stock Data ───────────────────────────────────────────────────────────────
STOCKS = {
    "AAPL":  {"name": "Apple Inc.",          "base": 150, "volatility": 0.018, "trend": 0.0004},
    "GOOGL": {"name": "Alphabet Inc.",        "base": 130, "volatility": 0.020, "trend": 0.0003},
    "MSFT":  {"name": "Microsoft Corp.",      "base": 310, "volatility": 0.016, "trend": 0.0005},
    "TSLA":  {"name": "Tesla Inc.",           "base": 200, "volatility": 0.035, "trend": 0.0002},
    "AMZN":  {"name": "Amazon.com Inc.",      "base": 180, "volatility": 0.022, "trend": 0.0004},
    "NVDA":  {"name": "NVIDIA Corp.",         "base": 450, "volatility": 0.030, "trend": 0.0008},
    "META":  {"name": "Meta Platforms Inc.",  "base": 300, "volatility": 0.025, "trend": 0.0006},
    "NFLX":  {"name": "Netflix Inc.",         "base": 400, "volatility": 0.028, "trend": 0.0003},
}

def generate_stock_data(ticker, days=365, seed=None):
    """Generates realistic OHLCV data using Geometric Brownian Motion."""
    if ticker not in STOCKS:
        raise ValueError(f"Unknown ticker: {ticker}. Choose from {list(STOCKS.keys())}")
    info = STOCKS[ticker]
    random.seed(seed or hash(ticker) % 10000)
    price = info["base"]
    data  = []
    start = datetime.now() - timedelta(days=days)
    for i in range(days):
        date = start + timedelta(days=i)
        if date.weekday() >= 5: continue
        ret   = info["trend"] + info["volatility"] * random.gauss(0, 1)
        open_p  = price
        close_p = round(open_p * (1 + ret), 2)
        high_p  = round(max(open_p, close_p) * (1 + abs(random.gauss(0, info["volatility"]/2))), 2)
        low_p   = round(min(open_p, close_p) * (1 - abs(random.gauss(0, info["volatility"]/2))), 2)
        volume  = max(int(random.gauss(50_000_000, 10_000_000)), 1_000_000)
        data.append({"date": date.strftime("%Y-%m-%d"), "open": open_p,
                     "high": high_p, "low": low_p, "close": close_p, "volume": volume})
        price = close_p
    return data, info["name"]

# ─── Technical Indicators ─────────────────────────────────────────────────────

def sma(closes, window):
    result = []
    for i in range(len(closes)):
        if i < window - 1: result.append(None)
        else: result.append(round(sum(closes[i-window+1:i+1]) / window, 2))
    return result

def ema(closes, window):
    k = 2 / (window + 1)
    result = [None] * (window - 1)
    e = sum(closes[:window]) / window
    result.append(round(e, 2))
    for price in closes[window:]:
        e = price * k + e * (1 - k)
        result.append(round(e, 2))
    return result

def bollinger_bands(closes, window=20, std_dev=2):
    upper, middle, lower = [], [], []
    for i in range(len(closes)):
        if i < window - 1:
            upper.append(None); middle.append(None); lower.append(None)
        else:
            w = closes[i-window+1:i+1]
            m = sum(w) / window
            s = math.sqrt(sum((x-m)**2 for x in w) / window)
            upper.append(round(m + std_dev*s, 2))
            middle.append(round(m, 2))
            lower.append(round(m - std_dev*s, 2))
    return upper, middle, lower

def rsi(closes, window=14):
    result = [None] * window
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0)); losses.append(abs(min(d, 0)))
    ag = sum(gains[:window]) / window
    al = sum(losses[:window]) / window
    for i in range(window, len(closes)):
        rs = ag / al if al > 0 else 100
        result.append(round(100 - 100/(1+rs), 2))
        ag = (ag*(window-1) + gains[i-1]) / window
        al = (al*(window-1) + losses[i-1]) / window
    return result

def macd(closes, fast=12, slow=26, signal=9):
    ef = ema(closes, fast); es = ema(closes, slow)
    macd_line = [round(f-s, 4) if f and s else None for f, s in zip(ef, es)]
    valid = [(i, v) for i, v in enumerate(macd_line) if v]
    sig_line = [None]*len(macd_line)
    if len(valid) >= signal:
        vals = [v for _, v in valid]
        es2  = ema(vals, signal)
        for j, (i, _) in enumerate(valid):
            sig_line[i] = es2[j]
    hist = [round(m-s, 4) if m and s else None for m, s in zip(macd_line, sig_line)]
    return macd_line, sig_line, hist

# ─── Stats ────────────────────────────────────────────────────────────────────

def compute_stats(data):
    closes = [d["close"] for d in data]
    vols   = [d["volume"] for d in data]
    rets   = [(closes[i]-closes[i-1])/closes[i-1]*100 for i in range(1, len(closes))]
    peak   = closes[0]; max_dd = 0
    for c in closes:
        peak   = max(peak, c)
        max_dd = max(max_dd, (peak-c)/peak*100)
    avg_r = sum(rets)/len(rets)
    std_r = math.sqrt(sum((r-avg_r)**2 for r in rets)/len(rets))
    return {
        "start_price":  closes[0],
        "end_price":    closes[-1],
        "total_return": round((closes[-1]-closes[0])/closes[0]*100, 2),
        "annual_vol":   round(math.sqrt(252)*std_r, 2),
        "sharpe":       round(avg_r/(std_r+1e-8)*math.sqrt(252), 2),
        "max_drawdown": round(max_dd, 2),
        "avg_volume":   int(sum(vols)/len(vols)),
        "all_time_high": max(d["high"] for d in data),
        "all_time_low":  min(d["low"]  for d in data),
        "win_rate":     round(sum(1 for r in rets if r>0)/len(rets)*100, 1),
        "best_day":     round(max(rets), 2),
        "worst_day":    round(min(rets), 2),
        "total_days":   len(rets),
        "positive_days":sum(1 for r in rets if r>0),
    }

def generate_signals(data, sma20, sma50, rsi_v):
    signals = []
    closes  = [d["close"] for d in data]
    for i in range(1, len(closes)):
        if sma20[i] and sma50[i] and sma20[i-1] and sma50[i-1]:
            if sma20[i-1] < sma50[i-1] and sma20[i] >= sma50[i]:
                signals.append({"date": data[i]["date"], "type": "BUY",  "reason": "Golden Cross (SMA20>SMA50)", "price": closes[i]})
            elif sma20[i-1] > sma50[i-1] and sma20[i] <= sma50[i]:
                signals.append({"date": data[i]["date"], "type": "SELL", "reason": "Death Cross (SMA20<SMA50)",  "price": closes[i]})
        if rsi_v[i] and rsi_v[i] < 30:
            signals.append({"date": data[i]["date"], "type": "BUY",  "reason": f"RSI Oversold ({rsi_v[i]})",   "price": closes[i]})
        elif rsi_v[i] and rsi_v[i] > 70:
            signals.append({"date": data[i]["date"], "type": "SELL", "reason": f"RSI Overbought ({rsi_v[i]})", "price": closes[i]})
    return signals[-8:]

# ─── Analyze ──────────────────────────────────────────────────────────────────

def analyze(ticker, days=365):
    data, name = generate_stock_data(ticker.upper(), days)
    closes = [d["close"] for d in data]
    s20, s50 = sma(closes, 20), sma(closes, 50)
    e20 = ema(closes, 20)
    bbu, bbm, bbl = bollinger_bands(closes)
    rsi_v = rsi(closes)
    ml, ms, mh = macd(closes)
    stats   = compute_stats(data)
    signals = generate_signals(data, s20, s50, rsi_v)
    return {"ticker": ticker.upper(), "name": name, "data": data,
            "closes": closes, "sma20": s20, "sma50": s50, "ema20": e20,
            "bb_upper": bbu, "bb_mid": bbm, "bb_lower": bbl,
            "rsi": rsi_v, "macd": ml, "macd_signal": ms, "macd_hist": mh,
            "stats": stats, "signals": signals}

def print_analysis(r):
    s = r["stats"]
    icon = "🟢" if s["total_return"] > 0 else "🔴"
    print(f"\n  {'='*55}")
    print(f"  📈 {r['name']} ({r['ticker']})")
    print(f"  Period       : {r['data'][0]['date']} → {r['data'][-1]['date']}")
    print(f"  Start Price  : ${s['start_price']:.2f}")
    print(f"  End Price    : ${s['end_price']:.2f}")
    print(f"  Total Return : {icon} {s['total_return']:+.2f}%")
    print(f"  Sharpe Ratio : {s['sharpe']:.2f}")
    print(f"  Max Drawdown : -{s['max_drawdown']:.2f}%")
    print(f"  Win Rate     : {s['win_rate']}%")
    print(f"  Best Day     : +{s['best_day']}%")
    print(f"  Worst Day    : {s['worst_day']}%")
    last_rsi = next((v for v in reversed(r["rsi"]) if v), None)
    if last_rsi:
        status = "🔴 Overbought" if last_rsi > 70 else "🟢 Oversold" if last_rsi < 30 else "⚪ Neutral"
        print(f"  RSI (14)     : {last_rsi} {status}")
    if r["signals"]:
        print(f"\n  🚦 Recent Signals:")
        for sig in r["signals"][-5:]:
            icon2 = "🟢 BUY" if sig["type"] == "BUY" else "🔴 SELL"
            print(f"     {icon2} on {sig['date']} — {sig['reason']} @ ${sig['price']:.2f}")
    print(f"  {'='*55}\n")

def run():
    print("="*55)
    print("   📈 Stock Price Analyzer")
    print(f"   Tickers: {', '.join(STOCKS.keys())}")
    print("="*55)
    while True:
        print("\n  1 → Analyze stock  2 → Compare two  3 → Quit")
        c = input("  Choice: ").strip()
        if c == "1":
            t = input("  Ticker: ").strip().upper()
            try: print_analysis(analyze(t))
            except ValueError as e: print(f"  ⚠️  {e}")
        elif c == "2":
            t1, t2 = input("  Ticker 1: ").strip().upper(), input("  Ticker 2: ").strip().upper()
            try:
                r1, r2 = analyze(t1), analyze(t2)
                print(f"\n  {'Metric':<18} {t1:<12} {t2}")
                print(f"  {'-'*42}")
                for label, v1, v2 in [
                    ("Total Return", f"{r1['stats']['total_return']:+.2f}%", f"{r2['stats']['total_return']:+.2f}%"),
                    ("Annual Vol",   f"{r1['stats']['annual_vol']:.2f}%",    f"{r2['stats']['annual_vol']:.2f}%"),
                    ("Sharpe",       f"{r1['stats']['sharpe']:.2f}",         f"{r2['stats']['sharpe']:.2f}"),
                    ("Max Drawdown", f"-{r1['stats']['max_drawdown']:.2f}%", f"-{r2['stats']['max_drawdown']:.2f}%"),
                    ("Win Rate",     f"{r1['stats']['win_rate']}%",          f"{r2['stats']['win_rate']}%"),
                ]: print(f"  {label:<18} {v1:<12} {v2}")
            except ValueError as e: print(f"  ⚠️  {e}")
        elif c == "3":
            print("\n  👋 Happy trading!\n"); break

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: print_analysis(analyze(sys.argv[1]))
    else: run()
