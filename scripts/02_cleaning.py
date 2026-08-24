"""
02_cleaning.py — Stock Market Data Cleaning & Feature Engineering
Loads raw NIFTY stock data, computes financial metrics,
engineers ML features, and saves processed files.
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

RAW  = "data/raw"
PROC = "data/processed"
os.makedirs(PROC, exist_ok=True)

print("=" * 60)
print("STEP 1 — Loading raw data")
print("=" * 60)

df  = pd.read_csv(f"{RAW}/nifty_stocks_5yr.csv", parse_dates=["date"])
mkt = pd.read_csv(f"{RAW}/nifty_index.csv",      parse_dates=["date"])

print(f"  Stock rows:   {len(df):,}")
print(f"  Stocks:       {df['ticker'].nunique()}")
print(f"  Sectors:      {df['sector'].nunique()}")
print(f"  Date range:   {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Index rows:   {len(mkt):,}")

print("\n" + "=" * 60)
print("STEP 2 — Data Cleaning")
print("=" * 60)

# Sort
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# Drop rows with null close prices
before = len(df)
df = df.dropna(subset=["close", "adj_close"])
print(f"  Null price rows removed: {before - len(df)}")

# Remove stocks with less than 200 trading days (insufficient data)
stock_counts = df.groupby("ticker")["date"].count()
valid_tickers = stock_counts[stock_counts >= 200].index
df = df[df["ticker"].isin(valid_tickers)]
print(f"  Stocks with 200+ trading days: {df['ticker'].nunique()}")

# Clip extreme single-day price moves > 50% (data errors)
df["daily_chg"] = df.groupby("ticker")["adj_close"].pct_change()
extreme = (df["daily_chg"].abs() > 0.50) & (df["daily_chg"].notna())
print(f"  Extreme price moves clipped:   {extreme.sum()}")
df = df[~extreme].copy()
df = df.drop(columns=["daily_chg"])

print("\n" + "=" * 60)
print("STEP 3 — Feature Engineering")
print("=" * 60)

# ── Daily Returns ──────────────────────────────────────────────────────────
df["return_1d"]  = df.groupby("ticker")["adj_close"].pct_change()
df["return_5d"]  = df.groupby("ticker")["adj_close"].pct_change(5)
df["return_20d"] = df.groupby("ticker")["adj_close"].pct_change(20)

# ── Moving Averages ────────────────────────────────────────────────────────
df["ma_10"]  = df.groupby("ticker")["adj_close"].transform(
    lambda x: x.rolling(10, min_periods=1).mean())
df["ma_50"]  = df.groupby("ticker")["adj_close"].transform(
    lambda x: x.rolling(50, min_periods=1).mean())
df["ma_200"] = df.groupby("ticker")["adj_close"].transform(
    lambda x: x.rolling(200, min_periods=1).mean())
df["ma_signal"] = (df["ma_10"] > df["ma_50"]).astype(int)

print("  Moving averages computed (10, 50, 200 day)")

# ── RSI (14-day) ───────────────────────────────────────────────────────────
def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["rsi_14"] = df.groupby("ticker")["adj_close"].transform(compute_rsi)
print("  RSI (14-day) computed")

# ── Rolling Volatility ─────────────────────────────────────────────────────
df["vol_20d"] = df.groupby("ticker")["return_1d"].transform(
    lambda x: x.rolling(20, min_periods=5).std() * np.sqrt(252))
print("  Rolling 20-day volatility computed")

# ── Volume Spike ───────────────────────────────────────────────────────────
df["vol_spike"] = df.groupby("ticker")["volume"].transform(
    lambda x: x / x.rolling(20, min_periods=5).mean())
print("  Volume spike feature computed")

# ── Merge Market Returns for Beta ──────────────────────────────────────────
mkt_ret = mkt[["date", "index_return"]].rename(
    columns={"index_return": "market_return"})
df = df.merge(mkt_ret, on="date", how="left")
print("  Market returns merged for Beta calculation")

# ── ML Target: Next Day Direction (1=UP, 0=DOWN) ──────────────────────────
df["target"] = (df.groupby("ticker")["return_1d"].shift(-1) > 0).astype(int)

print("\n" + "=" * 60)
print("STEP 4 — Computing Financial Metrics per Stock")
print("=" * 60)

risk_free_daily = 0.065 / 252   # India 10Y bond ~6.5%

results = []
for ticker, grp in df.groupby("ticker"):
    grp    = grp.sort_values("date")
    r      = grp["return_1d"].dropna()
    mr     = grp["market_return"].dropna()
    prices = grp["adj_close"].values
    sector = grp["sector"].iloc[0]

    if len(r) < 100:
        continue

    # Annualised return & volatility
    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (r.mean() - risk_free_daily) / r.std() * np.sqrt(252) \
             if r.std() > 0 else 0

    # Beta vs market
    common = pd.concat([r, mr], axis=1).dropna()
    if len(common) > 50:
        cov = np.cov(common.iloc[:,0], common.iloc[:,1])
        beta = cov[0,1] / cov[1,1] if cov[1,1] > 0 else 1.0
    else:
        beta = 1.0

    # Maximum Drawdown
    roll_max = np.maximum.accumulate(prices)
    drawdown = (prices - roll_max) / roll_max
    max_dd   = drawdown.min()

    # Calmar Ratio (return per unit of drawdown)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Win Rate
    win_rate = (r > 0).sum() / len(r)

    # 5-year total return
    total_ret = (prices[-1] - prices[0]) / prices[0] * 100

    results.append({
        "ticker":          ticker,
        "sector":          sector,
        "ann_return":      round(ann_ret,   4),
        "ann_volatility":  round(ann_vol,   4),
        "sharpe_ratio":    round(sharpe,    3),
        "beta":            round(beta,      3),
        "max_drawdown":    round(max_dd,    4),
        "calmar_ratio":    round(calmar,    3),
        "win_rate":        round(win_rate,  3),
        "total_return_pct":round(total_ret, 2),
    })

metrics = pd.DataFrame(results)

# ── Stock Composite Score (0–100) ──────────────────────────────────────────
def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

metrics["score_sharpe"]   = minmax(metrics["sharpe_ratio"])    * 35
metrics["score_drawdown"] = minmax(-metrics["max_drawdown"])   * 30
metrics["score_calmar"]   = minmax(metrics["calmar_ratio"])    * 20
metrics["score_winrate"]  = minmax(metrics["win_rate"])        * 15
metrics["stock_score"]    = (
    metrics["score_sharpe"] +
    metrics["score_drawdown"] +
    metrics["score_calmar"] +
    metrics["score_winrate"]
).clip(0, 100).round(1)

def get_tier(s):
    if s >= 60:   return "Elite"
    elif s >= 45: return "Strong"
    elif s >= 30: return "Average"
    else:         return "Weak"

metrics["tier"] = metrics["stock_score"].apply(get_tier)

print(f"  Metrics computed for {len(metrics)} stocks")
print(f"\n  Top 5 by Sharpe Ratio:")
top5 = metrics.nlargest(5, "sharpe_ratio")[
    ["ticker","sector","ann_return","sharpe_ratio","beta","stock_score"]]
print(top5.to_string(index=False))

print("\n" + "=" * 60)
print("STEP 5 — Saving Processed Files")
print("=" * 60)

df.to_csv(f"{PROC}/stocks_features.csv", index=False)
metrics.to_csv(f"{PROC}/stock_metrics.csv", index=False)

print(f"  stocks_features.csv → {len(df):,} rows × {df.shape[1]} columns")
print(f"  stock_metrics.csv   → {len(metrics)} stocks × {metrics.shape[1]} columns")

print("\n✅ Cleaning complete. Run 03_charts.py next.")
