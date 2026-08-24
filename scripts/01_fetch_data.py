"""
01_fetch_data.py — Fetch 5 Years of NIFTY 500 Stock Data
Pulls real historical data from Yahoo Finance using yfinance.
Covers 40 stocks across 8 sectors — 52,000+ rows total.
"""

import yfinance as yf
import pandas as pd
import os, time, warnings
warnings.filterwarnings("ignore")

RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)

START = "2019-01-01"
END   = "2023-12-31"

# ── 40 NIFTY stocks across 8 sectors ──────────────────────────────────────
STOCKS = {
    "IT": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"
    ],
    "Banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"
    ],
    "Pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"
    ],
    "Auto": [
        "MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"
    ],
    "Energy": [
        "RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "BPCL.NS"
    ],
    "Metals": [
        "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS", "VEDL.NS"
    ],
    "Infra": [
        "ADANIPORTS.NS", "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "SIEMENS.NS"
    ],
}

# ── Fetch each stock one by one ────────────────────────────────────────────
all_data = []
failed   = []

total = sum(len(v) for v in STOCKS.values())
count = 0

print("=" * 60)
print("Fetching 5-year NIFTY stock data from Yahoo Finance")
print(f"Stocks: {total}  |  Period: {START} → {END}")
print("=" * 60)

for sector, tickers in STOCKS.items():
    for ticker in tickers:
        count += 1
        try:
            print(f"  [{count:02d}/{total}] {ticker:<20s} ({sector})", end=" ... ")

            df = yf.download(
                ticker,
                start=START,
                end=END,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                print("NO DATA — skipped")
                failed.append(ticker)
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.strip() for c in df.columns]

            # Rename columns to standard names
            df = df.rename(columns={
                "Date":      "date",
                "Open":      "open",
                "High":      "high",
                "Low":       "low",
                "Close":     "close",
                "Volume":    "volume",
                "Adj Close": "adj_close",
            })

            # If adj_close missing (auto_adjust=True gives Close as adjusted)
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]

            df["ticker"] = ticker.replace(".NS", "")
            df["sector"] = sector

            all_data.append(df)
            print(f"{len(df)} rows ✓")

            # Small pause to avoid rate limiting
            time.sleep(0.3)

        except Exception as e:
            print(f"ERROR — {e}")
            failed.append(ticker)
            continue

# ── Combine and save ───────────────────────────────────────────────────────
print("\n" + "=" * 60)

if not all_data:
    print("❌ No data fetched. Check internet connection and try again.")
else:
    master = pd.concat(all_data, ignore_index=True)

    # Keep only needed columns
    cols = ["date", "ticker", "sector", "open", "high", "low",
            "close", "adj_close", "volume"]
    master = master[[c for c in cols if c in master.columns]]

    # Sort
    master = master.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Save
    master.to_csv(f"{RAW}/nifty_stocks_5yr.csv", index=False)

    print(f"✅ Data saved → data/raw/nifty_stocks_5yr.csv")
    print(f"   Stocks fetched:  {master['ticker'].nunique()}")
    print(f"   Sectors covered: {master['sector'].nunique()}")
    print(f"   Total rows:      {len(master):,}")
    print(f"   Date range:      {master['date'].min()} → {master['date'].max()}")

    if failed:
        print(f"\n⚠  Failed tickers ({len(failed)}): {', '.join(failed)}")
        print("   These can be skipped — 35+ stocks is enough for the project.")

# ── Fetch NIFTY 500 Index (benchmark for Beta calculation) ─────────────────
print("\n" + "=" * 60)
print("Fetching NIFTY 500 Index (benchmark)...")

try:
    nifty = yf.download("^CRSLDX", start=START, end=END,
                        progress=False, auto_adjust=True)

    if nifty.empty:
        # Fallback to NIFTY 50 if 500 not available
        nifty = yf.download("^NSEI", start=START, end=END,
                            progress=False, auto_adjust=True)

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    nifty = nifty.reset_index()
    nifty.columns = [c.strip().lower().replace(" ", "_") for c in nifty.columns]

    close_col = "close"
    nifty = nifty[["date", close_col]].rename(columns={close_col: "index_close"})
    nifty["index_return"] = nifty["index_close"].pct_change()
    nifty.to_csv(f"{RAW}/nifty_index.csv", index=False)

    print(f"✅ Index saved → data/raw/nifty_index.csv  ({len(nifty)} rows)")

except Exception as e:
    print(f"⚠  Index fetch failed: {e}")
    print("   Beta calculation will be skipped — everything else still works.")

print("\n" + "=" * 60)
print("✅ Data fetch complete. Run 02_cleaning.py next.")
print("=" * 60)
