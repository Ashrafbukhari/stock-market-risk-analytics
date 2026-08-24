"""
03_charts.py — Stock Market Risk Analytics
Generates 10 professional charts from real NIFTY 500 data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

PROC   = "data/processed"
RAW    = "data/raw"
CHARTS = "outputs/charts"
os.makedirs(CHARTS, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "figure.dpi":        130,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

NAVY   = "#0D3B6E"
BLUE   = "#1A56A0"
ORANGE = "#F97316"
GREEN  = "#16A34A"
RED    = "#DC2626"
PURPLE = "#7C3AED"
TEAL   = "#0891B2"
AMBER  = "#D97706"

SECTOR_COLORS = {
    "IT":      BLUE,
    "Banking": NAVY,
    "Pharma":  GREEN,
    "FMCG":    TEAL,
    "Auto":    ORANGE,
    "Energy":  AMBER,
    "Metals":  RED,
    "Infra":   PURPLE,
}

print("Loading processed data...")
df      = pd.read_csv(f"{PROC}/stocks_features.csv", parse_dates=["date"])
metrics = pd.read_csv(f"{PROC}/stock_metrics.csv")
mkt     = pd.read_csv(f"{RAW}/nifty_index.csv", parse_dates=["date"])

print(f"  Stocks: {metrics['ticker'].nunique()}  |  Rows: {len(df):,}")
print("\nGenerating charts...")

# ══════════════════════════════════════════════════════════════
# CHART 1 — NIFTY Index 5-Year Performance
# ══════════════════════════════════════════════════════════════
print("  Chart 1: NIFTY Index trend...")

mkt = mkt.dropna(subset=["index_close"])
fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(mkt["date"], mkt["index_close"], alpha=0.12, color=BLUE)
ax.plot(mkt["date"], mkt["index_close"], color=BLUE, lw=2)

# COVID crash annotation
covid_s = pd.Timestamp("2020-02-20")
covid_e = pd.Timestamp("2020-03-23")
recov   = pd.Timestamp("2020-09-30")
ax.axvspan(covid_s, covid_e, alpha=0.15, color=RED,   label="COVID Crash")
ax.axvspan(covid_e, recov,   alpha=0.08, color=GREEN, label="Recovery")

# Drop %
pre_val   = mkt[mkt["date"] <= covid_s]["index_close"].iloc[-1]
crash_val = mkt[mkt["date"] <= covid_e]["index_close"].iloc[-1]
drop_pct  = (crash_val - pre_val) / pre_val * 100

ax.annotate(f"Peak-to-trough\n{drop_pct:.1f}% drop",
    xy=(covid_e, crash_val),
    xytext=(60, -60), textcoords="offset points",
    fontsize=9, color=RED, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

start_val = mkt["index_close"].iloc[0]
end_val   = mkt["index_close"].iloc[-1]
total_ret = (end_val - start_val) / start_val * 100
ax.text(0.02, 0.92, f"5Y Return: +{total_ret:.1f}%",
    transform=ax.transAxes, fontsize=11, color=GREEN, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#D1FAE5", alpha=0.9))

ax.set_title("NIFTY Index — 5 Year Performance with COVID Crash (2019–2023)",
    fontsize=14, fontweight="bold")
ax.set_ylabel("Index Value")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS}/01_nifty_index.png", bbox_inches="tight")
plt.close()
print("    ✓ Chart 1 saved")

# ══════════════════════════════════════════════════════════════
# CHART 2 — Risk vs Return Scatter
# ══════════════════════════════════════════════════════════════
print("  Chart 2: Risk vs Return scatter...")

fig, ax = plt.subplots(figsize=(13, 7))
for sector in metrics["sector"].unique():
    s = metrics[metrics["sector"] == sector]
    ax.scatter(s["ann_volatility"] * 100, s["ann_return"] * 100,
               c=SECTOR_COLORS.get(sector, BLUE),
               s=120, alpha=0.85, label=sector,
               edgecolors="white", linewidths=0.8, zorder=3)
    for _, row in s.iterrows():
        ax.annotate(row["ticker"],
            (row["ann_volatility"] * 100, row["ann_return"] * 100),
            textcoords="offset points", xytext=(5, 4),
            fontsize=7.5, color="#374151")

ax.axhline(metrics["ann_return"].mean() * 100, color=NAVY,
           ls="--", alpha=0.4, lw=1.2, label="Avg Return")
ax.axvline(metrics["ann_volatility"].mean() * 100, color=RED,
           ls="--", alpha=0.4, lw=1.2, label="Avg Risk")

ax.set_xlabel("Annualised Volatility (Risk %)")
ax.set_ylabel("Annualised Return (%)")
ax.set_title("Risk vs Return — All NIFTY Stocks  (Top-Right = Ideal)",
    fontsize=14, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="lower right")
plt.tight_layout()
plt.savefig(f"{CHARTS}/02_risk_return_scatter.png", bbox_inches="tight")
plt.close()
print("    ✓ Chart 2 saved")

# ══════════════════════════════════════════════════════════════
# CHART 3 — Sharpe Ratio Ranking
# ══════════════════════════════════════════════════════════════
print("  Chart 3: Sharpe ratio ranking...")

m_sorted = metrics.sort_values("sharpe_ratio", ascending=True)
fig, ax  = plt.subplots(figsize=(12, 9))
bar_colors = [SECTOR_COLORS.get(s, BLUE) for s in m_sorted["sector"]]
ax.barh(m_sorted["ticker"], m_sorted["sharpe_ratio"],
        color=bar_colors, edgecolor="white", height=0.7)
ax.axvline(1.0, color=GREEN,  ls="--", lw=1.5, label="Sharpe = 1.0 (Good)")
ax.axvline(0.5, color=ORANGE, ls="--", lw=1.2, label="Sharpe = 0.5 (Ok)")
for bar, (_, row) in zip(ax.patches, m_sorted.iterrows()):
    ax.text(max(row["sharpe_ratio"] + 0.01, 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{row['sharpe_ratio']:.2f}", va="center", fontsize=8)

ax.set_xlabel("Sharpe Ratio (Risk-Adjusted Return)")
ax.set_title("Sharpe Ratio Ranking — All NIFTY Stocks",
    fontsize=14, fontweight="bold")
legend_p = [mpatches.Patch(color=c, label=s)
            for s, c in SECTOR_COLORS.items()]
ax.legend(handles=legend_p, fontsize=8, loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig(f"{CHARTS}/03_sharpe_ranking.png", bbox_inches="tight")
plt.close()
print("    ✓ Chart 3 saved")

# ══════════════════════════════════════════════════════════════
# CHART 4 — Sector Comparison (2x2 Grid)
# ══════════════════════════════════════════════════════════════
print("  Chart 4: Sector comparison grid...")

sec_met = metrics.groupby("sector").agg(
    Avg_Return    = ("ann_return",    "mean"),
    Avg_Sharpe    = ("sharpe_ratio",  "mean"),
    Avg_Beta      = ("beta",          "mean"),
    Avg_MaxDD     = ("max_drawdown",  "mean"),
).reset_index().sort_values("Avg_Return", ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Sector-Level Risk & Return Analysis",
    fontsize=14, fontweight="bold")
sec_colors = [SECTOR_COLORS.get(s, BLUE) for s in sec_met["sector"]]

axes[0,0].bar(sec_met["sector"], sec_met["Avg_Return"] * 100,
              color=sec_colors, edgecolor="white")
axes[0,0].set_title("Avg Annual Return (%)")
axes[0,0].tick_params(axis="x", rotation=35)
for i, v in enumerate(sec_met["Avg_Return"] * 100):
    axes[0,0].text(i, v + 0.3, f"{v:.1f}%",
                   ha="center", fontsize=9, fontweight="bold")

axes[0,1].bar(sec_met["sector"], sec_met["Avg_Sharpe"],
              color=sec_colors, edgecolor="white")
axes[0,1].axhline(1.0, color=GREEN, ls="--", lw=1.5)
axes[0,1].set_title("Avg Sharpe Ratio")
axes[0,1].tick_params(axis="x", rotation=35)
for i, v in enumerate(sec_met["Avg_Sharpe"]):
    axes[0,1].text(i, v + 0.01, f"{v:.2f}",
                   ha="center", fontsize=9)

axes[1,0].bar(sec_met["sector"], sec_met["Avg_Beta"],
              color=sec_colors, edgecolor="white")
axes[1,0].axhline(1.0, color=RED, ls="--", lw=1.5, label="Beta=1 (Market)")
axes[1,0].set_title("Avg Beta (Market Sensitivity)")
axes[1,0].tick_params(axis="x", rotation=35)
axes[1,0].legend(fontsize=8)

axes[1,1].bar(sec_met["sector"], abs(sec_met["Avg_MaxDD"]) * 100,
              color=sec_colors, edgecolor="white")
axes[1,1].set_title("Avg Max Drawdown % — Lower is Better")
axes[1,1].tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.savefig(f"{CHARTS}/04_sector_comparison.png", bbox_inches="tight")
plt.close()
print("    ✓ Chart 4 saved")

# ══════════════════════════════════════════════════════════════
# CHART 5 — COVID Crash & Recovery by Sector
# ══════════════════════════════════════════════════════════════
print("  Chart 5: COVID crash & recovery...")

covid_pre   = "2020-01-31"
covid_crash = "2020-03-23"
covid_recov = "2020-09-30"

rec_rows = []
for sector, grp in df.groupby("sector"):
    grp = grp.sort_values("date")
    pre   = grp[grp["date"] <= covid_pre]["adj_close"].mean()
    crash = grp[(grp["date"] >= covid_pre) &
                (grp["date"] <= covid_crash)]["adj_close"].mean()
    recov = grp[(grp["date"] >= covid_crash) &
                (grp["date"] <= covid_recov)]["adj_close"].mean()
    if pre > 0 and crash > 0:
        rec_rows.append({
            "Sector":       sector,
            "Crash_Pct":    (crash - pre) / pre * 100,
            "Recovery_Pct": (recov - crash) / crash * 100 if crash > 0 else 0,
        })

rec_df = pd.DataFrame(rec_rows).sort_values("Crash_Pct")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("COVID Crash & Recovery Analysis by Sector",
    fontsize=14, fontweight="bold")

cols_c = [RED if v < -15 else ORANGE for v in rec_df["Crash_Pct"]]
ax1.barh(rec_df["Sector"], rec_df["Crash_Pct"],
         color=cols_c, edgecolor="white")
ax1.set_title("Crash Severity (Jan–Mar 2020)")
ax1.set_xlabel("% Change from Pre-COVID Levels")
ax1.axvline(0, color="gray", lw=0.8)
for i, v in enumerate(rec_df["Crash_Pct"]):
    ax1.text(v - 0.5, i, f"{v:.1f}%", va="center",
             ha="right", fontsize=9, color="white", fontweight="bold")

cols_r = [GREEN if v > 15 else TEAL for v in rec_df["Recovery_Pct"]]
ax2.barh(rec_df["Sector"], rec_df["Recovery_Pct"],
         color=cols_r, edgecolor="white")
ax2.set_title("Recovery Speed (Mar–Sep 2020)")
ax2.set_xlabel("% Bounce from Crash Lows")
for i, v in enumerate(rec_df["Recovery_Pct"]):
    ax2.text(v + 0.5, i, f"{v:.1f}%", va="center",
             fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{CHARTS}/05_covid_recovery.png", bbox_inches="tight")
plt.close()
print("    ✓ Chart 5 saved")

# ══════════════════════════════════════════════════════════════
# CHART 6 — Sector Correlation Heatmap
# ══════════════════════════════════════════════════════════════
print("  Chart 6: Sector correlation heatmap...")

pivot = df.pivot_table(index="date", columns="sector",
                       values="return_1d", aggfunc="mean")
corr  = pivot.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, square=True,
            cbar_kws={"label": "Correlation"}, vmin=-0.2, vmax=1.0)
ax.set_title("Sector Correlation Matrix — Diversification Opportunities",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS}/06_sector_correlation.png", bbox_inches="tight")
plt.close()
print("    ✓ Chart 6 saved")

# ══════════════════════════════════════════════════════════════
# CHART 7 — Stock Composite Score (UNIQUE FEATURE)
# ══════════════════════════════════════════════════════════════
print("  Chart 7: Stock composite score...")

tier_colors = {
    "Elite":   GREEN,
    "Strong":  BLUE,
    "Average": ORANGE,
    "Weak":    RED,
}
m_score    = metrics.sort_values("stock_score", ascending=False)
bar_colors = [tier_colors.get(t, BLUE) for t in m_score["tier"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Stock Composite Score — Custom 0–100 KPI",
    fontsize=14, fontweight="bold")

ax1.barh(m_score["ticker"][::-1], m_score["stock_score"][::-1],
         color=bar_colors[::-1], edgecolor="white", height=0.7)
ax1.axvline(75, color=GREEN, ls="--", lw=1.5, label="Elite threshold (75)")
ax1.axvline(35, color=RED,   ls="--", lw=1.2, label="Weak threshold (35)")
ax1.set_xlabel("Composite Score (0–100)")
ax1.set_title("All Stocks Ranked by Composite Score")
ax1.legend(fontsize=8)

tier_counts = metrics["tier"].value_counts()
tier_order  = ["Elite", "Strong", "Average", "Weak"]
counts      = [tier_counts.get(t, 0) for t in tier_order]
colors_t    = [GREEN, BLUE, ORANGE, RED]
bars_t = ax2.bar(
    [t for t, c in zip(tier_order, counts) if c > 0],
    [c for c in counts if c > 0],
    color=[c for t, c, cnt in zip(tier_order, colors_t, counts) if cnt > 0],
    edgecolor="white"
)
for bar in bars_t:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
             str(int(h)), ha="center", fontsize=11, fontweight="bold")
ax2.set_ylabel("Number of Stocks")
ax2.set_title("Stocks by Performance Tier")

plt.tight_layout()
plt.savefig(f"{CHARTS}/07_stock_score.png", bbox_inches="tight")
plt.close()
print("    ✓ Chart 7 saved")

# ══════════════════════════════════════════════════════════════
# CHART 8 — Monte Carlo Portfolio Simulation (UNIQUE FEATURE)
# ══════════════════════════════════════════════════════════════
print("  Chart 8: Monte Carlo simulation (1,000 portfolios)...")

pivot_p  = df.pivot_table(index="date", columns="ticker",
                          values="return_1d")
pivot_p  = pivot_p.dropna(axis=1, thresh=int(len(pivot_p) * 0.8)).fillna(0)
tickers_p = list(pivot_p.columns)

N_sim = 1000
np.random.seed(42)
sim_ret = []; sim_vol = []; sim_sharpe = []; sim_weights = []

for _ in range(N_sim):
    w      = np.random.dirichlet(np.ones(len(tickers_p)))
    port_r = pivot_p[tickers_p].values @ w
    ann_r  = port_r.mean() * 252
    ann_v  = port_r.std()  * np.sqrt(252)
    sr     = (ann_r - 0.065) / ann_v if ann_v > 0 else 0
    sim_ret.append(ann_r); sim_vol.append(ann_v)
    sim_sharpe.append(sr); sim_weights.append(w)

sim_ret    = np.array(sim_ret)
sim_vol    = np.array(sim_vol)
sim_sharpe = np.array(sim_sharpe)
best_idx   = np.argmax(sim_sharpe)
best_w     = sim_weights[best_idx]
best_top5  = sorted(zip(tickers_p, best_w), key=lambda x: -x[1])[:5]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Monte Carlo Portfolio Simulation — {N_sim:,} Random Portfolios",
    fontsize=14, fontweight="bold")

sc = ax1.scatter(sim_vol * 100, sim_ret * 100,
                 c=sim_sharpe, cmap="RdYlGn",
                 alpha=0.5, s=12, zorder=2)
plt.colorbar(sc, ax=ax1, label="Sharpe Ratio")
ax1.scatter(sim_vol[best_idx] * 100, sim_ret[best_idx] * 100,
            c="gold", s=220, marker="*", zorder=5,
            label=f"Optimal (Sharpe={sim_sharpe[best_idx]:.2f})")
ax1.set_xlabel("Portfolio Volatility (%)")
ax1.set_ylabel("Portfolio Return (%)")
ax1.set_title("Efficient Frontier — Risk vs Return")
ax1.legend(fontsize=9)

labels  = [t for t, w in best_top5]
weights = [w * 100 for t, w in best_top5]
others  = 100 - sum(weights)
if others > 1:
    labels.append("Others")
    weights.append(others)

ax2.pie(weights, labels=labels, autopct="%1.1f%%",
        colors=[BLUE, GREEN, ORANGE, RED, PURPLE, TEAL, "#9CA3AF"][:len(labels)],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
ax2.set_title(
    f"Optimal Portfolio Allocation\n"
    f"Sharpe={sim_sharpe[best_idx]:.2f}  "
    f"Return={sim_ret[best_idx]*100:.1f}%  "
    f"Vol={sim_vol[best_idx]*100:.1f}%"
)

plt.tight_layout()
plt.savefig(f"{CHARTS}/08_monte_carlo.png", bbox_inches="tight")
plt.close()
print(f"    ✓ Chart 8 saved  |  Best Sharpe: {sim_sharpe[best_idx]:.2f}  "
      f"Return: {sim_ret[best_idx]*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# CHART 9 — ML Price Direction Predictor
# ══════════════════════════════════════════════════════════════
print("  Chart 9: ML model (Random Forest)...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import roc_auc_score

# Use TCS as the demo stock
stock_ml = df[df["ticker"] == "TCS"].copy().sort_values("date")
feats    = ["return_1d", "return_5d", "return_20d",
            "rsi_14", "ma_signal", "vol_20d", "vol_spike"]
ml_df    = stock_ml[feats + ["target", "date"]].dropna()

split  = int(len(ml_df) * 0.8)
X_tr   = ml_df[feats].iloc[:split]
X_te   = ml_df[feats].iloc[split:]
y_tr   = ml_df["target"].iloc[:split]
y_te   = ml_df["target"].iloc[split:]

model  = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
y_prob = model.predict_proba(X_te)[:, 1]
acc    = (y_pred == y_te).mean()
auc    = roc_auc_score(y_te, y_prob)
fi     = pd.Series(model.feature_importances_, index=feats).sort_values()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"ML Price Direction Predictor — TCS  "
    f"(Accuracy={acc:.1%}  AUC-ROC={auc:.2f})",
    fontsize=14, fontweight="bold"
)

ax1.barh(fi.index, fi.values * 100,
         color=[BLUE if v > fi.mean() else "#93C5FD" for v in fi.values],
         edgecolor="white")
ax1.set_xlabel("Feature Importance (%)")
ax1.set_title("What Drives Next-Day Price Direction?")
for i, v in enumerate(fi.values):
    ax1.text(v * 100 + 0.1, i, f"{v*100:.1f}%", va="center", fontsize=9)

te_dates  = ml_df["date"].iloc[split:].values
te_ret    = stock_ml["return_1d"].reindex(ml_df.index).iloc[split:]
strategy  = te_ret * y_pred
bh_cum    = te_ret.cumsum() * 100
strat_cum = strategy.cumsum() * 100

ax2.plot(te_dates, bh_cum.values,    color="#94A3B8", lw=1.5, label="Buy & Hold")
ax2.plot(te_dates, strat_cum.values, color=GREEN,     lw=2,   label="ML Strategy")
ax2.set_ylabel("Cumulative Return (%)")
ax2.set_xlabel("Date")
ax2.set_title("ML Strategy vs Buy & Hold (Out-of-Sample 2023)")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{CHARTS}/09_ml_model.png", bbox_inches="tight")
plt.close()
print(f"    ✓ Chart 9 saved  |  Accuracy: {acc:.1%}  AUC: {auc:.2f}")

# ══════════════════════════════════════════════════════════════
# CHART 10 — Executive KPI Dashboard
# ══════════════════════════════════════════════════════════════
print("  Chart 10: Executive dashboard...")

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("#F8FAFC")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.text(0.5, 0.7,
    "Stock Market Risk Analytics & Portfolio Intelligence",
    ha="center", va="center", fontsize=18,
    fontweight="bold", color=NAVY)
ax0.text(0.5, 0.15,
    f"NIFTY 500  ·  {metrics['ticker'].nunique()} Stocks  ·  "
    f"8 Sectors  ·  5 Years (2019–2023)  ·  {len(df):,}+ Rows",
    ha="center", va="center", fontsize=10, color="#64748B")

best_stock  = metrics.loc[metrics["sharpe_ratio"].idxmax()]
def_sector  = metrics.groupby("sector")["beta"].mean().idxmin()

kpis = [
    ("Stocks Analysed",  str(metrics["ticker"].nunique()),              BLUE),
    ("Sectors Covered",  "8",                                           NAVY),
    ("Trading Days",     "1,235",                                       PURPLE),
    ("Best Sharpe",
     f"{metrics['sharpe_ratio'].max():.2f}\n({best_stock['ticker']})", GREEN),
    ("Avg Ann. Return",
     f"{metrics['ann_return'].mean()*100:.1f}%",                       ORANGE),
    ("Avg Volatility",
     f"{metrics['ann_volatility'].mean()*100:.1f}%",                   TEAL),
    ("Monte Carlo Runs", "1,000",                                       AMBER),
    ("Defensive Sector", def_sector,                                    RED),
]

positions = [
    (1,0),(1,1),(1,2),(1,3),
    (2,0),(2,1),(2,2),(2,3),
]
for (r, c), (label, val, color) in zip(positions, kpis):
    ax = fig.add_subplot(gs[r, c])
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(2)
    ax.text(0.5, 0.62, val,
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.18, label,
            ha="center", va="center", fontsize=8.5,
            color="#64748B", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)

plt.savefig(f"{CHARTS}/10_executive_dashboard.png",
            bbox_inches="tight", dpi=140)
plt.close()
print("    ✓ Chart 10 saved")

print("\n" + "=" * 60)
print(f"✅ All 10 charts saved to outputs/charts/")
print("=" * 60)
for f in sorted(os.listdir(CHARTS)):
    print(f"   {f}")
