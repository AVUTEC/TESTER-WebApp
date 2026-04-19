"""Deep dive on inflation -> ruling party loss in democratic elections, 1980-2025.

Four angles:
  1. Full inflation-level response curve (finer bins + smoothed)
  2. Inflation SHOCK (change vs. prior year) — level vs. acceleration
  3. By decade — does the 2020-2024 inflation wave fit the pattern?
  4. Head-to-head: when growth is good but inflation is high (or vice versa),
     which dominates the outcome?
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
p = pd.read_csv(PROC / "panel.csv")

# Compute inflation shock at country level: change vs 2 years prior
p = p.sort_values(["country_text_id", "year"])
p["inflation_cpi_lag2"] = p.groupby("country_text_id")["inflation_cpi"].shift(2)
p["inflation_shock"] = p["inflation_cpi_lag1"] - p["inflation_cpi_lag2"]

d = p[
    (p["election_year"] == 1)
    & (p["v2x_polyarchy"] > 0.5)
    & p["v2elturnhog"].notna()
    & p["inflation_cpi_lag1"].notna()
].copy()
d["lost"] = (d["v2elturnhog"] == 2).astype(int)
print(f"Sample: {len(d)} democratic elections, 1980-{int(d.year.max())}\n")

# ---------- 1. Inflation level, finer bins ----------
print("=" * 60)
print("1. RULING-PARTY LOSS RATE BY PRE-ELECTION INFLATION")
print("=" * 60)
bins = [-100, -2, 0, 2, 4, 6, 8, 10, 15, 25, 1e9]
labels = ["<-2% (deflation)", "-2 to 0%", "0-2%", "2-4%", "4-6%",
          "6-8%", "8-10%", "10-15%", "15-25%", ">25%"]
d["infl_bin"] = pd.cut(d["inflation_cpi_lag1"], bins=bins, labels=labels)
tab = d.groupby("infl_bin", observed=True).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean() * 100, 1)),
    avg_infl=("inflation_cpi_lag1", lambda s: round(s.mean(), 1)),
)
print(tab.to_string())

# ---------- 2. Inflation SHOCK (change) ----------
print("\n" + "=" * 60)
print("2. LOSS RATE BY INFLATION SHOCK (change from 2yr prior)")
print("=" * 60)
ds = d.dropna(subset=["inflation_shock"])
bins = [-1e9, -3, -1, 1, 3, 5, 1e9]
labels = ["falling hard (<-3pp)", "falling (-3 to -1)", "stable (-1 to 1)",
          "rising (1-3pp)", "rising fast (3-5pp)", "spiking (>5pp)"]
ds["shock_bin"] = pd.cut(ds["inflation_shock"], bins=bins, labels=labels)
tab = ds.groupby("shock_bin", observed=True).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean() * 100, 1)),
    avg_shock=("inflation_shock", lambda s: round(s.mean(), 1)),
)
print(tab.to_string())

# ---------- 3. By decade ----------
print("\n" + "=" * 60)
print("3. LOSS RATE BY DECADE, SPLIT BY INFLATION")
print("=" * 60)
d["decade"] = (d["year"] // 10 * 10).astype(int).astype(str) + "s"
d["high_infl"] = (d["inflation_cpi_lag1"] > 5).astype(int)
pivot = d.groupby(["decade", "high_infl"]).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean() * 100, 1)),
).unstack()
print(pivot.to_string())

print("\n  Inflation >5% in each decade:")
for dec in sorted(d["decade"].unique()):
    sub = d[d["decade"] == dec]
    high = sub[sub["high_infl"] == 1]
    low = sub[sub["high_infl"] == 0]
    if len(high) >= 5 and len(low) >= 5:
        print(f"  {dec}: low-infl loss {low['lost'].mean()*100:4.1f}% (n={len(low):3d})   "
              f"high-infl loss {high['lost'].mean()*100:4.1f}% (n={len(high):3d})   "
              f"gap +{(high['lost'].mean()-low['lost'].mean())*100:.1f}pp")

# ---------- 4. Head-to-head: inflation vs growth ----------
print("\n" + "=" * 60)
print("4. HEAD-TO-HEAD: when growth and inflation disagree, which wins?")
print("=" * 60)
dg = d.dropna(subset=["gdp_growth_real_lag1"]).copy()
dg["good_growth"] = (dg["gdp_growth_real_lag1"] > 2).astype(int)
dg["low_infl"] = (dg["inflation_cpi_lag1"] < 5).astype(int)

quad = dg.groupby(["good_growth", "low_infl"]).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean() * 100, 1)),
)
quad.index = quad.index.map(lambda t: (
    "growth>2%" if t[0] else "growth<=2%",
    "infl<5%" if t[1] else "infl>=5%",
))
print(quad.to_string())

# ---------- Extreme cases ----------
print("\n" + "=" * 60)
print("5. ELECTIONS WITH VERY HIGH INFLATION (>20%) — did ruling party lose?")
print("=" * 60)
extreme = d[d["inflation_cpi_lag1"] > 20].sort_values("inflation_cpi_lag1", ascending=False)
print(f"{len(extreme)} elections")
print(extreme[["country_name", "year", "inflation_cpi_lag1", "gdp_growth_real_lag1", "lost"]]
      .head(15).to_string(index=False))
print(f"\nLoss rate in these {len(extreme)} high-inflation elections: "
      f"{extreme['lost'].mean()*100:.1f}%")
