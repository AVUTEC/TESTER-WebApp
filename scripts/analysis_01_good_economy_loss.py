"""How often does the ruling party lose when the economy is good vs bad?

Data: 959 democratic elections, 1980-2025 (V-Dem + WDI).

Outcome:
  v2elturnhog == 2  -> ruling PARTY lost (new party takes head of govt)
  v2elturnhog == 0  -> same head of govt stays
  v2elturnhog == 1  -> same party, new HoG (e.g. term-limited)

"Economy good" is defined two ways:
  A) Growth: real GDP growth in year before election > 0% (positive growth)
  B) Inflation: CPI inflation in year before election < 5% (low inflation)
  C) Combined: growth > 2% AND inflation < 5%
"""
from pathlib import Path
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
p = pd.read_csv(PROC / "panel.csv")

# Democratic elections with full data and a coded HoG turnover outcome
d = p[
    (p["election_year"] == 1)
    & (p["v2x_polyarchy"] > 0.5)
    & p["v2elturnhog"].notna()
    & p["inflation_cpi_lag1"].notna()
    & p["gdp_growth_real_lag1"].notna()
].copy()

d["ruling_party_lost"] = (d["v2elturnhog"] == 2).astype(int)

n = len(d)
base = d["ruling_party_lost"].mean() * 100
print(f"Sample: {n} democratic elections with full data, 1980-2025")
print(f"Base rate: ruling party lost in {base:.1f}% of elections\n")


def bucket(df, label, mask_good):
    good = df[mask_good]
    bad = df[~mask_good]
    print(f"--- {label} ---")
    print(f"  GOOD economy (n={len(good):4d}): ruling party lost {good['ruling_party_lost'].mean()*100:5.1f}%")
    print(f"  BAD  economy (n={len(bad):4d}): ruling party lost {bad['ruling_party_lost'].mean()*100:5.1f}%")
    diff = (bad['ruling_party_lost'].mean() - good['ruling_party_lost'].mean()) * 100
    print(f"  gap: +{diff:.1f} pp more likely to lose when economy is bad\n")


bucket(d, "A) Growth in prior year > 0%", d["gdp_growth_real_lag1"] > 0)
bucket(d, "B) Inflation in prior year < 5%", d["inflation_cpi_lag1"] < 5)
bucket(d, "C) Strong economy: growth > 2% AND inflation < 5%",
       (d["gdp_growth_real_lag1"] > 2) & (d["inflation_cpi_lag1"] < 5))

# Finer buckets by inflation
print("--- Inflation buckets (ruling-party loss rate) ---")
bins = [-100, 0, 2, 5, 10, 20, 1e9]
labels = ["<0%", "0-2%", "2-5%", "5-10%", "10-20%", ">20%"]
d["infl_bucket"] = pd.cut(d["inflation_cpi_lag1"], bins=bins, labels=labels)
print(d.groupby("infl_bucket", observed=True).agg(
    n=("ruling_party_lost", "size"),
    pct_lost=("ruling_party_lost", lambda s: f"{s.mean()*100:.1f}%"),
).to_string())

print("\n--- Growth buckets ---")
bins = [-100, -2, 0, 2, 4, 1e9]
labels = ["<-2%", "-2-0%", "0-2%", "2-4%", ">4%"]
d["gr_bucket"] = pd.cut(d["gdp_growth_real_lag1"], bins=bins, labels=labels)
print(d.groupby("gr_bucket", observed=True).agg(
    n=("ruling_party_lost", "size"),
    pct_lost=("ruling_party_lost", lambda s: f"{s.mean()*100:.1f}%"),
).to_string())
