"""It's the economy, stupid — but measured in VOTE SHARE.

Binary win/lose smooths out close races that flip on small swings.
Economic voting literature (Fair, Hibbs, Lewis-Beck) uses vote share.

Outcome: change in ruling party's vote share vs. its PREVIOUS election.
Inputs:  pre-election inflation, real growth.

Samples:
  A) All democratic elections with V-Party data
  B) Consolidated democracies only (v2x_polyarchy > 0.75)
  C) United States alone (Carville's context)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# Build ruling-party vote-share panel from V-Party
vp = pd.read_csv(PROC / "vparty_slim.csv")
vp = vp.dropna(subset=["v2pavote", "v2pagovsup"])
ruling = vp[vp["v2pagovsup"] == 0].copy()  # 0 = ruling party at the time of election

# For each ruling party, compute change in vote share vs its last election
ruling = ruling.sort_values(["country_text_id", "v2paenname", "year"])
ruling["vote_share_prev"] = ruling.groupby(["country_text_id", "v2paenname"])["v2pavote"].shift(1)
ruling["vote_share_chg"] = ruling["v2pavote"] - ruling["vote_share_prev"]

# Merge in economic lags from the panel
panel = pd.read_csv(PROC / "panel.csv")
econ = panel[["country_text_id", "year", "v2x_polyarchy",
              "inflation_cpi_lag1", "gdp_growth_real_lag1",
              "inflation_cpi", "gdp_growth_real"]].drop_duplicates()

d = ruling.merge(econ, on=["country_text_id", "year"], how="left")
d = d.dropna(subset=["vote_share_chg", "inflation_cpi_lag1", "v2x_polyarchy"])
# drop extreme FX-induced growth outliers
d = d[d["gdp_growth_real_lag1"].abs() < 50] if "gdp_growth_real_lag1" in d else d

print(f"Ruling-party-election observations with vote-share change + econ: {len(d)}")
print(f"  countries: {d['country_text_id'].nunique()}   "
      f"years: {int(d.year.min())}-{int(d.year.max())}\n")


def summarize(df, label):
    df = df.dropna(subset=["vote_share_chg"])
    print(f"--- {label}  (n={len(df)}) ---")
    if len(df) < 20:
        print("  too small\n"); return
    # Correlations
    ci = df[["inflation_cpi_lag1", "vote_share_chg"]].corr().iloc[0, 1]
    print(f"  corr(inflation, vote-share change): {ci:+.3f}")
    if "gdp_growth_real_lag1" in df.columns:
        sub = df.dropna(subset=["gdp_growth_real_lag1"])
        if len(sub) > 20:
            cg = sub[["gdp_growth_real_lag1", "vote_share_chg"]].corr().iloc[0, 1]
            print(f"  corr(growth,    vote-share change): {cg:+.3f}")

    # Mean vote-share change by inflation bin
    bins = [-100, 0, 2, 5, 10, 1e9]
    labels = ["<0%", "0-2%", "2-5%", "5-10%", ">10%"]
    df["infl_bin"] = pd.cut(df["inflation_cpi_lag1"], bins=bins, labels=labels)
    tab = df.groupby("infl_bin", observed=True).agg(
        n=("vote_share_chg", "size"),
        mean_chg=("vote_share_chg", "mean"),
        median_chg=("vote_share_chg", "median"),
    ).round(2)
    print("  ruling-party vote-share change by pre-election inflation:")
    print("   " + tab.to_string().replace("\n", "\n   "))
    print()


# A) All democratic elections
dem = d[d["v2x_polyarchy"] > 0.5]
summarize(dem, "A) All democratic elections")

# B) Consolidated democracies
cons = d[d["v2x_polyarchy"] > 0.75]
summarize(cons, "B) Consolidated democracies (v2x_polyarchy > 0.75)")

# C) US only
us = d[d["country_text_id"] == "USA"]
print("--- C) United States — every election, ruling party change ---")
if len(us):
    print(us[["year", "v2paenname", "vote_share_prev", "v2pavote",
              "vote_share_chg", "inflation_cpi_lag1", "gdp_growth_real_lag1"]]
          .round(2).to_string(index=False))
print()

# Misery index proxy (we lack unemployment, so use inflation + |recession|)
print("=" * 60)
print("MISERY PROXY: inflation + (negative growth, else 0)")
print("=" * 60)
dem = dem.dropna(subset=["gdp_growth_real_lag1"]).copy()
dem["misery"] = dem["inflation_cpi_lag1"] + dem["gdp_growth_real_lag1"].apply(lambda x: -x if x < 0 else 0)
bins = [-1e9, 2, 5, 10, 20, 1e9]
labels = ["<2 (very calm)", "2-5", "5-10", "10-20", ">20 (high misery)"]
dem["misery_bin"] = pd.cut(dem["misery"], bins=bins, labels=labels)
tab = dem.groupby("misery_bin", observed=True).agg(
    n=("vote_share_chg", "size"),
    mean_vote_chg=("vote_share_chg", lambda s: round(s.mean(), 2)),
    median_vote_chg=("vote_share_chg", lambda s: round(s.median(), 2)),
)
print(tab.to_string())

# Simple OLS: vote_share_chg = a + b1*inflation + b2*growth
print("\n" + "=" * 60)
print("OLS regression on consolidated democracies")
print("=" * 60)
sub = cons.dropna(subset=["vote_share_chg", "inflation_cpi_lag1", "gdp_growth_real_lag1"])
X = sub[["inflation_cpi_lag1", "gdp_growth_real_lag1"]].values
X = np.column_stack([np.ones(len(X)), X])
y = sub["vote_share_chg"].values
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ coef
r2 = 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
n, k = len(y), X.shape[1]
resid = y - y_hat
sigma2 = (resid ** 2).sum() / (n - k)
var_b = sigma2 * np.linalg.inv(X.T @ X).diagonal()
se = np.sqrt(var_b)
names = ["intercept", "inflation_t-1", "real_growth_t-1"]
print(f"  n = {n}   R² = {r2:.3f}")
for name, c, s in zip(names, coef, se):
    t = c / s
    print(f"    {name:18s}  coef={c:+.3f}  se={s:.3f}  t={t:+.2f}")
print("\n  interpretation: a 1-pp increase in inflation in the year before the")
print("  election is associated with ~X pp change in ruling-party vote share.")
