"""US + Western Europe established democracies only.

This is the setting where:
  - Party systems are stable (same party names over decades)
  - Responsibility for the economy is relatively clear
  - Classic economic voting literature (Fair, Lewis-Beck) focused here

Countries:
  USA, and Western Europe: AUT, BEL, DNK, FIN, FRA, DEU, GRC, IRL, ITA,
  LUX, NLD, PRT, ESP, SWE, GBR, NOR, CHE, ISL
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
panel = pd.read_csv(PROC / "panel.csv").sort_values(["country_text_id", "year"])
vp = pd.read_csv(PROC / "vparty_slim.csv")

WEST = ["USA", "AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "GRC", "IRL",
        "ITA", "LUX", "NLD", "PRT", "ESP", "SWE", "GBR", "NOR", "CHE", "ISL"]

# ---- Build panel subset ----
p = panel[panel["country_text_id"].isin(WEST)].copy()

# Recompute per-capita growth
p["gdp_pc_growth"] = p.groupby("country_text_id")["gdp_pc_usd"].pct_change() * 100
p["gdp_pc_growth_lag1"] = p.groupby("country_text_id")["gdp_pc_growth"].shift(1)

# inflation shock & z-score
p["infl_shock"] = p["inflation_cpi_lag1"] - p.groupby("country_text_id")["inflation_cpi"].shift(2)
p["infl_zscore"] = p.groupby("country_text_id")["inflation_cpi_lag1"].transform(
    lambda s: (s - s.expanding(min_periods=5).mean().shift(1))
              / s.expanding(min_periods=5).std().shift(1).replace(0, np.nan)
)

# ---- Election-level analysis ----
e = p[(p["election_year"] == 1) & p["v2elturnhog"].notna()
      & p["inflation_cpi_lag1"].notna() & p["gdp_growth_real_lag1"].notna()].copy()
e["lost"] = (e["v2elturnhog"] == 2).astype(int)

print(f"US + W. Europe: {len(e)} democratic elections ({e.country_text_id.nunique()} countries), 1980-{int(e.year.max())}")
print(f"Base loss rate: {e['lost'].mean()*100:.1f}%\n")

# Elections per country
print("Elections per country:")
print(e.groupby("country_text_id").size().sort_values(ascending=False).to_string())
print()

# ---- Cross-tabs ----
print("=" * 60)
print("LOSS RATE by inflation (lag1)")
print("=" * 60)
bins = [-100, 2, 4, 6, 10, 1e9]
labels = ["<2%", "2-4%", "4-6%", "6-10%", ">10%"]
e["ib"] = pd.cut(e["inflation_cpi_lag1"], bins=bins, labels=labels)
tab = e.groupby("ib", observed=True).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean()*100, 1)),
    avg=("inflation_cpi_lag1", lambda s: round(s.mean(), 1)),
)
print(tab.to_string())

print("\nLOSS RATE by real GDP growth (lag1)")
bins = [-100, -1, 1, 2, 3, 1e9]
labels = ["<-1% (recession)", "-1 to 1%", "1-2%", "2-3%", ">3%"]
e["gb"] = pd.cut(e["gdp_growth_real_lag1"], bins=bins, labels=labels)
tab = e.groupby("gb", observed=True).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean()*100, 1)),
    avg=("gdp_growth_real_lag1", lambda s: round(s.mean(), 1)),
)
print(tab.to_string())

print("\nLOSS RATE by per-capita GDP growth (lag1)")
bins = [-100, -2, 0, 2, 4, 1e9]
labels = ["<-2%", "-2 to 0", "0-2%", "2-4%", ">4%"]
e["pgb"] = pd.cut(e["gdp_pc_growth_lag1"], bins=bins, labels=labels)
tab = e.groupby("pgb", observed=True).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean()*100, 1)),
)
print(tab.to_string())

# ---- Correlations ----
print("\n" + "=" * 60)
print("CORRELATIONS (binary loss)")
print("=" * 60)
for c in ["inflation_cpi_lag1", "infl_shock", "infl_zscore",
          "gdp_growth_real_lag1", "gdp_pc_growth_lag1"]:
    sub = e.dropna(subset=[c])
    r = sub[[c, "lost"]].corr().iloc[0, 1]
    print(f"  corr({c:24s}, lost) = {r:+.3f}   n={len(sub)}")

# ---- Vote share analysis (V-Party) ----
print("\n" + "=" * 60)
print("VOTE-SHARE CHANGE (V-Party, US+W.Europe)")
print("=" * 60)
vp2 = vp[vp["country_text_id"].isin(WEST) & (vp["v2pagovsup"] == 0)].copy()
vp2 = vp2.dropna(subset=["v2pavote"]).sort_values(["country_text_id", "v2paenname", "year"])
vp2["vote_prev"] = vp2.groupby(["country_text_id", "v2paenname"])["v2pavote"].shift(1)
vp2["vote_chg"] = vp2["v2pavote"] - vp2["vote_prev"]

m = vp2.merge(
    p[["country_text_id", "year", "inflation_cpi_lag1", "infl_shock", "infl_zscore",
       "gdp_growth_real_lag1", "gdp_pc_growth_lag1"]],
    on=["country_text_id", "year"], how="left"
).dropna(subset=["vote_chg"])

print(f"n = {len(m)} ruling-party elections across {m.country_text_id.nunique()} countries")
print(f"mean vote-share change: {m['vote_chg'].mean():+.2f} pp (the 'cost of ruling')\n")

for c in ["inflation_cpi_lag1", "infl_shock", "infl_zscore",
          "gdp_growth_real_lag1", "gdp_pc_growth_lag1"]:
    sub = m.dropna(subset=[c])
    r = sub[[c, "vote_chg"]].corr().iloc[0, 1]
    print(f"  corr({c:24s}, vote_chg) = {r:+.3f}   n={len(sub)}")

# ---- OLS with country fixed effects ----
print("\n" + "=" * 60)
print("OLS with country fixed effects (dummy vars)")
print("=" * 60)
fe = m.dropna(subset=["vote_chg", "inflation_cpi_lag1", "gdp_pc_growth_lag1"]).copy()
country_dummies = pd.get_dummies(fe["country_text_id"], drop_first=True).astype(float)
X_cols = ["inflation_cpi_lag1", "gdp_pc_growth_lag1"]
X = np.column_stack([
    np.ones(len(fe)),
    fe[X_cols].values,
    country_dummies.values,
])
y = fe["vote_chg"].values
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ coef
r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
resid = y - yhat
sigma2 = (resid ** 2).sum() / (len(y) - X.shape[1])
se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X).diagonal())
print(f"  n = {len(y)}   R² = {r2:.3f}   (incl. country FE)")
for i, name in enumerate(["intercept"] + X_cols):
    print(f"    {name:24s}  coef={coef[i]:+.3f}  se={se[i]:.3f}  t={coef[i]/se[i]:+.2f}")

# ---- US-only: landmark test ----
print("\n" + "=" * 60)
print("US ONLY — does the classic 'economic voting' hold?")
print("=" * 60)
us = m[m["country_text_id"] == "USA"].copy()
print(f"n = {len(us)} US ruling-party elections")
print(us[["year", "v2paenname", "vote_prev", "v2pavote", "vote_chg",
         "inflation_cpi_lag1", "gdp_growth_real_lag1", "gdp_pc_growth_lag1"]].round(2).to_string(index=False))
if len(us) > 5:
    r_infl = us[["inflation_cpi_lag1", "vote_chg"]].corr().iloc[0, 1]
    r_gr = us[["gdp_growth_real_lag1", "vote_chg"]].corr().iloc[0, 1]
    r_pc = us[["gdp_pc_growth_lag1", "vote_chg"]].corr().iloc[0, 1]
    print(f"\n  corr(inflation, vote_chg) = {r_infl:+.3f}")
    print(f"  corr(real growth, vote_chg) = {r_gr:+.3f}")
    print(f"  corr(pc growth,  vote_chg) = {r_pc:+.3f}")
