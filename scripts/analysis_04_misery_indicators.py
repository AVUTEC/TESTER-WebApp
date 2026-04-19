"""Richer distress indicators and a WITHIN-COUNTRY test of 'it's the economy'.

The previous vote-share regression showed near-zero correlation with
absolute inflation. Two things may be hiding the effect:

  1. Absolute inflation means different things in different countries
     (Venezuela's baseline is not Denmark's). The right test is each
     country relative to ITS OWN history: demean by country.

  2. Level may not matter as much as SHOCK (acceleration), VOLATILITY,
     or CUMULATIVE inflation over the whole term.

Build these indicators and rerun, with country fixed effects.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
panel = pd.read_csv(PROC / "panel.csv").sort_values(["country_text_id", "year"])
vp = pd.read_csv(PROC / "vparty_slim.csv")

g = panel.groupby("country_text_id")

# 1. Inflation level
panel["infl_level"] = panel["inflation_cpi_lag1"]
# 2. Inflation SHOCK (change vs 2yr prior)
panel["infl_shock"] = panel["inflation_cpi_lag1"] - g["inflation_cpi"].shift(2)
# 3. Inflation VOLATILITY (rolling 3yr std of inflation)
panel["infl_vol"] = g["inflation_cpi"].shift(1).rolling(3).std().reset_index(0, drop=True)
# Recompute volatility properly (within country)
panel["infl_vol"] = (
    panel.groupby("country_text_id")["inflation_cpi"]
    .transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
)
# 4. Inflation Z-score: this year's inflation vs country's own history (mean, std up to yr-1)
def zscore_vs_history(s):
    mean = s.shift(1).expanding(min_periods=5).mean()
    std = s.shift(1).expanding(min_periods=5).std()
    return (s - mean) / std.replace(0, np.nan)
panel["infl_zscore"] = panel.groupby("country_text_id")["inflation_cpi_lag1"].transform(
    lambda s: (s - s.expanding(min_periods=5).mean().shift(1)) / s.expanding(min_periods=5).std().shift(1).replace(0, np.nan)
)

# 5. Cumulative inflation over 4 years prior (compounding)
def cum4(s):
    pct = s / 100.0
    return (((1 + pct.shift(1)) * (1 + pct.shift(2)) * (1 + pct.shift(3)) * (1 + pct.shift(4))) - 1) * 100
panel["infl_cum4yr"] = panel.groupby("country_text_id")["inflation_cpi"].transform(cum4)

# 6. Per-capita GDP growth (we already have gdp_pc_usd; diff year-over-year)
panel["gdp_pc_growth"] = panel.groupby("country_text_id")["gdp_pc_usd"].pct_change() * 100
panel["gdp_pc_growth_lag1"] = panel.groupby("country_text_id")["gdp_pc_growth"].shift(1)

# Build ruling-party vote-share change from V-Party
r = vp[vp["v2pagovsup"] == 0].dropna(subset=["v2pavote"]).copy()
r = r.sort_values(["country_text_id", "v2paenname", "year"])
r["vote_share_prev"] = r.groupby(["country_text_id", "v2paenname"])["v2pavote"].shift(1)
r["vote_share_chg"] = r["v2pavote"] - r["vote_share_prev"]

# Merge
cols = ["country_text_id", "year", "v2x_polyarchy",
        "infl_level", "infl_shock", "infl_vol", "infl_zscore",
        "infl_cum4yr", "gdp_growth_real_lag1", "gdp_pc_growth_lag1"]
d = r.merge(panel[cols], on=["country_text_id", "year"], how="left")
d = d.dropna(subset=["vote_share_chg"])
d = d[d["v2x_polyarchy"] > 0.5]
print(f"n = {len(d)} ruling-party elections with vote-share change (democracies)\n")


def corr_report(df, ycol, xcols):
    for x in xcols:
        sub = df.dropna(subset=[x, ycol])
        if len(sub) < 30: continue
        c = sub[[x, ycol]].corr().iloc[0, 1]
        print(f"  corr({x:22s}, {ycol}) = {c:+.3f}   n={len(sub)}")


print("=" * 60)
print("1) CORRELATION of each indicator with ruling-party vote-share change")
print("=" * 60)
corr_report(d, "vote_share_chg", ["infl_level", "infl_shock", "infl_vol",
                                   "infl_zscore", "infl_cum4yr",
                                   "gdp_growth_real_lag1", "gdp_pc_growth_lag1"])

# Same but for binary loss (use panel's election-level data to get v2elturnhog)
print("\n" + "=" * 60)
print("2) BINARY LOSS: each indicator's correlation with ruling-party loss")
print("=" * 60)
b = panel[(panel["election_year"] == 1) & (panel["v2x_polyarchy"] > 0.5) & panel["v2elturnhog"].notna()].copy()
b["lost"] = (b["v2elturnhog"] == 2).astype(int)
corr_report(b, "lost", ["infl_level", "infl_shock", "infl_vol", "infl_zscore",
                         "infl_cum4yr", "gdp_growth_real_lag1", "gdp_pc_growth_lag1"])

# Loss rate by inflation z-score bucket (within country)
print("\n  Loss rate by INFLATION Z-SCORE (vs country's own history):")
bins = [-100, -1, -0.5, 0, 0.5, 1, 2, 1e9]
labels = ["below avg (<-1)", "-1 to -.5", "-.5 to 0", "0 to .5", ".5 to 1",
          "1 to 2 (above)", ">2 (extreme shock)"]
bz = b.dropna(subset=["infl_zscore"]).copy()
bz["zb"] = pd.cut(bz["infl_zscore"], bins=bins, labels=labels)
tab = bz.groupby("zb", observed=True).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean() * 100, 1)),
    mean_z=("infl_zscore", lambda s: round(s.mean(), 2)),
)
print(tab.to_string())

print("\n  Loss rate by INFLATION VOLATILITY (3yr rolling std):")
bins = [-0.01, 1, 2, 5, 10, 1e9]
labels = ["<1pp (stable)", "1-2pp", "2-5pp", "5-10pp", ">10pp (erratic)"]
bv = b.dropna(subset=["infl_vol"]).copy()
bv["vb"] = pd.cut(bv["infl_vol"], bins=bins, labels=labels)
tab = bv.groupby("vb", observed=True).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean() * 100, 1)),
)
print(tab.to_string())

print("\n  Loss rate by CUMULATIVE 4-year inflation (total price rise over prior term):")
bins = [-100, 5, 15, 30, 60, 1e9]
labels = ["<5% (calm term)", "5-15%", "15-30%", "30-60%", ">60% (runaway)"]
bc = b.dropna(subset=["infl_cum4yr"]).copy()
bc["cb"] = pd.cut(bc["infl_cum4yr"], bins=bins, labels=labels)
tab = bc.groupby("cb", observed=True).agg(
    n=("lost", "size"),
    pct_lost=("lost", lambda s: round(s.mean() * 100, 1)),
)
print(tab.to_string())

# Within-country (fixed-effects) regression: demean by country
print("\n" + "=" * 60)
print("3) WITHIN-COUNTRY (fixed effects) regression on vote-share change")
print("=" * 60)
fe = d.dropna(subset=["vote_share_chg", "infl_zscore", "gdp_pc_growth_lag1"]).copy()
# Demean by country
for col in ["vote_share_chg", "infl_zscore", "gdp_pc_growth_lag1", "infl_vol", "infl_shock"]:
    fe[f"{col}_dm"] = fe[col] - fe.groupby("country_text_id")[col].transform("mean")

X = fe[["infl_zscore_dm", "gdp_pc_growth_lag1_dm"]].values
X = np.column_stack([np.ones(len(X)), X])
y = fe["vote_share_chg_dm"].values
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ coef
r2 = 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
resid = y - y_hat
sigma2 = (resid ** 2).sum() / (len(y) - X.shape[1])
se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X).diagonal())
print(f"  n = {len(y)}   R² (within) = {r2:.3f}")
for name, c, s in zip(["intercept", "infl_zscore (country-demeaned)", "gdp_pc_growth (country-demeaned)"], coef, se):
    print(f"    {name:40s}  coef={c:+.3f}  se={s:.3f}  t={c/s:+.2f}")

print("\n  Interpretation: when a country's inflation is 1 sd above ITS own history,")
print("  ruling party's vote share changes by the coefficient amount (pp).")
