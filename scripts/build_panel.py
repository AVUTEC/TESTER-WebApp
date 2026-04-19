"""Build analysis-ready country-year panel: elections x economy, 1980+.

Logic:
  - Start from V-Dem slim (country-year panel with democracy + election flags).
  - Flag election_year = 1 when any v2eltype_* has a non-null election indicator.
  - Merge in economic indicators (inflation, GDP, population) on iso3 (via COWcode->iso3 mapping)
    using V-Dem's country_text_id (already ISO-3).
  - Compute lagged inflation & growth (year-1) for pre-election conditions.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

vdem = pd.read_csv(PROC / "vdem_slim.csv")
econ = pd.read_csv(ROOT / "data" / "raw" / "economic_indicators.csv")
vparty = pd.read_csv(PROC / "vparty_slim.csv")

# Election year flag — v2eltype_* non-null indicates that election type occurred
eltype_cols = [c for c in vdem.columns if c.startswith("v2eltype_")]
vdem["election_year"] = vdem[eltype_cols].notna().any(axis=1).astype(int)
vdem["n_election_types"] = vdem[eltype_cols].notna().sum(axis=1)

# Derive real GDP growth (yr-over-yr on current USD is noisy; use nominal growth as proxy
# and inflation_cpi to estimate real — rough, but inflation_gdp_deflator is better where present)
econ = econ.sort_values(["iso3", "year"])
econ["gdp_growth_nominal"] = econ.groupby("iso3")["gdp_current_usd"].pct_change() * 100
econ["pop_growth"] = econ.groupby("iso3")["population"].pct_change() * 100
# Real GDP growth ≈ nominal growth - GDP deflator inflation
econ["gdp_growth_real"] = econ["gdp_growth_nominal"] - econ["inflation_gdp_deflator"]
econ["gdp_pc_usd"] = econ["gdp_current_usd"] / econ["population"]

# Merge
panel = vdem.merge(
    econ[["iso3", "year", "inflation_cpi", "inflation_gdp_deflator",
          "gdp_current_usd", "gdp_pc_usd", "gdp_growth_real", "gdp_growth_nominal",
          "population", "pop_growth"]],
    left_on=["country_text_id", "year"], right_on=["iso3", "year"], how="left"
).drop(columns=["iso3"])

# Lags for pre-election economy
for col in ["inflation_cpi", "gdp_growth_real"]:
    panel[f"{col}_lag1"] = panel.groupby("country_text_id")[col].shift(1)
    panel[f"{col}_lag2"] = panel.groupby("country_text_id")[col].shift(2)
    # Average over 2 years prior
    panel[f"{col}_pre2yr_avg"] = (panel[f"{col}_lag1"] + panel[f"{col}_lag2"]) / 2

panel = panel.sort_values(["country_text_id", "year"]).reset_index(drop=True)
panel.to_csv(PROC / "panel.csv", index=False)

# --- Summary ---
print("=" * 60)
print("PANEL COVERAGE SUMMARY")
print("=" * 60)
print(f"Total country-years:   {len(panel):,}")
print(f"Countries:             {panel['country_text_id'].nunique()}")
print(f"Years:                 {int(panel.year.min())}-{int(panel.year.max())}")
print(f"Election-years:        {int(panel['election_year'].sum()):,}")
print()

# Democracies only (electoral-democracy index > 0.5)
dem = panel[panel["v2x_polyarchy"] > 0.5]
print(f"Democratic country-years (v2x_polyarchy > 0.5): {len(dem):,}")
print(f"  countries: {dem['country_text_id'].nunique()}")
print(f"  election-years in democracies: {int(dem['election_year'].sum()):,}")
print()

# With economic data
has_econ = dem.dropna(subset=["inflation_cpi", "gdp_growth_real"])
print(f"Democratic election-years WITH econ data (inflation + growth):")
elec_econ = has_econ[has_econ["election_year"] == 1]
print(f"  {len(elec_econ):,} election-years across {elec_econ['country_text_id'].nunique()} countries")
print()

# V-Party coverage (for incumbent analysis)
print(f"V-Party rows:          {len(vparty):,}")
print(f"V-Party countries:     {vparty['country_name'].nunique()}")
print(f"V-Party years:         {int(vparty.year.min())}-{int(vparty.year.max())}")
ruling = vparty[vparty["v2pagovsup"] == 0]
print(f"Ruling-party rows:     {len(ruling):,}")

print()
print("Saved: data/processed/panel.csv")
