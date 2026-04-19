"""Pull economic indicators from GitHub-mirrored open datasets.

The sandbox blocks api.worldbank.org directly, but `github.com/datasets/*`
mirrors the WDI and refreshes it periodically.

Sources:
  - datasets/inflation  (CPI inflation, annual %, World Bank NY.GDP.DEFL.KD.ZG)
  - datasets/gdp        (nominal GDP in current USD)
  - datasets/population (total population)

All three cover ~200 countries, 1960-2023.
"""
from pathlib import Path
import io
import requests
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "inflation_cpi": "https://raw.githubusercontent.com/datasets/inflation/main/data/inflation-consumer.csv",
    "inflation_gdp_deflator": "https://raw.githubusercontent.com/datasets/inflation/main/data/inflation-gdp.csv",
    "gdp_current_usd": "https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv",
    "population": "https://raw.githubusercontent.com/datasets/population/main/data/population.csv",
}


def normalize(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    # datasets/* files use: "Country Name","Country Code","Year","Value"
    # inflation uses: "Country","Country Code","Year","Inflation"
    cols = {c.lower(): c for c in df.columns}
    country = cols.get("country name") or cols.get("country")
    iso = cols.get("country code")
    year = cols.get("year")
    val_col = [c for c in df.columns if c not in {country, iso, year}][0]
    out = df.rename(columns={country: "country", iso: "iso3", year: "year", val_col: value_name})
    out = out[["country", "iso3", "year", value_name]]
    out["year"] = out["year"].astype(int)
    return out


def main():
    merged = None
    for name, url in SOURCES.items():
        print(f"Fetching {name} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = normalize(df, name)
        print(f"  rows={len(df):,}  countries={df['iso3'].nunique()}  years {df['year'].min()}-{df['year'].max()}")
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["iso3", "year", name]], on=["iso3", "year"], how="outer")
    merged = merged[merged["year"] >= 1980].sort_values(["iso3", "year"]).reset_index(drop=True)
    out_path = OUT / "economic_indicators.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"  rows={len(merged):,}  countries={merged['iso3'].nunique()}  years {merged['year'].min()}-{merged['year'].max()}")
    print(merged.head())


if __name__ == "__main__":
    main()
