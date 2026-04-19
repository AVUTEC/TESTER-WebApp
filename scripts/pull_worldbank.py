"""Pull World Bank WDI indicators for all countries, 1980-present.

Indicators:
  FP.CPI.TOTL.ZG    - Inflation, consumer prices (annual %)
  NY.GDP.MKTP.KD.ZG - GDP growth (annual %)
  SL.UEM.TOTL.ZS    - Unemployment, total (% of labor force, ILO est.)
  NY.GDP.PCAP.KD.ZG - GDP per capita growth (annual %)
"""
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

INDICATORS = {
    "FP.CPI.TOTL.ZG": "inflation_cpi",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "SL.UEM.TOTL.ZS": "unemployment",
    "NY.GDP.PCAP.KD.ZG": "gdp_pc_growth",
}

BASE = "https://api.worldbank.org/v2/country/all/indicator/{code}"


def fetch(code: str) -> pd.DataFrame:
    rows = []
    page = 1
    while True:
        r = requests.get(
            BASE.format(code=code),
            params={"format": "json", "date": "1980:2024", "per_page": 20000, "page": page},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            break
        meta, data = payload[0], payload[1]
        rows.extend(data)
        if page >= meta.get("pages", 1):
            break
        page += 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[["countryiso3code", "country", "date", "value"]].copy()
    df["country"] = df["country"].apply(lambda x: x.get("value") if isinstance(x, dict) else x)
    df = df.rename(columns={"countryiso3code": "iso3", "date": "year"})
    df["year"] = df["year"].astype(int)
    return df


def main():
    merged = None
    for code, name in INDICATORS.items():
        print(f"Fetching {code} ({name}) ...")
        df = fetch(code)
        df = df.rename(columns={"value": name})
        df = df[df["iso3"].str.len() == 3]  # drop aggregates where iso3 is empty
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["iso3", "year", name]], on=["iso3", "year"], how="outer")
        print(f"  rows: {len(df):,}")
    merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)
    out_path = OUT / "worldbank_wdi.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  rows={len(merged):,}  countries={merged['iso3'].nunique()}")
    print(merged.head())


if __name__ == "__main__":
    main()
