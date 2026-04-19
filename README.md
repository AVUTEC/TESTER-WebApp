# Economy, Inflation & Democratic Elections

Research project on how macroeconomic conditions — especially inflation and GDP
growth — shape electoral outcomes in democracies worldwide since 1980.

## Status

Phase 1: **data collection**. See `docs/data_sources.md` for source notes.

## Datasets pulled (raw)

| File | Source | Shape | Coverage |
|---|---|---|---|
| `data/raw/economic_indicators.csv` | World Bank WDI (via `github.com/datasets/*` mirrors) | 11,915 × 7 | 265 countries, 1980–2024 |
| `data/raw/vdem.RData` | V-Dem Institute | 28,092 × 4,618 | 202 countries, 1789–2025 |
| `data/raw/vparty.RData` | V-Dem (V-Party) | 11,898 × 384 | 180 countries, 1900–2019 |

## Processed

| File | Description |
|---|---|
| `data/processed/vdem_slim.csv` | V-Dem filtered to 23 relevant columns, year ≥ 1980 |
| `data/processed/vparty_slim.csv` | V-Party filtered to party-election variables |
| `data/processed/panel.csv` | Country-year panel merging V-Dem + economic indicators with 1- and 2-year lags |

## Panel coverage

- 7,954 country-years across 181 countries, 1980–2025
- **2,174 election-years** flagged (any V-Dem `v2eltype_*` indicator)
- **1,096 election-years in democracies** (V-Dem `v2x_polyarchy > 0.5`)
- **959 democratic election-years with complete inflation + real-growth data**
- 1,357 ruling-party observations in V-Party (for incumbent vote-share analysis)

## Hypotheses we can test

1. **Economic voting**: does high pre-election inflation predict incumbent loss?
2. **Growth dividend**: does strong pre-election GDP growth boost incumbent vote share?
3. **Asymmetry**: are voters more sensitive to inflation or unemployment? (unemployment data still needs a source — see TODOs)
4. **Democratic resilience**: does inflation correlate with democratic backsliding (`v2x_polyarchy` decline)?
5. **Regime effects**: does the economy-vote link differ in electoral democracies vs. liberal democracies?

## Known limitations

- **GDP growth is approximated** as `(% change in nominal USD GDP) - (GDP deflator inflation)`. For countries with volatile exchange rates vs. the USD this is noisy. A proper real-GDP-growth or Penn World Table pull is TODO.
- **No unemployment data yet** — the `datasets/unemployment` mirror is US-only. Options: IMF WEO CSV, ILO bulk download, OECD.
- **Sandbox network restrictions**: direct access to `api.worldbank.org`, `v-dem.net`, `imf.org` is blocked. All data must come via GitHub-hosted mirrors.

## Reproducing

```bash
pip install pandas numpy requests pyreadr
python scripts/pull_economic.py
python scripts/pull_vdem.py
python scripts/build_panel.py
```

## Layout

```
data/
  raw/        # unmodified source files
  processed/  # cleaned/merged panels
scripts/      # data pulls + panel build
notebooks/    # (TBD) exploratory analysis
docs/         # source notes
```
