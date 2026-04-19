# Data sources

## Accessible from this sandbox

Only `github.com` / `raw.githubusercontent.com` are allowlisted. Direct APIs
(World Bank, IMF, V-Dem portal, FRED, IDEA) return `403 Host not in allowlist`.
Everything below is pulled via GitHub mirrors.

### Economic

| Indicator | URL | Years | Countries |
|---|---|---|---|
| CPI inflation (annual %) | `github.com/datasets/inflation/main/data/inflation-consumer.csv` | 1961–2023 | 261 |
| GDP deflator inflation | `github.com/datasets/inflation/main/data/inflation-gdp.csv` | 1960–2024 | 240 |
| Nominal GDP (USD) | `github.com/datasets/gdp/main/data/gdp.csv` | 1960–2023 | 262 |
| Population | `github.com/datasets/population/main/data/population.csv` | 1960–2024 | 265 |

All four are World Bank WDI mirrors, auto-refreshed quarterly.

### Political / elections / democracy

| Dataset | URL | Shape | Notes |
|---|---|---|---|
| V-Dem v16 | `github.com/vdeminstitute/vdemdata/master/data/vdem.RData` | 28k × 4618 | Country-year, 1789–2025 |
| V-Party | `github.com/vdeminstitute/vdemdata/master/data/vparty.RData` | 12k × 384 | Party-year, 1900–2019 |
| Polity5 panel | `github.com/xmarquez/democracyData/master/data-raw/polity5_panel.sav` | SPSS file | Polity IV/V scores |

Format note: V-Dem ships as `.RData`. We use `pyreadr` to load in Python.

## Key V-Dem columns in use

| Column | Description |
|---|---|
| `v2x_polyarchy` | Electoral democracy index (0–1) |
| `v2x_libdem` | Liberal democracy index |
| `v2x_regime` | 0 closed autoc, 1 electoral autoc, 2 electoral dem, 3 liberal dem |
| `v2eltype_0..6` | Election type indicators (legislative, presidential, etc.) |
| `v2elturnhog` | Turnover of head of government at election (0 none, 1 partial, 2 full) |
| `v2elvaptrn` | Voting-age-population turnout |
| `v2ellostsl` | Largest opposition vote share |
| `v2elvotlrg` | Vote share of largest party |

## Key V-Party columns

| Column | Description |
|---|---|
| `v2pavote` | Vote share (%) |
| `v2paseatshare` | Seat share (%) |
| `v2pagovsup` | 0 ruling, 1 support, 2 oppose |
| `v2pariglef` | Left-right scale |
| `v2xpa_popul` | Populism score |

## Still needed

- **Unemployment**: IMF WEO or ILO bulk. No current GitHub mirror found.
- **Real GDP growth**: Penn World Table or Maddison would be cleaner than
  deflating USD-denominated GDP.
- **Government composition**: ParlGov or DPI for OECD countries to identify
  incumbents more precisely than V-Party's `v2pagovsup`.
- **Election-specific fairness flags**: NELDA (currently only available in
  R-package form from `xmarquez/democracyData`, not mirrored as CSV).

## Blocked sources (fallback options)

- `api.worldbank.org` — WDI API directly
- `www.v-dem.net` — V-Dem download portal
- `www.imf.org` — WEO database
- `www.parlgov.org` — ParlGov
- `nelda.co` — NELDA

If the allowlist is expanded, `scripts/pull_worldbank.py` (already written) can
hit the WDI API directly for any indicator.
