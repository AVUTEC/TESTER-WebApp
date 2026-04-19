"""Pull V-Dem and V-Party datasets from the vdeminstitute/vdemdata repo.

V-Dem main: ~28k country-year rows, 1789-2025, 4600+ variables.
V-Party:    ~12k party-year rows, 1900-2019, vote & seat shares, ideology.

We keep only the variables we need for election/economy analysis and
filter to year>=1980 where relevant.
"""
from pathlib import Path
import requests
import pyreadr

OUT_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_PROC.mkdir(parents=True, exist_ok=True)

VDEM_URL = "https://raw.githubusercontent.com/vdeminstitute/vdemdata/master/data/vdem.RData"
VPARTY_URL = "https://raw.githubusercontent.com/vdeminstitute/vdemdata/master/data/vparty.RData"

VDEM_KEEP = [
    "country_name", "country_text_id", "country_id", "year", "COWcode",
    "v2x_polyarchy",        # electoral democracy index (0-1)
    "v2x_libdem",           # liberal democracy index
    "v2x_regime",           # regime type (0=closed autoc, 1=electoral autoc, 2=electoral dem, 3=liberal dem)
    "v2xlg_legcon",         # legislative constraints
    "v2elturnhog",          # turnover head of government at election
    "v2elturnhos",          # turnover head of state at election
    "v2eltype_0",           # election type indicators (0-6)
    "v2eltype_1", "v2eltype_2", "v2eltype_3",
    "v2eltype_4", "v2eltype_5", "v2eltype_6",
    "v2elvaptrn",           # voting-age-population turnout
    "v2elreggov",            # regional govt election
    "v2ellostsl",           # largest opposition vote share
    "v2ellostsw",           # largest opposition seat share
    "v2elvotlrg",           # vote share largest party
    "v2elvotsml",           # vote share smallest party in govt
    "v2ex_hosw",            # HOS power
    "v2exfemhog",           # female HOG
]

VPARTY_KEEP = [
    "country_name", "country_text_id", "country_id", "year",
    "v2paenname",           # party name
    "v2pashname",           # short name
    "v2pavote",             # vote share (%)
    "v2paseatshare",        # seat share (%)
    "v2panumbseat",         # number of seats
    "v2patotalseat",        # total seats in chamber
    "v2pagovsup",           # 0=ruling party, 1=support, 2=oppose, 3=no info
    "v2pariglef",           # left-right scale
    "v2xpa_popul",          # populism
    "v2paelcont",           # contested this election
]


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  cached: {dest}")
        return dest
    print(f"  downloading {url} -> {dest}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_rdata(path: Path):
    res = pyreadr.read_r(str(path))
    df = list(res.values())[0]
    return df


def main():
    # V-Dem main
    print("V-Dem main ...")
    p = download(VDEM_URL, OUT_RAW / "vdem.RData")
    vdem = load_rdata(p)
    present = [c for c in VDEM_KEEP if c in vdem.columns]
    missing = [c for c in VDEM_KEEP if c not in vdem.columns]
    if missing:
        print(f"  WARNING: missing cols: {missing}")
    vdem_slim = vdem[present].copy()
    vdem_slim = vdem_slim[vdem_slim["year"] >= 1980]
    vdem_slim.to_csv(OUT_PROC / "vdem_slim.csv", index=False)
    print(f"  rows={len(vdem_slim):,} countries={vdem_slim['country_name'].nunique()} years={int(vdem_slim.year.min())}-{int(vdem_slim.year.max())}")

    # V-Party
    print("\nV-Party ...")
    p = download(VPARTY_URL, OUT_RAW / "vparty.RData")
    vp = load_rdata(p)
    present = [c for c in VPARTY_KEEP if c in vp.columns]
    missing = [c for c in VPARTY_KEEP if c not in vp.columns]
    if missing:
        print(f"  WARNING: missing cols: {missing}")
    vp_slim = vp[present].copy()
    vp_slim = vp_slim[vp_slim["year"] >= 1980]
    vp_slim.to_csv(OUT_PROC / "vparty_slim.csv", index=False)
    print(f"  rows={len(vp_slim):,} countries={vp_slim['country_name'].nunique()} years={int(vp_slim.year.min())}-{int(vp_slim.year.max())}")


if __name__ == "__main__":
    main()
