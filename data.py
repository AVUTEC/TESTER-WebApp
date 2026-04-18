"""Koersdata ophalen of uit lokale CSV laden."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_prices(
    ticker: str = "MSE.PA",
    start: str = "2010-01-01",
    end: str | None = None,
    csv_path: str | Path | None = None,
) -> pd.Series:
    """Return een pandas Series van slotkoersen (index = datum).

    Voorkeur: lokale `prices.csv` (door fetch_data.ipynb in Colab gecommit).
    Anders: yfinance. Of een expliciete `csv_path`.
    """
    if csv_path is not None:
        return _from_csv(Path(csv_path))

    # Map ticker -> lokaal CSV
    local_map = {
        "SPY": "prices_spy.csv",
        "^GSPC": "prices_spy.csv",
        "QQQ": "prices_qqq.csv",
        "VWCE": "prices_vwce.csv",
        "VWCE.DE": "prices_vwce.csv",
        "BANKS": "prices_banks.csv",
        "EXV1.DE": "prices_banks.csv",
        "DEFENSE": "prices_defense.csv",
        "DFEN.DE": "prices_defense.csv",
    }
    local_name = local_map.get(ticker, "prices.csv")
    local = Path(__file__).parent / local_name
    if local.exists():
        s = _from_csv(local)
        if start:
            s = s.loc[s.index >= pd.Timestamp(start)]
        if end:
            s = s.loc[s.index <= pd.Timestamp(end)]
        return s

    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Geen data terug van yfinance voor {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return close.dropna()


def _from_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df["Close"].astype(float).dropna()
