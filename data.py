"""Koersdata ophalen of uit lokale CSV laden."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_prices(
    ticker: str = "LYSX.PA",
    start: str = "2010-01-01",
    end: str | None = None,
    csv_path: str | Path | None = None,
) -> pd.Series:
    """Return een pandas Series van slotkoersen (index = datum).

    Probeert eerst yfinance; als dat niet lukt (of als `csv_path` is opgegeven),
    valt het terug op een lokale CSV met kolommen Date,Close.
    """
    if csv_path is not None:
        return _from_csv(Path(csv_path))

    try:
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
    except Exception as exc:
        fallback = Path(__file__).parent / "prices.csv"
        if fallback.exists():
            print(f"[data] yfinance faalde ({exc}); gebruik {fallback}")
            return _from_csv(fallback)
        raise


def _from_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df["Close"].astype(float).dropna()
