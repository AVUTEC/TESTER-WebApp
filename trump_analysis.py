"""Analyseer of Trump-events het volgende MSE.PA-rendement voorspellen.

Werkt met dagelijkse close data die we al hebben (prices.csv). Voor elke event
kijkt 't naar het rendement op dag 0 (event_day -> volgende close),
dag +1, +3, +5, en +10 handelsdagen.

Daarna aggregeert 't per sentiment-label en vergelijkt met baseline (random dag).
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from data import load_prices


SENTIMENT_ORDER = ["very_bearish", "bearish", "neutral", "bullish", "very_bullish"]
SCORE = {"very_bearish": -2, "bearish": -1, "neutral": 0, "bullish": 1, "very_bullish": 2}


def next_trading_day_on_or_after(idx: pd.DatetimeIndex, when: pd.Timestamp) -> pd.Timestamp | None:
    pos = idx.searchsorted(when)
    if pos >= len(idx):
        return None
    return idx[pos]


def forward_returns(prices: pd.Series, events: pd.DataFrame, horizons=(1, 3, 5, 10)) -> pd.DataFrame:
    idx = prices.index
    rows = []
    for _, ev in events.iterrows():
        # Event-datum naar eerste handelsdag op of na die datum
        ev_date = pd.to_datetime(ev.date)
        # Als event >= 17:00 ET, reken dag erna als "event day" (post-close impact)
        try:
            et_hour = int(str(ev.event_time_et).split(":")[0])
        except Exception:
            et_hour = 12
        if et_hour >= 17:
            ev_date = ev_date + pd.Timedelta(days=1)

        day0 = next_trading_day_on_or_after(idx, ev_date)
        if day0 is None:
            continue
        p0_pos = idx.get_loc(day0)
        p0_prev = prices.iloc[p0_pos - 1] if p0_pos > 0 else prices.iloc[p0_pos]

        row = {
            "date": ev.date,
            "sentiment": ev.sentiment,
            "description": ev.description,
            "day0_date": day0.date(),
        }
        # dag 0 = rendement van pre-event close naar event-dag close
        row["d0"] = prices.iloc[p0_pos] / p0_prev - 1

        for h in horizons:
            tgt_pos = p0_pos + h
            if tgt_pos < len(prices):
                row[f"d+{h}"] = prices.iloc[tgt_pos] / p0_prev - 1
            else:
                row[f"d+{h}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_stats(prices: pd.Series, horizons=(1, 3, 5, 10)) -> dict:
    rets = {}
    rets["d0"] = prices.pct_change().mean()
    for h in horizons:
        rets[f"d+{h}"] = prices.pct_change(h + 1).mean()
    return rets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="trump_events.csv")
    ap.add_argument("--ticker", default="MSE.PA",
                    help="MSE.PA (Euro Stoxx 50), SPY (S&P 500), QQQ (Nasdaq 100)")
    args = ap.parse_args()

    prices = load_prices(ticker=args.ticker)
    print(f"[ticker] {args.ticker}")
    events = pd.read_csv(args.events)
    print(f"[data] {len(prices)} koers-dagen, {len(events)} events")
    print(f"       periode koers: {prices.index[0].date()} -> {prices.index[-1].date()}")

    # Filter events binnen koersperiode
    events = events[pd.to_datetime(events.date).between(prices.index[0], prices.index[-1])]

    fr = forward_returns(prices, events)
    print(f"       gebruikte events: {len(fr)}")

    # Rapport per event
    print("\n=== Per event ===")
    cols = ["day0_date", "sentiment", "d0", "d+1", "d+3", "d+5", "d+10"]
    with pd.option_context("display.max_colwidth", 60, "display.width", 180):
        show = fr.copy()
        for c in ["d0", "d+1", "d+3", "d+5", "d+10"]:
            show[c] = show[c].apply(lambda x: f"{x*100:+6.2f}%" if pd.notna(x) else "   -")
        show["description"] = fr["description"].str.slice(0, 55)
        print(show[["day0_date", "sentiment", "d0", "d+1", "d+3", "d+5", "d+10", "description"]].to_string(index=False))

    # Baseline (random dag)
    base = baseline_stats(prices)
    print("\n=== Baseline (gemiddeld dagrendement op ELKE dag, 2010-2026) ===")
    for k, v in base.items():
        print(f"  {k}: {v*100:+.3f}%")

    # Aggregaat per sentiment
    print("\n=== Gemiddeld rendement NA event, per sentiment ===")
    hdr = "{:<15} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "sentiment", "n", "d0", "d+1", "d+3", "d+5", "d+10")
    print(hdr)
    for sent in SENTIMENT_ORDER:
        sub = fr[fr.sentiment == sent]
        if len(sub) == 0:
            continue
        vals = [f"{sub[c].mean()*100:+6.2f}%" for c in ["d0", "d+1", "d+3", "d+5", "d+10"]]
        print(f"{sent:<15} {len(sub):>5} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} {vals[4]:>10}")

    # Correlatie sentiment-score vs rendement
    print("\n=== Correlatie sentiment-score met forward returns ===")
    fr["score"] = fr["sentiment"].map(SCORE)
    for c in ["d0", "d+1", "d+3", "d+5", "d+10"]:
        valid = fr[["score", c]].dropna()
        if len(valid) < 3:
            continue
        corr = valid["score"].corr(valid[c])
        print(f"  corr(score, {c}) = {corr:+.3f}  (n={len(valid)})")


if __name__ == "__main__":
    main()
