"""Test een hypothetische 'event-triggered' strategie op historische events.

Let op: dit is een ORACLE-test. De events en labels zijn door mij achteraf
toegekend. Dit toont of ER een exploiteerbaar signaal IS, niet of wij het
in realtime zouden hebben herkend.

Regels:
- op very_bearish event: sell-next-open, wacht N dagen, koop terug
- op very_bullish event: koop next-open (als in cash)
- anders: niks doen
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from data import load_prices


def run_strategy(
    prices: pd.Series,
    events: pd.DataFrame,
    start_capital: float = 10_000.0,
    fee_rate: float = 0.00044,
    cash_rate_annual: float = 0.01,
    hold_cash_days: int = 10,
    trigger_bearish: tuple = ("very_bearish",),
    trigger_bullish: tuple = ("very_bullish", "bullish"),
):
    daily_cash = (1 + cash_rate_annual) ** (1 / 252)
    idx = prices.index
    p = prices.values

    # Map event-datum -> eerste handelsdag op/na event (account for post-close timing)
    sell_days = set()
    buy_days = set()
    for _, ev in events.iterrows():
        ev_date = pd.to_datetime(ev.date)
        try:
            et_hour = int(str(ev.event_time_et).split(":")[0])
        except Exception:
            et_hour = 12
        if et_hour >= 16:
            ev_date = ev_date + pd.Timedelta(days=1)
        pos = idx.searchsorted(ev_date)
        if pos >= len(idx):
            continue
        day = idx[pos]
        if ev.sentiment in trigger_bearish:
            sell_days.add(day)
        if ev.sentiment in trigger_bullish:
            buy_days.add(day)

    invested = True
    invested_amount = start_capital / (1 + fee_rate)
    shares = invested_amount / p[0]
    cash = 0.0
    cooldown = 0
    trades = 1
    equity = np.empty(len(p))

    for i in range(len(p)):
        day = idx[i]
        if not invested:
            cash *= daily_cash
            cooldown -= 1

        if invested and day in sell_days:
            cash = shares * p[i] * (1 - fee_rate)
            shares = 0.0
            invested = False
            cooldown = hold_cash_days
            trades += 1
        elif not invested and (cooldown <= 0 or day in buy_days):
            shares = (cash / (1 + fee_rate)) / p[i]
            cash = 0.0
            invested = True
            trades += 1

        equity[i] = shares * p[i] + cash

    return pd.Series(equity, index=idx), trades


def metrics(eq: pd.Series, label: str):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    return f"{label:<35} eind EUR {eq.iloc[-1]:>10,.0f}  CAGR {cagr*100:5.2f}%  maxDD {dd*100:6.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="trump_events.csv")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--hold-days", type=int, default=10)
    args = ap.parse_args()

    prices = load_prices()
    events = pd.read_csv(args.events)

    # filter naar koersperiode
    events = events[pd.to_datetime(events.date).between(prices.index[0], prices.index[-1])]
    print(f"[data] {len(prices)} koers-dagen, {len(events)} events binnen periode")

    bh_shares = args.capital / (1 + 0.00044) / prices.iloc[0]
    bh = prices * bh_shares
    print("\n" + metrics(bh, "Buy & Hold (met fee)"))

    # Alleen very_bearish trigger
    eq1, tr1 = run_strategy(prices, events, args.capital, hold_cash_days=args.hold_days,
                             trigger_bearish=("very_bearish",),
                             trigger_bullish=("very_bullish",))
    print(metrics(eq1, f"exit op very_bearish ({tr1} trades)"))

    # very_bearish + gewoon bearish trigger
    eq2, tr2 = run_strategy(prices, events, args.capital, hold_cash_days=args.hold_days,
                             trigger_bearish=("very_bearish", "bearish"),
                             trigger_bullish=("very_bullish", "bullish"))
    print(metrics(eq2, f"exit op bearish ook ({tr2} trades)"))

    # verschillende hold-periodes voor very_bearish alleen
    print("\nHoud-periodes (alleen very_bearish trigger):")
    for h in [1, 3, 5, 10, 15, 20]:
        eq, tr = run_strategy(prices, events, args.capital, hold_cash_days=h,
                               trigger_bearish=("very_bearish",),
                               trigger_bullish=("very_bullish",))
        d = (eq.iloc[-1] - bh.iloc[-1]) / bh.iloc[-1] * 100
        print(f"  hold {h:>2}d: EUR {eq.iloc[-1]:>10,.0f}  ({d:+6.2f}% vs B&H, {tr} trades)")


if __name__ == "__main__":
    main()
