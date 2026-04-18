"""Vergelijk buy & hold met de wekelijkse 4%-switch strategie.

Gebruik:
    python run.py                      # default: LYSX.PA vanaf 2010
    python run.py --csv prices.csv     # lokale CSV (kolommen: Date,Close)
    python run.py --start 2015-01-01 --plot
"""
from __future__ import annotations

import argparse

from data import load_prices
from strategies import buy_and_hold, weekly_switch, SimResult


def _fmt_row(r: SimResult) -> str:
    return (
        f"{r.name:<28} "
        f"eind € {r.equity.iloc[-1]:>12,.2f}  "
        f"tot {r.total_return * 100:>7.2f}%  "
        f"CAGR {r.cagr * 100:>6.2f}%  "
        f"max DD {r.max_drawdown * 100:>7.2f}%  "
        f"trades {r.trades}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="LYSX.PA")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--csv", default=None, help="lokale CSV in plaats van yfinance")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--sell", type=float, default=0.04)
    parser.add_argument("--buy", type=float, default=-0.04)
    parser.add_argument("--cash-rate", type=float, default=0.01)
    parser.add_argument("--lookback", type=int, default=5, help="handelsdagen (~1 week = 5)")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    prices = load_prices(args.ticker, args.start, args.end, args.csv)
    print(f"[data] {len(prices)} dagen: {prices.index[0].date()}  ->  {prices.index[-1].date()}")

    bh = buy_and_hold(prices, args.capital)
    sw = weekly_switch(
        prices,
        start_capital=args.capital,
        sell_threshold=args.sell,
        buy_threshold=args.buy,
        cash_rate_annual=args.cash_rate,
        lookback_trading_days=args.lookback,
    )

    print()
    for r in (bh, sw):
        print(_fmt_row(r))

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        bh.equity.plot(ax=ax, label=bh.name)
        sw.equity.plot(ax=ax, label=sw.name)
        ax.set_title(f"{args.ticker} — portefeuillewaarde (start € {args.capital:,.0f})")
        ax.set_ylabel("EUR")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("comparison.png", dpi=120)
        print("[plot] comparison.png geschreven")


if __name__ == "__main__":
    main()
