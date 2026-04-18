"""Grid search: zoekt de beste (drempel, lookback) combinatie, met transactiekosten.

Usage:
    python optimize.py                      # default grid
    python optimize.py --capital 100000
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from data import load_prices
from strategies import buy_and_hold, weekly_switch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--fee-rate", type=float, default=0.00044,
                    help="proportionele kosten per transactie (0.00044 = 0,044%)")
    ap.add_argument("--cash-rate", type=float, default=0.01)
    ap.add_argument("--th-min", type=float, default=0.005)
    ap.add_argument("--th-max", type=float, default=0.10)
    ap.add_argument("--th-step", type=float, default=0.0025)
    ap.add_argument("--lb-min", type=int, default=2)
    ap.add_argument("--lb-max", type=int, default=30)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--heatmap", action="store_true")
    args = ap.parse_args()

    prices = load_prices()
    print(f"[data] {len(prices)} dagen: {prices.index[0].date()} -> {prices.index[-1].date()}")

    bh = buy_and_hold(prices, args.capital, fee_rate=args.fee_rate)
    print(f"\nBuy & Hold (incl. fee): eind \u20ac {bh.equity.iloc[-1]:>12,.2f}  "
          f"CAGR {bh.cagr*100:5.2f}%  max DD {bh.max_drawdown*100:6.2f}%\n")

    thresholds = np.arange(args.th_min, args.th_max + args.th_step / 2, args.th_step)
    lookbacks = np.arange(args.lb_min, args.lb_max + 1)

    rows = []
    for th in thresholds:
        for lb in lookbacks:
            r = weekly_switch(
                prices,
                start_capital=args.capital,
                sell_threshold=float(th),
                buy_threshold=-float(th),
                cash_rate_annual=args.cash_rate,
                lookback_trading_days=int(lb),
                fee_rate=args.fee_rate,
            )
            rows.append({
                "drempel": float(th),
                "lookback": int(lb),
                "eind": r.equity.iloc[-1],
                "CAGR": r.cagr,
                "maxDD": r.max_drawdown,
                "trades": r.trades,
                "vs_BH": r.equity.iloc[-1] / bh.equity.iloc[-1] - 1,
            })

    df = pd.DataFrame(rows).sort_values("eind", ascending=False).reset_index(drop=True)

    print(f"Grid: {len(thresholds)} drempels x {len(lookbacks)} lookbacks = {len(df)} combinaties")
    print(f"\nTop {args.top}:")
    header = "{:>8} {:>9} {:>12} {:>7} {:>8} {:>7} {:>8}".format(
        "drempel", "lookback", "eind EUR", "CAGR", "maxDD", "trades", "vs B&H")
    print(header)
    for _, r in df.head(args.top).iterrows():
        print(f"{r.drempel*100:>7.2f}% {int(r.lookback):>8}d "
              f"{r.eind:>12,.0f} {r.CAGR*100:>6.2f}% {r.maxDD*100:>7.2f}% "
              f"{int(r.trades):>7} {r.vs_BH*100:>+7.2f}%")

    best = df.iloc[0]
    print(f"\n>>> Optimum: drempel {best.drempel*100:.2f}%, lookback {int(best.lookback)}d "
          f"-> \u20ac {best.eind:,.0f} (+{best.vs_BH*100:.1f}% vs B&H)")

    if args.heatmap:
        import matplotlib.pyplot as plt

        pivot = df.pivot(index="drempel", columns="lookback", values="eind")
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       extent=[pivot.columns.min() - 0.5, pivot.columns.max() + 0.5,
                               pivot.index.min() - args.th_step / 2,
                               pivot.index.max() + args.th_step / 2],
                       cmap="RdYlGn")
        ax.set_xlabel("lookback (handelsdagen)")
        ax.set_ylabel("drempel")
        ax.set_title(f"Eindwaarde (\u20ac) bij start \u20ac {args.capital:,.0f}, fee {args.fee_rate*100:.3f}%")
        ax.contour(pivot.columns, pivot.index, pivot.values,
                   levels=[bh.equity.iloc[-1]], colors="black", linewidths=1)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("eindwaarde \u20ac")
        fig.tight_layout()
        fig.savefig("heatmap.png", dpi=120)
        print("[plot] heatmap.png geschreven (zwarte lijn = break-even met B&H)")


if __name__ == "__main__":
    main()
