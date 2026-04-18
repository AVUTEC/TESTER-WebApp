"""Scan de ratchet-strategie (geen tijdvenster) over een range drempels."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from data import load_prices
from strategies import buy_and_hold, ratchet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--fee-rate", type=float, default=0.00044)
    ap.add_argument("--cash-rate", type=float, default=0.01)
    ap.add_argument("--th-min", type=float, default=0.01)
    ap.add_argument("--th-max", type=float, default=0.30)
    ap.add_argument("--th-step", type=float, default=0.005)
    ap.add_argument("--test-years", type=float, default=2.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    prices = load_prices()
    cutoff = prices.index[-1] - pd.Timedelta(days=int(args.test_years * 365.25))
    train = prices.loc[prices.index < cutoff]
    test = prices.loc[prices.index >= cutoff]

    thresholds = np.arange(args.th_min, args.th_max + args.th_step / 2, args.th_step)

    def run_scan(series, label):
        bh = buy_and_hold(series, args.capital, fee_rate=args.fee_rate)
        rows = []
        for th in thresholds:
            r = ratchet(series, args.capital, float(th), float(th),
                        args.cash_rate, args.fee_rate)
            rows.append({"th": float(th),
                         "eind": r.equity.iloc[-1],
                         "CAGR": r.cagr,
                         "DD": r.max_drawdown,
                         "trades": r.trades,
                         "vs_BH": r.equity.iloc[-1] / bh.equity.iloc[-1] - 1})
        df = pd.DataFrame(rows)
        print(f"\n=== {label}: {series.index[0].date()} -> {series.index[-1].date()} "
              f"({len(series)} dagen) ===")
        print(f"B&H: \u20ac {bh.equity.iloc[-1]:,.0f}  CAGR {bh.cagr*100:.2f}%  DD {bh.max_drawdown*100:.2f}%")
        hdr = "{:>8} {:>12} {:>7} {:>8} {:>7} {:>8}".format(
            "drempel", "eind EUR", "CAGR", "maxDD", "trades", "vs B&H")
        print("\n" + hdr)
        for _, r in df.iterrows():
            print(f"{r.th*100:>7.2f}% {r.eind:>12,.0f} {r.CAGR*100:>6.2f}% "
                  f"{r.DD*100:>7.2f}% {int(r.trades):>7} {r.vs_BH*100:>+7.2f}%")
        best = df.loc[df.eind.idxmax()]
        print(f">>> beste op {label}: drempel {best.th*100:.2f}% "
              f"-> \u20ac {best.eind:,.0f} ({best.vs_BH*100:+.1f}% vs B&H)")
        return df, bh

    df_full, bh_full = run_scan(prices, "VOLLEDIG")
    df_train, bh_train = run_scan(train, "train")

    # walk-forward: neem beste drempel uit train en pas toe op test
    th_star = float(df_train.loc[df_train.eind.idxmax(), "th"])
    print(f"\n\n--- Walk-forward met drempel {th_star*100:.2f}% (optimum uit train) ---")
    r_test = ratchet(test, args.capital, th_star, th_star, args.cash_rate, args.fee_rate)
    bh_test = buy_and_hold(test, args.capital, fee_rate=args.fee_rate)
    d = (r_test.equity.iloc[-1] - bh_test.equity.iloc[-1]) / bh_test.equity.iloc[-1] * 100
    print(f"test periode {test.index[0].date()} -> {test.index[-1].date()}:")
    print(f"  B&H   : \u20ac {bh_test.equity.iloc[-1]:>10,.0f}  CAGR {bh_test.cagr*100:5.2f}%  DD {bh_test.max_drawdown*100:6.2f}%")
    print(f"  Ratchet: \u20ac {r_test.equity.iloc[-1]:>10,.0f}  CAGR {r_test.cagr*100:5.2f}%  DD {r_test.max_drawdown*100:6.2f}%  {r_test.trades} trades  vs B&H: {d:+.2f}%")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df_full.th * 100, df_full.eind, marker="o", label="volledig 2010-2026")
        ax.plot(df_train.th * 100, df_train.eind, marker="x", label=f"train tot {cutoff.date()}")
        ax.axhline(bh_full.equity.iloc[-1], color="gray", linestyle="--",
                   label=f"B&H volledig (\u20ac {bh_full.equity.iloc[-1]:,.0f})")
        ax.axhline(bh_train.equity.iloc[-1], color="gray", linestyle=":",
                   label=f"B&H train (\u20ac {bh_train.equity.iloc[-1]:,.0f})")
        ax.set_xlabel("drempel (%)")
        ax.set_ylabel("eindwaarde \u20ac")
        ax.set_title(f"Ratchet-strategie \u2014 start \u20ac {args.capital:,.0f}, fee {args.fee_rate*100:.3f}%")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("ratchet_scan.png", dpi=120)
        print("[plot] ratchet_scan.png geschreven")


if __name__ == "__main__":
    main()
