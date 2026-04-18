"""Walk-forward test: optimaliseer op trainingsdata, test 'blind' op de rest.

Default: train tot 2 jaar voor einde, test de laatste 2 jaar.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from data import load_prices
from strategies import buy_and_hold, weekly_switch


def grid_best(prices, capital, fee, cash_rate, thresholds, lookbacks):
    best = None
    for th in thresholds:
        for lb in lookbacks:
            r = weekly_switch(prices, capital, float(th), -float(th),
                              cash_rate, int(lb), fee_rate=fee)
            end = r.equity.iloc[-1]
            if best is None or end > best[0]:
                best = (end, float(th), int(lb), r)
    return best


def evaluate(prices, capital, fee, cash_rate, th, lb):
    bh = buy_and_hold(prices, capital, fee_rate=fee)
    sw = weekly_switch(prices, capital, th, -th, cash_rate, lb, fee_rate=fee)
    return bh, sw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--fee-rate", type=float, default=0.00044)
    ap.add_argument("--cash-rate", type=float, default=0.01)
    ap.add_argument("--test-years", type=float, default=2.0,
                    help="hoeveel jaar (aan het eind) voor out-of-sample test")
    ap.add_argument("--th-min", type=float, default=0.005)
    ap.add_argument("--th-max", type=float, default=0.10)
    ap.add_argument("--th-step", type=float, default=0.0025)
    ap.add_argument("--lb-min", type=int, default=2)
    ap.add_argument("--lb-max", type=int, default=30)
    args = ap.parse_args()

    prices = load_prices()
    cutoff = prices.index[-1] - pd.Timedelta(days=int(args.test_years * 365.25))
    train = prices.loc[prices.index < cutoff]
    test = prices.loc[prices.index >= cutoff]

    print(f"[data] totaal {len(prices)} dagen: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"       train: {len(train)} dagen ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"       test : {len(test)} dagen ({test.index[0].date()} -> {test.index[-1].date()})")

    thresholds = np.arange(args.th_min, args.th_max + args.th_step / 2, args.th_step)
    lookbacks = np.arange(args.lb_min, args.lb_max + 1)

    # 1. optimaliseer op trainingsperiode
    end_tr, th_star, lb_star, _ = grid_best(
        train, args.capital, args.fee_rate, args.cash_rate, thresholds, lookbacks
    )
    bh_tr = buy_and_hold(train, args.capital, fee_rate=args.fee_rate)

    # 2. pas optimum ongewijzigd toe op testperiode
    bh_te, sw_te = evaluate(test, args.capital, args.fee_rate,
                            args.cash_rate, th_star, lb_star)

    # 3. en op de volledige periode, ter referentie
    bh_all, sw_all = evaluate(prices, args.capital, args.fee_rate,
                              args.cash_rate, th_star, lb_star)

    def line(label, bh, sw):
        d = (sw.equity.iloc[-1] - bh.equity.iloc[-1]) / bh.equity.iloc[-1] * 100
        return (f"{label:<14} "
                f"B&H \u20ac {bh.equity.iloc[-1]:>10,.0f} (CAGR {bh.cagr*100:5.2f}%, DD {bh.max_drawdown*100:6.2f}%)  "
                f"Switch \u20ac {sw.equity.iloc[-1]:>10,.0f} (CAGR {sw.cagr*100:5.2f}%, DD {sw.max_drawdown*100:6.2f}%, {sw.trades} trades)  "
                f"vs B&H: {d:+6.2f}%")

    print(f"\n>>> Optimum op training: drempel {th_star*100:.2f}%, lookback {lb_star}d "
          f"(train: \u20ac {end_tr:,.0f} vs B&H \u20ac {bh_tr.equity.iloc[-1]:,.0f})\n")

    sw_tr = weekly_switch(train, args.capital, th_star, -th_star,
                          args.cash_rate, lb_star, fee_rate=args.fee_rate)
    print(line("train", bh_tr, sw_tr))
    print(line("TEST (blind)", bh_te, sw_te))
    print(line("volledig", bh_all, sw_all))


if __name__ == "__main__":
    main()
