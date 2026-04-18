"""Strategieën die een cash+positie-waarde reeks produceren."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass
class SimResult:
    name: str
    equity: pd.Series          # totale portefeuillewaarde per dag
    positions: pd.Series       # 1 = belegd, 0 = liquide
    trades: int                # aantal transacties (buy of sell)

    @property
    def total_return(self) -> float:
        return self.equity.iloc[-1] / self.equity.iloc[0] - 1

    @property
    def cagr(self) -> float:
        years = (self.equity.index[-1] - self.equity.index[0]).days / 365.25
        if years <= 0:
            return float("nan")
        return (self.equity.iloc[-1] / self.equity.iloc[0]) ** (1 / years) - 1

    @property
    def max_drawdown(self) -> float:
        peak = self.equity.cummax()
        dd = self.equity / peak - 1
        return dd.min()


def buy_and_hold(
    prices: pd.Series,
    start_capital: float = 10_000.0,
    fee_rate: float = 0.0,
) -> SimResult:
    invested_amount = start_capital / (1 + fee_rate)
    shares = invested_amount / prices.iloc[0]
    equity = prices * shares
    positions = pd.Series(1, index=prices.index)
    return SimResult("Buy & Hold", equity, positions, trades=1)


def weekly_switch(
    prices: pd.Series,
    start_capital: float = 10_000.0,
    sell_threshold: float = 0.04,
    buy_threshold: float = -0.04,
    cash_rate_annual: float = 0.01,
    lookback_trading_days: int = 5,
    fee_rate: float = 0.0,
) -> SimResult:
    """Verkoop als de koers in de laatste `lookback` dagen >= sell_threshold steeg.
    Koop terug als de koers in de laatste `lookback` dagen <= buy_threshold daalde.
    Liquide geld krijgt `cash_rate_annual` (simpele dagelijkse aangroei).
    `fee_rate` is proportioneel (0.00044 = 0,044% per transactie, op zowel buy als sell).
    """
    daily_cash_factor = (1 + cash_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR)

    p = prices.values
    idx = prices.index

    invested = True
    # initiele aankoop kost ook fee
    invested_amount = start_capital / (1 + fee_rate)
    shares = invested_amount / p[0]
    cash = 0.0

    equity = np.empty(len(p))
    pos = np.empty(len(p), dtype=np.int8)
    trades = 1  # initiele buy

    for i in range(len(p)):
        if not invested:
            cash *= daily_cash_factor

        if i >= lookback_trading_days:
            pct = p[i] / p[i - lookback_trading_days] - 1
            if invested and pct >= sell_threshold:
                proceeds = shares * p[i]
                cash = proceeds * (1 - fee_rate)
                shares = 0.0
                invested = False
                trades += 1
            elif not invested and pct <= buy_threshold:
                investable = cash / (1 + fee_rate)
                shares = investable / p[i]
                cash = 0.0
                invested = True
                trades += 1

        equity[i] = shares * p[i] + cash
        pos[i] = 1 if invested else 0

    return SimResult(
        name=f"Switch \u00b1{sell_threshold*100:.2f}% / {lookback_trading_days}d",
        equity=pd.Series(equity, index=idx),
        positions=pd.Series(pos, index=idx),
        trades=trades,
    )


def ratchet(
    prices: pd.Series,
    start_capital: float = 10_000.0,
    sell_threshold: float = 0.04,
    buy_threshold: float = 0.04,
    cash_rate_annual: float = 0.01,
    fee_rate: float = 0.0,
) -> SimResult:
    """Geen tijdvenster. Verkoop zodra koers >= (1+sell_th) * laatste_koopprijs.
    Koop terug zodra koers <= (1-buy_th) * laatste_verkoopprijs.
    """
    daily_cash_factor = (1 + cash_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR)

    p = prices.values
    idx = prices.index

    invested = True
    invested_amount = start_capital / (1 + fee_rate)
    shares = invested_amount / p[0]
    cash = 0.0
    last_buy = p[0]
    last_sell = None

    equity = np.empty(len(p))
    pos = np.empty(len(p), dtype=np.int8)
    trades = 1

    for i in range(len(p)):
        if not invested:
            cash *= daily_cash_factor

        if invested and p[i] >= last_buy * (1 + sell_threshold):
            proceeds = shares * p[i]
            cash = proceeds * (1 - fee_rate)
            shares = 0.0
            invested = False
            last_sell = p[i]
            trades += 1
        elif not invested and last_sell is not None and p[i] <= last_sell * (1 - buy_threshold):
            investable = cash / (1 + fee_rate)
            shares = investable / p[i]
            cash = 0.0
            invested = True
            last_buy = p[i]
            trades += 1

        equity[i] = shares * p[i] + cash
        pos[i] = 1 if invested else 0

    return SimResult(
        name=f"Ratchet \u00b1{sell_threshold*100:.2f}%",
        equity=pd.Series(equity, index=idx),
        positions=pd.Series(pos, index=idx),
        trades=trades,
    )


def overreaction(
    prices: pd.Series,
    start_capital: float = 10_000.0,
    sell_threshold: float = 0.02,   # dag-return waarboven we verkopen
    buy_threshold: float = -0.03,   # dag-return waaronder we kopen
    cash_rate_annual: float = 0.01,
    fee_rate: float = 0.0,
) -> SimResult:
    """Contrarian overreactie. Default fully invested.
    - Op dag met return >= sell_threshold: verkoop aan slot.
    - Op dag met return <= buy_threshold: koop aan slot.
    """
    daily_cash_factor = (1 + cash_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR)

    p = prices.values
    idx = prices.index

    invested = True
    invested_amount = start_capital / (1 + fee_rate)
    shares = invested_amount / p[0]
    cash = 0.0

    equity = np.empty(len(p))
    pos = np.empty(len(p), dtype=np.int8)
    trades = 1

    for i in range(len(p)):
        if not invested:
            cash *= daily_cash_factor

        if i > 0:
            day_ret = p[i] / p[i - 1] - 1
            if invested and day_ret >= sell_threshold:
                cash = shares * p[i] * (1 - fee_rate)
                shares = 0.0
                invested = False
                trades += 1
            elif not invested and day_ret <= buy_threshold:
                shares = (cash / (1 + fee_rate)) / p[i]
                cash = 0.0
                invested = True
                trades += 1

        equity[i] = shares * p[i] + cash
        pos[i] = 1 if invested else 0

    return SimResult(
        name=f"Overreact sell>+{sell_threshold*100:.1f}% buy<{buy_threshold*100:.1f}%",
        equity=pd.Series(equity, index=idx),
        positions=pd.Series(pos, index=idx),
        trades=trades,
    )
