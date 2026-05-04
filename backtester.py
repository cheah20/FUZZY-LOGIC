"""
backtester.py
=============
In proposal la, I foollow and do this module as the Historical Backtesting Engine.

i set this rules(i think this is the best way to do it, if got any update just inform me or do a comment):
Role in the project: QA & Backtester 
Purpose:
  This module evaluates how the FLC-based trading strategy would have performed
  on historical market data. It uses a train/test split methodology:
    - 70% of data: calibration period (used to observe and tune the system)
    - 30% of data: test period (simulated trading, performance is measured here)

Performance metrics reported:
  - Total Return        : Overall profit or loss as a percentage
  - Win Rate            : Percentage of completed trades that were profitable
  - Max Drawdown        : Worst peak-to-trough decline during the test period
  - Sharpe Ratio        : Risk-adjusted return (higher is better)
  - Buy & Hold Return   : Benchmark — what if we just bought and held instead?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Trading_Data import get_stock_data
from fuzzy_engine import compute_score


#Settings
# These can be overridden from interface.py when running the interactive backtest.

TICKER         = "AAPL"
INITIAL_CASH   = 10_000.0   # Starting capital in USD (or your guys need use RM???)
TRAIN_RATIO    = 0.70       # 70% calibration, 30% test (or 6 for cali and 40 test???)

# FLC signal thresholds  is  for the score at which we enter or exit a position.
BUY_THRESHOLD  = 3.5   # FLC score >= this value triggers a BUY
SELL_THRESHOLD = 2.5   # FLC score <= this value triggers a SELL
# Scores between the two thresholds result in no action (HOLD)


#Data Loading 

def load_and_split(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    here i set for 5 years only(but your guys need 20year/10year also can, i just for try only):
    Downloads 5 years of historical OHLCV data and splits it into
    a calibration set (70%) and a test set (30%).

    We use only the last 5 years 
    so the backtest reflects recent market conditions rather than historical regimes that may no longer be relevant.
    """
    print(f"\n[DATA] Downloading historical data for {ticker}...")
    df = get_stock_data(ticker)

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = df.tail(1260).copy()

    # Remove the initial rows where RSI and MACD cannot yet be computed
    macd_col = next((c for c in df.columns if c.startswith("MACDh_")), None)
    df = df.dropna(subset=["RSI", macd_col])

    split_idx = int(len(df) * TRAIN_RATIO)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    print(f"[DATA] Total rows : {len(df)}")
    print(f"[DATA] Train set  : {len(train_df)} rows  ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"[DATA] Test set   : {len(test_df)} rows  ({test_df.index[0].date()} to {test_df.index[-1].date()})")

    return train_df, test_df


#Signal Generation
# The compute_score function from fuzzy_engine.py is used to calculate the FLC score and corresponding label for each day in the test set.
def generate_signals(test_df: pd.DataFrame) -> pd.DataFrame:
    macd_col = next((c for c in test_df.columns if c.startswith("MACDh_")), None)

    scores, labels, actions = [], [], []
    print("\n[FLC] Computing signals for test set...")

    for date, row in test_df.iterrows():
        rsi    = float(row["RSI"])
        macd_h = float(row[macd_col]) if macd_col else 0.0

        result = compute_score(rsi, macd_h, sentiment=0.0)
        score  = result["score"]
        label  = result["label"]

        if score >= BUY_THRESHOLD:
            action = "BUY"
        elif score <= SELL_THRESHOLD:
            action = "SELL"
        else:
            action = "HOLD"

        scores.append(score)
        labels.append(label)
        actions.append(action)

    df             = test_df.copy()
    df["flc_score"] = scores
    df["flc_label"] = labels
    df["action"]    = actions

    counts = df["action"].value_counts()
    print(f"[FLC] BUY signals  : {counts.get('BUY',  0)}")
    print(f"[FLC] SELL signals : {counts.get('SELL', 0)}")
    print(f"[FLC] HOLD signals : {counts.get('HOLD', 0)}")

    return df


#Portfolio Simulation

def simulate_portfolio(signals_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    i setting:

    Trading rules:
      BUY  — Invest all available cash into shares (whole shares only)
      SELL — Liquidate the entire position
      HOLD — Take no action
    """
    cash    = INITIAL_CASH
    shares  = 0
    portfolio_values = []
    trades  = []        # Records (buy_price, sell_price) for each completed round trip
    buy_price = None

    print("\n[SIM] Running portfolio simulation...")

    for date, row in signals_df.iterrows():
        price  = float(row["Close"])
        action = row["action"]

        if action == "BUY" and shares == 0:
            shares_to_buy = int(cash // price)
            if shares_to_buy > 0:
                cash     -= shares_to_buy * price
                shares    = shares_to_buy
                buy_price = price

        elif action == "SELL" and shares > 0:
            cash += shares * price
            if buy_price is not None:
                trades.append((buy_price, price))
            shares    = 0
            buy_price = None

        portfolio_values.append(cash + shares * price)

    # Force-close any open position at the last day's price
    if shares > 0:
        last_price = float(signals_df.iloc[-1]["Close"])
        cash      += shares * last_price
        if buy_price is not None:
            trades.append((buy_price, last_price))
        portfolio_values[-1] = cash

    result_df = signals_df.copy()
    result_df["portfolio_value"] = portfolio_values

    print(f"[SIM] Total trades executed : {len(trades)}")
    print(f"[SIM] Final portfolio value : ${portfolio_values[-1]:,.2f}")

    return result_df, trades


#Performance Metrics

def calculate_metrics(result_df: pd.DataFrame, trades: list) -> dict:
    # This function calculates the key performance metrics for the backtest, including total return, win rate, maximum drawdown, and Sharpe ratio. 
    # It also computes the buy-and-hold return for comparison.
    portfolio = result_df["portfolio_value"]
    final_val = portfolio.iloc[-1]

    # for total return, we compare the final portfolio value to the initial cash.
    total_return_pct = (final_val - INITIAL_CASH) / INITIAL_CASH * 100

    # for win rate, we look at the list of completed trades (buy/sell pairs)
    # calculate the percentage that were profitable.
    if trades:
        wins     = sum(1 for buy, sell in trades if sell > buy)
        win_rate = wins / len(trades) * 100
    else:
        wins, win_rate = 0, 0.0

    # for max drawdown and calculate the rolling maximum of the portfolio value 
    # find the largest percentage drop from any peak to the subsequent trough.
    rolling_max  = portfolio.cummax()
    drawdown     = (portfolio - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

   # for sharpe ratio calculate the daily returns of the portfolio, subtract a risk-free rate
   # divide by the standard deviation of returns to get a risk-adjusted performance measure.
    daily_returns = portfolio.pct_change().dropna()
    risk_free_rate_daily = 0.05 / 252   # 5% annual risk-free rate
    excess_returns = daily_returns - risk_free_rate_daily
    if excess_returns.std() > 0:
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Buy & Hold baseline: benchmark return if we had simply bought at the
    # start of the test period and held until the end without any trading.
    start_price = float(result_df["Close"].iloc[0])
    end_price   = float(result_df["Close"].iloc[-1])
    bh_shares   = int(INITIAL_CASH // start_price)
    bh_leftover = INITIAL_CASH - bh_shares * start_price
    bh_final    = bh_shares * end_price + bh_leftover
    bh_return   = (bh_final - INITIAL_CASH) / INITIAL_CASH * 100

    return {
        "final_value"  : round(final_val, 2),
        "total_return" : round(total_return_pct, 2),
        "win_rate"     : round(win_rate, 2),
        "total_trades" : len(trades),
        "wins"         : wins,
        "max_drawdown" : round(max_drawdown, 2),
        "sharpe_ratio" : round(sharpe, 3),
        "bh_final_value": round(bh_final, 2),
        "bh_return"    : round(bh_return, 2),
    }


#Report

def print_report(metrics: dict, ticker: str):
    print("\n" + "=" * 55)
    print(f"  BACKTEST REPORT — {ticker}")
    print("=" * 55)
    print(f"  Starting Capital     : ${INITIAL_CASH:>10,.2f}")
    print(f"  Final Value (FLC)    : ${metrics['final_value']:>10,.2f}")
    print(f"  Total Return (FLC)   : {metrics['total_return']:>+9.2f}%")
    print(f"  Buy & Hold Return    : {metrics['bh_return']:>+9.2f}%")
    print("-" * 55)
    print(f"  Total Trades         : {metrics['total_trades']:>10}")
    print(f"  Winning Trades       : {metrics['wins']:>10}")
    print(f"  Win Rate             : {metrics['win_rate']:>9.1f}%")
    print(f"  Max Drawdown         : {metrics['max_drawdown']:>9.2f}%")
    print(f"  Sharpe Ratio         : {metrics['sharpe_ratio']:>10.3f}")
    print("=" * 55)
    diff = metrics["total_return"] - metrics["bh_return"]
    if diff > 0:
        print(f"  FLC OUTPERFORMED Buy & Hold by {diff:.2f}%")
    else:
        print(f"  FLC UNDERPERFORMED Buy & Hold by {abs(diff):.2f}%")
    print("=" * 55 + "\n")


# Chart 
def plot_results(result_df: pd.DataFrame, metrics: dict, ticker: str):
    """Generates a two-panel chart: portfolio value and FLC score over time."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    dates = result_df.index

    ax1.plot(dates, result_df["portfolio_value"],
             color="cyan", linewidth=1.5, label="FLC Strategy")

    start_price = float(result_df["Close"].iloc[0])
    bh_shares   = int(INITIAL_CASH // start_price)
    bh_leftover = INITIAL_CASH - bh_shares * start_price
    bh_curve    = result_df["Close"] * bh_shares + bh_leftover
    ax1.plot(dates, bh_curve, color="orange", linewidth=1.5,
             linestyle="--", label="Buy & Hold")

    ax1.axhline(INITIAL_CASH, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Portfolio Value (USD)")
    ax1.set_title(
        f"{ticker} — FLC Strategy vs Buy & Hold  |  "
        f"FLC: {metrics['total_return']:+.1f}%  |  "
        f"B&H: {metrics['bh_return']:+.1f}%"
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    buy_dates  = result_df[result_df["action"] == "BUY"].index
    sell_dates = result_df[result_df["action"] == "SELL"].index
    ax1.scatter(buy_dates,  result_df.loc[buy_dates,  "portfolio_value"],
                marker="^", color="lime", s=60, zorder=5, label="BUY")
    ax1.scatter(sell_dates, result_df.loc[sell_dates, "portfolio_value"],
                marker="v", color="red",  s=60, zorder=5, label="SELL")

    ax2.plot(dates, result_df["flc_score"], color="magenta",
             linewidth=1, label="FLC Score")
    ax2.axhline(BUY_THRESHOLD,  color="lime", linewidth=0.8, linestyle="--",
                label=f"Buy >= {BUY_THRESHOLD}")
    ax2.axhline(SELL_THRESHOLD, color="red",  linewidth=0.8, linestyle="--",
                label=f"Sell <= {SELL_THRESHOLD}")
    ax2.axhline(3.0, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_ylim(1, 5)
    ax2.set_ylabel("FLC Score (1–5)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    fig.tight_layout()

    output_path = f"backtest_{ticker}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Chart saved to {output_path}")
    plt.show()


# Main
def run_backtest(ticker: str = TICKER) -> dict:
    # runs the entire backtesting process for a given ticker: data loading, signal generation, portfolio simulation, metric calculation, and reporting. It returns the performance metrics as a dictionary for potential use in the UI or further analysis.
    train_df, test_df = load_and_split(ticker)
    signals_df        = generate_signals(test_df)
    result_df, trades = simulate_portfolio(signals_df)
    metrics           = calculate_metrics(result_df, trades)
    print_report(metrics, ticker)
    plot_results(result_df, metrics, ticker)
    return metrics


if __name__ == "__main__":
    run_backtest("AAPL")
