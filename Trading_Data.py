"""Price data and technical indicators.

Real prices come from yfinance. When that fails, or for the demo pages,
a simulated snapshot is used instead. Indicators: RSI, MACD, ATR,
Bollinger Bands and SMA, plus a simple sentiment value from momentum.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import random


def compute_rsi(prices: List[float], period: int = 14) -> float:
    """RSI from closing prices, 0 to 100."""
    if len(prices) < period + 1:
        return 50.0

    gains, losses = [], []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gains.append(delta if delta > 0 else 0.0)
        losses.append(abs(delta) if delta < 0 else 0.0)

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    prices: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[float, float, float]:
    """MACD line, signal line and histogram."""
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0

    def ema(data: List[float], period: int) -> float:
        multiplier = 2.0 / (period + 1)
        result = data[0]
        for val in data[1:]:
            result = (val - result) * multiplier + result
        return result

    fast_ema = ema(prices[-fast:], fast)
    slow_ema = ema(prices[-slow:], slow)
    macd_line = fast_ema - slow_ema

    # the signal line needs MACD history, so this is a rough
    signal_line = macd_line * 0.8  # approximation
    histogram = macd_line - signal_line

    return round(macd_line, 6), round(signal_line, 6), round(histogram, 6)


def compute_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> float:
    """Average true range, scaled to 0-10."""
    if len(closes) < period + 1:
        return 4.5

    true_ranges = []
    for i in range(1, len(closes)):
        h, l, prev_c = highs[i], lows[i], closes[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        true_ranges.append(tr)

    atr = sum(true_ranges[-period:]) / period
    price = closes[-1] if closes else 100.0
    # scale to 0-10
    normalized = min(10.0, (atr / price) * 1000)
    return round(normalized, 3)


def compute_sma(prices: List[float], period: int = 20) -> float:
    """Simple moving average."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    return sum(prices[-period:]) / period


def compute_bollinger_bands(
    prices: List[float],
    period: int = 20,
    std_mult: float = 2.0,
) -> Tuple[float, float, float]:
    """Bollinger Bands as (lower, middle, upper)."""
    if len(prices) < period:
        return 0.0, 0.0, 0.0

    middle = compute_sma(prices, period)
    variance = sum((p - middle) ** 2 for p in prices[-period:]) / period
    std_dev = math.sqrt(variance)

    return (
        round(middle - std_mult * std_dev, 4),
        round(middle, 4),
        round(middle + std_mult * std_dev, 4),
    )


def get_real_market_snapshot(symbol: str = "BTC-USD", period: str = "60d") -> Dict:
    """Get real prices from yfinance and work out the indicators.
    Falls back to simulated data if the download fails.

    symbol is a Yahoo symbol such as BTC-USD or AAPL, period something like '60d'.
    """
    try:
        import yfinance as yf
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=period, interval="1d")

        if hist.empty or len(hist) < 35:
            raise ValueError(f"Insufficient data: {len(hist)} bars")

        closes = [float(x) for x in hist["Close"]]
        highs  = [float(x) for x in hist["High"]]
        lows   = [float(x) for x in hist["Low"]]

        rsi              = compute_rsi(closes)
        _, _, macd_hist_raw = compute_macd(closes)
        # MACD is in dollars, so for BTC it is huge. Use it as a percent of price
        # so it fits the fuzzy range of -5 to 5.
        current_price    = closes[-1] if closes[-1] > 0 else 1.0
        macd_hist        = round(macd_hist_raw / current_price * 100, 4)
        atr              = compute_atr(highs, lows, closes)
        sma_20           = compute_sma(closes, 20)
        bb_lower, bb_mid, bb_upper = compute_bollinger_bands(closes)

        # sentiment stand-in: 5-day price change
        price_change = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0.0
        sentiment    = max(-1.0, min(1.0, price_change * 20))

        return {
            "symbol":            symbol,
            "price":             round(closes[-1], 2),
            "rsi":               round(rsi,       2),
            "macd_hist":         round(macd_hist, 4),
            "atr":               round(atr,       3),
            "sma_20":            round(sma_20,    2),
            "bollinger_lower":   bb_lower,
            "bollinger_middle":  bb_mid,
            "bollinger_upper":   bb_upper,
            "sentiment":         round(sentiment, 4),
            "volatility":        round(atr * closes[-1] / 1000, 2),
            "source":            "yfinance",
            "bars_used":         len(closes),
        }

    except Exception as exc:
        # use the simulation instead
        snap = get_simulated_market_snapshot()
        snap["source"]  = f"simulated (yfinance error: {exc})"
        snap["symbol"]  = symbol
        return snap


def normalize_ticker(ticker: str) -> str:
    """Turn what the user typed into a yfinance symbol, e.g. BTC -> BTC-USD."""
    t = ticker.strip().upper()
    CRYPTO = {"BTC","ETH","BNB","SOL","XRP","ADA","DOGE","AVAX","DOT","MATIC"}
    if t in CRYPTO:
        return f"{t}-USD"
    return t


def get_ticker_snapshot(ticker: str, period: str = "90d") -> Dict:
    """Snapshot for any stock or crypto ticker. Adds an 'error' key on failure."""
    symbol = normalize_ticker(ticker)
    snap   = get_real_market_snapshot(symbol, period=period)
    snap["ticker"]         = ticker.upper()
    snap["ticker_symbol"]  = symbol
    return snap


def get_ohlcv_chart_data(ticker: str, period: str = "3mo") -> Dict:
    """Daily open/high/low/close for the chart, plus volume scaled to 0-100.

    Returns a dict with ticker, symbol, dates, open, high, low, close and volume.
    """
    symbol = normalize_ticker(ticker)
    try:
        import yfinance as yf
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        hist = yf.Ticker(symbol).history(period=period, interval="1d")
        if hist.empty:
            return {"error": f"No data for {symbol}"}

        max_vol = hist["Volume"].max() or 1
        return {
            "ticker":  ticker.upper(),
            "symbol":  symbol,
            "dates":   [str(d.date()) for d in hist.index],
            "close":   [round(float(v), 2) for v in hist["Close"]],
            "open":    [round(float(v), 2) for v in hist["Open"]],
            "high":    [round(float(v), 2) for v in hist["High"]],
            "low":     [round(float(v), 2) for v in hist["Low"]],
            "volume":  [round(float(v) / max_vol * 100, 1) for v in hist["Volume"]],
        }

    except Exception as exc:
        return {"error": str(exc)}


def get_simulated_market_snapshot(
    base_price: float = 64231.0,
) -> Dict:
    """Made-up but realistic snapshot (price, RSI, MACD, ATR, sentiment) for the demo pages."""
    # make up a price series.
    # it needs at least 35 bars, otherwise compute_macd() returns (0, 0, 0)
    random.seed()
    price = base_price + random.uniform(-500, 500)
    prices = [price + random.uniform(-200, 200) * math.sin(i * 0.3) for i in range(35)]
    highs = [p + random.uniform(50, 200) for p in prices]
    lows = [p - random.uniform(50, 200) for p in prices]

    rsi = compute_rsi(prices)
    _, _, macd_hist = compute_macd(prices)
    atr = compute_atr(highs, lows, prices)
    sma_20 = compute_sma(prices, 20)
    bb_lower, bb_mid, bb_upper = compute_bollinger_bands(prices)

    # fake sentiment, nudged by the price move
    price_change = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0.0
    sentiment = max(-1.0, min(1.0, price_change * 15 + random.uniform(-0.2, 0.2)))

    return {
        "symbol": "BTC/USDT",
        "price": round(price, 2),
        "rsi": round(rsi, 2),
        "macd_hist": round(macd_hist, 4),
        "atr": round(atr, 3),
        "sma_20": round(sma_20, 2),
        "bollinger_lower": bb_lower,
        "bollinger_middle": bb_mid,
        "bollinger_upper": bb_upper,
        "sentiment": round(sentiment, 4),
        "volatility": round(random.uniform(600, 950), 2),
    }