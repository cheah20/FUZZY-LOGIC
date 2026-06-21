"""
Trading_Data.py ─ Market Data Layer
ASTDSS Terminal — Neuro-Symbolic Trading DSS

Purpose:
  Data-driven layer responsible for fetching market data and
  computing technical indicators.

Provides:
  - Simulated OHLCV data for development/demo
  - RSI, MACD, ATR, Bollinger Bands, SMA computation
  - Sentiment score extraction from news feed

In production, this module would connect to:
  - Binance WebSocket for real-time OHLCV
  - Bloomberg / Reuters APIs
  - News sentiment APIs

For now it provides realistic simulated data for UI development.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import random

# Technical Indicator Computation

def compute_rsi(prices: List[float], period: int = 14) -> float:
    """
    Compute Relative Strength Index.
    
    RSI = 100 - (100 / (1 + avg_gain / avg_loss))
    
    Returns 0–100 value.
    """
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
    """
    Compute MACD line, signal line, and histogram.
    
    Returns: (macd_line, signal_line, histogram)
    """
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

    # For signal line we'd need MACD history; simplified here
    signal_line = macd_line * 0.8  # approximation
    histogram = macd_line - signal_line

    return round(macd_line, 6), round(signal_line, 6), round(histogram, 6)


def compute_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> float:
    """Compute Average True Range (normalised 0–10)."""
    if len(closes) < period + 1:
        return 4.5

    true_ranges = []
    for i in range(1, len(closes)):
        h, l, prev_c = highs[i], lows[i], closes[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        true_ranges.append(tr)

    atr = sum(true_ranges[-period:]) / period
    price = closes[-1] if closes else 100.0
    # Normalize to 0–10
    normalized = min(10.0, (atr / price) * 1000)
    return round(normalized, 3)


def compute_sma(prices: List[float], period: int = 20) -> float:
    """Compute Simple Moving Average."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    return sum(prices[-period:]) / period


def compute_bollinger_bands(
    prices: List[float],
    period: int = 20,
    std_mult: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Compute Bollinger Bands.
    Returns: (lower, middle, upper)
    """
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


# Real Market Data via yfinance (PDF §20)

def get_real_market_snapshot(symbol: str = "BTC-USD", period: str = "60d") -> Dict:
    """
    Fetch real market data using yfinance and compute indicators.
    Falls back to simulated data if yfinance fails.

    Args:
        symbol: Yahoo Finance ticker symbol (BTC-USD, AAPL, etc.)
        period: lookback period string (e.g. '60d', '90d')
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
        # Normalize MACD histogram to percentage of price so it fits FLC [-5, 5].
        # Raw MACD for high-priced assets (BTC ~$67k) is in USD, not normalized.
        current_price    = closes[-1] if closes[-1] > 0 else 1.0
        macd_hist        = round(macd_hist_raw / current_price * 100, 4)
        atr              = compute_atr(highs, lows, closes)
        sma_20           = compute_sma(closes, 20)
        bb_lower, bb_mid, bb_upper = compute_bollinger_bands(closes)

        # Sentiment proxy from 5-day price momentum
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
        # Fall back to simulation silently
        snap = get_simulated_market_snapshot()
        snap["source"]  = f"simulated (yfinance error: {exc})"
        snap["symbol"]  = symbol
        return snap


def normalize_ticker(ticker: str) -> str:
    """Normalise user input to a valid yfinance symbol.
    BTC -> BTC-USD, ETH -> ETH-USD, AAPL -> AAPL, etc.
    """
    t = ticker.strip().upper()
    CRYPTO = {"BTC","ETH","BNB","SOL","XRP","ADA","DOGE","AVAX","DOT","MATIC"}
    if t in CRYPTO:
        return f"{t}-USD"
    return t


def get_ticker_snapshot(ticker: str, period: str = "90d") -> Dict:
    """
    Get market snapshot for any ticker (stocks or crypto).
    Wraps get_real_market_snapshot with ticker normalisation.
    Returns the snapshot dict plus an 'error' key on failure.
    """
    symbol = normalize_ticker(ticker)
    snap   = get_real_market_snapshot(symbol, period=period)
    snap["ticker"]         = ticker.upper()
    snap["ticker_symbol"]  = symbol
    return snap


def get_ohlcv_chart_data(ticker: str, period: str = "3mo") -> Dict:
    """
    Return OHLCV series suitable for front-end chart rendering.
    All price values are rounded; volume is normalised 0-100.

    Returns:
        {
            "ticker":  "AAPL",
            "dates":   ["2024-01-02", ...],
            "close":   [186.2, ...],
            "open":    [...],
            "high":    [...],
            "low":     [...],
            "volume":  [45, 72, ...]   # normalised 0-100
        }
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


# Simulated Market Data (for development)

def get_simulated_market_snapshot(
    base_price: float = 64231.0,
) -> Dict:
    """
    Generate a realistic simulated market snapshot for UI development.
    
    Returns current price, RSI, MACD, ATR, sentiment, etc.
    """
    # Generate a pseudo price series.
    # Needs >= 35 bars or compute_macd() silently returns (0, 0, 0) — see backtester.py.
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

    # Simulated sentiment (biased by price movement)
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