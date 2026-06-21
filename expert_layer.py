"""
expert_layer.py ─ Human Expert Heuristic Layer
ASTDSS Terminal — Neuro-Symbolic Trading DSS

Purpose:
  Captures human expert trading knowledge to detect the current
  market regime. This layer emulates how professional traders
  selectively prioritize information under different conditions.

Market Regimes Detected:
  - NEWS_DRIVEN_BEARISH   → Extreme negative sentiment dominates
  - NEWS_DRIVEN_BULLISH   → Extreme positive sentiment dominates
  - TECHNICAL_DRIVEN      → Neutral sentiment, rely on indicators
  - CONFLICTING_SIGNALS   → RSI & MACD disagree
  - HIGH_UNCERTAINTY      → Extreme volatility / high ATR
  - NORMAL                → Default balanced state

Rule Sources (see ASTDSS_FUNCTION_CODING_PLAN.md §23-30):
  1. Financial trading literature (fuzzy DSS, neuro-symbolic AI)
  2. Technical analysis theory (RSI, MACD, ATR principles)
  3. Practitioner heuristics (panic logic, uncertainty avoidance,
     news dominance, capital preservation behavior)
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

# Regime Type Definitions

REGIME_NEWS_DRIVEN_BEARISH = "NEWS_DRIVEN_BEARISH"
REGIME_NEWS_DRIVEN_BULLISH = "NEWS_DRIVEN_BULLISH"
REGIME_TECHNICAL_DRIVEN = "TECHNICAL_DRIVEN"
REGIME_CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
REGIME_HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
REGIME_NORMAL = "NORMAL"

# Thresholds (calibratable)
SENTIMENT_EXTREME_BEARISH = -0.65   # Below this → panic mode
SENTIMENT_EXTREME_BULLISH = 0.65    # Above this → euphoria mode
SENTIMENT_NEUTRAL_LO = -0.15        # Neutral band lower
SENTIMENT_NEUTRAL_HI = 0.15         # Neutral band upper
RSI_OVERSOLD = 30                   # Traditional oversold threshold
RSI_OVERBOUGHT = 70                 # Traditional overbought threshold
MACD_BULLISH_THRESHOLD = 0.2        # MACD histogram bullish
MACD_BEARISH_THRESHOLD = -0.2       # MACD histogram bearish
ATR_HIGH_THRESHOLD = 7.0            # ATR normalised high threshold


# Individual Expert Rules

def _rule_extreme_bearish_news(
    sentiment: float,
) -> Tuple[Optional[str], float, str]:
    """
    Rule 1 — Extreme Bearish News
    
    Practitioner heuristic:
      During extreme negative sentiment (e.g. black swan events,
      regulatory crackdowns, macroeconomic shocks), human traders
      prioritize capital preservation over technical bounce signals.
    
    Source: Financial trading psychology, risk management literature.
    """
    if sentiment <= SENTIMENT_EXTREME_BEARISH:
        conf = min(1.0, abs(sentiment + 0.65) / 0.35 * 0.5 + 0.7)
        return (
            REGIME_NEWS_DRIVEN_BEARISH,
            round(conf, 3),
            "Extreme negative sentiment detected. Human expert rule prioritizes "
            "sentiment over technical indicators to protect capital. "
            "Technical bounce signals (RSI oversold) are suppressed during panic conditions."
        )
    return None, 0.0, ""


def _rule_extreme_bullish_news(
    sentiment: float,
) -> Tuple[Optional[str], float, str]:
    """
    Rule 2 — Extreme Bullish News
    
    Practitioner heuristic:
      During extreme positive news (e.g. major ETF approval, landmark
      adoption), sentiment dominates and technical overbought warnings
      may be temporarily invalidated by market euphoria.
    """
    if sentiment >= SENTIMENT_EXTREME_BULLISH:
        conf = min(1.0, abs(sentiment - 0.65) / 0.35 * 0.5 + 0.7)
        return (
            REGIME_NEWS_DRIVEN_BULLISH,
            round(conf, 3),
            "Extreme positive sentiment detected. Human expert prioritizes "
            "sentiment-driven momentum over technical indicators. "
            "Overbought RSI warnings are temporarily suppressed."
        )
    return None, 0.0, ""


def _rule_technical_driven(
    sentiment: float,
) -> Tuple[Optional[str], float, str]:
    """
    Rule 3 — Technical-Driven Market
    
    When sentiment is neutral (no strong news direction), human
    traders typically revert to technical analysis. This is the
    standard mode where RSI and MACD carry full weight.
    
    Source: Technical analysis theory, academic trading literature.
    """
    if SENTIMENT_NEUTRAL_LO <= sentiment <= SENTIMENT_NEUTRAL_HI:
        conf = 0.75 + (0.25 * (1.0 - abs(sentiment) / 0.15))
        return (
            REGIME_TECHNICAL_DRIVEN,
            round(min(1.0, conf), 3),
            "Neutral sentiment environment. Human expert delegates to "
            "technical indicators (RSI, MACD) as primary signals. "
            "News sentiment is suppressed to avoid noise."
        )
    return None, 0.0, ""


def _rule_conflicting_signals(
    rsi: float,
    macd_hist: float,
    sentiment: float,
) -> Tuple[Optional[str], float, str]:
    """
    Rule 4 — Conflicting Signals
    
    Practitioner heuristic:
      When RSI says "buy" but MACD says "sell" (or vice versa),
      experienced traders reduce confidence and avoid aggressive entry.
      This is NOT the time for large positions.
    
    Source: Multi-indicator trading methodology, risk management.
    """
    # RSI oversold = buy signal, MACD bearish = sell signal → conflict
    rsi_buy = rsi < RSI_OVERSOLD
    rsi_sell = rsi > RSI_OVERBOUGHT
    macd_bullish = macd_hist > MACD_BULLISH_THRESHOLD
    macd_bearish = macd_hist < MACD_BEARISH_THRESHOLD

    conflict = (rsi_buy and macd_bearish) or (rsi_sell and macd_bullish)

    if conflict and not (sentiment <= SENTIMENT_EXTREME_BEARISH or sentiment >= SENTIMENT_EXTREME_BULLISH):
        # Strength of conflict based on how far apart the signals are
        conflict_strength = min(1.0, (abs(rsi - 50) / 30 + abs(macd_hist) / 2.0) * 0.5)
        conf = 0.65 + conflict_strength * 0.3
        return (
            REGIME_CONFLICTING_SIGNALS,
            round(conf, 3),
            "RSI and MACD indicators are in disagreement. "
            "Human expert reduces confidence to avoid whipsaw. "
            "All features receive partial weight for cautious assessment."
        )
    return None, 0.0, ""


def _rule_high_uncertainty(
    atr: Optional[float],
    sentiment: float,
) -> Tuple[Optional[str], float, str]:
    """
    Rule 5 — High Uncertainty (Volatility Spike)
    
    Practitioner heuristic:
      When ATR spikes dramatically, the market enters a high-uncertainty
      regime. Traders typically reduce position sizes or stay flat.
      Equal weighting is applied to avoid over-reliance on any single signal.
    
    Source: Volatility-based position sizing, institutional risk management.
    """
    if atr is not None and atr >= ATR_HIGH_THRESHOLD:
        conf = min(1.0, 0.7 + (atr - 7.0) / 3.0 * 0.3)
        # If also extreme sentiment, let sentiment rules take priority
        if sentiment <= SENTIMENT_EXTREME_BEARISH or sentiment >= SENTIMENT_EXTREME_BULLISH:
            return None, 0.0, ""
        return (
            REGIME_HIGH_UNCERTAINTY,
            round(conf, 3),
            "Elevated ATR indicates extreme market uncertainty. "
            "Human expert applies equal weighting across all features. "
            "Position sizing should be reduced. Conservative Hold/Reduce bias applied."
        )
    return None, 0.0, ""


def _rule_normal_market(
    sentiment: float,
) -> Tuple[Optional[str], float, str]:
    """
    Rule 6 — Normal Market (default)
    
    When no extreme conditions are detected, the market is in a normal
    balanced state. All features contribute with default weights.
    """
    # Only fire if not in any extreme
    if not (sentiment <= SENTIMENT_EXTREME_BEARISH or sentiment >= SENTIMENT_EXTREME_BULLISH):
        return (
            REGIME_NORMAL,
            0.70,
            "Balanced market conditions. All features contribute with "
            "standard weightage. Technical indicators slightly favored."
        )
    return None, 0.0, ""


# Main Detection Function

def detect_market_regime(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """
    Detect the current market regime using human expert heuristics.
    
    Priority order (higher rules override lower ones):
      1. Extreme Bearish News  (capital preservation)
      2. Extreme Bullish News  (momentum prioritization)
      3. High Uncertainty      (volatility-driven caution)
      4. Conflicting Signals   (indicator disagreement)
      5. Technical Driven      (neutral sentiment → technical focus)
      6. Normal                (default balanced state)
    
    Args:
        rsi: RSI value (0–100)
        macd_hist: MACD histogram value
        sentiment: Sentiment score (-1 to +1)
        atr: Optional ATR normalised value (0–10)
    
    Returns:
        {
            "regime": str,          # One of the six regime types
            "confidence": float,    # 0.0–1.0 confidence in detection
            "reason": str,          # Human-readable explanation
            "triggered_rule": str,  # Which expert rule fired
        }
    
    Example:
        >>> detect_market_regime(rsi=25, macd_hist=0.8, sentiment=-0.9)
        {
            "regime": "NEWS_DRIVEN_BEARISH",
            "confidence": 0.92,
            "reason": "Extreme negative sentiment detected...",
            "triggered_rule": "RULE_1_EXTREME_BEARISH_NEWS"
        }
    """
    # Validate inputs
    rsi = max(0.0, min(100.0, float(rsi)))
    macd_hist = max(-5.0, min(5.0, float(macd_hist)))
    sentiment = max(-1.0, min(1.0, float(sentiment)))
    if atr is not None:
        atr = max(0.0, min(10.0, float(atr)))

    # Evaluate rules in priority order
    rules = [
        ("RULE_1_EXTREME_BEARISH_NEWS", _rule_extreme_bearish_news(sentiment)),
        ("RULE_2_EXTREME_BULLISH_NEWS", _rule_extreme_bullish_news(sentiment)),
        ("RULE_5_HIGH_UNCERTAINTY", _rule_high_uncertainty(atr, sentiment)),
        ("RULE_4_CONFLICTING_SIGNALS", _rule_conflicting_signals(rsi, macd_hist, sentiment)),
        ("RULE_3_TECHNICAL_DRIVEN", _rule_technical_driven(sentiment)),
        ("RULE_6_NORMAL", _rule_normal_market(sentiment)),
    ]

    for rule_name, (regime, confidence, reason) in rules:
        if regime is not None:
            return {
                "regime": regime,
                "confidence": confidence,
                "reason": reason,
                "triggered_rule": rule_name,
            }

    # Absolute fallback (should never reach here)
    return {
        "regime": REGIME_NORMAL,
        "confidence": 0.50,
        "reason": "Fallback: No expert rule matched. Defaulting to normal balanced state.",
        "triggered_rule": "FALLBACK",
    }


# Quick diagnostic helper

def get_expert_layer_info() -> Dict:
    """Return metadata about the expert layer for UI display."""
    return {
        "name": "Human Expert Heuristic Layer",
        "version": "1.0.0",
        "rules": [
            {
                "id": "RULE_1",
                "name": "Extreme Bearish News",
                "description": "Sentiment ≤ -0.65 → NEWS_DRIVEN_BEARISH",
                "priority": 1,
                "action": "100% sentiment; RSI & MACD masked",
            },
            {
                "id": "RULE_2",
                "name": "Extreme Bullish News",
                "description": "Sentiment ≥ 0.65 → NEWS_DRIVEN_BULLISH",
                "priority": 2,
                "action": "100% sentiment; RSI & MACD masked",
            },
            {
                "id": "RULE_3",
                "name": "Technical-Driven Market",
                "description": "Sentiment neutral → TECHNICAL_DRIVEN",
                "priority": 5,
                "action": "RSI 60%, MACD 40%; Sentiment masked",
            },
            {
                "id": "RULE_4",
                "name": "Conflicting Signals",
                "description": "RSI vs MACD disagreement → CONFLICTING_SIGNALS",
                "priority": 4,
                "action": "Partial weight on all features; bias toward Hold",
            },
            {
                "id": "RULE_5",
                "name": "High Uncertainty",
                "description": "ATR ≥ 7.0 → HIGH_UNCERTAINTY",
                "priority": 3,
                "action": "All features at 25% equal weight; Hold/Reduce bias",
            },
            {
                "id": "RULE_6",
                "name": "Normal Market",
                "description": "Default → NORMAL",
                "priority": 6,
                "action": "RSI 40%, MACD 30%, Sentiment 30%",
            },
        ],
        "reference_sources": [
            "Financial trading literature (fuzzy DSS, expert systems)",
            "Technical analysis theory (RSI, MACD, ATR principles)",
            "Practitioner heuristics (panic logic, uncertainty avoidance)",
            "Institutional risk management behavior",
            "Neuro-symbolic AI architecture patterns",
        ],
    }