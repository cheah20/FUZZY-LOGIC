"""Works out the market regime the way a trader would, before the fuzzy engine runs.

Regimes: news driven bearish, news driven bullish, technical driven,
conflicting signals, high uncertainty and normal. The rules come from
technical analysis books and common trading habits such as protecting
capital when the news is very bad.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple


REGIME_NEWS_DRIVEN_BEARISH = "NEWS_DRIVEN_BEARISH"
REGIME_NEWS_DRIVEN_BULLISH = "NEWS_DRIVEN_BULLISH"
REGIME_TECHNICAL_DRIVEN = "TECHNICAL_DRIVEN"
REGIME_CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
REGIME_HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
REGIME_NORMAL = "NORMAL"

# thresholds, change these to tune the rules
SENTIMENT_EXTREME_BEARISH = -0.65   # below this: panic
SENTIMENT_EXTREME_BULLISH = 0.65    # above this: euphoria
SENTIMENT_NEUTRAL_LO = -0.15        # neutral band, low end
SENTIMENT_NEUTRAL_HI = 0.15         # neutral band, high end
RSI_OVERSOLD = 30                   # usual oversold level
RSI_OVERBOUGHT = 70                 # usual overbought level
MACD_BULLISH_THRESHOLD = 0.2        # MACD histogram above this is bullish
MACD_BEARISH_THRESHOLD = -0.2       # MACD histogram below this is bearish
ATR_HIGH_THRESHOLD = 7.0            # ATR above this counts as high


def _rule_extreme_bearish_news(
    sentiment: float,
) -> Tuple[Optional[str], float, str]:
    """Rule 1: very bad news. Traders protect capital and ignore bounce signals."""
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
    """Rule 2: very good news. Sentiment wins, even over an overbought RSI."""
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
    """Rule 3: no strong news, so traders go back to RSI and MACD."""
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
    """Rule 4: RSI and MACD disagree. Lower the confidence and avoid big entries."""
    # RSI oversold says buy, MACD bearish says sell
    rsi_buy = rsi < RSI_OVERSOLD
    rsi_sell = rsi > RSI_OVERBOUGHT
    macd_bullish = macd_hist > MACD_BULLISH_THRESHOLD
    macd_bearish = macd_hist < MACD_BEARISH_THRESHOLD

    conflict = (rsi_buy and macd_bearish) or (rsi_sell and macd_bullish)

    if conflict and not (sentiment <= SENTIMENT_EXTREME_BEARISH or sentiment >= SENTIMENT_EXTREME_BULLISH):
        # the further apart they are, the stronger the conflict
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
    """Rule 5: ATR spike. Very uncertain market, so every input gets the same weight."""
    if atr is not None and atr >= ATR_HIGH_THRESHOLD:
        conf = min(1.0, 0.7 + (atr - 7.0) / 3.0 * 0.3)
        # if sentiment is also extreme, the news rules go first
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
    """Rule 6: nothing unusual, default weights."""
    # only when nothing extreme is going on
    if not (sentiment <= SENTIMENT_EXTREME_BEARISH or sentiment >= SENTIMENT_EXTREME_BULLISH):
        return (
            REGIME_NORMAL,
            0.70,
            "Balanced market conditions. All features contribute with "
            "standard weightage. Technical indicators slightly favored."
        )
    return None, 0.0, ""


def detect_market_regime(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """Detect the market regime.

    Rules are checked in this order and the first match wins: bad news, good
    news, high uncertainty, conflicting signals, technical driven, normal.

    Returns regime, confidence (0-1), a reason for the UI and the rule that fired.

    >>> detect_market_regime(rsi=25, macd_hist=0.8, sentiment=-0.9)["regime"]
    'NEWS_DRIVEN_BEARISH'
    """
    # check inputs
    rsi = max(0.0, min(100.0, float(rsi)))
    macd_hist = max(-5.0, min(5.0, float(macd_hist)))
    sentiment = max(-1.0, min(1.0, float(sentiment)))
    if atr is not None:
        atr = max(0.0, min(10.0, float(atr)))

    # rules in priority order
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

    # should not get here
    return {
        "regime": REGIME_NORMAL,
        "confidence": 0.50,
        "reason": "Fallback: No expert rule matched. Defaulting to normal balanced state.",
        "triggered_rule": "FALLBACK",
    }


# quick check

def get_expert_layer_info() -> Dict:
    """Info about the expert layer for the UI."""
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