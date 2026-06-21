"""
feature_selector.py ─ Dynamic Feature Masking
ASTDSS Terminal — Neuro-Symbolic Trading DSS

Purpose:
  Decides which input features should be active based on the
  detected market regime. This directly addresses the supervisor's
  comment: "Not every time 4 features used — maybe sometimes
  input only needs 1 feature."

Provides:
  - select_features()     → Returns active/masked features + weights
  - get_feature_selection_info() → Metadata for UI display

Weightage Design (from ASTDSS_FUNCTION_CODING_PLAN.md §7):
  NEWS_DRIVEN_BEARISH  → Sentiment 100%, RSI 0%, MACD 0%, ATR 0%
  NEWS_DRIVEN_BULLISH  → Sentiment 100%, RSI 0%, MACD 0%, ATR 0%
  TECHNICAL_DRIVEN     → RSI 60%, MACD 40%, Sentiment 0%
  CONFLICTING_SIGNALS  → RSI 30%, MACD 30%, Sentiment 20%, ATR 20%
  HIGH_UNCERTAINTY     → Equal 25% each
  NORMAL               → RSI 40%, MACD 30%, Sentiment 30%
"""

from __future__ import annotations
from typing import Dict, List, Optional

# Import regime constants from expert_layer
from expert_layer import (
    REGIME_NEWS_DRIVEN_BEARISH,
    REGIME_NEWS_DRIVEN_BULLISH,
    REGIME_TECHNICAL_DRIVEN,
    REGIME_CONFLICTING_SIGNALS,
    REGIME_HIGH_UNCERTAINTY,
    REGIME_NORMAL,
)

# Feature Weightage per Regime

# Each entry: {feature_name: weight (0.0–1.0)}
# Features with weight 0.0 are MASKED (inactive)

REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    REGIME_NEWS_DRIVEN_BEARISH: {
        "sentiment": 1.0,
        "rsi": 0.0,
        "macd_hist": 0.0,
        "atr": 0.0,
    },
    REGIME_NEWS_DRIVEN_BULLISH: {
        "sentiment": 1.0,
        "rsi": 0.0,
        "macd_hist": 0.0,
        "atr": 0.0,
    },
    REGIME_TECHNICAL_DRIVEN: {
        "sentiment": 0.0,
        "rsi": 0.6,
        "macd_hist": 0.4,
        "atr": 0.0,
    },
    # CONFLICTING_SIGNALS: all weights lowered so competing signals
    # cancel in centroid defuzz → output biases toward Hold (PDF §11).
    # RSI/MACD carry equal small weight so neither overrides the other.
    REGIME_CONFLICTING_SIGNALS: {
        "sentiment": 0.15,
        "rsi": 0.15,
        "macd_hist": 0.15,
        "atr": 0.20,
    },
    REGIME_HIGH_UNCERTAINTY: {
        "sentiment": 0.25,
        "rsi": 0.25,
        "macd_hist": 0.25,
        "atr": 0.25,
    },
    REGIME_NORMAL: {
        "sentiment": 0.3,
        "rsi": 0.4,
        "macd_hist": 0.3,
        "atr": 0.0,
    },
}

# Output tendency per regime (for UI explainability)
REGIME_TENDENCY: Dict[str, str] = {
    REGIME_NEWS_DRIVEN_BEARISH: "Strong Sell",
    REGIME_NEWS_DRIVEN_BULLISH: "Buy / Strong Buy",
    REGIME_TECHNICAL_DRIVEN: "Depends on RSI+MACD",
    REGIME_CONFLICTING_SIGNALS: "Hold",
    REGIME_HIGH_UNCERTAINTY: "Hold / Reduce Exposure",
    REGIME_NORMAL: "Balanced (slight technical bias)",
}

# Regime display names (UI-friendly)
REGIME_DISPLAY_NAMES: Dict[str, str] = {
    REGIME_NEWS_DRIVEN_BEARISH: "News-Driven Bearish",
    REGIME_NEWS_DRIVEN_BULLISH: "News-Driven Bullish",
    REGIME_TECHNICAL_DRIVEN: "Technical-Driven",
    REGIME_CONFLICTING_SIGNALS: "Conflicting Signals",
    REGIME_HIGH_UNCERTAINTY: "High Uncertainty",
    REGIME_NORMAL: "Normal Market",
}


# Main Selection Function

def select_features(
    regime_result: Dict,
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """
    Select active features and assign weights based on market regime.
    
    Args:
        regime_result: Output dict from expert_layer.detect_market_regime()
        rsi, macd_hist, sentiment, atr: Raw values (unused in selection
            logic but accepted for future adaptive weighting)
    
    Returns:
        {
            "active_features": ["sentiment"],     # Features with weight > 0
            "masked_features": ["rsi", "macd_hist"],  # Features with weight = 0
            "weights": {"sentiment": 1.0, ...},
            "regime": "NEWS_DRIVEN_BEARISH",
            "reason": "News-driven bearish condition detected...",
            "output_tendency": "Strong Sell"
        }
    """
    regime = regime_result.get("regime", REGIME_NORMAL)
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[REGIME_NORMAL]).copy()

    # Remove ATR from weights if no ATR value was provided
    if atr is None:
        weights.pop("atr", None)

    active_features = [f for f, w in weights.items() if w > 0.0]
    masked_features = [f for f, w in weights.items() if w == 0.0]

    # Build a human-readable reason
    regime_display = REGIME_DISPLAY_NAMES.get(regime, regime)
    tendency = REGIME_TENDENCY.get(regime, "Unknown")

    active_display = ", ".join(active_features) if active_features else "None"
    masked_display = ", ".join(masked_features) if masked_features else "None"

    reason = (
        f"Market regime classified as **{regime_display}**. "
        f"Active features: {active_display}. "
        f"Masked features: {masked_display}. "
        f"Expected output tendency: {tendency}."
    )

    return {
        "active_features": active_features,
        "masked_features": masked_features,
        "weights": weights,
        "regime": regime,
        "regime_display": regime_display,
        "reason": reason,
        "output_tendency": tendency,
    }


# Info helper for UI

def get_weightage_table() -> List[Dict]:
    """Return a display-friendly weightage table for the UI."""
    rows = []
    for regime, weights in REGIME_WEIGHTS.items():
        rows.append({
            "regime": REGIME_DISPLAY_NAMES.get(regime, regime),
            "sentiment": f"{weights['sentiment']*100:.0f}%",
            "rsi": f"{weights['rsi']*100:.0f}%",
            "macd_hist": f"{weights['macd_hist']*100:.0f}%",
            "atr": f"{weights.get('atr', 0)*100:.0f}%",
            "tendency": REGIME_TENDENCY.get(regime, ""),
        })
    return rows