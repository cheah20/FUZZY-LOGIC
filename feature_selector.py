"""Pick which indicators the fuzzy engine uses in each market regime.

In a news-driven market only sentiment counts. In a calm market RSI,
MACD and sentiment share the weight.
"""

from __future__ import annotations
from typing import Dict, List, Optional

# regime names come from expert_layer
from expert_layer import (
    REGIME_NEWS_DRIVEN_BEARISH,
    REGIME_NEWS_DRIVEN_BULLISH,
    REGIME_TECHNICAL_DRIVEN,
    REGIME_CONFLICTING_SIGNALS,
    REGIME_HIGH_UNCERTAINTY,
    REGIME_NORMAL,
)

# weight of each feature per regime, 0 means masked


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
    # conflicting signals: every weight stays low so they cancel out
    # and the centroid ends up near Hold
    # RSI and MACD get the same weight so neither one wins
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

# what each regime usually turns into, shown in the UI
REGIME_TENDENCY: Dict[str, str] = {
    REGIME_NEWS_DRIVEN_BEARISH: "Strong Sell",
    REGIME_NEWS_DRIVEN_BULLISH: "Buy / Strong Buy",
    REGIME_TECHNICAL_DRIVEN: "Depends on RSI+MACD",
    REGIME_CONFLICTING_SIGNALS: "Hold",
    REGIME_HIGH_UNCERTAINTY: "Hold / Reduce Exposure",
    REGIME_NORMAL: "Balanced (slight technical bias)",
}

# regime names as the UI shows them
REGIME_DISPLAY_NAMES: Dict[str, str] = {
    REGIME_NEWS_DRIVEN_BEARISH: "News-Driven Bearish",
    REGIME_NEWS_DRIVEN_BULLISH: "News-Driven Bullish",
    REGIME_TECHNICAL_DRIVEN: "Technical-Driven",
    REGIME_CONFLICTING_SIGNALS: "Conflicting Signals",
    REGIME_HIGH_UNCERTAINTY: "High Uncertainty",
    REGIME_NORMAL: "Normal Market",
}


def select_features(
    regime_result: Dict,
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """Return the active and masked features, their weights and a short reason
    for this regime. The raw indicator values are accepted but not used yet.
    """
    regime = regime_result.get("regime", REGIME_NORMAL)
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[REGIME_NORMAL]).copy()

    # no ATR value given, so leave it out
    if atr is None:
        weights.pop("atr", None)

    active_features = [f for f, w in weights.items() if w > 0.0]
    masked_features = [f for f, w in weights.items() if w == 0.0]

    # short sentence for the UI
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