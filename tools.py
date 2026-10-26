"""Small functions the cognition agent calls to run the fuzzy analysis.

They are plain functions now and could become CrewAI tools later.
"""

from __future__ import annotations
from typing import Dict, Optional
import json


def fuzzy_analysis_tool(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
    *,
    mode: str = "expert",
) -> Dict:
    """Full analysis for the cognition agent: expert layer, feature
    selection, fuzzy engine.

    Returns score (-2..2), label, confidence, market_regime, active and
    masked features, weights, reason, rule_firings and mode.
    """
    from fuzzy_engine import compute_full_pipeline

    result = compute_full_pipeline(
        rsi=rsi,
        macd_hist=macd_hist,
        sentiment=sentiment,
        atr=atr,
        mode=mode,
    )

    # shape the result for the agent
    formatted = {
        "score": result.get("score", 0.0),
        "label": result.get("label", "Hold"),
        "confidence": result.get("confidence", 0.5),
        "market_regime": result.get("market_regime"),
        "active_features": result.get("active_features", []),
        "masked_features": result.get("masked_features", []),
        "weights": result.get("feature_weights", {}),
        "reason": result.get("selection_reason", ""),
        "rule_firings": result.get("rule_firings", []),
        "rules_fired": result.get("rules_fired", 0),
        "label_memberships": result.get("label_memberships", {}),
        "mode": result.get("mode", mode),
    }

    return formatted


def baseline_fuzzy_tool(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """Baseline result for comparison: every input counts the same."""
    from fuzzy_engine import compute_score

    result = compute_score(rsi, macd_hist, sentiment, atr)
    return {
        "score": result.get("score", 0.0),
        "label": result.get("label", "Hold"),
        "confidence": result.get("confidence", 0.5),
        "rule_firings": result.get("rule_firings", []),
        "rules_fired": result.get("rules_fired", 0),
        "mode": "baseline",
    }


def market_regime_tool(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """Market regime, using the expert rules."""
    from expert_layer import detect_market_regime
    return detect_market_regime(rsi, macd_hist, sentiment, atr)


def feature_selection_tool(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """Active and masked features with their weights for a regime."""
    from expert_layer import detect_market_regime
    from feature_selector import select_features

    regime = detect_market_regime(rsi, macd_hist, sentiment, atr)
    selection = select_features(regime, rsi, macd_hist, sentiment, atr)
    return selection


def compare_modes_tool(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """Baseline and expert results side by side."""
    baseline = baseline_fuzzy_tool(rsi, macd_hist, sentiment, atr)
    expert = fuzzy_analysis_tool(rsi, macd_hist, sentiment, atr, mode="expert")

    return {
        "baseline": baseline,
        "expert": expert,
        "delta": round(expert["score"] - baseline["score"], 4),
        "consensus": (
            "Agree" if baseline["label"] == expert["label"]
            else "Diverge"
        ),
    }


# list of tools for the UI

TOOL_REGISTRY = [
    {
        "name": "fuzzy_analysis_tool",
        "description": "Full expert-adjusted fuzzy logic analysis with market regime detection and dynamic feature masking",
        "agent": "Cognition Agent",
        "inputs": ["rsi", "macd_hist", "sentiment", "atr (optional)"],
        "outputs": ["score", "label", "confidence", "market_regime", "active_features", "masked_features", "reason"],
    },
    {
        "name": "baseline_fuzzy_tool",
        "description": "Baseline FLC without expert adjustments — for comparison",
        "agent": "Cognition Agent",
        "inputs": ["rsi", "macd_hist", "sentiment", "atr (optional)"],
        "outputs": ["score", "label", "confidence"],
    },
    {
        "name": "market_regime_tool",
        "description": "Detect current market regime using human expert heuristics",
        "agent": "Perception Agent / Cognition Agent",
        "inputs": ["rsi", "macd_hist", "sentiment", "atr (optional)"],
        "outputs": ["regime", "confidence", "reason", "triggered_rule"],
    },
    {
        "name": "compare_modes_tool",
        "description": "Side-by-side comparison of baseline vs expert-adjusted FLC",
        "agent": "Cognition Agent / Execution Agent",
        "inputs": ["rsi", "macd_hist", "sentiment", "atr (optional)"],
        "outputs": ["baseline", "expert", "delta", "consensus"],
    },
]