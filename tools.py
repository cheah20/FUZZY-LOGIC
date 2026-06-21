"""
tools.py ─ CrewAI Agent Tools / Fuzzy Analysis Bridge
ASTDSS Terminal — Neuro-Symbolic Trading DSS

Purpose:
  Provides tool functions that CrewAI agents (or the standalone
  pipeline) call to perform fuzzy logic analysis. This is the bridge
  between the Cognition Agent and the Fuzzy Logic Controller.

  When CrewAI is fully integrated, agents will use these as @tool
  decorated functions. For now, they work as standalone callables.

Tools:
  - fuzzy_analysis_tool()      → Full expert-adjusted FLC analysis
  - baseline_fuzzy_tool()      → Baseline FLC (for comparison)
  - market_regime_tool()       → Detect current market regime
  - feature_selection_tool()   → Get feature weights/masking
  - compare_modes_tool()       → Baseline vs Expert side-by-side
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
    """
    Primary fuzzy analysis tool called by the Cognition Agent.
    
    Runs the full expert-adjusted pipeline:
      expert_layer → feature_selector → fuzzy_engine
    
    Args:
        rsi: RSI value (0–100)
        macd_hist: MACD histogram value
        sentiment: Sentiment score (-1 to +1)
        atr: Optional ATR normalised value (0–10)
        mode: "baseline" or "expert"
    
    Returns:
        {
            "score": float,           # Crisp defuzzified score (-2 to +2)
            "label": str,             # "Strong Sell" / "Sell" / "Hold" / "Buy" / "Strong Buy"
            "confidence": float,      # 0.0–1.0
            "market_regime": {...},   # Regime detection result
            "active_features": [...], # Features with weight > 0
            "masked_features": [...], # Features with weight = 0
            "weights": {...},         # Feature weight map
            "reason": str,            # Human-readable explanation
            "rule_firings": [...],    # Which fuzzy rules fired
            "mode": str               # "baseline" or "expert"
        }
    """
    from fuzzy_engine import compute_full_pipeline

    result = compute_full_pipeline(
        rsi=rsi,
        macd_hist=macd_hist,
        sentiment=sentiment,
        atr=atr,
        mode=mode,
    )

    # Format for CrewAI agent consumption
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
    """
    Baseline FLC tool — for comparison against expert mode.
    All inputs treated equally; no feature masking.
    """
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
    """
    Detect the current market regime using human expert heuristics.
    Called by the Cognition Agent to understand market context.
    """
    from expert_layer import detect_market_regime
    return detect_market_regime(rsi, macd_hist, sentiment, atr)


def feature_selection_tool(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """
    Get feature selection based on market regime.
    Returns active/masked features with weights.
    """
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
    """
    Compare baseline FLC vs expert-adjusted FLC side-by-side.
    Used for backtesting comparison and UI display.
    """
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


# Tool metadata for UI display / CrewAI registration

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