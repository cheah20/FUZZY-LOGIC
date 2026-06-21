"""
interface.py — UI Integration Layer
ASTDSS Terminal — Neuro-Symbolic Trading DSS

Purpose:
  Provides structured data for the UI Expert Reasoning Panel.
  Bridges the backend modules (expert_layer, feature_selector,
  fuzzy_engine) to the Flask/Jinja frontend.

Provides:
  - get_expert_reasoning_data() → Full explainability data for UI
  - compare_baseline_vs_expert()  → Side-by-side comparison
  - get_system_status()           → Overall system health
"""

from __future__ import annotations
from typing import Dict, List, Optional


def get_expert_reasoning_data(
    rsi: float = 74.2,
    macd_hist: float = -0.35,
    sentiment: float = -0.15,
    atr: Optional[float] = None,
) -> Dict:
    """
    Get complete explainability data for the Expert Reasoning Panel.
    
    This is the primary function the UI calls to populate the
    Expert Reasoning Panel on the Decision Logic page.
    
    Returns:
        {
            "rsi": 74.2,
            "macd_hist": -0.35,
            "sentiment": -0.15,
            "atr": 5.2,
            "market_regime": "...",
            "regime_confidence": 0.85,
            "regime_reason": "...",
            "triggered_rule": "RULE_3_TECHNICAL_DRIVEN",
            "active_features": ["rsi", "macd_hist"],
            "masked_features": ["sentiment"],
            "weights": {"rsi": 0.6, "macd_hist": 0.4, ...},
            "fuzzy_score": 0.45,
            "fuzzy_label": "Hold",
            "confidence": 0.72,
            "label_memberships": {"Strong Sell": 0.0, ...},
            "rule_firings": [...],
            "weightage_table": [...],
            "expert_rules": [...],
        }
    """
    from expert_layer import detect_market_regime, get_expert_layer_info
    from feature_selector import select_features, get_weightage_table
    from tools import fuzzy_analysis_tool

    regime = detect_market_regime(rsi, macd_hist, sentiment, atr)
    selection = select_features(regime, rsi, macd_hist, sentiment, atr)
    fuzzy_result = fuzzy_analysis_tool(rsi, macd_hist, sentiment, atr, mode="expert")
    expert_info = get_expert_layer_info()
    weightage_table = get_weightage_table()

    # Build display-ready input rows for the Input Variables panel
    # Colours reflect whether the input is active/masked in the current regime.
    active = selection["active_features"]

    def _bar_color(feat: str, val: float) -> str:
        if feat not in active:
            return "rgba(255,255,255,0.18)"   # muted = masked
        if feat == "rsi":
            if val > 70: return "#EF4444"
            if val < 30: return "#10B981"
            return "#7cd0ff"
        if feat in ("macd_hist", "sentiment"):
            if val > 0.05:  return "#10B981"
            if val < -0.05: return "#EF4444"
            return "rgba(255,255,255,0.4)"
        return "var(--gold-soft)"

    # Normalise raw values to 0–100 for progress bars
    rsi_pct    = int(max(0, min(100, rsi)))
    macd_pct   = int(max(0, min(100, ((macd_hist + 5) / 10) * 100)))
    sent_pct   = int(max(0, min(100, ((sentiment  + 1) / 2) * 100)))

    input_display = [
        {
            "label":  "RSI INTENSITY",
            "value":  f"{rsi:.1f}",
            "pct":    rsi_pct,
            "color":  _bar_color("rsi", rsi),
            "active": "rsi" in active,
        },
        {
            "label":  "MACD HISTOGRAM",
            "value":  f"{macd_hist:+.3f}",
            "pct":    macd_pct,
            "color":  _bar_color("macd_hist", macd_hist),
            "active": "macd_hist" in active,
        },
        {
            "label":  "SENTIMENT SCORE",
            "value":  f"{sentiment:+.2f}",
            "pct":    sent_pct,
            "color":  _bar_color("sentiment", sentiment),
            "active": "sentiment" in active,
        },
    ]
    if atr is not None:
        atr_pct = int(max(0, min(100, (atr / 10) * 100)))
        input_display.append({
            "label":  "ATR (VOLATILITY)",
            "value":  f"{atr:.2f}",
            "pct":    atr_pct,
            "color":  _bar_color("atr", atr),
            "active": "atr" in active,
        })

    return {
        "inputs": {
            "rsi": rsi,
            "macd_hist": macd_hist,
            "sentiment": sentiment,
            "atr": atr,
        },
        "market_regime": regime["regime"],
        "regime_display": selection.get("regime_display", regime["regime"]),
        "regime_confidence": regime["confidence"],
        "regime_reason": regime["reason"],
        "triggered_rule": regime.get("triggered_rule", ""),
        "active_features": selection["active_features"],
        "masked_features": selection["masked_features"],
        "weights": selection["weights"],
        "output_tendency": selection["output_tendency"],
        "fuzzy_score": fuzzy_result["score"],
        "fuzzy_label": fuzzy_result["label"],
        "confidence": fuzzy_result["confidence"],
        "label_memberships": fuzzy_result.get("label_memberships", {}),
        "rule_firings": fuzzy_result.get("rule_firings", []),
        "rules_fired": fuzzy_result.get("rules_fired", 0),
        "weightage_table": weightage_table,
        "expert_rules": expert_info["rules"],
        "reference_sources": expert_info["reference_sources"],
        "input_display": input_display,
    }


def compare_baseline_vs_expert(
    rsi: float = 74.2,
    macd_hist: float = -0.35,
    sentiment: float = -0.15,
    atr: Optional[float] = None,
) -> Dict:
    """Side-by-side comparison for UI display."""
    from tools import compare_modes_tool
    return compare_modes_tool(rsi, macd_hist, sentiment, atr)


def get_system_status() -> Dict:
    """Get overall system status for UI dashboard."""
    return {
        "fuzzy_engine": "Active",
        "expert_layer": "Active",
        "feature_selector": "Active",
        "mode": "Expert-Adjusted FLC",
        "sampling_rate": "100ms",
        "rules_available": 26,
        "regimes_supported": 6,
        "version": "v5.0.0-EXPERT",
    }