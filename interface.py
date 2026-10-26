"""Builds the data the Flask templates show: expert reasoning, the baseline
vs expert comparison and the system status.
"""

from __future__ import annotations
from typing import Dict, List, Optional


def get_expert_reasoning_data(
    rsi: float = 74.2,
    macd_hist: float = -0.35,
    sentiment: float = -0.15,
    atr: Optional[float] = None,
) -> Dict:
    """Everything the expert reasoning panel needs for one set of inputs.

    Includes the regime and why, the active and masked features with their
    weights, the fuzzy score and label, rule firings, the weightage table and
    the expert rules.
    """
    from expert_layer import detect_market_regime, get_expert_layer_info
    from feature_selector import select_features, get_weightage_table
    from tools import fuzzy_analysis_tool

    regime = detect_market_regime(rsi, macd_hist, sentiment, atr)
    selection = select_features(regime, rsi, macd_hist, sentiment, atr)
    fuzzy_result = fuzzy_analysis_tool(rsi, macd_hist, sentiment, atr, mode="expert")
    expert_info = get_expert_layer_info()
    weightage_table = get_weightage_table()

    # rows for the inputs panel; the colour shows
    # whether the input is used or masked in this regime
    active = selection["active_features"]

    def _bar_color(feat: str, val: float) -> str:
        if feat not in active:
            return "rgba(20, 26, 34, 0.18)"   # grey means masked
        if feat == "rsi":
            if val > 70: return "#e5443b"
            if val < 30: return "#12a46b"
            return "#141a22"
        if feat in ("macd_hist", "sentiment"):
            if val > 0.05:  return "#12a46b"
            if val < -0.05: return "#e5443b"
            return "rgba(20, 26, 34, 0.45)"
        return "var(--gold-soft)"

    # 0-100 for the bars
    rsi_pct    = int(max(0, min(100, rsi)))
    macd_pct   = int(max(0, min(100, ((macd_hist + 5) / 10) * 100)))
    sent_pct   = int(max(0, min(100, ((sentiment  + 1) / 2) * 100)))

    input_display = [
        {
            "label":  "RSI",
            "value":  f"{rsi:.1f}",
            "pct":    rsi_pct,
            "color":  _bar_color("rsi", rsi),
            "active": "rsi" in active,
        },
        {
            "label":  "MACD histogram",
            "value":  f"{macd_hist:+.3f}",
            "pct":    macd_pct,
            "color":  _bar_color("macd_hist", macd_hist),
            "active": "macd_hist" in active,
        },
        {
            "label":  "Sentiment",
            "value":  f"{sentiment:+.2f}",
            "pct":    sent_pct,
            "color":  _bar_color("sentiment", sentiment),
            "active": "sentiment" in active,
        },
    ]
    if atr is not None:
        atr_pct = int(max(0, min(100, (atr / 10) * 100)))
        input_display.append({
            "label":  "ATR (volatility)",
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
    """Baseline and expert next to each other."""
    from tools import compare_modes_tool
    return compare_modes_tool(rsi, macd_hist, sentiment, atr)


def get_system_status() -> Dict:
    """System status for the dashboard."""
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