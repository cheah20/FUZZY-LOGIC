"""Prepared cases where the LLM gets it wrong and the validation gate catches it.

Each case says what went wrong and which rule was broken. Used on the
Agentic Layer page.
"""

from __future__ import annotations


FAILURE_SCENARIOS = [
    {
        "scenario_id": "FAIL-001",
        "title": "ATR Misinterpretation — Volatility Treated as Direction",
        "description": (
            "High ATR combined with negative market sentiment causes the LLM reasoning "
            "layer to over-react and propose a Strong Sell signal, citing high volatility "
            "as the primary directional evidence. The Human Expert Validation Gate rejects "
            "this suggestion because ATR is a volatility measure, not a directional indicator."
        ),
        "market_context": {
            "symbol":        "DEMO",
            "rsi":           52.0,
            "macd_hist":     -0.12,
            "atr":           7.8,
            "sentiment":     -0.38,
            "market_regime": "HIGH_UNCERTAINTY",
            "price":         185.40,
        },
        "llm_suggestion": {
            "proposed_action": (
                "Issue a Strong Sell signal. Market volatility is extremely high (ATR 7.8) "
                "and sentiment is negative, indicating bearish pressure. Recommend reducing "
                "long exposure immediately."
            ),
            "reason": (
                "ATR is at 7.8 (above the high-volatility threshold of 6.0), indicating "
                "very high market volatility. Combined with a negative sentiment reading "
                "of -0.38, this suggests strong bearish conditions. The volatility spike "
                "poses significant downside risk, justifying a Strong Sell direction call."
            ),
            "fuzzy_shift": {
                "score_delta":      -0.40,
                "confidence_delta": 0.15,
                "direction":        "boost_sell",
            },
            "confidence": 0.82,
            "scenario_tag": "FAIL_ATR_DIRECTION",
            "mode":         "demo",
        },
        "validation_result": {
            "status": "REJECTED",
            "reason": (
                "Rejected: Critical rule violations ATR-001, ATR-002. "
                "ATR (Average True Range) is a volatility measure — it indicates how much "
                "price moves, NOT in which direction. Using ATR as a directional Sell trigger "
                "is a fundamental misapplication of this indicator. The correct response to "
                "high ATR is to increase the risk penalty and reduce position confidence, "
                "not to issue a directional Strong Sell signal. RSI (52.0) is in neutral "
                "territory and MACD histogram (-0.12) shows only weak bearish momentum — "
                "neither indicator independently justifies a Strong Sell. The sentiment "
                "reading alone is also insufficient per MULTI-001 (single-category evidence)."
            ),
            "applied_shift": {},
            "violated_rules": ["ATR-001", "ATR-002", "MULTI-001"],
            "supporting_rules": [],
        },
        "reflection": {
            "root_cause": (
                "The LLM conflated volatility magnitude with price direction. High ATR "
                "simply means the price range per bar is large — it does not indicate whether "
                "the net move will be up or down. This is one of the most common "
                "misinterpretations of volatility indicators in automated trading systems."
            ),
            "corrective_principle": (
                "Do NOT allow ATR alone (or in combination with sentiment only) to determine "
                "market price direction. ATR's correct role is in position sizing, stop-loss "
                "distance calculation, and risk-penalty adjustments. Any directional signal "
                "must come from RSI, MACD, or Bollinger Band positions — indicators that "
                "carry directional information."
            ),
            "what_should_have_happened": (
                "The LLM should have proposed: 'Apply a risk penalty by reducing confidence "
                "due to high ATR (7.8). This is a risk management adjustment, NOT a directional "
                "call. The fuzzy score is held at its current level while confidence is reduced "
                "by -0.10 to reflect elevated uncertainty in a high-volatility environment.' "
                "This would have been consistent with ATR-002 and would have passed the "
                "Human Expert Validation Gate."
            ),
            "kb_rules_violated": ["ATR-001", "ATR-002", "MULTI-001"],
            "kb_rules_corrective": ["ATR-002"],
            "lesson": (
                "Volatility != Direction. ATR measures uncertainty about where price will go, "
                "not the direction in which it will go. Human expert knowledge encoded in "
                "ATR-001 prevents this systematic error from propagating to the final decision."
            ),
        },
    },
    {
        "scenario_id": "FAIL-002",
        "title": "Single-Indicator Overconfidence — MACD Without Confirmation",
        "description": (
            "A strongly positive MACD histogram causes the LLM to propose a high-confidence "
            "Strong Buy signal citing MACD alone. The gate modifies this because MACD-only "
            "signals lack the multi-indicator confirmation required by the KB."
        ),
        "market_context": {
            "symbol":        "DEMO",
            "rsi":           58.0,
            "macd_hist":     0.85,
            "atr":           3.2,
            "sentiment":     0.20,
            "market_regime": "NORMAL",
            "price":         210.55,
        },
        "llm_suggestion": {
            "proposed_action": (
                "Issue a high-confidence Strong Buy signal. MACD histogram is strongly positive "
                "at 0.85, confirming significant upside momentum."
            ),
            "reason": (
                "The MACD histogram is at 0.85, well above zero, indicating strong and "
                "accelerating bullish momentum. This is a high-confidence buy signal."
            ),
            "fuzzy_shift": {
                "score_delta":      0.35,
                "confidence_delta": 0.20,
                "direction":        "boost_buy",
            },
            "confidence": 0.85,
            "scenario_tag": "FAIL_MACD_ONLY",
            "mode":         "demo",
        },
        "validation_result": {
            "status": "MODIFIED",
            "reason": (
                "Partially approved with reduced magnitude (50%). Non-critical violations: "
                "MACD-003, MULTI-001. MACD-only signals should be confirmed by RSI or "
                "Bollinger Bands before high confidence is assigned. RSI at 58 is neutral "
                "and does not confirm overbought/oversold conditions. Shift reduced from "
                "+0.35 to +0.175 and confidence delta halved."
            ),
            "applied_shift": {
                "score_delta":      0.175,
                "confidence_delta": 0.10,
                "direction":        "boost_buy",
            },
            "violated_rules": ["MACD-003", "MULTI-001"],
            "supporting_rules": ["MACD-001"],
        },
        "reflection": {
            "root_cause": (
                "The LLM placed excessive weight on a single momentum indicator (MACD) "
                "without seeking confirmation from an oscillator (RSI) or volatility band "
                "(Bollinger). Strong MACD readings can persist in trending markets for extended "
                "periods and do not guarantee near-term price continuation."
            ),
            "corrective_principle": (
                "Strong directional signals require at least two independent indicator "
                "categories. MACD (momentum) should be confirmed by RSI (oscillator) or "
                "Bollinger Band position (trend/range) before high confidence is assigned. "
                "MULTI-001 enforces this principle."
            ),
            "what_should_have_happened": (
                "The LLM should have noted RSI is in neutral territory (58) and therefore "
                "not confirming overbought/oversold extremes, limiting the confidence of the "
                "buy signal. A moderate score increase (+0.10 to +0.15) with moderate "
                "confidence would have passed the gate without modification."
            ),
            "kb_rules_violated": ["MACD-003", "MULTI-001"],
            "kb_rules_corrective": ["MACD-001", "MULTI-001"],
            "lesson": (
                "Momentum without oscillator confirmation is insufficient for strong signals. "
                "Multi-indicator convergence is required to reduce the false positive rate."
            ),
        },
    },
]


def get_failure_demo(scenario_index: int = 0) -> dict:
    """One failure case for the UI.

    0 = ATR read as a direction, 1 = trusting MACD alone.
    """
    idx = scenario_index % len(FAILURE_SCENARIOS)
    scenario = FAILURE_SCENARIOS[idx]
    return {
        "scenario_id":       scenario["scenario_id"],
        "title":             scenario["title"],
        "description":       scenario["description"],
        "market_context":    scenario["market_context"],
        "llm_suggestion":    scenario["llm_suggestion"],
        "validation_result": scenario["validation_result"],
        "reflection":        scenario["reflection"],
        "total_scenarios":   len(FAILURE_SCENARIOS),
    }


def get_all_failure_demos() -> list:
    """All failure cases."""
    return [
        {
            "scenario_id": s["scenario_id"],
            "title":       s["title"],
            "description": s["description"],
            "market_context":    s["market_context"],
            "llm_suggestion":    s["llm_suggestion"],
            "validation_result": s["validation_result"],
            "reflection":        s["reflection"],
        }
        for s in FAILURE_SCENARIOS
    ]


def get_reflection_summary() -> list:
    """Short list of the lessons from every case."""
    return [
        {
            "scenario_id": s["scenario_id"],
            "title":       s["title"],
            "status":      s["validation_result"]["status"],
            "violated":    s["validation_result"]["violated_rules"],
            "lesson":      s["reflection"]["lesson"],
        }
        for s in FAILURE_SCENARIOS
    ]
