"""
human_expert_kb.py — Human Expert Knowledge Base
ASTDSS Terminal — PA3 Agentic Fuzzy Trading DSS

Knowledge is captured from peer-reviewed technical analysis literature and
professional trading research. The LLM's proposed adjustments are validated
against these rules before any fuzzy output is modified.

Sources:
  [W78]  Wilder, J.W. (1978). New Concepts in Technical Trading Systems.
  [A05]  Appel, G. (2005). Technical Analysis: Power Tools for Active Investors.
  [M99]  Murphy, J.J. (1999). Technical Analysis of the Financial Markets.
  [B02]  Bollinger, J. (2002). Bollinger on Bollinger Bands.
  [S66]  Sharpe, W. (1966). Mutual Fund Performance. Journal of Business.
  [P08]  Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies.
  [E93]  Elder, A. (1993). Trading for a Living.
"""

KNOWLEDGE_BASE = [
    # RSI Rules
    {
        "id": "RSI-001",
        "category": "RSI",
        "captured_knowledge": (
            "RSI ≤ 30 indicates an oversold condition. Prices have fallen rapidly "
            "and a mean-reversion bounce is statistically more likely. This increases "
            "the confidence weight toward a Buy signal."
        ),
        "validation_rule": (
            "If the LLM's proposed action is directional Buy and RSI ≤ 30, "
            "the rule provides supporting evidence. Approve the directional component."
        ),
        "source_name": "Wilder (1978) — New Concepts in Technical Trading Systems [W78]",
        "system_action": "INCREASE buy_confidence when RSI ≤ 30",
    },
    {
        "id": "RSI-002",
        "category": "RSI",
        "captured_knowledge": (
            "RSI ≥ 70 indicates an overbought condition. Momentum has become "
            "stretched and price is susceptible to a reversal. This increases "
            "the confidence weight toward a Sell signal."
        ),
        "validation_rule": (
            "If the LLM's proposed action is directional Sell and RSI ≥ 70, "
            "the rule provides supporting evidence. Approve the directional component."
        ),
        "source_name": "Wilder (1978) — New Concepts in Technical Trading Systems [W78]",
        "system_action": "INCREASE sell_confidence when RSI ≥ 70",
    },
    {
        "id": "RSI-003",
        "category": "RSI",
        "captured_knowledge": (
            "RSI alone is insufficient to trigger a high-confidence Buy or Sell decision. "
            "RSI can remain in overbought/oversold territory for extended periods during "
            "strong trends. It must be confirmed by at least one other indicator category."
        ),
        "validation_rule": (
            "If the LLM's proposed action claims high confidence (> 0.70) based solely "
            "on RSI without mentioning any other indicator, REJECT or MODIFY the suggestion "
            "to reduce confidence below the 'Strong' threshold."
        ),
        "source_name": "Murphy (1999) — Technical Analysis of the Financial Markets [M99]",
        "system_action": "CAP confidence at 0.60 if RSI is the only cited evidence",
    },

    # MACD Rules
    {
        "id": "MACD-001",
        "category": "MACD",
        "captured_knowledge": (
            "A positive MACD histogram (fast EMA above slow EMA) indicates bullish "
            "short-term momentum. The histogram expanding positively signals accelerating "
            "upside momentum."
        ),
        "validation_rule": (
            "If the LLM's proposed action is bullish and MACD histogram > 0, "
            "approve the momentum directional component."
        ),
        "source_name": "Appel (2005) — Technical Analysis: Power Tools for Active Investors [A05]",
        "system_action": "SUPPORT bullish signals when MACD histogram > 0",
    },
    {
        "id": "MACD-002",
        "category": "MACD",
        "captured_knowledge": (
            "A negative MACD histogram (fast EMA below slow EMA) indicates bearish "
            "momentum. The histogram expanding negatively signals accelerating downside "
            "momentum."
        ),
        "validation_rule": (
            "If the LLM's proposed action is bearish and MACD histogram < 0, "
            "approve the momentum directional component."
        ),
        "source_name": "Appel (2005) — Technical Analysis: Power Tools for Active Investors [A05]",
        "system_action": "SUPPORT bearish signals when MACD histogram < 0",
    },
    {
        "id": "MACD-003",
        "category": "MACD",
        "captured_knowledge": (
            "MACD signals are momentum-based and can produce false signals in "
            "sideways or low-volatility markets. A MACD-only signal should always "
            "be confirmed by RSI or another independent indicator before generating "
            "a high-confidence directional decision."
        ),
        "validation_rule": (
            "If the LLM cites MACD only (no RSI or Bollinger mention) for a strong "
            "directional signal, MODIFY to reduce confidence or flag for review."
        ),
        "source_name": "Murphy (1999) — Technical Analysis of the Financial Markets [M99]",
        "system_action": "REQUIRE RSI or BB confirmation for MACD-only strong signals",
    },

    # ATR Rules
    {
        "id": "ATR-001",
        "category": "ATR",
        "captured_knowledge": (
            "Average True Range (ATR) is a measure of market volatility — how much "
            "price moves on average per period. It does NOT indicate price direction. "
            "High ATR means price is moving widely in either direction, not necessarily "
            "downward."
        ),
        "validation_rule": (
            "If the LLM proposes a Sell or bearish signal and cites ATR or 'high volatility' "
            "as the primary or sole reason (without RSI/MACD directional evidence), REJECT "
            "the directional component immediately."
        ),
        "source_name": "Wilder (1978) — New Concepts in Technical Trading Systems [W78]",
        "system_action": "REJECT directional Sell/Buy signal if ATR is cited as the direction reason",
    },
    {
        "id": "ATR-002",
        "category": "ATR",
        "captured_knowledge": (
            "High ATR correctly justifies increasing the risk penalty, widening stop-loss "
            "distances, and reducing position sizing. It does not justify a directional Sell "
            "signal. The correct response to high ATR is risk management adjustment, not "
            "a directional trade call."
        ),
        "validation_rule": (
            "If the LLM suggests increasing a risk penalty or reducing confidence due to "
            "high ATR, APPROVE the risk-management adjustment. Reject only if the LLM frames "
            "high ATR as a directional sell trigger."
        ),
        "source_name": "Elder (1993) — Trading for a Living [E93]",
        "system_action": "APPROVE risk_penalty increase for high ATR; REJECT direction calls from ATR alone",
    },

    # Bollinger Band Rules
    {
        "id": "BB-001",
        "category": "Bollinger Bands",
        "captured_knowledge": (
            "Bollinger Bands are constructed from a 20-period moving average with upper "
            "and lower bands at ±2 standard deviations. Price movements beyond the bands "
            "are statistically unusual (occurring ~5% of the time) and may signal "
            "overextension or a breakout."
        ),
        "validation_rule": (
            "Bollinger Band position alone is not sufficient for a directional decision. "
            "It must be combined with RSI or MACD to confirm overbought/oversold conditions."
        ),
        "source_name": "Bollinger (2002) — Bollinger on Bollinger Bands [B02]",
        "system_action": "USE BB position as supporting confirmation, not standalone trigger",
    },
    {
        "id": "BB-002",
        "category": "Bollinger Bands",
        "captured_knowledge": (
            "When price is near or below the lower Bollinger Band AND RSI is simultaneously "
            "in oversold territory (RSI ≤ 35), both indicators converge to support a Buy "
            "signal. This dual confirmation increases reliability."
        ),
        "validation_rule": (
            "If the LLM proposes a Buy signal with both lower-band touch and RSI oversold, "
            "APPROVE with increased confidence. This is a two-indicator confirmation per MULTI-001."
        ),
        "source_name": "Bollinger (2002) — Bollinger on Bollinger Bands [B02]",
        "system_action": "BOOST buy_confidence for lower-band touch + RSI oversold combination",
    },
    {
        "id": "BB-003",
        "category": "Bollinger Bands",
        "captured_knowledge": (
            "When price is near or above the upper Bollinger Band AND RSI is simultaneously "
            "in overbought territory (RSI ≥ 65), both indicators converge to support a Sell "
            "signal. This dual confirmation reduces false signal risk."
        ),
        "validation_rule": (
            "If the LLM proposes a Sell signal with both upper-band touch and RSI overbought, "
            "APPROVE with increased confidence. This satisfies the two-indicator rule."
        ),
        "source_name": "Bollinger (2002) — Bollinger on Bollinger Bands [B02]",
        "system_action": "BOOST sell_confidence for upper-band touch + RSI overbought combination",
    },

    # Multi-Indicator Confirmation
    {
        "id": "MULTI-001",
        "category": "Multi-Indicator Confirmation",
        "captured_knowledge": (
            "Strong trading decisions require confirmation from at least two independent "
            "indicator categories (e.g. momentum + trend, or oscillator + volatility band). "
            "Single-indicator high-confidence signals have historically produced more false "
            "positives, particularly in volatile or trending markets."
        ),
        "validation_rule": (
            "If the LLM proposes a signal with confidence > 0.65 and cites only one "
            "indicator category, MODIFY the suggestion to cap confidence at 0.60 and "
            "flag that multi-indicator confirmation is required."
        ),
        "source_name": "Murphy (1999) — Technical Analysis of the Financial Markets [M99]",
        "system_action": "CAP confidence at 0.60 for single-indicator signals; APPROVE multi-indicator",
    },

    # Backtesting Validation Rules
    {
        "id": "BT-001",
        "category": "Backtesting Validation",
        "captured_knowledge": (
            "Any strategy adjustment must be evaluated using four core performance metrics: "
            "Sharpe Ratio (risk-adjusted return), Maximum Drawdown (worst peak-to-trough loss), "
            "Win Rate (percentage of profitable trades), and Total Return. No adjustment should "
            "be considered valid without at least one of these metrics showing improvement."
        ),
        "validation_rule": (
            "If the LLM proposes a parameter change with a score delta > ±0.3 but cites no "
            "backtesting evidence (Sharpe, drawdown, win rate, return), mark as NEEDS_REVIEW. "
            "Do not apply the adjustment without performance justification."
        ),
        "source_name": "Sharpe (1966) — Mutual Fund Performance [S66]; Pardo (2008) [P08]",
        "system_action": "FLAG as NEEDS_REVIEW if large adjustment proposed without backtest evidence",
    },
    {
        "id": "BT-002",
        "category": "Backtesting Validation",
        "captured_knowledge": (
            "If a proposed fuzzy rule adjustment improves Total Return but causes Maximum "
            "Drawdown to worsen by more than 5 percentage points, the adjustment should be "
            "rejected. Capital preservation is prioritised over return maximisation when "
            "drawdown deterioration is significant."
        ),
        "validation_rule": (
            "If the LLM suggests an adjustment that would worsen maximum drawdown beyond "
            "an acceptable threshold while improving return, REJECT the adjustment. "
            "The risk/reward trade-off is unfavourable."
        ),
        "source_name": "Pardo (2008) — The Evaluation and Optimization of Trading Strategies [P08]",
        "system_action": "REJECT adjustments where MaxDD worsens by > 5pp even if return improves",
    },
    {
        "id": "BT-003",
        "category": "Backtesting Validation",
        "captured_knowledge": (
            "If a fuzzy rule adjustment shows improved performance on training data (in-sample) "
            "but degraded performance on testing data (out-of-sample), this pattern is indicative "
            "of curve-fitting or overfitting. The adjustment should be flagged as an overfitting "
            "risk and rejected or significantly reduced."
        ),
        "validation_rule": (
            "If the LLM's proposed adjustment shows train improvement but test degradation, "
            "flag OVERFITTING_RISK and recommend reducing the adjustment magnitude by at least 50%."
        ),
        "source_name": "Pardo (2008) — The Evaluation and Optimization of Trading Strategies [P08]",
        "system_action": "FLAG OVERFITTING_RISK if train improves but test degrades; reduce shift 50%",
    },

    # System Governance
    {
        "id": "SYS-001",
        "category": "System Governance",
        "captured_knowledge": (
            "All LLM-generated fuzzy adjustment suggestions must pass through the Human Expert "
            "Knowledge Base validation process before they can be applied to the final fuzzy "
            "decision output. The LLM is a reasoning assistant, not an authority. "
            "Human expert knowledge encoded in this KB is the authoritative gate."
        ),
        "validation_rule": (
            "This rule is always active. Any suggestion that bypasses or overrides the "
            "validation gate must be immediately rejected. The validation gate is non-optional."
        ),
        "source_name": "ASTDSS PA3 Design Principle — Literature-Grounded Human Expert Validated DSS",
        "system_action": "ENFORCE KB validation for all LLM suggestions; no bypass permitted",
    },
]

# Category index for quick lookup
_KB_BY_CATEGORY: dict[str, list] = {}
for rule in KNOWLEDGE_BASE:
    cat = rule["category"]
    _KB_BY_CATEGORY.setdefault(cat, []).append(rule)

_KB_BY_ID: dict[str, dict] = {rule["id"]: rule for rule in KNOWLEDGE_BASE}


def get_knowledge_base() -> list:
    """Return the full knowledge base."""
    return KNOWLEDGE_BASE


def get_rule_by_id(rule_id: str) -> dict | None:
    """Return a specific rule by its ID (e.g. 'RSI-001')."""
    return _KB_BY_ID.get(rule_id)


def get_rules_by_category(category: str) -> list:
    """Return all rules in a given category."""
    return _KB_BY_CATEGORY.get(category, [])


def get_categories() -> list[str]:
    """Return all distinct categories in the KB."""
    return list(_KB_BY_CATEGORY.keys())


def get_kb_summary() -> dict:
    """Return a compact summary for UI display."""
    return {
        "total_rules": len(KNOWLEDGE_BASE),
        "categories": get_categories(),
        "rules_by_category": {cat: len(rules) for cat, rules in _KB_BY_CATEGORY.items()},
        "rule_ids": [r["id"] for r in KNOWLEDGE_BASE],
    }
