"""
expert_validation_gate.py — Human Expert Validation Gate
ASTDSS Terminal — PA3 Agentic Fuzzy Trading DSS

Receives an LLM-generated suggestion and validates it against the
Human Expert Knowledge Base. Returns a structured validation result
indicating whether the suggestion should be APPROVED, REJECTED,
MODIFIED, or flagged as NEEDS_REVIEW.

Validation logic is rule-based and derived entirely from the KB.
The gate does not itself use an LLM — it applies deterministic checks.
"""

from __future__ import annotations
from typing import Optional
from human_expert_kb import get_rule_by_id, KNOWLEDGE_BASE


def validate_llm_suggestion(
    llm_suggestion: dict,
    market_context: dict,
    backtest_delta: Optional[dict] = None,
) -> dict:
    """
    Validate an LLM suggestion against the Human Expert Knowledge Base.

    Args:
        llm_suggestion:  Output from llm_reasoning.generate_llm_suggestion().
        market_context:  Current market indicators (rsi, macd_hist, atr, sentiment …).
        backtest_delta:  Optional dict with performance impact estimates
                         {"return_delta": float, "maxdd_delta": float,
                          "train_sharpe": float, "test_sharpe": float}.

    Returns:
        {
            "status":           "APPROVED" | "REJECTED" | "MODIFIED" | "NEEDS_REVIEW",
            "reason":           str,
            "applied_shift":    dict,   # fuzzy_shift to actually apply (may be zeroed/halved)
            "violated_rules":   list,   # KB rule IDs that were violated
            "supporting_rules": list,   # KB rule IDs that supported the suggestion
        }
    """
    action   = llm_suggestion.get("proposed_action", "").lower()
    reason   = llm_suggestion.get("reason", "").lower()
    shift    = llm_suggestion.get("fuzzy_shift", {})
    llm_conf = float(llm_suggestion.get("confidence", 0.5))
    direction = shift.get("direction", "hold_neutral")
    score_delta = float(shift.get("score_delta", 0.0))

    rsi       = float(market_context.get("rsi", 50))
    macd_hist = float(market_context.get("macd_hist", 0))
    atr       = float(market_context.get("atr", 3))

    violated_rules: list[str] = []
    supporting_rules: list[str] = []

    # 1. ATR Direction Violation (ATR-001, ATR-002)
    is_bearish_action = any(w in action for w in ("sell", "bear", "bearish", "short"))
    cites_atr_as_direction = (
        ("atr" in reason or "volatil" in reason)
        and is_bearish_action
        and "rsi" not in reason
        and "macd" not in reason
        and "bollinger" not in reason
        and "band" not in reason
    )
    if cites_atr_as_direction:
        violated_rules += ["ATR-001", "ATR-002"]

    # ATR risk penalty approval (ATR-002 supporting case)
    if "atr" in reason and direction == "risk_penalty" and not is_bearish_action:
        supporting_rules.append("ATR-002")

    # 2. RSI Evidence Check (RSI-001, RSI-002, RSI-003)
    is_bullish_action = any(w in action for w in ("buy", "bull", "bullish", "long"))

    if rsi <= 30 and is_bullish_action:
        supporting_rules.append("RSI-001")
    if rsi >= 70 and is_bearish_action:
        supporting_rules.append("RSI-002")

    # RSI-003: single-indicator high-confidence check
    cites_rsi_only = (
        "rsi" in reason
        and "macd" not in reason
        and "bollinger" not in reason
        and "band" not in reason
        and "atr" not in reason
    )
    if cites_rsi_only and llm_conf > 0.70 and score_delta != 0:
        violated_rules.append("RSI-003")

    # 3. MACD Evidence Check (MACD-001, MACD-002, MACD-003)
    if macd_hist > 0 and is_bullish_action:
        supporting_rules.append("MACD-001")
    if macd_hist < 0 and is_bearish_action:
        supporting_rules.append("MACD-002")

    cites_macd_only = (
        "macd" in reason
        and "rsi" not in reason
        and "bollinger" not in reason
        and "band" not in reason
    )
    if cites_macd_only and llm_conf > 0.65 and (is_bullish_action or is_bearish_action):
        violated_rules.append("MACD-003")

    # 4. Multi-Indicator Confirmation (MULTI-001)
    indicator_count = sum([
        "rsi" in reason,
        "macd" in reason,
        "bollinger" in reason or "band" in reason,
        "atr" in reason,
    ])
    if indicator_count >= 2:
        supporting_rules.append("MULTI-001")
    elif llm_conf > 0.65 and abs(score_delta) > 0.05:
        violated_rules.append("MULTI-001")

    # Conflict-reduction: reducing confidence on CONFLICTING_SIGNALS is always good
    if "conflict" in reason and direction in ("reduce_confidence", "hold_neutral"):
        supporting_rules.append("MULTI-001")

    # 5. Backtesting Evidence Check (BT-001)
    # When real backtest_delta evidence is supplied, BT-001 checks whether the
    # evidence actually supports a large shift (numeric check) instead of only
    # checking whether the LLM wrote backtest-sounding words (keyword check).
    has_bt_reference = any(t in reason for t in (
        "sharpe", "drawdown", "win rate", "total return", "backtest", "out-of-sample"
    ))
    if abs(score_delta) > 0.30:
        if backtest_delta is not None:
            evidence_supports_shift = float(backtest_delta.get("return_delta", 0)) > 0
            if not evidence_supports_shift:
                violated_rules.append("BT-001")
        elif not has_bt_reference:
            violated_rules.append("BT-001")

    # 6. Backtest Delta Checks (BT-002, BT-003)
    if backtest_delta:
        return_delta = float(backtest_delta.get("return_delta", 0))
        maxdd_delta  = float(backtest_delta.get("maxdd_delta", 0))
        train_sharpe = float(backtest_delta.get("train_sharpe", 0))
        test_sharpe  = float(backtest_delta.get("test_sharpe", 0))

        if return_delta > 0 and maxdd_delta > 5.0:
            violated_rules.append("BT-002")

        if train_sharpe > 0 and test_sharpe < 0:
            violated_rules.append("BT-003")

    # Deduplicate
    violated_rules   = list(dict.fromkeys(violated_rules))
    supporting_rules = list(dict.fromkeys(supporting_rules))

    # Decision Logic
    critical_violations = {"ATR-001", "ATR-002", "BT-002", "BT-003"}
    has_critical = bool(critical_violations.intersection(set(violated_rules)))

    # BT-001: large shift without backtest → NEEDS_REVIEW
    if "BT-001" in violated_rules and not has_critical:
        return {
            "status": "NEEDS_REVIEW",
            "reason": (
                "Proposed score adjustment exceeds ±0.30 without any backtesting evidence "
                "(Sharpe Ratio, Maximum Drawdown, Win Rate, Total Return). Human review is "
                "required before this adjustment can be applied. Rule BT-001 requires "
                "performance justification for significant parameter changes."
            ),
            "applied_shift": {},
            "violated_rules": violated_rules,
            "supporting_rules": supporting_rules,
        }

    # Critical violations → REJECT
    if has_critical:
        viol_labels = [f"{vid} ({_rule_short(vid)})" for vid in violated_rules if vid in critical_violations]
        return {
            "status": "REJECTED",
            "reason": (
                f"Rejected by Human Expert Validation Gate. Critical rule violations detected: "
                f"{', '.join(viol_labels)}. "
                f"{_get_rejection_explanation(violated_rules)}"
            ),
            "applied_shift": {},
            "violated_rules": violated_rules,
            "supporting_rules": supporting_rules,
        }

    # Non-critical violations → MODIFIED (halve the shift)
    if violated_rules:
        reduced = {
            "score_delta":      round(score_delta * 0.5, 4),
            "confidence_delta": round(float(shift.get("confidence_delta", 0)) * 0.5, 4),
            "direction":        direction,
        }
        return {
            "status": "MODIFIED",
            "reason": (
                f"Partially approved with reduced magnitude (50%). Non-critical violations: "
                f"{', '.join(violated_rules)}. The directional intent is acceptable but the "
                f"suggested magnitude exceeds what the KB rules currently support. "
                f"Supporting evidence: {', '.join(supporting_rules) if supporting_rules else 'none'}."
            ),
            "applied_shift": reduced,
            "violated_rules": violated_rules,
            "supporting_rules": supporting_rules,
        }

    # No violations → APPROVE
    return {
        "status": "APPROVED",
        "reason": (
            f"LLM suggestion validated against Human Expert Knowledge Base. "
            f"No rule violations detected. Supporting rules: "
            f"{', '.join(supporting_rules) if supporting_rules else 'General consistency with KB'}."
        ),
        "applied_shift": {
            "score_delta":      round(score_delta, 4),
            "confidence_delta": round(float(shift.get("confidence_delta", 0)), 4),
            "direction":        direction,
        },
        "violated_rules": [],
        "supporting_rules": supporting_rules,
    }


def _rule_short(rule_id: str) -> str:
    """Return a one-line description for a rule ID."""
    rule = get_rule_by_id(rule_id)
    if rule:
        kb = rule["captured_knowledge"]
        return kb[:80] + "..."
    return rule_id


def _get_rejection_explanation(violated_rules: list[str]) -> str:
    """Return a targeted explanation for the most critical violation."""
    if "ATR-001" in violated_rules or "ATR-002" in violated_rules:
        return (
            "ATR (Average True Range) is a volatility measure — it indicates how much price "
            "moves, NOT in which direction. Using ATR as a directional Sell trigger is a "
            "fundamental misapplication of this indicator. The correct response to high ATR "
            "is to increase the risk penalty and reduce position confidence, not to issue a "
            "directional trade signal."
        )
    if "BT-002" in violated_rules:
        return (
            "The proposed adjustment would worsen Maximum Drawdown beyond acceptable limits "
            "even if Total Return improves. Capital preservation takes precedence over "
            "return maximisation when drawdown deterioration is material."
        )
    if "BT-003" in violated_rules:
        return (
            "The proposed adjustment exhibits an overfitting pattern: in-sample performance "
            "improves but out-of-sample performance degrades. This pattern indicates the "
            "adjustment is curve-fitted to historical noise, not a genuine edge."
        )
    return "Refer to Human Expert Knowledge Base for detailed rule explanations."
