"""The full agentic run, from market data to the logged decision.

Steps: get data, run the baseline and expert fuzzy engines (and ANFIS),
build the context, ask the LLM for a shift, check it with the validation
gate, apply what was approved, score stability, write the log entry.

Start with run_agentic_analysis(symbol, demo).
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional


_SCORE_LABELS = [
    (-2.00, -1.20, "Strong Sell"),
    (-1.20, -0.40, "Sell"),
    (-0.40,  0.40, "Hold"),
    ( 0.40,  1.20, "Buy"),
    ( 1.20,  2.00, "Strong Buy"),
]


def _score_to_label(score: float) -> str:
    """Map a combined score to a trading label."""
    for lo, hi, label in _SCORE_LABELS:
        if lo <= score < hi:
            return label
    return "Strong Buy" if score >= 1.20 else "Strong Sell"


def _detect_indicator_conflict(rsi: float, macd_hist: float) -> bool:
    """True when RSI and the MACD histogram clearly point opposite ways."""
    rsi_bullish  = rsi < 45
    rsi_bearish  = rsi > 55
    macd_bullish = macd_hist > 0.10
    macd_bearish = macd_hist < -0.10
    return (rsi_bullish and macd_bearish) or (rsi_bearish and macd_bullish)


def _compute_stability_score(
    expert_result: dict,
    validation_result: dict,
    has_conflict: bool,
) -> float:
    """Stability score from 0 to 1.

    Starts at the expert confidence, +0.10 if approved, -0.05 if modified,
    -0.15 if rejected, and -0.10 more when RSI and MACD disagree.
    """
    base = float(expert_result.get("confidence", 0.5))
    status = validation_result.get("status", "UNKNOWN")

    adjustment = {
        "APPROVED":     +0.10,
        "MODIFIED":     -0.05,
        "REJECTED":     -0.15,
        "NEEDS_REVIEW": -0.08,
    }.get(status, 0.0)

    score = base + adjustment - (0.10 if has_conflict else 0.0)
    return round(max(0.0, min(1.0, score)), 3)


def _build_reflection_note(validation_result: dict, has_conflict: bool) -> str:
    """One-line note for the experience log."""
    status = validation_result.get("status", "UNKNOWN")
    violated = validation_result.get("violated_rules", [])

    if status == "APPROVED":
        return "Agentic adjustment approved and applied. Indicator alignment is consistent with Human Expert KB."
    if status == "REJECTED":
        return (
            f"Adjustment rejected by Human Expert Validation Gate. Violated rules: {', '.join(violated)}. "
            "Baseline expert fuzzy output retained without modification."
        )
    if status == "MODIFIED":
        return (
            f"Adjustment partially approved with 50% magnitude reduction. Violated: {', '.join(violated)}. "
            "Full shift was not justified by current indicator evidence."
        )
    if status == "NEEDS_REVIEW":
        return (
            "Large adjustment proposed without backtesting evidence (BT-001). "
            "Flagged for human review. No shift applied pending validation."
        )
    if has_conflict:
        return "Conflicting RSI/MACD signals detected. Confidence dampened per CONFLICTING_SIGNALS regime rule."
    return "Agentic cycle completed. No unusual conditions detected."


def run_agentic_analysis(
    symbol: str = "AAPL",
    demo: bool = True,
    rsi: Optional[float] = None,
    macd_hist: Optional[float] = None,
    atr: Optional[float] = None,
    sentiment: Optional[float] = None,
) -> dict:
    """Run the whole pipeline and return every stage for the UI.

    demo=True uses simulated data and the demo LLM. demo=False tries yfinance
    and the real LLM (needs an API key). rsi, macd_hist, atr and sentiment can
    be passed in to skip the data step.
    """
    from tools import fuzzy_analysis_tool, baseline_fuzzy_tool
    from Trading_Data import get_simulated_market_snapshot, get_ticker_snapshot
    from llm_reasoning import generate_llm_suggestion
    from expert_validation_gate import validate_llm_suggestion
    from memory_log import append_experience_log, append_llm_text_log, build_experience_record
    from backtester import get_backtest_evidence
    import anfis

    # 1. market data
    if rsi is not None:
        # values given by the caller
        snapshot = {
            "rsi":        rsi,
            "macd_hist":  macd_hist or 0.0,
            "atr":        atr or 3.0,
            "sentiment":  sentiment or 0.0,
            "price":      0.0,
            "volatility": 0.0,
            "source":     "manual_input",
        }
    elif demo:
        snapshot = get_simulated_market_snapshot()
    else:
        snapshot = get_ticker_snapshot(symbol, period="90d")
        if "simulated" in snapshot.get("source", ""):
            snapshot = get_simulated_market_snapshot()

    _rsi       = float(snapshot.get("rsi", 50))
    _macd_hist = float(snapshot.get("macd_hist", 0))
    _atr       = float(snapshot.get("atr", 3))
    _sentiment = float(snapshot.get("sentiment", 0))
    _price     = float(snapshot.get("price", 0))

    # 2. baseline and expert fuzzy results
    baseline_result = baseline_fuzzy_tool(_rsi, _macd_hist, _sentiment, _atr)
    expert_result   = fuzzy_analysis_tool(_rsi, _macd_hist, _sentiment, _atr, mode="expert")

    # 2b. ANFIS (see anfis.py). It runs next to the expert engine and is never
    # applied directly; its result goes through the same validation gate
    # as the LLM suggestion.
    try:
        anfis_result = anfis.predict_score(_rsi, _macd_hist, _atr)
    except Exception as exc:
        anfis_result = {"score": 0.0, "label": "Hold", "confidence": 0.0,
                         "mode": "error", "note": str(exc)}

    market_regime_info = expert_result.get("market_regime") or {}
    market_regime = (
        market_regime_info.get("regime", "NORMAL")
        if isinstance(market_regime_info, dict)
        else str(market_regime_info)
    )

    has_conflict = _detect_indicator_conflict(_rsi, _macd_hist)

    # 3. context for the LLM
    market_context = {
        "symbol":           symbol.upper(),
        "rsi":              _rsi,
        "macd_hist":        _macd_hist,
        "atr":              _atr,
        "sentiment":        _sentiment,
        "price":            _price,
        "volatility":       float(snapshot.get("volatility", 0)),
        "market_regime":    market_regime,
        "baseline_score":   baseline_result.get("score", 0.0),
        "baseline_label":   baseline_result.get("label", "Hold"),
        "expert_score":     expert_result.get("score", 0.0),
        "expert_label":     expert_result.get("label", "Hold"),
        "anfis_score":      anfis_result.get("score", 0.0),
        "anfis_label":      anfis_result.get("label", "Hold"),
        "indicator_conflict": has_conflict,
        "demo_mode":        demo,
    }

    # 4. LLM suggestion
    llm_suggestion = generate_llm_suggestion(
        market_context=market_context,
        fuzzy_output=expert_result,
        demo=demo,
    )

    # 5. validation gate
    # backtest numbers for rules BT-001 to BT-003 come from the cached
    # baseline vs expert comparison, the same one the Backtesting page shows,
    # so the gate looks at real results and not only the text
    backtest_delta = get_backtest_evidence()
    validation_result = validate_llm_suggestion(llm_suggestion, market_context, backtest_delta)

    # 6. apply the approved shift
    base_score     = expert_result.get("score", 0.0)
    base_conf      = expert_result.get("confidence", 0.5)
    applied_shift  = validation_result.get("applied_shift", {})

    final_score = base_score + float(applied_shift.get("score_delta", 0.0))
    final_conf  = base_conf  + float(applied_shift.get("confidence_delta", 0.0))
    final_score = round(max(-2.0, min(2.0, final_score)), 4)
    final_conf  = round(max(0.0,  min(1.0, final_conf)), 3)
    final_label = _score_to_label(final_score)

    final_decision = {
        "score":      final_score,
        "label":      final_label,
        "confidence": final_conf,
    }

    # 7. stability
    stability_score = _compute_stability_score(expert_result, validation_result, has_conflict)

    # 8. experience log
    reflection_note = _build_reflection_note(validation_result, has_conflict)
    record = build_experience_record(
        market_context=market_context,
        baseline_result=baseline_result,
        llm_suggestion=llm_suggestion,
        validation_result=validation_result,
        final_decision=final_decision,
        stability_score=stability_score,
        reflection_note=reflection_note,
        anfis_result=anfis_result,
    )
    try:
        append_experience_log(record)
    except Exception:
        pass  # a logging problem should never stop the run
    try:
        append_llm_text_log(record)
    except Exception:
        pass  # same for the text copy of the log

    # 9. everything the UI needs
    return {
        "symbol":          symbol.upper(),
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "demo_mode":       demo,
        # market
        "market_context":  market_context,
        # fuzzy results
        "baseline_result": {
            "score":      baseline_result.get("score", 0.0),
            "label":      baseline_result.get("label", "Hold"),
            "confidence": baseline_result.get("confidence", 0.5),
        },
        "expert_result": {
            "score":         expert_result.get("score", 0.0),
            "label":         expert_result.get("label", "Hold"),
            "confidence":    expert_result.get("confidence", 0.5),
            "market_regime": market_regime,
            "active_features": expert_result.get("active_features", []),
            "masked_features": expert_result.get("masked_features", []),
        },
        "anfis_result":    anfis_result,
        # agentic part
        "llm_suggestion":    llm_suggestion,
        "validation_result": validation_result,
        "final_decision":    final_decision,
        "indicator_conflict": has_conflict,
        "stability_score":   stability_score,
        "reflection_note":   reflection_note,
    }
