"""
agentic_pipeline.py — Agentic Analysis Pipeline
ASTDSS Terminal — PA3 Agentic Fuzzy Trading DSS

Orchestrates the full agentic decision cycle:
  1. Retrieve or simulate market data + indicators
  2. Run baseline and expert fuzzy analysis (existing engine)
  3. Build structured market context
  4. Call LLM reasoning layer (demo or real)
  5. Validate suggestion via Human Expert Validation Gate
  6. Apply approved fuzzy shift (or keep expert baseline)
  7. Compute stability score
  8. Save structured experience log entry
  9. Return full output dict for UI and PA3 presentation

Entry point: run_agentic_analysis(symbol, demo)
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional


# Label helpers

_SCORE_LABELS = [
    (-2.00, -1.20, "Strong Sell"),
    (-1.20, -0.40, "Sell"),
    (-0.40,  0.40, "Hold"),
    ( 0.40,  1.20, "Buy"),
    ( 1.20,  2.00, "Strong Buy"),
]


def _score_to_label(score: float) -> str:
    for lo, hi, label in _SCORE_LABELS:
        if lo <= score < hi:
            return label
    return "Strong Buy" if score >= 1.20 else "Strong Sell"


def _detect_indicator_conflict(rsi: float, macd_hist: float) -> bool:
    """True when RSI and MACD histogram point in opposite directions at notable levels."""
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
    """
    Compute a 0–1 stability score reflecting signal coherence and validation outcome.

      Base:   expert result confidence
      +0.10   if APPROVED (clean validation pass)
      -0.05   if MODIFIED (partial pass)
      -0.15   if REJECTED (fundamental violation)
      -0.10   if indicator conflict detected
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
    """Generate a short reflection note for the experience log."""
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


# Main Pipeline

def run_agentic_analysis(
    symbol: str = "AAPL",
    demo: bool = True,
    rsi: Optional[float] = None,
    macd_hist: Optional[float] = None,
    atr: Optional[float] = None,
    sentiment: Optional[float] = None,
) -> dict:
    """
    Run the full agentic analysis pipeline.

    Args:
        symbol:    Ticker symbol (used for real data fetch or labelling in demo).
        demo:      True  → simulated market data + demo LLM mode.
                   False → attempt real yfinance data + real LLM (requires API key).
        rsi, macd_hist, atr, sentiment:
                   Optional override values for indicators (used when called
                   from UI with specific inputs).

    Returns:
        Full output dict containing all pipeline stages for UI rendering.
    """
    from tools import fuzzy_analysis_tool, baseline_fuzzy_tool
    from Trading_Data import get_simulated_market_snapshot, get_ticker_snapshot
    from llm_reasoning import generate_llm_suggestion
    from expert_validation_gate import validate_llm_suggestion
    from memory_log import append_experience_log, append_llm_text_log, build_experience_record
    from backtester import get_backtest_evidence
    import anfis

    # Step 1: Market Data
    if rsi is not None:
        # Caller supplied specific indicator values
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

    # Step 2: Baseline + Expert Fuzzy Analysis
    baseline_result = baseline_fuzzy_tool(_rsi, _macd_hist, _sentiment, _atr)
    expert_result   = fuzzy_analysis_tool(_rsi, _macd_hist, _sentiment, _atr, mode="expert")

    # Step 2b: ANFIS — independent data-driven proposer (Jang 1993 hybrid-trained
    # neuro-fuzzy network, see anfis.py). Runs in parallel with the expert engine;
    # never applied directly — its suggestion still passes through the same
    # Human Expert Validation Gate as the LLM suggestion (Step 5).
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

    # Step 3: Market Context
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

    # Step 4: LLM Reasoning
    llm_suggestion = generate_llm_suggestion(
        market_context=market_context,
        fuzzy_output=expert_result,
        demo=demo,
    )

    # Step 5: Human Expert Validation Gate
    # Real numeric backtest evidence (BT-001/BT-002/BT-003) from the cached
    # Baseline vs Expert 70/30 split comparison — same numbers shown on the
    # Backtesting page, so the gate checks actual performance, not just text.
    backtest_delta = get_backtest_evidence()
    validation_result = validate_llm_suggestion(llm_suggestion, market_context, backtest_delta)

    # Step 6: Apply Approved Shift
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

    # Step 7: Stability Score
    stability_score = _compute_stability_score(expert_result, validation_result, has_conflict)

    # Step 8: Experience Log
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
        pass  # Never let logging failure break the pipeline
    try:
        append_llm_text_log(record)
    except Exception:
        pass  # Same — txt mirror is a convenience, not a hard dependency

    # Step 9: Return Full Output
    return {
        "symbol":          symbol.upper(),
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "demo_mode":       demo,
        # Market state
        "market_context":  market_context,
        # Fuzzy analysis layers
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
        # Agentic layer
        "llm_suggestion":    llm_suggestion,
        "validation_result": validation_result,
        "final_decision":    final_decision,
        "indicator_conflict": has_conflict,
        "stability_score":   stability_score,
        "reflection_note":   reflection_note,
    }
