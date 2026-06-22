"""
memory_log.py — Structured Experience Log
ASTDSS Terminal — PA3 Agentic Fuzzy Trading DSS

Persists each agentic decision cycle to data/experience_log.json.
Each record captures the full context-to-decision chain for analysis,
auditing, and PA3 demonstration purposes.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_LOG_FILE = os.path.join(_DATA_DIR, "experience_log.json")
_TXT_LOG_FILE = os.path.join(_DATA_DIR, "llm_prompt_log.txt")


def _ensure_data_dir() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)


def _atomic_write_json(data: list) -> None:
    """Write JSON to a temp file then rename, so a crash mid-write can't corrupt the log."""
    tmp_path = _LOG_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, _LOG_FILE)


def append_experience_log(record: dict) -> None:
    """
    Append a single experience record to the log file.

    The record should contain:
      - timestamp              : ISO-format datetime string
      - market_context         : dict of indicator values
      - baseline_fuzzy_output  : dict (score, label, confidence)
      - llm_suggestion         : dict from llm_reasoning
      - expert_validation_result: dict from expert_validation_gate
      - final_decision         : dict (score, label)
      - fuzzy_shift            : dict (actually applied shift)
      - stability_score        : float 0–1
      - reflection_note        : str (human-readable summary)
    """
    _ensure_data_dir()
    existing = load_experience_log()
    existing.append(record)
    _atomic_write_json(existing)


def load_experience_log() -> list:
    """Return the full experience log as a list of dicts (oldest first)."""
    _ensure_data_dir()
    if not os.path.exists(_LOG_FILE):
        return []
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def get_recent_experience(limit: int = 5) -> list:
    """Return the most recent `limit` experience records (newest first)."""
    return list(reversed(load_experience_log()))[:limit]


def clear_experience_log() -> None:
    """Erase all experience log entries. Used for testing/reset."""
    _ensure_data_dir()
    _atomic_write_json([])


def build_experience_record(
    market_context: dict,
    baseline_result: dict,
    llm_suggestion: dict,
    validation_result: dict,
    final_decision: dict,
    stability_score: float,
    reflection_note: str = "",
    anfis_result: Optional[dict] = None,
) -> dict:
    """
    Helper to assemble a well-structured experience record.
    Call this before append_experience_log().
    """
    applied_shift = validation_result.get("applied_shift", {})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "market_context": {
            "symbol":       market_context.get("symbol", "DEMO"),
            "rsi":          round(float(market_context.get("rsi", 50)), 2),
            "macd_hist":    round(float(market_context.get("macd_hist", 0)), 4),
            "atr":          round(float(market_context.get("atr", 3)), 2),
            "sentiment":    round(float(market_context.get("sentiment", 0)), 3),
            "market_regime": market_context.get("market_regime", "NORMAL"),
            "price":        round(float(market_context.get("price", 0)), 2),
        },
        "baseline_fuzzy_output": {
            "score":      round(float(baseline_result.get("score", 0)), 4),
            "label":      baseline_result.get("label", "Hold"),
            "confidence": round(float(baseline_result.get("confidence", 0.5)), 3),
        },
        "llm_suggestion": {
            "proposed_action": llm_suggestion.get("proposed_action", "")[:200],
            "confidence":      round(float(llm_suggestion.get("confidence", 0.5)), 3),
            "scenario_tag":    llm_suggestion.get("scenario_tag", "UNKNOWN"),
            "mode":            llm_suggestion.get("mode", "demo"),
            "score_delta":     float(llm_suggestion.get("fuzzy_shift", {}).get("score_delta", 0)),
            "direction":       llm_suggestion.get("fuzzy_shift", {}).get("direction", "hold_neutral"),
        },
        # Full prompt/response audit trail — answers "where can I see the
        # exact prompt sent to the LLM and what it returned" (PA3 review).
        # Kept separate from llm_suggestion above so the summary table stays
        # short while the raw record below remains fully inspectable.
        "llm_prompt_log": {
            "system_prompt": (llm_suggestion.get("prompt_log") or {}).get("system_prompt"),
            "user_prompt":   (llm_suggestion.get("prompt_log") or {}).get("user_prompt"),
            "raw_response":  (llm_suggestion.get("prompt_log") or {}).get("raw_response"),
            "note":          (llm_suggestion.get("prompt_log") or {}).get("note"),
        },
        "anfis_result": {
            "score":      round(float((anfis_result or {}).get("score", 0.0)), 4),
            "label":      (anfis_result or {}).get("label", "Hold"),
            "confidence": round(float((anfis_result or {}).get("confidence", 0.0)), 4),
            "mode":       (anfis_result or {}).get("mode", "untrained"),
        },
        "expert_validation_result": {
            "status":           validation_result.get("status", "UNKNOWN"),
            "reason":           validation_result.get("reason", "")[:300],
            "violated_rules":   validation_result.get("violated_rules", []),
            "supporting_rules": validation_result.get("supporting_rules", []),
        },
        "final_decision": {
            "score":      round(float(final_decision.get("score", 0)), 4),
            "label":      final_decision.get("label", "Hold"),
            "confidence": round(float(final_decision.get("confidence", 0.5)), 3),
        },
        "fuzzy_shift": {
            "score_delta":      round(float(applied_shift.get("score_delta", 0)), 4),
            "confidence_delta": round(float(applied_shift.get("confidence_delta", 0)), 4),
            "direction":        applied_shift.get("direction", "none"),
        },
        "stability_score": round(float(stability_score), 3),
        "reflection_note": reflection_note or validation_result.get("reason", "")[:200],
    }


def append_llm_text_log(record: dict) -> None:
    """
    Append a plain-text, human-readable mirror of one experience record's
    LLM prompt/response to data/llm_prompt_log.txt.

    This exists alongside the structured JSON log in experience_log.json on
    purpose: it's a redundant, independently-inspectable record so anyone
    (including a non-technical reviewer) can open a .txt file and see, in
    full, exactly what was sent to the LLM and what came back — without
    needing to parse JSON. Because both logs are built from the same
    `record` dict in the same call (see agentic_pipeline.py), the timestamp
    in this file should always match an entry in experience_log.json; if it
    ever doesn't, that itself is a signal something broke between the two
    writes.

    Also doubles as a quick "is the LLM actually running" check: when
    DEEPSEEK_API_KEY is missing/invalid, every block here will show
    mode=demo and empty SYSTEM PROMPT / USER PROMPT / RAW RESPONSE fields.
    """
    _ensure_data_dir()

    ctx = record.get("market_context", {})
    llm = record.get("llm_suggestion", {})
    plog = record.get("llm_prompt_log", {})

    def _block(label: str, text) -> str:
        text = text or "(empty — demo mode or no live LLM call was made)"
        return f"{label}:\n{text}\n"

    lines = [
        "=" * 80,
        f"TIMESTAMP : {record.get('timestamp', '')}",
        f"SYMBOL    : {ctx.get('symbol', 'DEMO')}",
        f"MODE      : {llm.get('mode', 'demo')}",
        "-" * 80,
        _block("SYSTEM PROMPT", plog.get("system_prompt")),
        _block("USER PROMPT", plog.get("user_prompt")),
        _block("RAW LLM RESPONSE", plog.get("raw_response")),
        "-" * 80,
        f"PARSED proposed_action : {llm.get('proposed_action', '')}",
        f"PARSED scenario_tag    : {llm.get('scenario_tag', '')}",
        f"PARSED score_delta     : {llm.get('score_delta', '')}",
        f"PARSED direction       : {llm.get('direction', '')}",
        f"PARSED confidence      : {llm.get('confidence', '')}",
    ]
    if plog.get("note"):
        lines.append(f"NOTE                    : {plog.get('note')}")
    lines.append("=" * 80)
    lines.append("")

    with open(_TXT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
