"""Saves every agentic run to data/experience_log.json."""

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
    """Write to a temp file and rename it, so a crash halfway cannot break the log."""
    tmp_path = _LOG_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, _LOG_FILE)


def append_experience_log(record: dict) -> None:
    """Add one record to the log.

    A record has timestamp, market_context, baseline_fuzzy_output,
    llm_suggestion, expert_validation_result, final_decision, fuzzy_shift,
    stability_score (0-1) and reflection_note.
    """
    _ensure_data_dir()
    existing = load_experience_log()
    existing.append(record)
    _atomic_write_json(existing)


def load_experience_log() -> list:
    """The whole log, oldest first."""
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
    """The last `limit` records, newest first."""
    return list(reversed(load_experience_log()))[:limit]


def clear_experience_log() -> None:
    """Empty the log (for tests)."""
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
    """Build a record for append_experience_log()."""
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
        # the exact prompt and reply, kept apart from llm_suggestion
        # so the summary table stays short
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
    """Write the LLM prompt and reply of one record to data/llm_prompt_log.txt.

    It repeats what is already in experience_log.json, but as plain text, so
    you can read exactly what was sent and what came back without opening
    JSON. Both logs come from the same record, so the timestamps should match.
    When DEEPSEEK_API_KEY is missing, every entry shows mode=demo and empty
    prompts, which is a quick way to see the LLM is not running.
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
