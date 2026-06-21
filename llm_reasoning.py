"""
llm_reasoning.py — LLM Reasoning Layer
ASTDSS Terminal — PA3 Agentic Fuzzy Trading DSS

Provides two modes:
  demo  — Deterministic scenario-based output for presentation/testing.
           No API key required. Reproduces five distinct reasoning scenarios
           based on the current market context.
  real  — Optional DeepSeek API call. Only activated when DEEPSEEK_API_KEY
           is present in the environment (.env file). Falls back to demo if
           the key is missing or the API call fails.

Output contract (both modes):
  {
      "proposed_action": str,       # Human-readable action description
      "reason":          str,       # Reasoning chain referencing indicators
      "fuzzy_shift": {
          "score_delta":      float,  # Additive shift to the fuzzy score (-2..+2)
          "confidence_delta": float,  # Additive shift to confidence (0..1)
          "direction":        str,    # "boost_buy" | "boost_sell" | "reduce_confidence"
                                      # | "risk_penalty" | "hold_neutral"
      },
      "confidence":          float,  # LLM's self-assessed confidence (0..1)
      "scenario_tag":        str,    # Internal scenario identifier for logging
      "mode":                str,    # "demo" | "real"
  }
"""

from __future__ import annotations
import os
from typing import Optional


# Demo scenario definitions

_SCENARIOS = {
    "CONFLICT_RSI_MACD": {
        "proposed_action": (
            "Reduce confidence and bias toward Hold. RSI indicates overbought territory "
            "while MACD histogram is simultaneously negative, creating a conflicting signal "
            "environment. The opposing momentum and oscillator readings reduce certainty."
        ),
        "reason": (
            "RSI is above 65 (overbought), which normally supports a Sell signal. However, "
            "the MACD histogram is negative (bearish momentum confirmed), yet both indicators "
            "together create a CONFLICTING_SIGNALS regime. When RSI and MACD disagree on "
            "direction at these extreme levels, historical analysis shows a higher rate of "
            "false directional signals. The recommended adjustment is to reduce overall fuzzy "
            "confidence and dampen the score toward a neutral Hold position rather than "
            "committing to a directional trade."
        ),
        "fuzzy_shift": {
            "score_delta": -0.15,
            "confidence_delta": -0.12,
            "direction": "reduce_confidence",
        },
        "confidence": 0.72,
    },
    "STRONG_BUY_CONFLUENCE": {
        "proposed_action": (
            "Moderately boost buy confidence. RSI is in oversold territory while MACD "
            "histogram is positive, creating bullish confluence across two independent "
            "indicator categories (oscillator + momentum)."
        ),
        "reason": (
            "RSI is below 35 (oversold), indicating prices have fallen rapidly and a "
            "mean-reversion bounce is statistically probable. Simultaneously, the MACD "
            "histogram is positive, confirming upside momentum is present. This two-indicator "
            "confluence (RSI oscillator + MACD momentum) meets the multi-indicator confirmation "
            "requirement. The fuzzy score adjustment is kept moderate (+0.15) to avoid "
            "overstating confidence in a single-session signal."
        ),
        "fuzzy_shift": {
            "score_delta": 0.15,
            "confidence_delta": 0.10,
            "direction": "boost_buy",
        },
        "confidence": 0.78,
    },
    "HIGH_VOLATILITY_RISK_PENALTY": {
        "proposed_action": (
            "Apply a risk penalty by reducing confidence. ATR is elevated above 6.0, "
            "indicating unusually wide price swings. This is a risk management adjustment, "
            "NOT a directional call."
        ),
        "reason": (
            "ATR is above 6.0 (on the normalised 0–10 scale), signalling high market "
            "volatility. In high-ATR environments, stop-loss distances must be widened and "
            "position sizing reduced to manage adverse excursion risk. The correct response "
            "is to reduce the fuzzy output confidence — reflecting increased uncertainty — "
            "not to issue a directional Sell signal. ATR measures how much price moves, not "
            "in which direction it will move. The risk-penalty confidence reduction of -0.10 "
            "is calibrated to the ATR normalised value."
        ),
        "fuzzy_shift": {
            "score_delta": 0.0,
            "confidence_delta": -0.10,
            "direction": "risk_penalty",
        },
        "confidence": 0.80,
    },
    "STRONG_SELL_CONFLUENCE": {
        "proposed_action": (
            "Moderately boost sell confidence. RSI is in overbought territory while MACD "
            "histogram is negative, providing bearish confluence across oscillator and "
            "momentum categories."
        ),
        "reason": (
            "RSI is above 68 (overbought), indicating a stretched rally susceptible to "
            "reversal. The MACD histogram is simultaneously negative, confirming that short-term "
            "momentum has already turned bearish while the oscillator remains extended. "
            "This dual confirmation (RSI oscillator + MACD momentum, both bearish) satisfies "
            "the multi-indicator confirmation requirement and supports a moderate sell bias."
        ),
        "fuzzy_shift": {
            "score_delta": -0.15,
            "confidence_delta": 0.08,
            "direction": "boost_sell",
        },
        "confidence": 0.74,
    },
    "NORMAL_MINOR_ADJUSTMENT": {
        "proposed_action": (
            "Apply a minor upward confidence alignment. Indicators are in moderate ranges "
            "with no extremes detected. The fuzzy output appears well-calibrated to current "
            "market conditions; only a marginal adjustment is warranted."
        ),
        "reason": (
            "RSI is in the neutral zone (35–65), MACD histogram magnitude is low, and ATR "
            "is within normal range (2–6). No single indicator is at an extreme that would "
            "justify a directional boost. The existing fuzzy score and market regime capture "
            "the current conditions adequately. A small positive confidence adjustment (+0.05) "
            "reflects the absence of contradictory signals rather than any strong directional "
            "conviction."
        ),
        "fuzzy_shift": {
            "score_delta": 0.05,
            "confidence_delta": 0.05,
            "direction": "hold_neutral",
        },
        "confidence": 0.60,
    },
}


def _select_demo_scenario(market_context: dict) -> str:
    """Deterministically pick a scenario tag from market context values."""
    rsi = market_context.get("rsi", 50)
    macd_hist = market_context.get("macd_hist", 0)
    atr = market_context.get("atr", 3)

    # Priority-ordered checks
    if rsi > 65 and macd_hist < -0.15:
        return "CONFLICT_RSI_MACD"
    if rsi < 35 and macd_hist > 0.1:
        return "STRONG_BUY_CONFLUENCE"
    if rsi > 68 and macd_hist < 0:
        return "STRONG_SELL_CONFLUENCE"
    if atr > 6.0:
        return "HIGH_VOLATILITY_RISK_PENALTY"
    return "NORMAL_MINOR_ADJUSTMENT"


def _build_demo_output(market_context: dict) -> dict:
    tag = _select_demo_scenario(market_context)
    scenario = _SCENARIOS[tag]
    return {
        "proposed_action": scenario["proposed_action"],
        "reason": scenario["reason"],
        "fuzzy_shift": dict(scenario["fuzzy_shift"]),
        "confidence": scenario["confidence"],
        "scenario_tag": tag,
        "mode": "demo",
    }


def _build_real_output(market_context: dict, fuzzy_output: dict) -> dict:
    """
    Call the DeepSeek API to generate a structured reasoning output.
    DeepSeek is OpenAI-API-compatible; uses the openai package with a custom base_url.

    Requires DEEPSEEK_API_KEY in environment (loaded from .env via python-dotenv).
    Falls back to demo mode silently if the key is missing or the call fails.
    """
    import json
    import re

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-your-"):
        return _build_demo_output(market_context)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        system_prompt = (
            "You are a structured financial reasoning assistant for an academic "
            "Decision Support System (DSS). Your role is to propose a CANDIDATE "
            "fuzzy rule adjustment based on current market indicator values. "
            "You are NOT the final authority — your suggestion will be validated "
            "by a Human Expert Knowledge Base before it can be applied.\n\n"
            "Rules you must follow:\n"
            "- ATR measures VOLATILITY only, not price DIRECTION. Never cite ATR as a reason to Sell.\n"
            "- Do not base high-confidence signals on a single indicator. Require 2+ indicators.\n"
            "- Keep score_delta within [-0.5, +0.5]. Keep confidence_delta within [-0.3, +0.3].\n"
            "- Be conservative. Prefer small adjustments unless clearly justified.\n\n"
            "Respond ONLY with valid JSON, no markdown, no explanation outside JSON:\n"
            "{\n"
            '  "proposed_action": "One sentence describing the recommended adjustment",\n'
            '  "reason": "2-3 sentences referencing RSI, MACD, ATR, or Bollinger Band values",\n'
            '  "fuzzy_shift": {\n'
            '    "score_delta": <float -0.5 to 0.5>,\n'
            '    "confidence_delta": <float -0.3 to 0.3>,\n'
            '    "direction": "<boost_buy|boost_sell|reduce_confidence|risk_penalty|hold_neutral>"\n'
            "  },\n"
            '  "confidence": <float 0.0 to 1.0>\n'
            "}"
        )

        user_prompt = (
            "Market Context (current indicator values):\n"
            f"{json.dumps(market_context, indent=2)}\n\n"
            "Current Expert Fuzzy Output:\n"
            f"{json.dumps(fuzzy_output, indent=2)}\n\n"
            "Propose a candidate fuzzy adjustment. "
            "Reference the exact RSI, MACD, ATR values in your reason. "
            "Respond with JSON only."
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=500,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            data["scenario_tag"] = "DEEPSEEK_REAL"
            data["mode"] = "real"
            # Enforce numeric safety bounds before passing to validation gate
            shift = data.get("fuzzy_shift", {})
            shift["score_delta"]      = max(-0.5, min(0.5, float(shift.get("score_delta", 0))))
            shift["confidence_delta"] = max(-0.3, min(0.3, float(shift.get("confidence_delta", 0))))
            data["fuzzy_shift"]  = shift
            data["confidence"]   = max(0.0, min(1.0, float(data.get("confidence", 0.6))))
            return data

    except Exception as exc:
        # Never break the pipeline over LLM unavailability — but log it so a
        # failed real-API call isn't mistaken for a successful one.
        print(f"[llm_reasoning] DeepSeek API call failed, falling back to demo mode: {exc}")

    return _build_demo_output(market_context)


def generate_llm_suggestion(
    market_context: dict,
    fuzzy_output: Optional[dict] = None,
    demo: bool = True,
) -> dict:
    """
    Generate a candidate fuzzy adjustment suggestion.

    Args:
        market_context: Dict with rsi, macd_hist, atr, sentiment, market_regime, etc.
        fuzzy_output:   Current fuzzy engine output (score, label, confidence).
        demo:           If True, use deterministic demo mode. If False, attempt real API call.

    Returns:
        Structured suggestion dict (see module docstring for contract).
    """
    if fuzzy_output is None:
        fuzzy_output = {}

    if demo:
        return _build_demo_output(market_context)
    return _build_real_output(market_context, fuzzy_output)
