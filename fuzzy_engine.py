"""Mamdani fuzzy engine for RSI, MACD histogram, sentiment and ATR.

compute_score() treats every input the same (baseline).
compute_expert_adjusted_score() uses the weights from the expert layer.
Rules use min for AND, the output is found with the centroid method.
"""

from __future__ import annotations
import math
from typing import Dict, List, Literal, Optional, Tuple


SignalLabel = Literal["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]

# membership functions

def _trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangle membership. x == a or x == c gives 0, x == b gives 1.
    Also works when a == b.
    """
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if a <= x < b:
        return (x - a) / (b - a) if b != a else 1.0
    if b < x <= c:
        return (c - x) / (c - b) if c != b else 0.0
    return 0.0


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid membership.

    The outer checks are strict (x < a, x > d) so values right on the edge,
    like sentiment = -1 or rsi = 100, still get full membership. Works with
    a == b or c == d too.
    """
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a <= x < b:
        return (x - a) / (b - a) if b != a else 1.0
    if c < x <= d:
        return (d - x) / (d - c) if d != c else 0.0
    return 0.0


# RSI, 0-100

def _rsi_oversold(rsi: float) -> float:
    """RSI oversold: full membership below 25."""
    return _trapmf(rsi, 0, 0, 20, 35)


def _rsi_neutral(rsi: float) -> float:
    """RSI neutral: triangle around 50."""
    return _trimf(rsi, 30, 50, 70)


def _rsi_overbought(rsi: float) -> float:
    """RSI overbought: full membership above 75."""
    return _trapmf(rsi, 65, 80, 100, 100)


# MACD histogram, -5..5

def _macd_bearish(hist: float) -> float:
    return _trapmf(hist, -5, -5, -0.8, -0.2)


def _macd_neutral(hist: float) -> float:
    return _trimf(hist, -0.5, 0, 0.5)


def _macd_bullish(hist: float) -> float:
    return _trapmf(hist, 0.2, 0.8, 5, 5)


# sentiment, -1..1

def _sent_bearish(s: float) -> float:
    return _trapmf(s, -1, -1, -0.55, -0.25)


def _sent_neutral(s: float) -> float:
    return _trimf(s, -0.35, 0, 0.35)


def _sent_bullish(s: float) -> float:
    return _trapmf(s, 0.25, 0.55, 1, 1)


# ATR, 0-10

def _atr_low(atr: float) -> float:
    return _trapmf(atr, 0, 0, 1.5, 3.0)


def _atr_normal(atr: float) -> float:
    return _trimf(atr, 2.0, 4.5, 7.0)


def _atr_high(atr: float) -> float:
    return _trapmf(atr, 6.0, 8.5, 10, 10)


# output score, -2..2

OUTPUT_TERMS: Dict[str, Tuple[float, float, float]] = {
    "Strong Sell": (-2.0, -2.0, -1.0),
    "Sell":        (-1.5, -0.8, -0.2),
    "Hold":        (-0.4,  0.0,  0.4),
    "Buy":         ( 0.2,  0.8,  1.5),
    "Strong Buy":  ( 1.0,  2.0,  2.0),
}


def _output_mu(score: float, label: str) -> float:
    a, b, c = OUTPUT_TERMS[label]
    return _trimf(score, a, b, c)


# rules

MEMBERSHIP_FNS = {
    "rsi": {
        "oversold": _rsi_oversold,
        "neutral": _rsi_neutral,
        "overbought": _rsi_overbought,
    },
    "macd_hist": {
        "bearish": _macd_bearish,
        "neutral": _macd_neutral,
        "bullish": _macd_bullish,
    },
    "sentiment": {
        "bearish": _sent_bearish,
        "neutral": _sent_neutral,
        "bullish": _sent_bullish,
    },
    "atr": {
        "low": _atr_low,
        "normal": _atr_normal,
        "high": _atr_high,
    },
}

RULES = [
    # RSI + MACD
    ({"rsi": ("overbought", 1.0), "macd_hist": ("bearish", 1.0)}, "Strong Sell"),
    ({"rsi": ("overbought", 1.0), "macd_hist": ("neutral", 0.6)}, "Sell"),
    ({"rsi": ("overbought", 1.0), "macd_hist": ("bullish", 0.5)}, "Hold"),
    ({"rsi": ("oversold", 1.0), "macd_hist": ("bullish", 1.0)}, "Strong Buy"),
    ({"rsi": ("oversold", 1.0), "macd_hist": ("neutral", 0.6)}, "Buy"),
    ({"rsi": ("oversold", 1.0), "macd_hist": ("bearish", 0.5)}, "Hold"),
    ({"rsi": ("neutral", 1.0), "macd_hist": ("bullish", 0.7)}, "Buy"),
    ({"rsi": ("neutral", 1.0), "macd_hist": ("bearish", 0.7)}, "Sell"),
    ({"rsi": ("neutral", 1.0), "macd_hist": ("neutral", 1.0)}, "Hold"),

    # RSI + sentiment
    ({"rsi": ("overbought", 1.0), "sentiment": ("bearish", 1.0)}, "Strong Sell"),
    ({"rsi": ("oversold", 1.0), "sentiment": ("bullish", 1.0)}, "Strong Buy"),
    ({"rsi": ("neutral", 1.0), "sentiment": ("bearish", 0.8)}, "Sell"),
    ({"rsi": ("neutral", 1.0), "sentiment": ("bullish", 0.8)}, "Buy"),

    # MACD + sentiment
    ({"macd_hist": ("bearish", 1.0), "sentiment": ("bearish", 1.0)}, "Strong Sell"),
    ({"macd_hist": ("bullish", 1.0), "sentiment": ("bullish", 1.0)}, "Strong Buy"),
    ({"macd_hist": ("bearish", 1.0), "sentiment": ("bullish", 0.5)}, "Hold"),
    ({"macd_hist": ("bullish", 1.0), "sentiment": ("bearish", 0.5)}, "Hold"),

    # ATR / volatility
    ({"atr": ("high", 1.0)}, "Hold"),
    ({"atr": ("low", 1.0), "rsi": ("overbought", 0.7)}, "Sell"),
    ({"atr": ("low", 1.0), "rsi": ("oversold", 0.7)}, "Buy"),

    # one input only
    ({"rsi": ("overbought", 0.5)}, "Sell"),
    ({"rsi": ("oversold", 0.5)}, "Buy"),
    ({"sentiment": ("bearish", 0.6)}, "Sell"),
    ({"sentiment": ("bullish", 0.6)}, "Buy"),

    # strong single-sentiment rules. In a news-driven regime RSI and MACD
    # are masked, and without these the result could only reach Sell or Buy,
    # never Strong Sell or Strong Buy
    ({"sentiment": ("bearish", 1.0)}, "Strong Sell"),
    ({"sentiment": ("bullish", 1.0)}, "Strong Buy"),
]


# inference

def _fuzzify(inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Crisp inputs to membership values."""
    result: Dict[str, Dict[str, float]] = {}
    for var_name, value in inputs.items():
        if var_name in MEMBERSHIP_FNS:
            result[var_name] = {}
            for term_name, fn in MEMBERSHIP_FNS[var_name].items():
                result[var_name][term_name] = round(fn(value), 6)
    return result


def _evaluate_rules(
    fuzzy_inputs: Dict[str, Dict[str, float]],
    input_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """Fire every rule, using min for AND."""
    firing: List[Tuple[str, float]] = []
    if input_weights is None:
        input_weights = {}

    for conditions, consequent in RULES:
        strengths = []
        for var_name, (term, rule_weight) in conditions.items():
            if var_name in fuzzy_inputs and term in fuzzy_inputs[var_name]:
                mu = fuzzy_inputs[var_name][term]
                feature_wt = input_weights.get(var_name, 1.0)
                strengths.append(mu * rule_weight * feature_wt)

        if strengths:
            firing_strength = min(strengths)
            if firing_strength > 0.005:
                firing.append((consequent, firing_strength))

    return firing


def _centroid_defuzzify(
    firing: List[Tuple[str, float]],
    resolution: int = 200,
) -> Tuple[float, Dict[str, float]]:
    """Centroid over the output range -2..2."""
    if not firing:
        return 0.0, {"Hold": 1.0}

    x_min, x_max = -2.0, 2.0
    step = (x_max - x_min) / resolution

    numerator = 0.0
    denominator = 0.0
    label_max: Dict[str, float] = {label: 0.0 for label in OUTPUT_TERMS}

    for i in range(resolution + 1):
        x = x_min + i * step
        agg_mu = 0.0
        for label, strength in firing:
            out_mu = min(strength, _output_mu(x, label))
            agg_mu = max(agg_mu, out_mu)

        numerator += x * agg_mu
        denominator += agg_mu

    # highest membership per label
    for label in OUTPUT_TERMS:
        best = 0.0
        for sl, s in firing:
            if sl == label:
                for i in range(resolution + 1):
                    x = x_min + i * step
                    best = max(best, min(s, _output_mu(x, label)))
        label_max[label] = round(best, 4)

    if denominator < 0.0001:
        return 0.0, label_max

    score = numerator / denominator
    return round(score, 4), label_max


def _score_to_label(score: float) -> str:
    """Score to a label such as Buy or Strong Sell."""
    if score <= -1.2:
        return "Strong Sell"
    elif score <= -0.4:
        return "Sell"
    elif score < 0.4:
        return "Hold"
    elif score < 1.2:
        return "Buy"
    else:
        return "Strong Buy"


def compute_score(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
) -> Dict:
    """Baseline: every input counts the same.

    Ranges: rsi 0-100, macd_hist -5..5, sentiment -1..1, atr 0-10 (optional).
    Returns score, label, rule_firings and membership_details.
    """
    inputs = {
        "rsi": max(0.0, min(100.0, float(rsi))),
        "macd_hist": max(-5.0, min(5.0, float(macd_hist))),
        "sentiment": max(-1.0, min(1.0, float(sentiment))),
    }
    if atr is not None:
        inputs["atr"] = max(0.0, min(10.0, float(atr)))

    fuzzy_in = _fuzzify(inputs)
    firing = _evaluate_rules(fuzzy_in)
    score, label_memberships = _centroid_defuzzify(firing)
    label = _score_to_label(score)

    return {
        "score": score,
        "label": label,
        "confidence": round(max(label_memberships.values()) if label_memberships else 0.5, 4),
        "label_memberships": label_memberships,
        "rules_fired": len(firing),
        "rule_firings": [
            {"consequent": lbl, "strength": round(s, 4)} for lbl, s in firing
        ],
        "mode": "baseline",
    }


def compute_expert_adjusted_score(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
    *,
    active_features: Optional[List[str]] = None,
    feature_weights: Optional[Dict[str, float]] = None,
    market_regime: Optional[Dict] = None,
) -> Dict:
    """Fuzzy result using the expert layer's weights.

    Masked features (weight 0) are set to neutral values so they do not
    affect the result. market_regime is the dict from
    expert_layer.detect_market_regime().
    """
    inputs = {
        "rsi": max(0.0, min(100.0, float(rsi))),
        "macd_hist": max(-5.0, min(5.0, float(macd_hist))),
        "sentiment": max(-1.0, min(1.0, float(sentiment))),
    }
    if atr is not None:
        inputs["atr"] = max(0.0, min(10.0, float(atr)))

    # masked features go to neutral
    adjusted_inputs = dict(inputs)
    neutral_map = {"rsi": 50.0, "macd_hist": 0.0, "sentiment": 0.0, "atr": 4.5}
    if feature_weights:
        for feat, wt in feature_weights.items():
            if wt == 0.0 and feat in adjusted_inputs:
                adjusted_inputs[feat] = neutral_map.get(feat, adjusted_inputs[feat])

    fuzzy_in = _fuzzify(adjusted_inputs)
    firing = _evaluate_rules(fuzzy_in, input_weights=feature_weights)
    score, label_memberships = _centroid_defuzzify(firing)

    # conflicting signals: a trader would be less sure and lean to Hold,
    # so pull the score towards 0
    from expert_layer import REGIME_CONFLICTING_SIGNALS
    if (market_regime and
            market_regime.get("regime") == REGIME_CONFLICTING_SIGNALS):
        score = round(score * 0.35, 4)

    label = _score_to_label(score)

    masked = [f for f, w in (feature_weights or {}).items() if w == 0.0]
    active = [f for f, w in (feature_weights or {}).items() if w > 0.0] if feature_weights else list(inputs.keys())

    return {
        "score": score,
        "label": label,
        "confidence": round(max(label_memberships.values()) if label_memberships else 0.5, 4),
        "label_memberships": label_memberships,
        "rules_fired": len(firing),
        "rule_firings": [
            {"consequent": lbl, "strength": round(s, 4)} for lbl, s in firing
        ],
        "mode": "expert",
        "market_regime": market_regime,
        "active_features": active,
        "masked_features": masked,
        "feature_weights": feature_weights or {},
    }


def compute_full_pipeline(
    rsi: float,
    macd_hist: float,
    sentiment: float,
    atr: Optional[float] = None,
    *,
    mode: str = "expert",
) -> Dict:
    """Expert layer, then feature selection, then the fuzzy engine.
    mode is 'baseline' or 'expert'.
    """
    if mode == "baseline":
        return compute_score(rsi, macd_hist, sentiment, atr)

    # imported here to avoid a circular import
    from expert_layer import detect_market_regime
    from feature_selector import select_features

    regime = detect_market_regime(rsi, macd_hist, sentiment, atr)
    selection = select_features(regime, rsi, macd_hist, sentiment, atr)
    result = compute_expert_adjusted_score(
        rsi, macd_hist, sentiment, atr,
        active_features=selection.get("active_features"),
        feature_weights=selection.get("weights"),
        market_regime=regime,
    )
    result["selection_reason"] = selection.get("reason", "")
    return result