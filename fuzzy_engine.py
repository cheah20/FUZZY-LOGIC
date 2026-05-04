"""
fuzzy_engine.py
===============
ok first in my idea ,i write down first (if got any update ,just do a comment or inform group to me)

Core Fuzzy Logic Controller (FLC) for the trading DSS.

This module implements the Mamdani style fuzzy inference system (follow by proposal la)

My idea design is:
  Inputs:
    RSI       — Relative Strength Index (0 to 100)
                Measures whether a stock is overbought or oversold.
    MACD Hist — MACD Histogram (typically -3 to +3)
                Measures the momentum and direction of price movement.
    Sentiment — News sentiment score (-1.0 to +1.0)
                Derived from recent news headlines via the Daily Data Agent.

  Output:
    Score (1 to 5) mapped to linguistic labels:
      1 = Strong Sell
      2 = Sell
      3 = Hold
      4 = Buy
      5 = Strong Buy
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_flc() -> ctrl.ControlSystemSimulation:
    """
    My idea and research:
    Constructs and returns the fuzzy control system simulation object.
    This function is called once and the result is cached (singleton pattern).
    """

    # Step 1: define fuzzy variables (antecedents and consequents) and their universes
    # These are the numerical ranges each input/output variable can take.
    # (can adjusted based on real data distribution ba, i try only la)
    rsi       = ctrl.Antecedent(np.arange(0, 101, 1),     "rsi")
    macd_hist = ctrl.Antecedent(np.arange(-3, 3.01, 0.01), "macd_hist")
    sentiment = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), "sentiment")
    score     = ctrl.Consequent(np.arange(1, 5.01, 0.01),  "score")

    #Step 2: Define membership functions :
    # My research in google scholar and AI:

    # Each linguistic label ("oversold") is mapped to a fuzzy set
    # using trapezoidal (trapmf) or triangular (trimf) membership functions.
#_________________________________________________________________________________________
    # RSI membership functions
    # Oversold region (RSI below 30 signals possible buying opportunity)
    rsi["oversold"]   = fuzz.trapmf(rsi.universe, [0,  0,  30, 45])
    # Neutral RSI (between overbought and oversold)
    rsi["neutral"]    = fuzz.trimf(rsi.universe,  [30, 50, 70])
    # Overbought region (RSI above 70 signals possible selling pressure)
    rsi["overbought"] = fuzz.trapmf(rsi.universe, [55, 70, 100, 100])

    # MACD Histogram membership functions
    # Bearish: histogram is negative (price momentum falling)
    macd_hist["bearish"] = fuzz.trapmf(macd_hist.universe, [-3, -3, -0.5,  0])
    # Neutral: histogram near zero (no strong trend)
    macd_hist["neutral"] = fuzz.trimf(macd_hist.universe,  [-0.3, 0, 0.3])
    # Bullish: histogram is positive (price momentum rising)
    macd_hist["bullish"] = fuzz.trapmf(macd_hist.universe, [0, 0.5,  3,  3])

    # Sentiment membership functions
    # Negative: recent news is bearish or concerning
    sentiment["negative"] = fuzz.trapmf(sentiment.universe, [-1, -1, -0.3,  0])
    # Neutral: no strong news signal
    sentiment["neutral"]  = fuzz.trimf(sentiment.universe,  [-0.2, 0, 0.2])
    # Positive: recent news is bullish or optimistic
    sentiment["positive"] = fuzz.trapmf(sentiment.universe, [0, 0.3,  1,  1])

    # Output score membership functions (1 to 5 scale)
    score["strong_sell"] = fuzz.trimf(score.universe, [1,   1,   1.8])
    score["sell"]        = fuzz.trimf(score.universe, [1.5, 2,   2.5])
    score["hold"]        = fuzz.trimf(score.universe, [2.5, 3,   3.5])
    score["buy"]         = fuzz.trimf(score.universe, [3.5, 4,   4.5])
    score["strong_buy"]  = fuzz.trimf(score.universe, [4.2, 5,   5])


    # Step 3: Define fuzzy if-then rules 
    # This step recommends by gemini and gpt and qwen (3 AI also same idea)

    # These rules encode the trading logic agreed upon with our financial expert.
    # RSI provides the primary signal; MACD and Sentiment provide confirmation.

    rules = [
        # Oversold conditions — potential buying opportunities
        ctrl.Rule(rsi["oversold"] & macd_hist["bullish"],                           score["strong_buy"]),
        ctrl.Rule(rsi["oversold"] & macd_hist["neutral"],                           score["buy"]),
        ctrl.Rule(rsi["oversold"] & macd_hist["bearish"],                           score["hold"]),

        # Neutral RSI — MACD and sentiment determine the direction
        ctrl.Rule(rsi["neutral"] & macd_hist["bullish"] & sentiment["positive"],    score["buy"]),
        ctrl.Rule(rsi["neutral"] & macd_hist["bullish"] & sentiment["neutral"],     score["buy"]),
        ctrl.Rule(rsi["neutral"] & macd_hist["bullish"] & sentiment["negative"],    score["hold"]),
        ctrl.Rule(rsi["neutral"] & macd_hist["neutral"],                            score["hold"]),
        ctrl.Rule(rsi["neutral"] & macd_hist["bearish"] & sentiment["negative"],    score["sell"]),
        ctrl.Rule(rsi["neutral"] & macd_hist["bearish"] & sentiment["neutral"],     score["sell"]),
        ctrl.Rule(rsi["neutral"] & macd_hist["bearish"] & sentiment["positive"],    score["hold"]),

        # Overbought conditions — potential selling pressure
        ctrl.Rule(rsi["overbought"] & macd_hist["bearish"],                         score["strong_sell"]),
        ctrl.Rule(rsi["overbought"] & macd_hist["neutral"],                         score["sell"]),
        ctrl.Rule(rsi["overbought"] & macd_hist["bullish"],                         score["hold"]),

        # Extreme sentiment overrides — news can reinforce technical signals
        ctrl.Rule(sentiment["positive"] & rsi["oversold"],                          score["strong_buy"]),
        ctrl.Rule(sentiment["negative"] & rsi["overbought"],                        score["strong_sell"]),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


# Singleton instance — the FLC is built once and reused for all calls.
# Rebuilding it on every call would add unnecessary overhead.
_sim = None

def _get_sim():
    global _sim
    if _sim is None:
        _sim = build_flc()
    return _sim


def compute_score(rsi: float, macd_hist: float, sentiment: float = 0.0) -> dict:
    """

    #ok here  is setting is below :
    # if you guys need have more knowledge can change the setting :

    Runs the FLC inference engine and returns the trading score and labels.
    Parameters
    ----------
    rsi       : RSI value (0 to 100)
    macd_hist : MACD histogram value (clipped to -3 to +3 internally)
    sentiment : News sentiment score (-1.0 to +1.0), defaults to 0.0 (neutral)

    Returns
    -------
    dict with keys:
        score      — float between 1.0 and 5.0
        label      — "Strong Sell" / "Sell" / "Hold" / "Buy" / "Strong Buy"
        rsi_label  — dominant RSI linguistic state
        macd_label — dominant MACD linguistic state
    """

    sim = _get_sim()

    # Clip inputs to their valid ranges to prevent FLC computation errors
    rsi_val  = float(np.clip(rsi,       0,   100))
    macd_val = float(np.clip(macd_hist, -3,    3))
    sent_val = float(np.clip(sentiment, -1,    1))

    sim.input["rsi"]       = rsi_val
    sim.input["macd_hist"] = macd_val
    sim.input["sentiment"] = sent_val
    sim.compute()

    raw_score = float(sim.output["score"])
    label     = _score_to_label(raw_score)

    rsi_label  = _dominant_label(rsi_val,  {
        "oversold":   [0,  0,  30, 45],
        "neutral":    [30, 50, 70],
        "overbought": [55, 70, 100, 100],
    })
    macd_label = _dominant_label(macd_val, {
        "bearish": [-3, -3, -0.5, 0],
        "neutral": [-0.3, 0, 0.3],
        "bullish": [0, 0.5, 3, 3],
    })

    return {
        "score"      : round(raw_score, 2),
        "label"      : label,
        "rsi_label"  : rsi_label,
        "macd_label" : macd_label,
    }


def _score_to_label(s: float) -> str:
    """Maps a numeric FLC score to the corresponding linguistic trading signal."""
    if s < 1.8: return "Strong Sell"
    if s < 2.5: return "Sell"
    if s < 3.5: return "Hold"
    if s < 4.3: return "Buy"
    return "Strong Buy"


def _dominant_label(value: float, labels: dict) -> str:
    """
    Returns the linguistic label with the highest membership degree
    for a given crisp input value and same time Used for diagnostic display in the UI. (got problem here, still need time fix it)
    """
    best, best_mu = "unknown", -1.0
    u = np.array([value])
    for name, params in labels.items():
        mu = float(fuzz.trimf(u, params)[0] if len(params) == 3
                   else fuzz.trapmf(u, params)[0])
        if mu > best_mu:
            best_mu, best = mu, name
    return best


if __name__ == "__main__":
    # Quick sanity check for \ run a few representative test cases
    test_cases = [
        ("Overbought + Bearish MACD", 78, -0.8, -0.3),
        ("Oversold + Bullish MACD",   22,  0.9,  0.4),
        ("Neutral conditions",         50,  0.1,  0.0),
        ("Extreme fear scenario",      18, -1.2, -0.8),
    ]
    for name, rsi, macd, sent in test_cases:
        result = compute_score(rsi, macd, sent)
        print(f"[{name}]")
        print(f"  Input : RSI={rsi}, MACD={macd}, Sentiment={sent}")
        print(f"  Output: Score={result['score']}/5  ->  {result['label']}\n")
