"""
anfis.py — Adaptive Neuro-Fuzzy Inference System (ANFIS)
ASTDSS Terminal — Neuro-Symbolic Trading DSS

Implements the standard 5-layer Sugeno-type ANFIS architecture (Jang, 1993,
"ANFIS: Adaptive-Network-Based Fuzzy Inference System", IEEE Trans. SMC):

  Layer 1 (Fuzzification)  — Gaussian membership grade of each crisp input
                              for each linguistic term.
  Layer 2 (Rule firing)    — Product T-norm of membership grades per rule.
  Layer 3 (Normalisation)  — Each rule's firing strength divided by the sum
                              of all firing strengths.
  Layer 4 (Consequent)     — First-order Sugeno linear function of the raw
                              inputs, scaled by the normalised firing strength.
  Layer 5 (Summation)      — Sum of all rule contributions = crisp output.

Inputs: RSI, MACD histogram, ATR — the same three indicators whose expert
boundaries are cited in human_expert_kb.py (Wilder 1978 RSI/ATR, Appel 2005
MACD; see docs/research/). Those cited boundaries seed the initial Gaussian
premise parameters (Layer 1) below, so the network starts from literature-
grounded membership functions rather than arbitrary numbers. Training then
*adapts* those parameters from real historical price data — this is what
distinguishes ANFIS from the static Mamdani engine in fuzzy_engine.py, and
why the network is described as "neuro-fuzzy": fixed expert structure +
learned parameters.

This module deliberately does NOT replace fuzzy_engine.py / expert_layer.py.
It runs as an independent, parallel proposer in agentic_pipeline.py — its
output sits alongside baseline_result/expert_result and is still subject to
the Human Expert Validation Gate before influencing the final decision. Pure
data-driven ANFIS output is never applied unchecked (see tests/run_test.py
§21 "ANFIS only data-driven" — rejected as the *sole* signal source, accepted
here as one more validated input).

Hybrid learning (Jang's two-pass algorithm):
  Forward pass:  consequent parameters solved in closed form via least
                 squares (numpy.linalg.lstsq) for fixed premise parameters.
  Backward pass: premise parameters (Gaussian centre/width) updated by
                 gradient descent, with the gradient estimated by central
                 finite differences (numerically robust, avoids deriving
                 analytic gradients through the normalisation layer by hand).

Public API:
  train_from_history(symbol, period, epochs)  → trains + caches params
  predict_score(rsi, macd_hist, atr)          → dict (score, label, confidence, ...)
  load_params() / params_exist()
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_PARAMS_FILE = os.path.join(_DATA_DIR, "anfis_params.json")

# Input domains (match fuzzy_engine.py so ANFIS output is directly comparable)
INPUT_NAMES = ["rsi", "macd_hist", "atr"]
INPUT_RANGES = {"rsi": (0.0, 100.0), "macd_hist": (-5.0, 5.0), "atr": (0.0, 10.0)}
TERMS = ["low", "mid", "high"]  # generic 3-term partition per input
N_RULES = len(TERMS) ** len(INPUT_NAMES)  # 27

# Output score domain, shared with fuzzy_engine.py OUTPUT_TERMS [-2, 2]
SCORE_MIN, SCORE_MAX = -2.0, 2.0

# Initial Gaussian premise parameters (centre, sigma), seeded from the
# literature-cited boundaries already used in fuzzy_engine.py / human_expert_kb.py:
#   RSI:       Wilder (1978) oversold < 30, overbought > 70
#   MACD hist: Appel (2005) zero-line crossover as the neutral pivot
#   ATR:       Wilder (1978) volatility bands used for the 0-10 normalised scale
_INIT_PREMISE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "rsi": {
        "low":  (20.0, 12.0),
        "mid":  (50.0, 14.0),
        "high": (80.0, 12.0),
    },
    "macd_hist": {
        "low":  (-1.2, 0.9),
        "mid":  (0.0, 0.4),
        "high": (1.2, 0.9),
    },
    "atr": {
        "low":  (1.5, 1.3),
        "mid":  (4.5, 1.8),
        "high": (8.0, 1.3),
    },
}

_RULE_COMBOS: List[Tuple[str, str, str]] = [
    (a, b, c) for a in TERMS for b in TERMS for c in TERMS
]  # 27 rules, fixed order, indexes Layer 2/3/4 consistently


# Layer 1 — Fuzzification

def _gaussmf(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


def _premise_to_array(premise: Dict[str, Dict[str, Tuple[float, float]]]) -> np.ndarray:
    """Flatten premise dict to a fixed-order array [c,sigma]*9 for optimisation."""
    out = []
    for inp in INPUT_NAMES:
        for term in TERMS:
            c, s = premise[inp][term]
            out.extend([c, s])
    return np.array(out, dtype=float)


def _array_to_premise(arr: np.ndarray) -> Dict[str, Dict[str, Tuple[float, float]]]:
    premise: Dict[str, Dict[str, Tuple[float, float]]] = {}
    i = 0
    for inp in INPUT_NAMES:
        premise[inp] = {}
        for term in TERMS:
            c, s = arr[i], max(abs(arr[i + 1]), 1e-3)
            premise[inp][term] = (float(c), float(s))
            i += 2
    return premise


def _layer1_memberships(
    X: np.ndarray, premise: Dict[str, Dict[str, Tuple[float, float]]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """X: (n_samples, 3) ordered [rsi, macd_hist, atr]. Returns mu[input][term] -> (n,) array."""
    mu: Dict[str, Dict[str, np.ndarray]] = {}
    for j, inp in enumerate(INPUT_NAMES):
        mu[inp] = {}
        for term in TERMS:
            c, s = premise[inp][term]
            mu[inp][term] = _gaussmf(X[:, j], c, s)
    return mu


# Layer 2/3 — Rule firing + normalisation

def _layer2_3_firing(mu: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    """Returns normalised firing strengths, shape (n_samples, N_RULES)."""
    n = len(next(iter(mu["rsi"].values())))
    w = np.ones((n, N_RULES))
    for r, (t_rsi, t_macd, t_atr) in enumerate(_RULE_COMBOS):
        w[:, r] = mu["rsi"][t_rsi] * mu["macd_hist"][t_macd] * mu["atr"][t_atr]
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-9] = 1e-9
    return w / row_sum


# Layer 4/5 — Consequent + summation

def _design_matrix(norm_w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Build the LSE design matrix: for each rule, columns [w_bar, w_bar*x1, w_bar*x2, w_bar*x3]."""
    n = X.shape[0]
    n_inputs = X.shape[1]
    cols_per_rule = n_inputs + 1
    A = np.zeros((n, N_RULES * cols_per_rule))
    for r in range(N_RULES):
        wr = norm_w[:, r]
        base = r * cols_per_rule
        A[:, base] = wr
        for j in range(n_inputs):
            A[:, base + 1 + j] = wr * X[:, j]
    return A


def _forward(
    X: np.ndarray,
    premise: Dict[str, Dict[str, Tuple[float, float]]],
    consequent: np.ndarray,
) -> np.ndarray:
    mu = _layer1_memberships(X, premise)
    norm_w = _layer2_3_firing(mu)
    A = _design_matrix(norm_w, X)
    return A @ consequent


def _fit_consequent_lse(
    norm_w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 1.0
) -> np.ndarray:
    """
    Ridge-regularised least squares for the consequent parameters.

    With 27 rules x 4 params = 108 consequent parameters, most rules only
    have a handful of supporting training samples (real price data clusters
    in the "mid" region of each input range; the "high"/"low" extremes of
    rarely-visited combinations like RSI=mid + ATR=high are sparsely or
    never represented). Plain least squares lets those near-empty columns
    take on huge, poorly-constrained coefficients that then extrapolate to
    nonsensical scores (e.g. flipping sign) the moment a live input falls in
    that sparse region. L2 regularisation shrinks unsupported coefficients
    toward zero instead, keeping out-of-distribution predictions bounded.
    """
    A = _design_matrix(norm_w, X)
    n_params = A.shape[1]
    AtA = A.T @ A + ridge * np.eye(n_params)
    Aty = A.T @ y
    theta = np.linalg.solve(AtA, Aty)
    return theta


def _mse(
    premise_arr: np.ndarray, consequent: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    premise = _array_to_premise(premise_arr)
    y_pred = _forward(X, premise, consequent)
    return float(np.mean((y_pred - y) ** 2))


# Hybrid (Jang) Training

def _train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    lr: float = 0.05,
    eps: float = 1e-3,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], np.ndarray, List[float]]:
    """
    Jang's hybrid learning algorithm.

    Each epoch:
      1. Forward pass: compute normalised firing strengths for current premise
         parameters, solve consequent parameters in closed form via LSE.
      2. Backward pass: estimate dMSE/d(premise) via central finite differences
         and take one gradient-descent step.
    """
    premise = {k: dict(v) for k, v in _INIT_PREMISE.items()}
    premise_arr = _premise_to_array(premise)
    consequent = np.zeros((N_RULES) * (X.shape[1] + 1))
    loss_history: List[float] = []

    for epoch in range(epochs):
        # Forward pass — closed-form consequent fit
        premise = _array_to_premise(premise_arr)
        mu = _layer1_memberships(X, premise)
        norm_w = _layer2_3_firing(mu)
        consequent = _fit_consequent_lse(norm_w, X, y)

        loss = _mse(premise_arr, consequent, X, y)
        loss_history.append(loss)

        # Backward pass — numerical gradient on premise parameters only
        grad = np.zeros_like(premise_arr)
        for i in range(len(premise_arr)):
            perturbed = premise_arr.copy()
            perturbed[i] += eps
            loss_plus = _mse(perturbed, consequent, X, y)
            perturbed[i] -= 2 * eps
            loss_minus = _mse(perturbed, consequent, X, y)
            grad[i] = (loss_plus - loss_minus) / (2 * eps)

        premise_arr = premise_arr - lr * grad
        # Keep sigmas positive and away from zero
        for i in range(1, len(premise_arr), 2):
            premise_arr[i] = max(abs(premise_arr[i]), 0.25)

    premise = _array_to_premise(premise_arr)
    return premise, consequent, loss_history


# Training Data — built from real historical OHLC via Trading_Data.py

def _build_training_dataset(
    symbol: str = "AAPL",
    period: str = "2y",
    horizon: int = 5,
    return_scale: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roll a window over real historical price data (yfinance), computing
    RSI/MACD-hist/ATR at each day and labelling with the forward `horizon`-day
    percentage return, scaled into the same [-2, 2] score domain used
    throughout the rest of the pipeline (fuzzy_engine.py OUTPUT_TERMS).
    """
    import yfinance as yf
    from Trading_Data import compute_rsi, compute_macd, compute_atr

    hist = yf.Ticker(symbol).history(period=period, interval="1d")
    if hist.empty or len(hist) < 60:
        raise ValueError(f"Insufficient history for {symbol}: {len(hist)} bars")

    closes = [float(v) for v in hist["Close"]]
    highs = [float(v) for v in hist["High"]]
    lows = [float(v) for v in hist["Low"]]

    window = 35  # matches compute_macd's minimum bar requirement
    rows_X: List[List[float]] = []
    rows_y: List[float] = []

    for i in range(window, len(closes) - horizon):
        c_slice = closes[: i + 1]
        h_slice = highs[: i + 1]
        l_slice = lows[: i + 1]

        rsi = compute_rsi(c_slice)
        _, _, macd_hist_raw = compute_macd(c_slice)
        macd_hist = macd_hist_raw / c_slice[-1] * 100  # normalise like Trading_Data.py
        atr = compute_atr(h_slice, l_slice, c_slice)

        future_return = (closes[i + horizon] - closes[i]) / closes[i]
        label = max(SCORE_MIN, min(SCORE_MAX, future_return * return_scale))

        rows_X.append([rsi, macd_hist, atr])
        rows_y.append(label)

    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float)


# Public API

def params_exist() -> bool:
    return os.path.exists(_PARAMS_FILE)


def load_params() -> Optional[Dict]:
    if not params_exist():
        return None
    try:
        with open(_PARAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_params(
    premise: Dict[str, Dict[str, Tuple[float, float]]],
    consequent: np.ndarray,
    meta: Dict,
) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    payload = {
        "premise": premise,
        "consequent": consequent.tolist(),
        "meta": meta,
    }
    tmp_path = _PARAMS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, _PARAMS_FILE)


def train_from_history(
    symbol: str = "AAPL",
    period: str = "2y",
    horizon: int = 5,
    epochs: int = 40,
) -> Dict:
    """Build a training set from real price history and run hybrid ANFIS training."""
    X, y = _build_training_dataset(symbol, period, horizon)
    premise, consequent, loss_history = _train(X, y, epochs=epochs)

    meta = {
        "symbol": symbol,
        "period": period,
        "horizon": horizon,
        "epochs": epochs,
        "n_samples": int(X.shape[0]),
        "final_mse": loss_history[-1] if loss_history else None,
        "loss_history": [round(l, 6) for l in loss_history],
    }
    _save_params(premise, consequent, meta)
    return meta


def _score_to_label(score: float) -> str:
    if score <= -1.2:
        return "Strong Sell"
    elif score <= -0.4:
        return "Sell"
    elif score < 0.4:
        return "Hold"
    elif score < 1.2:
        return "Buy"
    return "Strong Buy"


def predict_score(rsi: float, macd_hist: float, atr: float) -> Dict:
    """
    Run a forward pass through the trained ANFIS network for a single input.
    Falls back to a neutral Hold if the network hasn't been trained yet
    (params_exist() == False) rather than raising — this is a parallel
    proposer, never a hard pipeline dependency.
    """
    cached = load_params()
    if cached is None:
        return {
            "score": 0.0,
            "label": "Hold",
            "confidence": 0.0,
            "mode": "untrained",
            "note": "ANFIS has not been trained yet. Run anfis.train_from_history().",
        }

    premise = cached["premise"]
    consequent = np.array(cached["consequent"], dtype=float)

    X = np.array([[
        max(0.0, min(100.0, float(rsi))),
        max(-5.0, min(5.0, float(macd_hist))),
        max(0.0, min(10.0, float(atr))),
    ]])

    mu = _layer1_memberships(X, premise)
    norm_w = _layer2_3_firing(mu)
    A = _design_matrix(norm_w, X)
    score = float((A @ consequent)[0])
    score = round(max(SCORE_MIN, min(SCORE_MAX, score)), 4)

    # Per-rule contribution magnitude -> confidence proxy (top rule's normalised weight)
    confidence = round(float(norm_w.max()), 4)

    top_rule_idx = int(norm_w.argmax())
    top_rule_terms = _RULE_COMBOS[top_rule_idx]

    return {
        "score": score,
        "label": _score_to_label(score),
        "confidence": confidence,
        "mode": "trained",
        "dominant_rule": {
            "rsi": top_rule_terms[0],
            "macd_hist": top_rule_terms[1],
            "atr": top_rule_terms[2],
            "firing_strength": confidence,
        },
        "trained_on": cached.get("meta", {}),
    }


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    period = sys.argv[2] if len(sys.argv) > 2 else "2y"
    print(f"Training ANFIS on {symbol} ({period}) ...")
    meta = train_from_history(symbol=symbol, period=period)
    print(f"Done. {meta['n_samples']} samples, final MSE={meta['final_mse']:.5f}")
    print(f"Params cached to {_PARAMS_FILE}")
