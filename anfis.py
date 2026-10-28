"""ANFIS: a 5-layer Sugeno neuro-fuzzy network (Jang, 1993).

Inputs are RSI, the MACD histogram and ATR. The starting membership
functions use the same textbook limits as human_expert_kb.py (Wilder for
RSI and ATR, Appel for MACD), then training on real price history adjusts
them. That learning step is the difference from the fixed Mamdani engine
in fuzzy_engine.py.

It does not replace the expert engine. agentic_pipeline.py runs it next to
the other results, and its output still has to pass the validation gate.

Training is Jang's hybrid method: least squares for the consequent
parameters, then a gradient step on the Gaussian centres and widths. The
gradient is estimated with central differences instead of being derived
by hand.

Main functions: train_from_history(), predict_score(), load_params(),
params_exist().
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_PARAMS_FILE = os.path.join(_DATA_DIR, "anfis_params.json")

# input ranges, same as fuzzy_engine.py so the scores compare
INPUT_NAMES = ["rsi", "macd_hist", "atr"]
INPUT_RANGES = {"rsi": (0.0, 100.0), "macd_hist": (-5.0, 5.0), "atr": (0.0, 10.0)}
TERMS = ["low", "mid", "high"]  # three terms per input
N_RULES = len(TERMS) ** len(INPUT_NAMES)  # 27

# output score range, same -2..2 as fuzzy_engine.py
SCORE_MIN, SCORE_MAX = -2.0, 2.0

# starting Gaussian (centre, sigma) for each term, from the same limits
# used in fuzzy_engine.py and human_expert_kb.py:
#   RSI: Wilder (1978), oversold below 30, overbought above 70
#   MACD hist: Appel (2005), zero line is neutral
#   ATR: Wilder (1978), on the 0-10 scale
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
]  # the 27 rules in a fixed order


# layer 1: fuzzify

def _gaussmf(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    """Gaussian membership function used by the ANFIS inputs."""
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


def _premise_to_array(premise: Dict[str, Dict[str, Tuple[float, float]]]) -> np.ndarray:
    """Premise dict to a flat array of [c, sigma] pairs, always in the same order."""
    out = []
    for inp in INPUT_NAMES:
        for term in TERMS:
            c, s = premise[inp][term]
            out.extend([c, s])
    return np.array(out, dtype=float)


def _array_to_premise(arr: np.ndarray) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Turn the flat optimiser array back into centre and width per input term."""
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
    """X has columns rsi, macd_hist, atr. Returns mu[input][term] as arrays."""
    mu: Dict[str, Dict[str, np.ndarray]] = {}
    for j, inp in enumerate(INPUT_NAMES):
        mu[inp] = {}
        for term in TERMS:
            c, s = premise[inp][term]
            mu[inp][term] = _gaussmf(X[:, j], c, s)
    return mu


# layers 2 and 3: fire and normalise

def _layer2_3_firing(mu: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    """Normalised firing strength of each rule, shape (n_samples, N_RULES)."""
    n = len(next(iter(mu["rsi"].values())))
    w = np.ones((n, N_RULES))
    for r, (t_rsi, t_macd, t_atr) in enumerate(_RULE_COMBOS):
        w[:, r] = mu["rsi"][t_rsi] * mu["macd_hist"][t_macd] * mu["atr"][t_atr]
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-9] = 1e-9
    return w / row_sum


# layers 4 and 5: consequents and sum

def _design_matrix(norm_w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Design matrix for least squares: per rule [w, w*x1, w*x2, w*x3]."""
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
    """Run all ANFIS layers and return the predicted scores."""
    mu = _layer1_memberships(X, premise)
    norm_w = _layer2_3_firing(mu)
    A = _design_matrix(norm_w, X)
    return A @ consequent


def _fit_consequent_lse(
    norm_w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 1.0
) -> np.ndarray:
    """Least squares with a small ridge penalty for the consequent parameters.

    There are 108 of them (27 rules x 4) and real data mostly sits in the
    middle of each input range, so some rules barely see any samples. Plain
    least squares gives those rules huge coefficients that blow up as soon as
    a live input lands there. The ridge term keeps them small.
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
    """Mean squared error of the predictions, used as the training loss."""
    premise = _array_to_premise(premise_arr)
    y_pred = _forward(X, premise, consequent)
    return float(np.mean((y_pred - y) ** 2))


def _train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    lr: float = 0.05,
    eps: float = 1e-3,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], np.ndarray, List[float]]:
    """Jang's hybrid training.

    Each epoch: fit the consequent parameters with least squares for the
    current premises, then estimate the error gradient for the premises with
    central differences and take one step.
    """
    premise = {k: dict(v) for k, v in _INIT_PREMISE.items()}
    premise_arr = _premise_to_array(premise)
    consequent = np.zeros((N_RULES) * (X.shape[1] + 1))
    loss_history: List[float] = []

    for epoch in range(epochs):
        # forward pass: least squares for the consequents
        premise = _array_to_premise(premise_arr)
        mu = _layer1_memberships(X, premise)
        norm_w = _layer2_3_firing(mu)
        consequent = _fit_consequent_lse(norm_w, X, y)

        loss = _mse(premise_arr, consequent, X, y)
        loss_history.append(loss)

        # backward pass: numeric gradient, premises only
        grad = np.zeros_like(premise_arr)
        for i in range(len(premise_arr)):
            perturbed = premise_arr.copy()
            perturbed[i] += eps
            loss_plus = _mse(perturbed, consequent, X, y)
            perturbed[i] -= 2 * eps
            loss_minus = _mse(perturbed, consequent, X, y)
            grad[i] = (loss_plus - loss_minus) / (2 * eps)

        premise_arr = premise_arr - lr * grad
        # sigma must stay positive and not too small
        for i in range(1, len(premise_arr), 2):
            premise_arr[i] = max(abs(premise_arr[i]), 0.25)

    premise = _array_to_premise(premise_arr)
    return premise, consequent, loss_history


def _build_training_dataset(
    symbol: str = "AAPL",
    period: str = "2y",
    horizon: int = 5,
    return_scale: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Training samples from real price history.

    For each day: RSI, MACD histogram and ATR as inputs, and the return over
    the next `horizon` days, scaled to the -2..2 score range, as the target.
    """
    import yfinance as yf
    from Trading_Data import compute_rsi, compute_macd, compute_atr

    hist = yf.Ticker(symbol).history(period=period, interval="1d")
    if hist.empty or len(hist) < 60:
        raise ValueError(f"Insufficient history for {symbol}: {len(hist)} bars")

    closes = [float(v) for v in hist["Close"]]
    highs = [float(v) for v in hist["High"]]
    lows = [float(v) for v in hist["Low"]]

    window = 35  # compute_macd needs this many bars
    rows_X: List[List[float]] = []
    rows_y: List[float] = []

    for i in range(window, len(closes) - horizon):
        c_slice = closes[: i + 1]
        h_slice = highs[: i + 1]
        l_slice = lows[: i + 1]

        rsi = compute_rsi(c_slice)
        _, _, macd_hist_raw = compute_macd(c_slice)
        macd_hist = macd_hist_raw / c_slice[-1] * 100  # same scaling as Trading_Data.py
        atr = compute_atr(h_slice, l_slice, c_slice)

        future_return = (closes[i + horizon] - closes[i]) / closes[i]
        label = max(SCORE_MIN, min(SCORE_MAX, future_return * return_scale))

        rows_X.append([rsi, macd_hist, atr])
        rows_y.append(label)

    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float)


def params_exist() -> bool:
    """True when trained ANFIS parameters are saved on disk."""
    return os.path.exists(_PARAMS_FILE)


def load_params() -> Optional[Dict]:
    """Load the saved ANFIS parameters, or None if missing or unreadable."""
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
    """Save the trained parameters to disk in one atomic write."""
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
    """Build the training set from price history and train the network."""
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
    """Map an ANFIS score to a trading label, from Strong Sell to Strong Buy."""
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
    """Score one input with the trained network.

    If it has not been trained yet it returns a neutral Hold instead of
    raising, because the pipeline must work without it.
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

    # confidence: weight of the strongest rule
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
