# Research Literature

This folder contains the academic papers that ground the Human Expert Knowledge Base
(`human_expert_kb.py`) and the fuzzy logic / feature selection design used throughout
the ASTDSS Terminal. These papers are reference material, not runtime dependencies —
no Python module reads from this folder at runtime.

**Note on attribution:** the mapping below is based on each paper's stated subject
matter and the design rationale it supports — not a claim that every paper was used
to derive an exact formula. Where a paper's precise contribution is general
background (e.g. arXiv preprints on ML/AI trading), it's listed as supporting the
overall hybrid-AI design philosophy rather than a specific rule.

| Area | Used For | Example Files |
|---|---|---|
| RSI | Oversold/overbought expert rules (RSI-001/002/003 in `human_expert_kb.py`) | `Stock_Price_Prediction_Based_on_Relative_Strength_Index_Moving_Average_Convergence_Divergence_and_XGBoost_Model.pdf` |
| MACD | Momentum confirmation, multi-indicator confluence (MACD-001/002/003, MULTI-001) | `Stock_Price_Prediction_Based_on_Relative_Strength_Index_Moving_Average_Convergence_Divergence_and_XGBoost_Model.pdf` |
| ATR | Volatility/risk-penalty justification — explicitly **not** a directional signal (ATR-001/002) | `Average-True-Range.pdf` |
| Bollinger Bands | Volatility-relative bands, regime-detection support (BB-001/002/003) | `Bollinger_Bands_Thirty_Years_Later.pdf` |
| Technical Indicator ML/AI | General justification for using technical indicators as predictive/fuzzy input features | `IJISAE_50_Rahul+Marui+Dhokane_3_1867.pdf`, `entropy-27-00550-v2.pdf` |
| Neuro-Fuzzy / Computational Intelligence | Background for combining fuzzy inference with learned/adaptive components | `PubSub10294_702a_McGinnity.pdf` |
| Hybrid AI Trading | Supports the overall AI + technical/fuzzy hybrid design approach | `2206.12282v1.pdf`, `2111.13364v2.pdf`, `1212.4890v2.pdf`, `ARBMSV14_N1_P35_41.pdf` |
| Backtesting Metrics | Sharpe ratio, max drawdown, win-rate evaluation methodology (BT-001/002/003) | `1-s2.0-S0957417422008089-main.pdf` / `S0957417422008089.htm` |

For the full rule-by-rule citation (rule ID → paper → captured knowledge), see
`human_expert_kb.py` — each of the 16 rules carries a `source_name` field pointing to
the specific paper and author/year it draws from (e.g. Wilder 1978, Bollinger 2002,
Appel 2005, Murphy 1999, Sharpe 1966, Pardo 2008, Elder 1993).
