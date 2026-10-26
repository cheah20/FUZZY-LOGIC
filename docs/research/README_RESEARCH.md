# Research Literature

Papers behind the human expert knowledge base (`human_expert_kb.py`) and the fuzzy
logic and feature selection design. They are reading material only; no code reads
this folder.

The PDFs are kept on the developer's machine and are not in the repository, because
the publishers hold the copyright. Use the links below to read them.

The table shows which part of the design each paper backs up. Most papers gave
background and limits rather than an exact formula; the arXiv papers on ML trading
support the general hybrid idea, not one specific rule.

| Area | Used For | Papers |
|---|---|---|
| RSI | Oversold/overbought expert rules (RSI-001/002/003 in `human_expert_kb.py`) | [8] |
| MACD | Momentum confirmation, multi-indicator confluence (MACD-001/002/003, MULTI-001) | [8] |
| ATR | Volatility and the risk penalty; ATR is **not** a direction signal (ATR-001/002) | [10] |
| Bollinger Bands | Volatility-relative bands, regime-detection support (BB-001/002/003) | [2] |
| Technical Indicator ML/AI | General justification for using technical indicators as predictive/fuzzy input features | [6], [9] |
| Neuro-Fuzzy / Computational Intelligence | Background for combining fuzzy inference with learned/adaptive components | [7] |
| Hybrid AI Trading | Supports the overall AI + technical/fuzzy hybrid design approach | [3], [4], [2], [5] |
| Backtesting Metrics | Sharpe ratio, max drawdown, win-rate evaluation methodology (BT-001/002/003) | [1] |

## Papers

1. R. L. de Almeida and R. F. Neves, "Stock market prediction and portfolio composition using a hybrid approach combined with self-adaptive evolutionary algorithm," *Expert Systems with Applications*, vol. 204, 117478, 2022. https://doi.org/10.1016/j.eswa.2022.117478
2. M. Leeds, "Bollinger Bands Thirty Years Later," arXiv:1212.4890, 2013. https://arxiv.org/abs/1212.4890
3. P. S. K. Prasad, V. Madhav, R. Lal and V. Ravi, "Optimal Technical Indicator-based Trading Strategies Using NSGA-II," arXiv:2111.13364, 2021. https://arxiv.org/abs/2111.13364
4. P. T. Chio, "A comparative study of the MACD-base trading strategies: evidence from the US stock market," arXiv:2206.12282, 2022. https://arxiv.org/abs/2206.12282
5. B. J. Zaini, R. Mansor, N. Yusof and B. H. Sang, "Classify Stock Market Movement Based on Technical Analysis Indicators Using Logistic Regression," *Journal of Advanced Research in Business and Management Studies*, vol. 14, no. 1, pp. 35–41, 2019.
6. R. M. Dhokane and S. Agarwal, "LSTM Deep Learning Based Stock Price Prediction with Bollinger Band, RSI, MACD, and OHLC Features," *International Journal of Intelligent Systems and Applications in Engineering*, vol. 12, no. 3, pp. 1169–1176, 2024.
7. Y. Shynkevich, T. M. McGinnity, S. Coleman, A. Belatreche and Y. Li, "Forecasting Price Movements using Technical Indicators: Investigating the Impact of Varying Input Window Length," *Neurocomputing*, 2017.
8. L. Ming and G. Chen, "Stock price prediction based on relative strength index, moving average convergence divergence and XGBoost model," in *Proc. Data Driven Control and Learning Systems Conference (DDCLS)*, 2024. https://doi.org/10.1109/DDCLS61622.2024.10606580
9. G. Cohen, A. Aiche and R. Eichel, "Artificial Intelligence Models for Predicting Stock Returns Using Fundamental, Technical, and Entropy-Based Strategies: A Semantic-Augmented Hybrid Approach," *Entropy*, vol. 27, no. 6, 550, 2025. https://doi.org/10.3390/e27060550
10. "Average True Range Indicator Trading Strategy," practitioner guide (PDF).

Each of the 16 rules in `human_expert_kb.py` has a `source_name` field with the
book or paper it comes from (Wilder 1978, Bollinger 2002,
Appel 2005, Murphy 1999, Sharpe 1966, Pardo 2008, Elder 1993).
