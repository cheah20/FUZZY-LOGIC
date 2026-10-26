# NeuralTrade

**Fuzzy Logic Stock Trading Decision Support System**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-3f6fb6)

NeuralTrade is a web app that looks at a stock or crypto ticker and suggests **Buy, Sell or Hold**, explains how it got there, and compares itself with a simple baseline and with buy-and-hold. A Mamdani fuzzy engine does the main work. A rule-based expert layer decides which inputs to trust in each kind of market. An LLM (DeepSeek) can suggest small changes, but each suggestion has to pass a rule check first, so the LLM never decides on its own.

Undergraduate Fuzzy Logic course project (PA3: Agentic Fuzzy Logic), developed May – June 2026.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Objectives](#project-objectives)
- [System Architecture](#system-architecture)
- [Fuzzy Logic Design](#fuzzy-logic-design)
- [Features](#features)
- [Development Timeline](#development-timeline)
- [Test Results](#test-results)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Pages](#pages)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [References](#references)

## Project Overview

Technical indicators such as RSI, MACD and ATR often disagree, and their meaning depends on the market situation. A strong news event can make every chart signal irrelevant, while in a calm market the indicators matter most. Rigid "if RSI < 30 then buy" rules ignore this.

NeuralTrade handles the uncertainty with fuzzy logic. Each indicator is turned into degrees of membership (for example, RSI is 0.7 "oversold" and 0.3 "neutral"), 26 fuzzy rules combine them, and centroid defuzzification turns the result into a trading signal. On top of this, a Human Expert layer detects the market regime and masks or re-weights inputs, following how professional traders focus on different information in different conditions.

The agentic layer adds an LLM that reads the full market context and proposes a small shift to the fuzzy score. A deterministic validation gate, built from 16 literature-based expert rules, approves, modifies or rejects each proposal, and every run is logged for review.

## Project Objectives

1. Model trading decisions under uncertainty with a Mamdani fuzzy inference system using RSI, MACD histogram, news sentiment and ATR.
2. Capture human expert knowledge as market-regime rules that decide which inputs are active and how much each one counts.
3. Show that the expert-adjusted system behaves differently from, and more sensibly than, a baseline where all inputs count equally.
4. Add an agentic AI layer whose suggestions are always checked against a rule-based knowledge base before they can change a decision.
5. Evaluate the strategies with backtesting on real price data (total return, win rate, maximum drawdown, Sharpe ratio).
6. Explain every decision in the interface: active inputs, fired rules, the AI proposal, and why it was accepted or rejected.

## System Architecture

```text
Market data (yfinance) + news (Marketaux) ──► Trading_Data.py: RSI, MACD, ATR, Bollinger Bands, sentiment
                                                     │
                       ┌─────────────────────────────┼─────────────────────────────┐
                       ▼                             ▼                             ▼
              Baseline fuzzy engine        Human Expert layer              ANFIS (neuro-fuzzy)
              all inputs equal             regime → feature masking        learned parameters
                       │                             │                             │
                       └──────────────► Expert-adjusted fuzzy score ◄──────────────┘
                                                     │
                                          LLM proposes a score shift
                                         (DeepSeek, or demo scenarios)
                                                     │
                                    Human Expert Validation Gate (16 rules)
                                    APPROVED / MODIFIED / REJECTED / NEEDS_REVIEW
                                                     │
                                Final decision + explanation + experience log
```

## Fuzzy Logic Design

### Inputs and outputs

| Variable | Linguistic terms |
|---|---|
| RSI | oversold, neutral, overbought |
| MACD histogram | bearish, neutral, bullish |
| News sentiment | bearish, neutral, bullish |
| ATR (volatility) | low, normal, high |
| **Output: signal** | Strong Sell, Sell, Hold, Buy, Strong Buy |

Inference: Mamdani, 26 rules, centroid defuzzification (`fuzzy_engine.py`).

### Market regimes and feature weights

The Human Expert layer (`expert_layer.py`) detects the regime; `feature_selector.py` then decides which inputs are used.

| Regime | When | Input weights |
|---|---|---|
| NEWS_DRIVEN_BEARISH | Extreme negative sentiment | Sentiment 100% |
| NEWS_DRIVEN_BULLISH | Extreme positive sentiment | Sentiment 100% |
| TECHNICAL_DRIVEN | Neutral sentiment | RSI 60%, MACD 40% |
| CONFLICTING_SIGNALS | RSI and MACD disagree | RSI 30%, MACD 30%, Sentiment 20%, ATR 20% |
| HIGH_UNCERTAINTY | Extreme volatility | 25% each |
| NORMAL | Default | RSI 40%, MACD 30%, Sentiment 30% |

### Human Expert Knowledge Base

16 rules in `human_expert_kb.py`, each with a rule ID, the captured knowledge, a validation rule, the action, and its source (Wilder 1978, Murphy 1999, Appel 2005, Elder 1993, Bollinger 2002, Sharpe 1966, Pardo 2008). The validation gate applies them deterministically; it does not use an LLM.

### ANFIS layer

`anfis.py` implements a 5-layer Sugeno-type ANFIS (Jang, 1993) on RSI, MACD histogram and ATR. Its Gaussian membership functions start from the same literature-based boundaries and are then trained on historical prices with Jang's hybrid learning (least squares for the consequents, gradient descent for the premises). It runs as a parallel proposer and is still checked by the validation gate.

## Features

| Feature | Status |
|---|---|
| Market data and indicators: RSI, MACD, ATR, Bollinger Bands, SMA, sentiment | Done |
| Baseline Mamdani fuzzy engine (all inputs equal) | Done |
| Human Expert regime detection (6 regimes) | Done |
| Dynamic feature masking and weighting per regime | Done |
| Expert-adjusted fuzzy score and baseline comparison | Done |
| Backtesting with 70/30 calibration/testing split; return, win rate, max drawdown, Sharpe ratio | Done |
| Market Intelligence: live Marketaux news with AI market brief, demo feed fallback | Done |
| Agentic pipeline: LLM proposal → validation gate → final decision | Done |
| Human Expert Validation Gate with 16 literature-based rules | Done |
| Failure scenarios with structured reflection (AI wrong, gate correct) | Done |
| Experience log of every agentic run, plus full LLM prompt/response log | Done |
| ANFIS neuro-fuzzy layer with hybrid learning | Done |
| Step-by-step Tutorial page explaining the pipeline | Done |
| Works with no API keys (demo mode for AI and news) | Done |
| Broker-style interface: candle chart, glass price card, side menu that slides out on hover, keys 1-7 to switch pages | Done |

## Development Timeline

| Period | Work | Status |
|---|---|---|
| Early May 2026 | Repository set-up; market data layer and technical indicators | Done |
| May 2026 | Baseline Mamdani fuzzy engine: membership functions, 26 rules, defuzzification | Done |
| Mid – late May 2026 | Human Expert layer, regime detection and dynamic feature selection | Done |
| Late May 2026 | Backtesting engine and baseline vs expert comparison | Done |
| Early June 2026 | Flask web interface: Dashboard, Decision Logic, Backtesting, Market Intelligence | Done |
| Mid June 2026 | Agentic layer: LLM reasoning, Human Expert Knowledge Base, validation gate, failure reflection | Done |
| 21 June 2026 | End-to-end testing and submission | Done |
| 22 June 2026 | ANFIS neuro-fuzzy layer and full LLM prompt/response logging | Done |

**Status:** completed in June 2026. The interface and the repository were tidied up later.

## Test Results

| Script | What it checks | Result |
|---|---|---|
| `tests/e2e_test.py` | Every page, API endpoint, agent run and data flow, plus response time | 65 / 65 pass |
| `tests/validate_interaction.py` | Charts, period selector, and that the Human Expert layer changes decisions | 44 / 44 pass |
| `tests/run_test.py` | Indicator ranges, fuzzy engine, expert vs baseline behaviour, all project phases | All phases pass |

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Fuzzy engine, expert layer, ANFIS, backtester |
| Flask | Web pages and API routes |
| NumPy | ANFIS training (least squares, gradients) |
| yfinance | Historical stock and crypto prices |
| DeepSeek API (OpenAI client) | LLM proposals and market brief (optional) |
| Marketaux API | Live finance news (optional) |
| HTML, CSS, JavaScript | Interface with SVG charts |

## Getting Started

You need **Python 3.10 or newer**. API keys are optional: without them the app runs in demo mode.

```bash
git clone https://github.com/cheah20/NeuralTrade.git
cd NeuralTrade
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env
```

| Variable in `.env` | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | Real LLM proposals and market brief. Get one at https://platform.deepseek.com/api_keys |
| `MARKETAUX_API_KEY` | Live news on Market Intelligence. Free plan at https://www.marketaux.com/ |
| `NEWS_API_PROVIDER` | News source, `marketaux` |
| `NEWS_REFRESH_MINUTES` | How long news is cached (default 15) |

Never commit your real `.env`; it is already in `.gitignore`.

| Action | Command |
|---|---|
| Start | `python app.py`, then open http://127.0.0.1:5000 |
| Stop | `Ctrl + C` |
| Run tests | `python tests/e2e_test.py` and `python tests/validate_interaction.py` |

## Pages

| Page | What it shows |
|---|---|
| Dashboard | Pick a ticker from the watchlist (or type one) for a daily candle chart, the expert signal, the indicator weights and the reasoning |
| Market Intelligence | Finance news feed and an AI market brief |
| Decision Logic | How the fuzzy engine decided: active or masked inputs, fired rules, final signal |
| Backtesting | Strategy vs baseline vs buy-and-hold on historical data |
| Agentic Layer | The full pipeline: baseline → expert → LLM proposal → validation gate → final decision |
| Tutorial | A slower walkthrough of the same pipeline with an explanation at each stage |
| Settings | Illustrative settings panel (the real controls are the engine files) |

## Troubleshooting

| Problem | Fix |
|---|---|
| Port 5000 already in use | Another `app.py` is running. Close that terminal first |
| Market Intelligence shows "DEMO SAMPLE FEED" | `MARKETAUX_API_KEY` is missing or not yet valid. The app falls back to demo news on purpose |
| `ModuleNotFoundError` | Activate the virtual environment before `pip install` or `python app.py` |
| Changed `.env` but nothing changed | Restart `app.py`; `.env` is only read at start-up |

## Project Structure

```text
app.py                     Flask routes (pages + API endpoints)
fuzzy_engine.py            Mamdani fuzzy logic controller
expert_layer.py            Human Expert regime detection
feature_selector.py        Input masking and weights per regime
anfis.py                   ANFIS neuro-fuzzy layer
backtester.py              Strategy evaluation on historical prices
agentic_pipeline.py        Full agentic decision cycle
llm_reasoning.py           DeepSeek proposals (demo mode without a key)
expert_validation_gate.py  Rule-based check of each LLM proposal
human_expert_kb.py         16-rule Human Expert Knowledge Base
failure_reflection.py      Failure scenarios and reflections
memory_log.py              Experience log of every agentic run
news_service.py            Marketaux news + AI market brief
Trading_Data.py            Price data and technical indicators
crew.py, tools.py          Multi-agent orchestration and tool functions
agent_profile.py           Agent mission, constraints and governance
interface.py               Data for the explainability panels
templates/, static/        HTML pages, CSS and JavaScript
tests/                     End-to-end and interaction tests
data/                      ANFIS parameters and the experience log
docs/research/             Reading list of the papers behind the design
```

## References

Books and classic papers behind the Human Expert Knowledge Base:

- J. W. Wilder, *New Concepts in Technical Trading Systems*, Trend Research, 1978.
- J. J. Murphy, *Technical Analysis of the Financial Markets*, New York Institute of Finance, 1999.
- G. Appel, *Technical Analysis: Power Tools for Active Investors*, FT Prentice Hall, 2005.
- A. Elder, *Trading for a Living*, Wiley, 1993.
- J. Bollinger, *Bollinger on Bollinger Bands*, McGraw-Hill, 2002.
- W. F. Sharpe, "Mutual Fund Performance," *Journal of Business*, vol. 39, no. 1, pp. 119–138, 1966.
- R. Pardo, *The Evaluation and Optimization of Trading Strategies*, 2nd ed., Wiley, 2008.
- J.-S. R. Jang, "ANFIS: Adaptive-Network-Based Fuzzy Inference System," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 23, no. 3, pp. 665–685, 1993.

Research papers:

1. R. L. de Almeida and R. F. Neves, "Stock market prediction and portfolio composition using a hybrid approach combined with self-adaptive evolutionary algorithm," *Expert Systems with Applications*, vol. 204, 117478, 2022. https://doi.org/10.1016/j.eswa.2022.117478
2. M. Leeds, "Bollinger Bands Thirty Years Later," arXiv:1212.4890, 2013. https://arxiv.org/abs/1212.4890
3. P. S. K. Prasad, V. Madhav, R. Lal and V. Ravi, "Optimal Technical Indicator-based Trading Strategies Using NSGA-II," arXiv:2111.13364, 2021. https://arxiv.org/abs/2111.13364
4. P. T. Chio, "A comparative study of the MACD-base trading strategies: evidence from the US stock market," arXiv:2206.12282, 2022. https://arxiv.org/abs/2206.12282
5. B. J. Zaini, R. Mansor, N. Yusof and B. H. Sang, "Classify Stock Market Movement Based on Technical Analysis Indicators Using Logistic Regression," *Journal of Advanced Research in Business and Management Studies*, vol. 14, no. 1, pp. 35–41, 2019.
6. R. M. Dhokane and S. Agarwal, "LSTM Deep Learning Based Stock Price Prediction with Bollinger Band, RSI, MACD, and OHLC Features," *International Journal of Intelligent Systems and Applications in Engineering*, vol. 12, no. 3, pp. 1169–1176, 2024.
7. Y. Shynkevich, T. M. McGinnity, S. Coleman, A. Belatreche and Y. Li, "Forecasting Price Movements using Technical Indicators: Investigating the Impact of Varying Input Window Length," *Neurocomputing*, 2017.
8. L. Ming and G. Chen, "Stock price prediction based on relative strength index, moving average convergence divergence and XGBoost model," in *Proc. Data Driven Control and Learning Systems Conference (DDCLS)*, 2024. https://doi.org/10.1109/DDCLS61622.2024.10606580
9. G. Cohen, A. Aiche and R. Eichel, "Artificial Intelligence Models for Predicting Stock Returns Using Fundamental, Technical, and Entropy-Based Strategies: A Semantic-Augmented Hybrid Approach," *Entropy*, vol. 27, no. 6, 550, 2025. https://doi.org/10.3390/e27060550
10. "Average True Range Indicator Trading Strategy," practitioner guide.

Which paper supports which rule: [docs/research/README_RESEARCH.md](docs/research/README_RESEARCH.md).

---

**Project:** NeuralTrade — Fuzzy Logic Stock Trading Decision Support System
**Project Type:** Undergraduate Fuzzy Logic course project (PA3)
**Development Period:** May – June 2026
**Status:** Completed
