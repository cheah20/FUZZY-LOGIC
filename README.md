# Fuzzy Logic Adaptive Stock Trading Decision Support System

## Setup Instructions （认真看，不会问AI，写很清楚了）

### 1. Install dependencies （第一步，first！！！！）

```
pip install -r requirements.txt
```

### 2. Install and start Ollama (required for AI agents)

### 用ollama 是因为免费而且不泄露API KEY, 缺点是看你电脑spec 导致你慢or 快

### 不要问问什么不用其他，我是不想share API KEY，if 你们不介意share 你们的， 我是ok for this la

Download Ollama from https://ollama.com, then run:

```
ollama pull llama3.1:8b
ollama serve
```

Keep `ollama serve` running in a separate terminal whenever you use the app.

### 3. Run the application

```
streamlit run interface.py
```

---

## How to Use

### Live Analysis tab

- Enter a stock ticker (e.g. `AAPL`, `MSFT`, `TSLA`)
- Click **ANALYZE (FLC only)** — instant result, no Ollama needed
- Click **RUN AGENTS (LLM)** — full 4-agent AI analysis (requires Ollama running)

### Backtesting tab

- Enter a ticker and adjust Buy/Sell thresholds using the sliders
- Click **RUN BACKTEST** to simulate the FLC strategy on 5 years of historical data
- Results are compared against a Buy & Hold benchmark

---

## Project Structure

```
Trading Project/
├── interface.py          Main Streamlit dashboard
├── fuzzy_engine.py       Fuzzy Logic Controller (FLC)
├── backtester.py         Historical backtest engine
├── Trading_Data.py       Stock data fetching (yfinance + pandas-ta)
├── requirements.txt      Python dependencies
└── agents/
    ├── crew.py           CrewAI 4-agent pipeline
    ├── tools.py          CrewAI tools (news, indicators, FLC)
    └── config/
        ├── agents.yaml   Agent role/goal/backstory definitions
        └── tasks.yaml    Task descriptions and dependencies
```

### .yaml 是crew ai auto create 的file
