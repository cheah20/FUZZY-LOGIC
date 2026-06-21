# ASTDSS Terminal — Fuzzy Logic Trading Decision Support System

This is a web app (built with Python + Flask) that shows how a fuzzy logic
system can analyze a stock or crypto ticker and suggest a trading decision
(Buy / Sell / Hold), explain *why* it made that decision, and compare itself
against a simple baseline. It also has an "Agentic Layer" where an AI model
(DeepSeek) proposes adjustments, but a rule-based "Human Expert" layer has
the final say — the AI never makes the final call by itself.

This README explains how to get it running on your own computer, step by
step, even if you've never set up a Python project before.

## What you need before starting

1. **Python 3.10 or newer.** Check what you have by opening a terminal and
   typing:
   ```
   python --version
   ```
   If that command isn't found, or shows Python 2.x, download Python from
   https://www.python.org/downloads/ and install it. On Windows, tick the
   box that says "Add Python to PATH" during install.

2. **A code editor** (VS Code is fine, or anything you like).

3. **A DeepSeek API key.** This is free to create. Go to
   https://platform.deepseek.com/api_keys, sign up, and create a key. You'll
   need it in step 5 below.

4. (Optional) **A Marketaux API key**, if you want real finance news on the
   Market Intelligence page instead of sample demo news. Get one free at
   https://www.marketaux.com/ (look for the free plan — 100 requests/day).
   If you skip this, the app still works fine, it just shows demo news
   instead of live news.

## Step-by-step setup

**1. Get the code onto your computer**

If you haven't already, clone this repository:
```
git clone https://github.com/cheah20/FUZZY-LOGIC.git
cd FUZZY-LOGIC
```

**2. Create a virtual environment**

A virtual environment keeps this project's Python packages separate from
everything else on your computer. From inside the project folder:
```
python -m venv venv
```

**3. Activate the virtual environment**

On Windows (Command Prompt or PowerShell):
```
venv\Scripts\activate
```
On Mac/Linux:
```
source venv/bin/activate
```
You'll know it worked because your terminal prompt will now show `(venv)`
at the start of the line. You need to do this every time you open a new
terminal to work on this project.

**4. Install the required packages**

```
pip install -r requirements.txt
```
This downloads Flask, the DeepSeek client, and everything else the app
needs. It might take a minute or two.

**5. Set up your API keys**

Copy the example environment file:
```
copy .env.example .env
```
(On Mac/Linux, use `cp .env.example .env` instead.)

Open the new `.env` file in your editor and fill in:
- `DEEPSEEK_API_KEY` — paste the key you created in the "What you need"
  section above.
- `MARKETAUX_API_KEY` — optional, paste your Marketaux key if you have one.
  If you leave the placeholder text here, the app just falls back to demo
  news automatically — nothing breaks.

**Never commit your real `.env` file to GitHub.** It's already listed in
`.gitignore` so Git will ignore it automatically, but don't paste your real
keys anywhere else (chat messages, screenshots, etc.).

**6. Run the app**

```
python app.py
```
You should see Flask print something like
`Running on http://127.0.0.1:5000`. Leave this terminal window open — it's
your running server.

**7. Open it in your browser**

Go to: http://127.0.0.1:5000

If you see the dashboard, you're done — the system is running.

## A quick tour of the pages

- **Dashboard** — type a ticker (e.g. `AAPL`, `BTC`) and click Analyze to
  see a live price chart, RSI/ATR readings, and the current expert signal.
- **Market Intelligence** — finance news feed (real if you set up a
  Marketaux key, demo sample otherwise) plus an AI-generated market brief.
- **Decision Logic** — shows exactly how the fuzzy logic engine reached its
  decision: which inputs were active or masked, the rules that fired, and
  the final signal.
- **Backtesting** — run the strategy against real historical price data and
  compare it to a simple baseline and buy-and-hold.
- **Agentic Layer** — run the full pipeline: baseline fuzzy logic → expert
  fuzzy logic → an LLM proposes a tweak → a rule-based validation gate
  approves, rejects, or modifies that tweak → final decision.
- **Tutorial** — a slower, step-by-step walkthrough of the same agentic
  pipeline, with an explanation at every stage.
- **Settings** — a visual/illustrative settings panel (not wired to a real
  backend — the real controls are the engine files themselves).

## Running the test scripts

There are a few scripts in `tests/` that check the system is working
correctly end-to-end:
```
python tests/e2e_test.py
python tests/validate_interaction.py
```
Both should print `PASS` for every check. If something fails, read the
error message — it'll tell you which route or function broke.

## Troubleshooting

- **"Address already in use" / port 5000 busy** — you already have another
  copy of `app.py` running somewhere. Close that terminal window first, or
  find and stop the process using that port.
- **Market Intelligence always shows "DEMO SAMPLE FEED"** — this means
  either you didn't set `MARKETAUX_API_KEY` in `.env`, or the key isn't
  valid yet (e.g. email not verified on marketaux.com). The app is working
  correctly — it's designed to fall back to demo news instead of crashing.
- **`ModuleNotFoundError` for some package** — you probably forgot to
  activate the virtual environment (step 3) before running `pip install` or
  `python app.py`.
- **Changed `.env` but nothing changed in the browser** — environment
  variables are only read once, when `app.py` starts. Stop the server
  (Ctrl+C in its terminal) and run `python app.py` again.

## Project structure

```
app.py                 Flask routes (web pages + API endpoints)
fuzzy_engine.py         Core fuzzy logic math
expert_layer.py         Human Expert regime detection
feature_selector.py     Decides which inputs to trust per regime
backtester.py           Runs the strategy against historical price data
agentic_pipeline.py     Orchestrates the full agentic decision flow
llm_reasoning.py         Talks to DeepSeek for the LLM proposal step
expert_validation_gate.py  Rule-based check on the LLM's proposal
human_expert_kb.py       The rule knowledge base used by the validation gate
memory_log.py            Saves a record of every agentic run
news_service.py          Marketaux news + AI market brief
Trading_Data.py          Fetches real price data (via yfinance)
templates/               HTML pages
static/                  CSS and JavaScript
tests/                   End-to-end test scripts
docs/research/           Academic papers behind the indicators/methods used
```
