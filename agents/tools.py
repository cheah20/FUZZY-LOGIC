"""
agents/tools.py
===============
This module defines the three tools used by the CrewAI agents in our
Fuzzy Logic Adaptive Stock Trading DSS.

Each tool is a Python class that wraps a specific data-fetching or
computation function. By registering these as BaseTool subclasses,
the CrewAI agents can call them autonomously during the pipeline.

Tools defined here:
  FetchNewsSentiment       — Used by the Daily Data Agent
  FetchTechnicalIndicators — Used by the Perception Agent
  RunFuzzyAnalysis         — Used by the Cognition Agent
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

from Trading_Data import get_stock_data
from fuzzy_engine import compute_score


# ── Input Schemas ──────────────────────────────────────────────────────────────
# Pydantic models define and validate the arguments each tool accepts.
# The LLM reads these field descriptions to know what inputs to provide.

class TickerInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL, MSFT, TSLA")


class FuzzyInput(BaseModel):
    rsi:       float = Field(description="RSI value between 0 and 100")
    macd_hist: float = Field(description="MACD histogram value, typically -3 to +3")
    sentiment: float = Field(default=0.0, description="Sentiment score from -1.0 to +1.0")


# ── Tool 1: News Sentiment ─────────────────────────────────────────────────────
# The Daily Data Agent uses this tool to collect market news and produce
# the sentiment score (S) that feeds into our Fuzzy Logic Controller.
# News source: Yahoo Finance's built-in headline feed via yfinance.

class FetchNewsSentiment(BaseTool):
    name: str = "fetch_news_sentiment"
    description: str = (
        "Fetch the 5 most recent news headlines for a stock ticker using Yahoo Finance. "
        "Returns the headlines so the agent can assess the overall market sentiment. "
        "Use this to obtain the sentiment input for the Fuzzy Logic Controller. "
        "Input: stock ticker symbol (e.g. AAPL, MSFT, TSLA)"
    )
    args_schema: type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        import yfinance as yf
        try:
            stock = yf.Ticker(ticker.strip().upper())
            news_items = stock.news or []

            if not news_items:
                return f"No recent news found for {ticker}. Use sentiment = 0.0 (neutral)."

            headlines = []
            for item in news_items[:5]:
                content = item.get("content", {})
                title = content.get("title", "") if isinstance(content, dict) else str(content)
                if title:
                    headlines.append(f"- {title}")

            if not headlines:
                return f"Could not extract headlines for {ticker}. Use sentiment = 0.0."

            result  = f"Latest news headlines for {ticker}:\n"
            result += "\n".join(headlines)
            result += (
                "\n\nBased on these headlines, rate the overall market sentiment "
                "as a number from -1.0 (very negative/bearish) to +1.0 (very positive/bullish). "
                "0.0 = neutral."
            )
            return result

        except Exception as e:
            return f"News fetch failed for {ticker}: {e}. Use sentiment = 0.0 (neutral)."


# ── Tool 2: Technical Indicators ───────────────────────────────────────────────
# The Perception Agent uses this tool to retrieve the two technical indicators
# defined in our proposal: RSI(14) and MACD histogram.
# These are the quantitative inputs (R and M) to the FLC.

class FetchTechnicalIndicators(BaseTool):
    name: str = "fetch_technical_indicators"
    description: str = (
        "Fetch the latest RSI(14) and MACD histogram for a stock ticker. "
        "Returns the exact numeric values required by the Fuzzy Logic Controller. "
        "Input: stock ticker symbol (e.g. AAPL, MSFT, TSLA)"
    )
    args_schema: type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        df = get_stock_data(ticker.strip().upper())

        if df is None or df.empty:
            return f"ERROR: No data found for ticker '{ticker}'."

        last = df.iloc[-1]
        rsi  = last.get("RSI", None)

        # The MACD histogram column is named by pandas_ta as "MACDh_12_26_9"
        macd_h_col = next((c for c in df.columns if c.startswith("MACDh_")), None)
        macd_hist  = float(last[macd_h_col]) if macd_h_col else 0.0

        if rsi is None or str(rsi) == "nan":
            return f"ERROR: RSI could not be computed for '{ticker}'."

        rsi  = float(rsi)
        date = df.index[-1].strftime("%Y-%m-%d")

        return (
            f"Ticker : {ticker.upper()}\n"
            f"Date   : {date}\n"
            f"Price  : ${float(last['Close']):.2f}\n"
            f"RSI(14): {rsi:.2f}\n"
            f"MACD-H : {macd_hist:.4f}\n"
        )


# ── Tool 3: Fuzzy Logic Controller ────────────────────────────────────────────
# The Cognition Agent uses this tool to run the FLC inference engine.
# It takes the three inputs (RSI, MACD, Sentiment) collected by the
# previous agents and returns a trading score from 1 to 5.
# Score mapping: 1=Strong Sell, 2=Sell, 3=Hold, 4=Buy, 5=Strong Buy

class RunFuzzyAnalysis(BaseTool):
    name: str = "run_fuzzy_analysis"
    description: str = (
        "Run the Fuzzy Logic Controller (FLC) to compute a trading score from 1 to 5. "
        "Inputs: rsi (0 to 100), macd_hist (MACD histogram value), "
        "sentiment (-1.0 to +1.0, default 0.0 if not available). "
        "Call this after obtaining RSI and MACD from fetch_technical_indicators "
        "and sentiment from fetch_news_sentiment."
    )
    args_schema: type[BaseModel] = FuzzyInput

    def _run(self, rsi: float, macd_hist: float, sentiment: float = 0.0) -> str:
        result = compute_score(float(rsi), float(macd_hist), float(sentiment))
        return (
            f"FLC Result:\n"
            f"  Score      : {result['score']} / 5\n"
            f"  Signal     : {result['label']}\n"
            f"  RSI state  : {result['rsi_label']}\n"
            f"  MACD state : {result['macd_label']}\n"
        )


# Instantiate once — imported by crew.py
fetch_news_sentiment       = FetchNewsSentiment()
fetch_technical_indicators = FetchTechnicalIndicators()
run_fuzzy_analysis         = RunFuzzyAnalysis()
