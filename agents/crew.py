"""
agents/crew.py
==============
This module implements the multi-agent orchestration layer of our
Fuzzy Logic Adaptive Stock Trading Decision Support System (DSS).

System Design (as per project proposal):
  The DSS uses a 4-agent CrewAI pipeline where each agent has a distinct
  role in the trading decision process:

    1. Daily Data Agent    — collects and interprets market news sentiment
    2. Perception Agent    — retrieves quantitative technical indicators (RSI, MACD)
    3. Cognition Agent     — feeds data into the Fuzzy Logic Controller (FLC)
    4. Execution Agent     — translates the FLC score into a human-readable recommendation

  Agent roles, goals, and backstories are defined in agents.yaml and tasks.yaml
  rather than hardcoded here. This separation makes it easy to adjust agent
  behaviour without modifying the Python code.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from agents.tools import fetch_news_sentiment, fetch_technical_indicators, run_fuzzy_analysis

load_dotenv()

# The local Ollama model used for all agents.
# We use llama3.1:8b as it provides reliable tool-calling support
# and runs efficiently on an RTX 4070 GPU (~4.7 GB VRAM).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def _get_llm() -> LLM:
    """
    Returns a CrewAI LLM instance connected to the local Ollama server.

    We chose Ollama over a cloud API so the system runs entirely offline —
    no subscription fees and no data leaving the machine. This is consistent
    with the proposal's goal of a self-contained trading DSS.

    Temperature is set to 0 so the agents produce deterministic, consistent
    outputs rather than creative but unpredictable ones.
    """
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url="http://localhost:11434",
        temperature=0.0,
    )


def _load_yaml_config() -> tuple[dict, dict]:
    """
    Loads agent and task definitions from the YAML configuration files.

    Storing agent properties (role, goal, backstory) in YAML rather than
    Python code follows the CrewAI best-practice pattern and allows the
    financial domain expert on our team to adjust agent behaviour without
    touching the codebase.
    """
    import yaml
    config_dir = Path(__file__).parent / "config"

    with open(config_dir / "agents.yaml", encoding="utf-8") as f:
        agents_cfg = yaml.safe_load(f)

    with open(config_dir / "tasks.yaml", encoding="utf-8") as f:
        tasks_cfg = yaml.safe_load(f)

    return agents_cfg, tasks_cfg


def _fill_placeholder(text: str, ticker: str) -> str:
    """Substitutes the {ticker} placeholder in YAML config strings at runtime."""
    return text.replace("{ticker}", ticker)


def run_trading_crew(ticker: str, sentiment_override: float = None) -> dict:
    """
    Runs the full 4-agent trading analysis pipeline for a given stock ticker.

    Pipeline flow (sequential):
      Daily Data Agent  →  fetches and scores news sentiment
      Perception Agent  →  retrieves RSI(14) and MACD histogram
      Cognition Agent   →  runs the Fuzzy Logic Controller with all three inputs
      Execution Agent   →  writes a plain-English investment recommendation

    Parameters
    ----------
    ticker            : Stock symbol to analyse, e.g. "AAPL"
    sentiment_override: Optional pre-computed sentiment score (skips news fetch)

    Returns
    -------
    dict with keys:
        score          — FLC score from 1 (Strong Sell) to 5 (Strong Buy)
        label          — Linguistic signal label
        sentiment      — News sentiment score (-1.0 to +1.0)
        recommendation — Plain-English brief from the Execution Agent
        logs           — Step-by-step pipeline log for the UI
    """
    llm  = _get_llm()
    logs = [f"[CREW]  Initializing pipeline for {ticker}..."]

    agents_cfg, tasks_cfg = _load_yaml_config()

    # Resolve {ticker} placeholders in all YAML string values
    def cfg(key: str, section: dict) -> dict:
        filled = {}
        for k, v in section[key].items():
            filled[k] = _fill_placeholder(str(v), ticker) if isinstance(v, str) else v
        return filled

    # ── Agent 1: Daily Data Agent ──────────────────────────────────────────────
    # Responsible for news sentiment analysis as described in the proposal.
    # Uses Yahoo Finance headlines as the sentiment data source.
    daily_agent = Agent(
        **cfg("daily_data_agent", agents_cfg),
        tools=[fetch_news_sentiment],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ── Agent 2: Perception Agent ──────────────────────────────────────────────
    # Retrieves the technical indicators that feed into the FLC.
    # RSI and MACD are the two quantitative inputs defined in the proposal.
    perception_agent = Agent(
        **cfg("perception_agent", agents_cfg),
        tools=[fetch_technical_indicators],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ── Agent 3: Cognition Agent ───────────────────────────────────────────────
    # The bridge between raw data and the fuzzy inference system.
    # Extracts numeric values from previous agents and invokes the FLC tool.
    cognition_agent = Agent(
        **cfg("cognition_agent", agents_cfg),
        tools=[run_fuzzy_analysis],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ── Agent 4: Execution Agent ───────────────────────────────────────────────
    # Converts the FLC score into a concise recommendation for retail investors.
    # No tools needed — this agent only generates text output.
    execution_agent = Agent(
        **cfg("execution_agent", agents_cfg),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ── Task Definitions ───────────────────────────────────────────────────────
    # Each task is assigned to one agent. The context parameter ensures that
    # downstream tasks receive the outputs of upstream tasks before executing.
    task_sentiment = Task(
        description=cfg("task_daily_sentiment", tasks_cfg)["description"],
        expected_output=cfg("task_daily_sentiment", tasks_cfg)["expected_output"],
        agent=daily_agent,
    )

    task_perception = Task(
        description=cfg("task_perception", tasks_cfg)["description"],
        expected_output=cfg("task_perception", tasks_cfg)["expected_output"],
        agent=perception_agent,
    )

    # Cognition waits for both sentiment and technical indicators before running
    task_cognition = Task(
        description=cfg("task_cognition", tasks_cfg)["description"],
        expected_output=cfg("task_cognition", tasks_cfg)["expected_output"],
        agent=cognition_agent,
        context=[task_perception, task_sentiment],
    )

    # Execution agent receives all prior outputs to write a complete brief
    task_execution = Task(
        description=cfg("task_execution", tasks_cfg)["description"],
        expected_output=cfg("task_execution", tasks_cfg)["expected_output"],
        agent=execution_agent,
        context=[task_perception, task_sentiment, task_cognition],
    )

    # ── Run the Crew ───────────────────────────────────────────────────────────
    # Sequential process ensures the pipeline runs in the correct order.
    # Each agent's output becomes available to the next agent via context.
    crew = Crew(
        agents=[daily_agent, perception_agent, cognition_agent, execution_agent],
        tasks=[task_sentiment, task_perception, task_cognition, task_execution],
        process=Process.sequential,
        verbose=False,
    )

    logs.append("[CREW]  Running 4-agent pipeline...")
    final_result = crew.kickoff()
    logs.append("[CREW]  Pipeline complete.")

    # Parse the FLC score out of the cognition agent's text output
    cognition_raw = task_cognition.output.raw if task_cognition.output else ""
    flc_score, flc_label = _parse_flc_score(cognition_raw)

    # Parse the sentiment score out of the daily agent's text output
    sentiment_raw = task_sentiment.output.raw if task_sentiment.output else ""
    sentiment = _parse_sentiment(sentiment_raw)

    logs.append(f"[FLC]   Score={flc_score}/5  Signal={flc_label}")
    logs.append(f"[SENT]  Sentiment={sentiment:+.2f}")

    return {
        "score"          : flc_score,
        "label"          : flc_label,
        "sentiment"      : sentiment,
        "recommendation" : str(final_result.raw),
        "logs"           : logs,
    }


# ── Output Parsers ─────────────────────────────────────────────────────────────
# These functions extract numeric values from the agents' free-text outputs.
# Regex is used because the LLM may vary its phrasing slightly between runs.

def _parse_flc_score(text: str) -> tuple[float, str]:
    """Extracts the FLC score and signal label from the cognition agent output."""
    score_m = re.search(r"Score\s*[:\-]\s*([\d.]+)", text, re.IGNORECASE)
    label_m = re.search(r"(Strong Buy|Buy|Hold|Sell|Strong Sell)", text, re.IGNORECASE)
    score = float(score_m.group(1)) if score_m else 3.0
    label = label_m.group(1).title() if label_m else "Hold"
    return score, label


def _parse_sentiment(text: str) -> float:
    """Extracts the numeric sentiment score from the daily agent output."""
    m = re.search(r"[-+]?\d+\.?\d*", text)
    if m:
        val = float(m.group())
        return max(-1.0, min(1.0, val))
    return 0.0
