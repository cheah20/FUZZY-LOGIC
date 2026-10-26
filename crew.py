"""Runs the four agents in order: daily data, perception, cognition, execution.

The cognition agent is the one that does the work: it reads the market
regime and runs the fuzzy engine with the expert rules. This is plain
Python, not CrewAI, but the tool functions could be used by CrewAI later.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time


@dataclass
class AgentState:
    name: str
    source: str
    status: str = "Idle"  # Idle, Processing, Synced, Waiting, Error
    badge_color: str = "gray"
    last_output: Optional[Dict] = None


class AgentPipeline:
    """Four-agent pipeline without CrewAI."""

    def __init__(self):
        self.agents = [
            AgentState(name="Daily Data", source="Binance WS", status="Synced", badge_color="green"),
            AgentState(name="Perception", source="Model-VT Market", status="Idle", badge_color="gray"),
            AgentState(name="Cognition", source="Fuzzy Logic Synthesis", status="Processing", badge_color="orange"),
            AgentState(name="Execution", source="ArKey Active", status="Waiting", badge_color="gray"),
        ]
        self.log: List[str] = []

    def run_pipeline(
        self,
        rsi: float,
        macd_hist: float,
        sentiment: float,
        atr: Optional[float] = None,
        *,
        mode: str = "expert",
    ) -> Dict:
        """Run all four agents and return the final signal with every step's output."""
        self._log("Pipeline initialised")

        # agent 1: daily data
        self._set_agent("Daily Data", "Processing", "orange")
        normalized = self._daily_data_agent(rsi, macd_hist, sentiment, atr)
        self._set_agent("Daily Data", "Synced", "green")
        self._log("Daily Data: Inputs normalized")

        # agent 2: perception
        self._set_agent("Perception", "Processing", "orange")
        trends = self._perception_agent(normalized)
        self._set_agent("Perception", "Synced", "green")
        self._log(f"Perception: RSI signal={trends['rsi_signal']}, MACD={trends['macd_signal']}")

        # agent 3: cognition
        self._set_agent("Cognition", "Processing", "orange")
        from tools import fuzzy_analysis_tool
        cognition_result = fuzzy_analysis_tool(
            rsi=normalized["rsi"],
            macd_hist=normalized["macd_hist"],
            sentiment=normalized["sentiment"],
            atr=normalized.get("atr"),
            mode=mode,
        )
        self._set_agent("Cognition", "Synced", "green")
        regime_info = cognition_result.get("market_regime", {})
        self._log(
            f"Cognition: Regime={regime_info.get('regime','?')}, "
            f"Score={cognition_result['score']}, Label={cognition_result['label']}"
        )

        # agent 4: execution
        self._set_agent("Execution", "Processing", "orange")
        execution_result = self._execution_agent(cognition_result)
        self._set_agent("Execution", "Synced", "green")
        self._log(f"Execution: Signal emitted -> {execution_result['final_signal']}")

        return {
            "final_signal": execution_result["final_signal"],
            "cognition": cognition_result,
            "perception": trends,
            "timestamp": time.time(),
            "mode": mode,
        }


    def _daily_data_agent(
        self,
        rsi: float,
        macd_hist: float,
        sentiment: float,
        atr: Optional[float],
    ) -> Dict:
        return {
            "rsi": max(0.0, min(100.0, float(rsi))),
            "macd_hist": max(-5.0, min(5.0, float(macd_hist))),
            "sentiment": max(-1.0, min(1.0, float(sentiment))),
            "atr": max(0.0, min(10.0, float(atr))) if atr is not None else None,
        }

    def _perception_agent(self, data: Dict) -> Dict:
        rsi = data["rsi"]
        macd = data["macd_hist"]
        sentiment = data["sentiment"]

        # rough trend and sentiment signals
        if rsi < 30:
            rsi_signal = "Oversold"
        elif rsi > 70:
            rsi_signal = "Overbought"
        else:
            rsi_signal = "Neutral"

        if macd > 0.2:
            macd_signal = "Bullish"
        elif macd < -0.2:
            macd_signal = "Bearish"
        else:
            macd_signal = "Neutral"

        if sentiment < -0.35:
            sent_signal = "Bearish"
        elif sentiment > 0.35:
            sent_signal = "Bullish"
        else:
            sent_signal = "Neutral"

        return {
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "sentiment_signal": sent_signal,
            "rsi": rsi,
            "macd_hist": macd,
            "sentiment": sentiment,
        }

    def _execution_agent(self, cognition: Dict) -> Dict:
        label = cognition.get("label", "Hold")
        score = cognition.get("score", 0.0)
        confidence = cognition.get("confidence", 0.5)

        # is the signal strong enough to act on
        actionable = abs(score) >= 0.5

        return {
            "final_signal": label,
            "score": score,
            "confidence": confidence,
            "actionable": actionable,
            "recommendation": (
                f"Execute {label} with {confidence*100:.1f}% confidence"
                if actionable
                else f"Hold position — signal below action threshold ({label}, conf={confidence*100:.1f}%)"
            ),
        }


    def _set_agent(self, name: str, status: str, badge: str):
        for agent in self.agents:
            if agent.name == name:
                agent.status = status
                agent.badge_color = badge
                break

    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {msg}")

    def get_agent_status(self) -> List[Dict]:
        return [
            {
                "name": a.name,
                "source": a.source,
                "status": a.status,
                "badge_color": a.badge_color,
            }
            for a in self.agents
        ]


# one shared pipeline

_pipeline_instance: Optional[AgentPipeline] = None


def get_pipeline() -> AgentPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AgentPipeline()
    return _pipeline_instance


# shortcut for other modules

def run_cognition_analysis(
    rsi: float = 74.2,
    macd_hist: float = 0.35,
    sentiment: float = -0.15,
    atr: Optional[float] = None,
    mode: str = "expert",
) -> Dict:
    """Run the cognition agent on its own. This is what the UI and API call."""
    from tools import fuzzy_analysis_tool
    return fuzzy_analysis_tool(rsi, macd_hist, sentiment, atr, mode=mode)