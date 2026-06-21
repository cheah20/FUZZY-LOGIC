"""
news_service.py — Finance/Trading News Layer
ASTDSS Terminal — Market Intelligence News Provider

Provides:
  - get_market_news(symbol=None) -> structured payload for the Market
    Intelligence "AI Market Brief" panel.

Behaviour:
  - If MARKETAUX_API_KEY is set, fetches real finance/trading headlines
    from Marketaux (https://www.marketaux.com/).
  - Caches the result to data/news_cache.json and reuses it until
    NEWS_REFRESH_MINUTES has elapsed (default 15).
  - Falls back to a small demo sample feed if no API key is configured,
    the API call fails, or it returns no articles. Never raises — the
    Flask app must keep working with zero API keys configured.
  - If DEEPSEEK_API_KEY is set, asks DeepSeek for a short market brief
    summarising the headlines. Falls back to a rule-based summary
    (bullish/bearish headline count) if DeepSeek is unavailable or fails.
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "news_cache.json")

# Demo sample feed (used when no API key / API failure / empty result)
_DEMO_ARTICLES = [
    {
        "time": "14:20:00",
        "content": "S&P 500 futures dip on hawkish Fed comments regarding sustained high interest rates through Q3.",
        "source": "Demo Sample",
        "sentiment": "BEARISH",
        "tags": ["BEARISH SENTIMENT", "MACRO POLICY"],
        "url": "",
    },
    {
        "time": "14:15:22",
        "content": "Tech sector rally gains momentum as major chipmaker announces breakthrough in 2nm production efficiency.",
        "source": "Demo Sample",
        "sentiment": "BULLISH",
        "tags": ["BULLISH SENTIMENT", "SEMICONDUCTORS"],
        "url": "",
    },
    {
        "time": "14:02:49",
        "content": "Crude oil volatility spikes following unscheduled OPEC technical meeting announcement.",
        "source": "Demo Sample",
        "sentiment": "NEUTRAL",
        "tags": ["HIGH VOLATILITY", "ENERGY"],
        "url": "",
    },
    {
        "time": "13:55:12",
        "content": "European Central Bank maintains deposit facility rate, signals \"data-dependent\" approach for upcoming sessions.",
        "source": "Demo Sample",
        "sentiment": "NEUTRAL",
        "tags": ["UNCERTAIN SUPPLY", "CENTRAL BANKS"],
        "url": "",
    },
]


def _ensure_data_dir() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)


def _atomic_write_json(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def _load_cache() -> Optional[Dict]:
    _ensure_data_dir()
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _cache_is_fresh(cache: Dict, refresh_minutes: int) -> bool:
    if not cache or "fetched_at" not in cache:
        return False
    try:
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
    except (ValueError, TypeError):
        return False
    age_minutes = (datetime.now() - fetched_at).total_seconds() / 60.0
    return age_minutes < refresh_minutes


def _demo_articles() -> List[Dict]:
    return [dict(a) for a in _DEMO_ARTICLES]


def _sentiment_from_entities(entities: List[Dict]) -> str:
    """Marketaux gives a sentiment_score per entity (-1..1, may be None). Average them."""
    scores = [e.get("sentiment_score") for e in entities if e.get("sentiment_score") is not None]
    if not scores:
        return "NEUTRAL"
    avg = sum(scores) / len(scores)
    if avg >= 0.15:
        return "BULLISH"
    if avg <= -0.15:
        return "BEARISH"
    return "NEUTRAL"


def _fetch_marketaux(symbol: Optional[str], api_key: str) -> List[Dict]:
    """Fetch live finance/trading headlines from Marketaux. Raises on HTTP/network failure."""
    import requests

    params = {
        "api_token": api_key,
        "language": "en",
        "limit": 6,
        "filter_entities": "true",
    }
    if symbol:
        params["symbols"] = symbol.upper()

    resp = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=8)
    resp.raise_for_status()
    payload = resp.json()

    articles = []
    for item in payload.get("data", []):
        entities = item.get("entities") or []
        published = item.get("published_at", "")
        time_str = published[11:19] if len(published) >= 19 else published
        articles.append({
            "time": time_str or "—",
            "content": item.get("title", ""),
            "description": item.get("description", "") or "",
            "url": item.get("url", ""),
            "source": (item.get("source") or "Marketaux"),
            "sentiment": _sentiment_from_entities(entities),
            "tags": [e.get("symbol") for e in entities if e.get("symbol")][:2] or ["MARKET"],
        })
    return articles


def _rule_based_brief(articles: List[Dict]) -> str:
    bullish = sum(1 for a in articles if a.get("sentiment") == "BULLISH")
    bearish = sum(1 for a in articles if a.get("sentiment") == "BEARISH")
    neutral = len(articles) - bullish - bearish
    if not articles:
        return "No headlines available to summarise."
    if bullish > bearish:
        lean = "leaning bullish"
    elif bearish > bullish:
        lean = "leaning bearish"
    else:
        lean = "mixed / no clear lean"
    return (
        f"Rule-based fallback summary: {len(articles)} headlines reviewed "
        f"({bullish} bullish, {bearish} bearish, {neutral} neutral) — overall tone is {lean}."
    )


def _deepseek_brief(articles: List[Dict]) -> Optional[str]:
    """Ask DeepSeek to summarise the headlines. Returns None if unavailable/failed."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        headlines = "\n".join(f"- {a.get('content', '')}" for a in articles[:6] if a.get("content"))
        if not headlines:
            return None

        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=180,
            temperature=0.4,
            messages=[
                {"role": "system", "content": (
                    "You are a concise financial news summarizer for an academic trading "
                    "decision-support system. Summarise the given headlines in 2-3 plain "
                    "sentences, noting the overall market tone. Do not invent facts not "
                    "present in the headlines."
                )},
                {"role": "user", "content": f"Headlines:\n{headlines}\n\nWrite the market brief."},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception as exc:
        print(f"[news_service] DeepSeek brief failed, using rule-based fallback: {exc}")
        return None


def get_market_news(symbol: Optional[str] = None, force_refresh: bool = False) -> Dict:
    """
    Return the Market Intelligence news payload.

    Returns:
        {
            "source": "marketaux" | "demo",
            "is_live": bool,
            "last_updated": str (ISO timestamp),
            "symbol": str,
            "articles": [...],
            "ai_brief": {"text": str, "mode": "deepseek" | "rule_based"},
        }
    """
    try:
        refresh_minutes = int(os.environ.get("NEWS_REFRESH_MINUTES", "15") or 15)
    except ValueError:
        refresh_minutes = 15
    provider = os.environ.get("NEWS_API_PROVIDER", "marketaux").strip().lower()
    api_key = os.environ.get("MARKETAUX_API_KEY", "").strip()

    cache_symbol = (symbol or "GENERAL").upper()
    cache = _load_cache()
    if (not force_refresh and cache and cache.get("symbol") == cache_symbol
            and _cache_is_fresh(cache, refresh_minutes)):
        return cache["payload"]

    is_live = False
    source = "demo"
    articles: List[Dict] = []

    if provider == "marketaux" and api_key:
        try:
            fetched = _fetch_marketaux(symbol, api_key)
            if fetched:
                articles = fetched
                is_live = True
                source = "marketaux"
        except Exception as exc:
            # requests' HTTPError messages embed the full request URL (including
            # api_token=...) — redact the key before logging so it never leaks.
            safe_msg = str(exc).replace(api_key, "***REDACTED***")
            print(f"[news_service] Marketaux fetch failed, falling back to demo feed: {safe_msg}")

    if not articles:
        articles = _demo_articles()

    brief_text = _deepseek_brief(articles)
    if brief_text:
        ai_brief = {"text": brief_text, "mode": "deepseek"}
    else:
        ai_brief = {"text": _rule_based_brief(articles), "mode": "rule_based"}

    payload = {
        "source": source,
        "is_live": is_live,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "symbol": cache_symbol,
        "articles": articles,
        "ai_brief": ai_brief,
    }

    try:
        _ensure_data_dir()
        _atomic_write_json(_CACHE_FILE, {
            "symbol": cache_symbol,
            "fetched_at": datetime.now().isoformat(),
            "payload": payload,
        })
    except Exception:
        pass  # Caching is best-effort — never break the request over a disk write failure.

    return payload
