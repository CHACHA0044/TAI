"""
NewsAPI integration service — quota-safe, cached, smart-triggered.

Key design decisions:
  - Reads NEWS_API from environment (HuggingFace Spaces secret / .env).
  - File-based cache with configurable TTL (default 12 h) so quota is shared
    across workers and survives restarts even when Redis is unavailable.
  - Separate file-based daily quota tracker (resets at UTC midnight).
  - Smart news-relevance gate: call NewsAPI *only* when the text is likely a
    news claim; skip for opinions, personal text, products, generic queries.
  - Query optimizer: stopword removal, entity-first keyword selection.
  - Semantic similarity matching via sentence-transformers (lazy-loaded; falls
    back to substring overlap when the model is unavailable).
  - Graceful degradation: returns None / empty results on quota exhaustion,
    network errors or missing API key — never blocks the main pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger("truthguard.news_api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_TIMEOUT = 6          # seconds per request
NEWSAPI_MAX_RESULTS = 5      # articles to fetch per query
CACHE_TTL_SECONDS = 43_200   # 12 hours
DAILY_QUOTA = 95             # leave a 5-request safety buffer from the 100-limit
QUOTA_EXCEEDED_MSG = "External news verification unavailable (quota exhausted)."
MAX_QUERY_KEYWORDS = 6       # maximum terms in an optimised search query
MAX_ENTITY_KEYWORDS = 3      # highest-priority entity slots in query
# Corroboration score adjustment constants:
#   Corroboration is centred at 0.50 (neutral baseline — neither supports nor contradicts).
#   A fully corroborating result shifts the midpoint toward 1.0 and vice-versa.
#   The multiplier (0.12) limits the influence to ±0.06 per analysis pass to
#   prevent a single news check from dominating the model-based truth score.
NEWS_CORROBORATION_MIDPOINT = 0.50
NEWS_CORROBORATION_MULTIPLIER = 0.12

# File paths (relative to this file → under backend/)
_BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = _BASE_DIR / ".news_cache"
QUOTA_FILE = _BASE_DIR / ".news_quota.json"

# ---------------------------------------------------------------------------
# Stopwords for query optimisation
# ---------------------------------------------------------------------------
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "them", "us", "my", "our",
    "your", "their", "his", "who", "which", "what", "when", "where", "how",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "just",
    "also", "too", "very", "quite", "about", "up", "out", "there", "then",
    "than", "more", "most", "such", "each", "all", "any", "some", "said",
    "says", "like", "into", "over", "after", "before", "between", "during",
    "according", "new", "one", "two", "first", "second", "last", "own",
})

# ---------------------------------------------------------------------------
# News-relevance patterns — only fire NewsAPI for news-like content
# ---------------------------------------------------------------------------
NEWS_TRIGGER_PATTERNS: List[re.Pattern] = [p for p in [
    re.compile(r"\b(breaking|breaking news|just in|report[s]?)\b", re.I),
    re.compile(r"\b(president|prime minister|government|congress|senate|parliament|minister|official|spokesperson)\b", re.I),
    re.compile(r"\b(killed|arrested|charged|convicted|elected|resigned|fired|banned|sanctioned)\b", re.I),
    re.compile(r"\b(protest[s]?|demonstrat|riot|strike|coup|conflict|war|attack|bombing|shooting)\b", re.I),
    re.compile(r"\b(earthquake|flood|hurricane|wildfire|disaster|explosion|crash|collapse)\b", re.I),
    re.compile(r"\b(climate|inflation|economy|gdp|unemployment|recession|interest rate|federal reserve)\b", re.I),
    re.compile(r"\b(covid|pandemic|outbreak|virus|vaccine|epidemic|health crisis)\b", re.I),
    re.compile(r"\b(election|vote|ballot|referendum|poll|campaign)\b", re.I),
    re.compile(r"\b(court|trial|verdict|lawsuit|indictment|plea|sentence)\b", re.I),
    re.compile(r"\b(hacked|breach|leak|cyber|ransomware)\b", re.I),
    re.compile(r"\b(trump|biden|putin|xi|zelenskyy|modi|macron|sunak|zelensky)\b", re.I),
    re.compile(r"\b(ukraine|russia|israel|gaza|china|taiwan|nato|un|imf|who|wto)\b", re.I),
    re.compile(r"\b(billion|trillion|million)\s+dollar", re.I),
    re.compile(r"\b(according to|sources say|reports say|officials confirm|confirmed by)\b", re.I),
    re.compile(r"\b\d{4}\b.*\b(died|killed|arrested|elected|resigned|announced|confirmed)\b", re.I),
]]

# Patterns that indicate the text is *not* news-relevant
NOT_NEWS_PATTERNS: List[re.Pattern] = [p for p in [
    re.compile(r"\b(recipe|ingredient[s]?|cook|bake|meal|dinner|lunch|breakfast)\b", re.I),
    re.compile(r"\b(i think|i feel|in my opinion|personally|my favorite|i love|i hate)\b", re.I),
    re.compile(r"\b(buy now|sale|discount|promo|coupon|free shipping|order today)\b", re.I),
    re.compile(r"\b(movie review|book review|album|playlist|song|artist|band)\b", re.I),
    re.compile(r"\b(how to|tutorial|step by step|diy|tips and tricks)\b", re.I),
]]


# ---------------------------------------------------------------------------
# Internal: cache helpers
# ---------------------------------------------------------------------------

def _cache_key(query: str) -> str:
    return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:24]


def _read_cache(key: str) -> Optional[Dict]:
    try:
        path = CACHE_DIR / f"{key}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text("utf-8"))
        if time.time() - data.get("ts", 0) > CACHE_TTL_SECONDS:
            path.unlink(missing_ok=True)
            return None
        return data
    except Exception:
        return None


def _write_cache(key: str, payload: Dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{key}.json"
        path.write_text(json.dumps({**payload, "ts": time.time()}), "utf-8")
    except Exception as exc:
        logger.warning(f"NewsAPI cache write failed: {exc}")


# ---------------------------------------------------------------------------
# Internal: quota tracker
# ---------------------------------------------------------------------------

def _load_quota() -> Dict:
    try:
        if QUOTA_FILE.exists():
            data = json.loads(QUOTA_FILE.read_text("utf-8"))
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if data.get("date") == today:
                return data
    except Exception:
        pass
    return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "used": 0}


def _save_quota(data: Dict) -> None:
    try:
        QUOTA_FILE.write_text(json.dumps(data), "utf-8")
    except Exception as exc:
        logger.warning(f"NewsAPI quota save failed: {exc}")


def _quota_remaining() -> int:
    q = _load_quota()
    return max(0, DAILY_QUOTA - q.get("used", 0))


def _increment_quota() -> None:
    q = _load_quota()
    q["used"] = q.get("used", 0) + 1
    _save_quota(q)


# ---------------------------------------------------------------------------
# Query optimiser
# ---------------------------------------------------------------------------

def _extract_named_entities(text: str) -> List[str]:
    """Very lightweight NER: consecutive capitalised tokens (no model needed)."""
    tokens = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    seen: Dict[str, int] = {}
    for t in tokens:
        seen[t] = seen.get(t, 0) + 1
    # Return multi-word or repeated single-word entities first
    return [t for t, c in sorted(seen.items(), key=lambda x: -x[1]) if len(t.split()) > 1 or c > 1][:4]


def _build_optimised_query(text: str) -> str:
    """Produce a compact ≤6-keyword query string from input text."""
    # Extract named entities first (highest priority)
    entities = _extract_named_entities(text)

    # Then grab top non-stopword content words from the first 300 chars
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text[:300])
    content_words = [w for w in words if w.lower() not in STOPWORDS and not any(e.lower().startswith(w.lower()) for e in entities)]
    # Deduplicate preserving order
    seen: Set[str] = set()
    unique_content: List[str] = []
    for w in content_words:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            unique_content.append(w)

    # Combine: entities first, then up to remaining content words
    parts = entities[:MAX_ENTITY_KEYWORDS] + unique_content[:max(1, MAX_QUERY_KEYWORDS - len(entities))]
    query = " ".join(parts[:MAX_QUERY_KEYWORDS])
    logger.debug(f"NewsAPI optimised query: {query!r}")
    return query


# ---------------------------------------------------------------------------
# Similarity matching (lazy sentence-transformers, fallback to overlap)
# ---------------------------------------------------------------------------

_sim_model = None


def _get_sim_model():
    global _sim_model
    if _sim_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _sim_model = False  # sentinel: unavailable
    return _sim_model


def _similarity(text: str, headline: str) -> float:
    """Return cosine similarity or simple overlap fallback."""
    model = _get_sim_model()
    if model:
        try:
            from sentence_transformers import util
            embs = model.encode([text, headline], convert_to_tensor=True)
            return float(util.cos_sim(embs[0], embs[1]))
        except Exception:
            pass
    # Fallback: word overlap
    a = set(re.findall(r"\b\w{3,}\b", text.lower()))
    b = set(re.findall(r"\b\w{3,}\b", headline.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a | b) + 1e-6)


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class NewsAPIService:
    """
    Quota-safe NewsAPI.org client with caching, smart triggering, and
    semantic similarity matching.

    Public interface:
      is_news_relevant(text)           → bool
      verify(text, max_results=5)      → NewsVerificationResult | None
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("NEWS_API", "").strip()
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.info("NewsAPIService: NEWS_API env var not set — disabled.")
        else:
            logger.info("NewsAPIService: initialised (quota remaining today: %d)", _quota_remaining())

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def is_news_relevant(self, text: str) -> bool:
        """Return True if the text is likely a news-type claim worth querying."""
        if not text or len(text.split()) < 5:
            return False
        # Hard-exclude non-news patterns first
        for pat in NOT_NEWS_PATTERNS:
            if pat.search(text):
                return False
        # Require at least one news trigger
        for pat in NEWS_TRIGGER_PATTERNS:
            if pat.search(text):
                return True
        return False

    def verify(self, text: str, max_results: int = NEWSAPI_MAX_RESULTS) -> Optional[Dict[str, Any]]:
        """
        Run full news verification for *text*.
        Returns a dict suitable for the `news_verification` response field,
        or None if the service is disabled / quota exhausted / text not news-like.
        """
        if not self.enabled:
            return None
        if not self.is_news_relevant(text):
            logger.debug("NewsAPIService: text not news-relevant — skipped.")
            return None
        if _quota_remaining() <= 0:
            logger.warning("NewsAPIService: daily quota exhausted.")
            return {
                "available": False,
                "message": QUOTA_EXCEEDED_MSG,
                "corroboration_score": None,
                "source_count": 0,
                "articles": [],
                "verification_label": "Unavailable",
                "contradiction_detected": False,
            }

        query = _build_optimised_query(text)
        cache_key = _cache_key(query)
        cached = _read_cache(cache_key)

        if cached:
            logger.info("NewsAPIService: cache hit for query %r", query)
            articles = cached.get("articles", [])
        else:
            articles = self._fetch(query, max_results)
            if articles is not None:
                _write_cache(cache_key, {"articles": articles})
            else:
                articles = []

        return self._build_result(text, articles)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fetch(self, query: str, max_results: int) -> Optional[List[Dict]]:
        """Call NewsAPI.org /v2/everything. Returns list of articles or None on error."""
        for attempt in range(2):
            try:
                resp = requests.get(
                    NEWSAPI_BASE_URL,
                    params={
                        "q": query[:100],
                        "language": "en",
                        "sortBy": "relevancy",
                        "pageSize": max_results,
                        "apiKey": self.api_key,
                    },
                    timeout=NEWSAPI_TIMEOUT,
                )
                if resp.status_code == 426:
                    logger.warning("NewsAPIService: 426 rate-limited — quota may be exhausted.")
                    return None
                resp.raise_for_status()
                _increment_quota()
                raw = resp.json().get("articles", [])
                return [
                    {
                        "title": a.get("title", "").strip(),
                        "source": a.get("source", {}).get("name", "Unknown"),
                        "url": a.get("url", ""),
                        "published_at": a.get("publishedAt", ""),
                        "description": (a.get("description") or "").strip()[:200],
                    }
                    for a in raw
                    if a.get("title") and "[Removed]" not in a.get("title", "")
                ]
            except requests.Timeout:
                logger.warning("NewsAPIService: request timed out (attempt %d)", attempt + 1)
            except Exception as exc:
                logger.warning("NewsAPIService: fetch error: %s", exc)
                break
        return None

    def _build_result(self, text: str, articles: List[Dict]) -> Dict[str, Any]:
        """Compute similarity scores and produce the verification result dict."""
        if not articles:
            return {
                "available": True,
                "message": "No corroborating reporting found.",
                "corroboration_score": 0.20,
                "source_count": 0,
                "articles": [],
                "verification_label": "No corroboration found",
                "contradiction_detected": False,
            }

        scores: List[Tuple[float, Dict]] = []
        for article in articles:
            headline = article.get("title", "")
            description = article.get("description", "")
            combined = f"{headline}. {description}" if description else headline
            sim = _similarity(text[:500], combined)
            scores.append((sim, article))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score = scores[0][0]
        top_articles = [a for _, a in scores[:5]]

        # Map similarity → corroboration score
        if best_score >= 0.75:
            corroboration = 0.90
            label = "Verified by multiple independent sources"
            n_verified = sum(1 for s, _ in scores if s >= 0.50)
            message = f"Verified by {n_verified} independent source{'s' if n_verified != 1 else ''}."
        elif best_score >= 0.55:
            corroboration = 0.68
            label = "Claim partially matches reported news"
            message = "Claim partially matches reported news."
        elif best_score >= 0.38:
            corroboration = 0.45
            label = "Weak corroboration — limited matching reporting"
            message = "Weak corroboration — limited matching reporting found."
        else:
            corroboration = 0.18
            label = "No corroborating reporting found"
            message = "No corroborating reporting found."

        # Naïve contradiction heuristic: when the best semantic similarity is very
        # low (< 0.30) but NewsAPI returned several articles on the same topic, it
        # suggests the query matched a news cluster that diverges from the claim —
        # a weak signal that major reporting contradicts (or simply does not match)
        # the user's statement. This is a heuristic, not a definitive fact-check.
        contradiction_detected = best_score < 0.30 and len(articles) >= 3

        if contradiction_detected:
            label = "Claim contradicts major reporting"
            message = "Claim contradicts or is absent from major reporting on this topic."
            corroboration = max(0.10, corroboration - 0.15)

        return {
            "available": True,
            "message": message,
            "corroboration_score": round(corroboration, 4),
            "source_count": len(articles),
            "articles": top_articles,
            "verification_label": label,
            "contradiction_detected": contradiction_detected,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: Optional[NewsAPIService] = None


def get_news_api_service() -> NewsAPIService:
    global _service
    if _service is None:
        _service = NewsAPIService()
    return _service
