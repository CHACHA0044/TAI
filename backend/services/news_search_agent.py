import os
import logging
import requests
from typing import List, Dict

logger = logging.getLogger("truthguard.news_search")

# Lazy-load sentence-transformers to avoid heavy startup cost
_similarity_model = None


def _get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        from sentence_transformers import SentenceTransformer
        _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _similarity_model


class NewsSearchAgent:
    """
    Searches for news headlines that match a claim and returns a consistency
    score (0.0–1.0).  Higher = more consistent with verified reporting.

    Sources (in priority order):
      1. GNews API  — enabled when GNEWS_API_KEY env-var is set.
      2. Offline mock headlines — used as fallback / during development.
    """

    GNEWS_API_URL = "https://gnews.io/api/v4/search"

    MOCK_HEADLINES: List[str] = [
        "Official reports confirm no evidence of widespread election fraud.",
        "Scientists reveal new study on climate change impact in the Arctic.",
        "Economic indicators show steady growth in the tech sector.",
        "Government spokesperson denies rumors of secret government programs.",
        "Researchers publish peer-reviewed findings on vaccine safety.",
        "Health authorities update guidelines based on latest clinical data.",
        "International observers verify results of democratic elections.",
        "Central bank announces monetary policy decision after board meeting.",
    ]

    def __init__(self) -> None:
        self.api_key = os.getenv("GNEWS_API_KEY", "").strip()
        self.use_api = bool(self.api_key)
        mode = "GNews API" if self.use_api else "mock headlines"
        logger.info(f"NewsSearchAgent initialized — using {mode}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, claim: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Return a list of relevant news headlines for *claim*."""
        if self.use_api:
            return self._search_gnews(claim, max_results)
        return self._mock_results()

    def get_consistency_score(self, claim: str) -> float:
        """
        Compute a news-consistency score in [0.0, 1.0].

        The score reflects how semantically similar the best-matching headline
        is to the claim.  A low score (e.g. < 0.35) suggests the claim has no
        close counterpart in known reliable reporting.
        """
        try:
            headlines = [item["headline"] for item in self.search(claim) if item.get("headline")]
            if not headlines:
                return 0.3 # Lowered from 0.5 to trigger skepticism penalty

            from sentence_transformers import util
            model = _get_similarity_model()

            claim_emb = model.encode(claim, convert_to_tensor=True)
            head_embs = model.encode(headlines, convert_to_tensor=True)
            scores = util.cos_sim(claim_emb, head_embs)[0]

            best = float(scores.max())
            logger.info(f"NewsSearchAgent: best headline similarity = {best:.3f}")

            if best > 0.75:
                return 0.90
            if best > 0.60:
                return 0.72
            if best > 0.45:
                return 0.55
            if best > 0.30:
                return 0.38
            return 0.20

        except Exception as exc:
            logger.error(f"NewsSearchAgent.get_consistency_score failed: {exc}")
            return 0.3 # Lowered from 0.5

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_gnews(self, claim: str, max_results: int) -> List[Dict[str, str]]:
        """Call the GNews REST API."""
        try:
            params = {
                "q": claim[:120].strip(),
                "lang": "en",
                "max": max_results,
                "apikey": self.api_key,
            }
            resp = requests.get(self.GNEWS_API_URL, params=params, timeout=6)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [
                {"headline": a.get("title", ""), "url": a.get("url", "")}
                for a in articles
                if a.get("title")
            ]
        except Exception as exc:
            logger.warning(f"GNews request failed ({exc}); falling back to mock headlines")
            return self._mock_results()

    def _mock_results(self) -> List[Dict[str, str]]:
        return [{"headline": h, "url": ""} for h in self.MOCK_HEADLINES]
