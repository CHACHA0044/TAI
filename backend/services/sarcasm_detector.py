import os
import re
from typing import Any, Dict, List


class HuggingFaceSarcasmAdapter:
    """
    Optional adapter for local HuggingFace sarcasm integration.
    Stubbed unless a model is explicitly wired.
    """

    def __init__(self):
        self.model_name = os.getenv("SARCASM_HF_MODEL")

    def detect(self, text: str) -> Dict[str, Any]:
        return {
            "sarcasm": False,
            "score": 0.0,
            "provider": "hf_stub",
            "indicators": [],
        }


class SarcasmDetector:
    MARKER_WEIGHT = 0.22
    HYPERBOLE_WEIGHT = 0.18
    CONTRADICTION_WEIGHT = 0.22
    PUNCTUATION_WEIGHT = 0.12
    SARCASM_THRESHOLD = 0.52

    def __init__(self):
        self.hf_adapter = HuggingFaceSarcasmAdapter()
        self.markers = [
            "yeah right",
            "yeah, because",
            "sure, because",
            "obviously",
            "as if",
            "/s",
            "what could possibly go wrong",
            "great job",
            "totally",
        ]
        self.hyperbole = [
            "everyone knows",
            "literally impossible",
            "best thing ever",
            "worst thing ever",
            "absolutely perfect",
        ]

    def _regex_hits(self, text: str, patterns: List[str]) -> List[str]:
        return [pattern for pattern in patterns if re.search(pattern, text)]

    def detect(self, text: str) -> Dict[str, Any]:
        normalized = (text or "").lower()
        marker_hits = [marker for marker in self.markers if marker in normalized]
        hyperbole_hits = [phrase for phrase in self.hyperbole if phrase in normalized]

        contradiction_hits = self._regex_hits(
            normalized,
            [
                r"\byeah[, ]+because\b",
                r"\bobviously\b.+\b(flat|fake|magic|impossible)\b",
                r"\bof course\b.+\bnot\b",
            ],
        )

        punctuation_cue = normalized.count("!") > 0 and normalized.count("?") > 0

        heuristic_score = min(
            1.0,
            (len(marker_hits) * self.MARKER_WEIGHT)
            + (len(hyperbole_hits) * self.HYPERBOLE_WEIGHT)
            + (len(contradiction_hits) * self.CONTRADICTION_WEIGHT)
            + (self.PUNCTUATION_WEIGHT if punctuation_cue else 0.0),
        )

        hf_signal = self.hf_adapter.detect(text)
        score = max(heuristic_score, float(hf_signal.get("score", 0.0)))
        sarcasm = score >= self.SARCASM_THRESHOLD or bool(hf_signal.get("sarcasm", False))

        indicators = marker_hits + hyperbole_hits + contradiction_hits
        if punctuation_cue:
            indicators.append("mixed_exclamatory_question_pattern")
        indicators.extend(hf_signal.get("indicators", []))

        return {
            "sarcasm": sarcasm,
            "score": round(score, 4),
            "provider": "heuristic+hf_stub",
            "indicators": indicators[:10],
        }
