import os
from typing import Dict, Any


class HuggingFaceSarcasmAdapter:
    """
    Placeholder adapter for future HuggingFace sarcasm integration.
    Set SARCASM_HF_MODEL to enable custom model wiring later.
    """

    def __init__(self):
        self.model_name = os.getenv("SARCASM_HF_MODEL")

    def detect(self, text: str) -> Dict[str, Any]:
        # Stub: no remote/model call yet.
        return {
            "sarcasm": False,
            "score": 0.0,
            "provider": "hf_stub",
            "indicators": [],
        }


class SarcasmDetector:
    MARKER_WEIGHT = 0.25
    PUNCTUATION_WEIGHT = 0.2
    SARCASM_THRESHOLD = 0.55

    def __init__(self):
        self.hf_adapter = HuggingFaceSarcasmAdapter()
        self.markers = [
            "yeah right",
            "sure, because",
            "obviously",
            "totally",
            "as if",
            "/s",
            "what could possibly go wrong",
            "great job",
        ]

    def detect(self, text: str) -> Dict[str, Any]:
        normalized = (text or "").lower()
        marker_hits = [m for m in self.markers if m in normalized]
        punctuation_cue = "!" in normalized and "?" in normalized
        heuristic_score = min(
            1.0,
            (len(marker_hits) * self.MARKER_WEIGHT)
            + (self.PUNCTUATION_WEIGHT if punctuation_cue else 0.0),
        )
        hf_signal = self.hf_adapter.detect(text)
        score = max(heuristic_score, float(hf_signal.get("score", 0.0)))
        sarcasm = score >= self.SARCASM_THRESHOLD or bool(hf_signal.get("sarcasm", False))

        indicators = list(marker_hits)
        if punctuation_cue:
            indicators.append("mixed_exclamatory_question_pattern")
        indicators.extend(hf_signal.get("indicators", []))

        return {
            "sarcasm": sarcasm,
            "score": round(score, 4),
            "provider": "heuristic+hf_stub",
            "indicators": indicators[:8],
        }
