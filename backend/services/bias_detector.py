from typing import Any, Dict, List


LOADED_ADJECTIVES = [
    "corrupt",
    "evil",
    "disgusting",
    "radical",
    "propaganda",
    "traitorous",
]

DEMONIZATION_PATTERNS = [
    "enemy of the people",
    "vermin",
    "subhuman",
    "destroy our country",
]

US_VS_THEM_PATTERNS = [
    "us vs them",
    "their agenda",
    "our side",
    "those people",
    "mainstream media",
]


def _find_hits(text: str, patterns: List[str]) -> List[str]:
    return [pattern for pattern in patterns if pattern in text]


def detect_bias(text: str, model_bias_score: float = 0.0) -> Dict[str, Any]:
    normalized = (text or "").lower()
    loaded_hits = _find_hits(normalized, LOADED_ADJECTIVES)
    demonization_hits = _find_hits(normalized, DEMONIZATION_PATTERNS)
    framing_hits = _find_hits(normalized, US_VS_THEM_PATTERNS)

    rule_score = min(
        1.0,
        (len(loaded_hits) * 0.35) + (len(demonization_hits) * 0.45) + (len(framing_hits) * 0.35),
    )
    score = max(rule_score, float(model_bias_score))

    return {
        "score": round(score, 4),
        "indicators": (loaded_hits + demonization_hits + framing_hits)[:10],
    }
