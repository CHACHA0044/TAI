from typing import Any, Dict, List


EMOTIONAL_PRESSURE_PATTERNS = [
    "if you care",
    "your family",
    "wake up",
    "before it's too late",
]

URGENCY_OR_COERCION_PATTERNS = [
    "must buy",
    "act now",
    "limited time",
    "buy this today",
    "you must",
]

FEAR_AND_MORALIZING_PATTERNS = [
    "or else",
    "only a fool would",
    "you have to",
    "do the right thing",
]


def _find_hits(text: str, patterns: List[str]) -> List[str]:
    return [pattern for pattern in patterns if pattern in text]


def detect_manipulation(text: str, style_metrics: Dict[str, float]) -> Dict[str, Any]:
    normalized = (text or "").lower()
    emotional_hits = _find_hits(normalized, EMOTIONAL_PRESSURE_PATTERNS)
    urgency_hits = _find_hits(normalized, URGENCY_OR_COERCION_PATTERNS)
    fear_hits = _find_hits(normalized, FEAR_AND_MORALIZING_PATTERNS)
    exclamation_component = 0.08 if "!" in normalized else 0.0

    # Urgency/coercion gets the highest weight because it is the strongest manipulation marker.
    rule_score = min(
        1.0,
        (len(emotional_hits) * 0.20)
        + (len(urgency_hits) * 0.24)
        + (len(fear_hits) * 0.16)
        + exclamation_component
        + (0.12 * float(style_metrics.get("lexical_repetition", 0.0))),
    )

    return {
        "score": round(rule_score, 4),
        "indicators": (emotional_hits + urgency_hits + fear_hits)[:10],
    }
