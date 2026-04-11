import re
from typing import Any, Dict, List


CLAIM_TYPES = {
    "FACTUAL_CLAIM",
    "OPINION",
    "PERSUASIVE_COPY",
    "SARCASTIC",
    "UNVERIFIABLE_SPECULATION",
    "MIXED",
}


SARCASM_PATTERNS = [
    r"\byeah[, ]+because\b",
    r"\byeah right\b",
    r"\bsure[, ]+because\b",
    r"\bobviously\b",
    r"\bas if\b",
    r"/s\b",
]

OPINION_PATTERNS = [
    r"\bbetter than\b",
    r"\bbetter\b.*\bthan\b",
    r"\bworse than\b",
    r"\bin my opinion\b",
    r"\bi think\b",
    r"\bi feel\b",
    r"\bi prefer\b",
    r"\bmeaningless\b",
    r"\bbest\b",
    r"\bworst\b",
    r"\bshould\b",
]

PERSUASIVE_PATTERNS = [
    r"\bif you care\b",
    r"\byou must\b",
    r"\bmust buy\b",
    r"\bact now\b",
    r"\blimited time\b",
    r"\bdon't miss\b",
    r"\bbuy this today\b",
]

SPECULATIVE_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\blikely\b",
    r"\bclaims? to\b",
    r"\breportedly\b",
    r"\bexpected to\b",
    r"\bpossible\b",
]

FACTUAL_ANCHOR_PATTERNS = [
    r"\b\d{1,4}\b",
    r"\baccording to\b",
    r"\bstudy\b",
    r"\breport\b",
    r"\bdata\b",
    r"\b(is|are|was|were|has|have)\b.+\b(in|at|on|from)\b",
]


def _matches(patterns: List[str], text: str) -> List[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


def classify_claim_type(text: str) -> Dict[str, Any]:
    normalized = (text or "").strip().lower()
    if not normalized:
        return {
            "claim_type": "MIXED",
            "signals": [],
            "reason": "empty_content",
        }

    sarcasm_hits = _matches(SARCASM_PATTERNS, normalized)
    if sarcasm_hits:
        return {
            "claim_type": "SARCASTIC",
            "signals": sarcasm_hits,
            "reason": "sarcasm_markers_detected",
        }

    opinion_hits = _matches(OPINION_PATTERNS, normalized)
    if opinion_hits:
        return {
            "claim_type": "OPINION",
            "signals": opinion_hits,
            "reason": "subjective_or_preference_language",
        }

    persuasive_hits = _matches(PERSUASIVE_PATTERNS, normalized)
    if persuasive_hits:
        return {
            "claim_type": "PERSUASIVE_COPY",
            "signals": persuasive_hits,
            "reason": "persuasive_or_sales_pressure_markers",
        }

    speculative_hits = _matches(SPECULATIVE_PATTERNS, normalized)
    if speculative_hits:
        return {
            "claim_type": "UNVERIFIABLE_SPECULATION",
            "signals": speculative_hits,
            "reason": "speculative_or_unconfirmed_language",
        }

    factual_hits = _matches(FACTUAL_ANCHOR_PATTERNS, normalized)
    if factual_hits:
        return {
            "claim_type": "FACTUAL_CLAIM",
            "signals": factual_hits,
            "reason": "factual_anchors_present",
        }

    return {
        "claim_type": "MIXED",
        "signals": [],
        "reason": "no_dominant_claim_type_detected",
    }
