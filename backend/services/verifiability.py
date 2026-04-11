import re
from typing import Dict, Any


SUBJECTIVE_PATTERNS = [
    r"\bi think\b",
    r"\bi feel\b",
    r"\bin my opinion\b",
    r"\bseems\b",
    r"\blooks like\b",
    r"\bbest\b",
    r"\bworst\b",
]

MARKETING_PATTERNS = [
    r"\brevolutionary\b",
    r"\bbreakthrough\b",
    r"\bguaranteed\b",
    r"\bworld[- ]?class\b",
    r"\bnumber one\b",
]

RECENCY_LOCAL_PATTERNS = [
    r"\btoday\b",
    r"\byesterday\b",
    r"\bjust now\b",
    r"\blocal\b",
    r"\bmy city\b",
    r"\bmy town\b",
]


def assess_claim_verifiability(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    normalized = content.lower()
    if not content:
        return {
            "claim_verifiable": False,
            "opinion_detected": False,
            "reason": "empty_content",
        }

    if any(re.search(pattern, normalized) for pattern in SUBJECTIVE_PATTERNS):
        return {
            "claim_verifiable": False,
            "opinion_detected": True,
            "reason": "subjective_statement",
        }

    if any(re.search(pattern, normalized) for pattern in MARKETING_PATTERNS):
        return {
            "claim_verifiable": False,
            "opinion_detected": False,
            "reason": "marketing_or_promotional_claim",
        }

    if any(re.search(pattern, normalized) for pattern in RECENCY_LOCAL_PATTERNS):
        return {
            "claim_verifiable": False,
            "opinion_detected": False,
            "reason": "recent_or_local_hard_to_source_claim",
        }

    has_verifiable_signal = bool(re.search(r"\b\d{4}\b|\b\d+%|\baccording to\b|\bstudy\b|\breport\b", normalized))
    return {
        "claim_verifiable": has_verifiable_signal or len(content.split()) > 8,
        "opinion_detected": False,
        "reason": "likely_verifiable" if has_verifiable_signal or len(content.split()) > 8 else "insufficient_factual_signal",
    }
