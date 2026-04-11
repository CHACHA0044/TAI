import re
from typing import Dict, Any

from services.claim_type_detector import classify_claim_type


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
    # Keep this list focused on locality/recency signals that are hard to verify globally.
    # Previously included "today"/"yesterday", which overfired on ordinary time-bound text.
    r"\bjust now\b",
    r"\blocal\b",
    r"\bmy city\b",
    r"\bmy town\b",
]

SPECULATION_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bclaims? to\b",
    r"\bexpected to\b",
    r"\bpossible\b",
]

FACTUAL_STRUCTURE_PATTERNS = [
    r"\b(is|are|was|were|has|have)\b.+\b(in|at|on|from)\b",
    r"\bhas \d+\b",
]

# Lowered from 8 to 6 as a practical heuristic so short factual claims
# (e.g. "The Eiffel Tower is in Berlin") are testable instead of auto-unverified.
MIN_WORDS_FOR_VERIFIABILITY = 6
YEAR_PATTERN = r"\b\d{4}\b"
PERCENTAGE_PATTERN = r"\b\d+%"
SOURCE_CITATION_PATTERN = r"\baccording to\b"
STUDY_PATTERN = r"\bstudy\b"
REPORT_PATTERN = r"\breport\b"


def assess_claim_verifiability(text: str, claim_type: str = "") -> Dict[str, Any]:
    content = (text or "").strip()
    normalized = content.lower()
    inferred_claim_type = claim_type or classify_claim_type(content).get("claim_type", "MIXED")
    if not content:
        return {
            "claim_verifiable": False,
            "opinion_detected": False,
            "reason": "empty_content",
            "claim_type": inferred_claim_type,
        }

    if inferred_claim_type == "OPINION" or any(re.search(pattern, normalized) for pattern in SUBJECTIVE_PATTERNS):
        return {
            "claim_verifiable": False,
            "opinion_detected": True,
            "reason": "subjective_statement",
            "claim_type": inferred_claim_type,
        }

    if any(re.search(pattern, normalized) for pattern in MARKETING_PATTERNS):
        return {
            "claim_verifiable": False,
            "opinion_detected": False,
            "reason": "marketing_or_promotional_claim",
            "claim_type": inferred_claim_type,
        }

    if inferred_claim_type == "UNVERIFIABLE_SPECULATION" or any(re.search(pattern, normalized) for pattern in SPECULATION_PATTERNS):
        return {
            "claim_verifiable": False,
            "opinion_detected": False,
            "reason": "speculative_or_unconfirmed_claim",
            "claim_type": inferred_claim_type,
        }

    if any(re.search(pattern, normalized) for pattern in RECENCY_LOCAL_PATTERNS):
        return {
            "claim_verifiable": False,
            "opinion_detected": False,
            "reason": "recent_or_local_hard_to_source_claim",
            "claim_type": inferred_claim_type,
        }

    has_verifiable_signal = any(
        re.search(pattern, normalized)
        for pattern in [
            YEAR_PATTERN,
            PERCENTAGE_PATTERN,
            SOURCE_CITATION_PATTERN,
            STUDY_PATTERN,
            REPORT_PATTERN,
        ]
    )
    has_factual_structure = any(re.search(pattern, normalized) for pattern in FACTUAL_STRUCTURE_PATTERNS)
    claim_verifiable = has_verifiable_signal or has_factual_structure or len(content.split()) >= MIN_WORDS_FOR_VERIFIABILITY

    return {
        "claim_verifiable": claim_verifiable,
        "opinion_detected": False,
        "reason": "likely_verifiable"
        if claim_verifiable
        else "insufficient_factual_signal",
        "claim_type": inferred_claim_type,
    }
