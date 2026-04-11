from typing import Dict, Any, Tuple

PRIMARY_VERDICTS = {
    "VERIFIED_FACT",
    "FALSE_FACT",
    "UNVERIFIED_CLAIM",
    "OPINION",
    "BIASED_CONTENT",
    "MANIPULATIVE_CONTENT",
    "SATIRE_OR_SARCASM",
    "CONSPIRACY_OR_EXTRAORDINARY_CLAIM",
    "LIKELY_AI_GENERATED",
    "MIXED_ANALYSIS",
}


def aggregate_primary_verdict(
    truth_score: float,
    ai_score: float,
    bias_score: float,
    manipulation_score: float,
    sarcasm_detected: bool,
    conspiracy_flag: bool,
    claim_verifiable: bool,
    opinion_detected: bool = False,
) -> Tuple[str, str]:
    """
    Deterministic verdict aggregation with strict precedence.
    Returns (verdict, triggered_rule).
    """
    if sarcasm_detected:
        return "SATIRE_OR_SARCASM", "RULE_1_SARCASM"
    if ai_score > 0.85 and truth_score > 0.50:
        return "LIKELY_AI_GENERATED", "RULE_2_AI_HIGH_TRUTH_MID"
    if truth_score < 0.30 and claim_verifiable:
        return "FALSE_FACT", "RULE_3_FALSE_VERIFIABLE"
    if not claim_verifiable:
        if opinion_detected:
            return "OPINION", "RULE_4A_OPINION_NON_VERIFIABLE"
        return "UNVERIFIED_CLAIM", "RULE_4_UNVERIFIED"
    if bias_score > 0.75 and truth_score > 0.50:
        return "BIASED_CONTENT", "RULE_5_BIAS_HIGH_TRUTH_MID"
    if manipulation_score > 0.75:
        return "MANIPULATIVE_CONTENT", "RULE_6_MANIPULATION_HIGH"
    if conspiracy_flag:
        return "CONSPIRACY_OR_EXTRAORDINARY_CLAIM", "RULE_7_CONSPIRACY_FLAG"
    if truth_score > 0.75:
        return "VERIFIED_FACT", "RULE_8_VERIFIED_FACT"
    return "MIXED_ANALYSIS", "RULE_9_MIXED"


def get_verdict_and_risk(
    truth_score: float,
    ai_score: float,
    bias_score: float,
    confidence: float,
    *,
    manipulation_score: float = 0.0,
    sarcasm_detected: bool = False,
    conspiracy_flag: bool = False,
    claim_verifiable: bool = True,
    opinion_detected: bool = False,
) -> Dict[str, Any]:
    """Returns both new deterministic verdict fields and legacy risk summary fields."""
    key_factors = []

    primary_verdict, triggered_rule = aggregate_primary_verdict(
        truth_score=truth_score,
        ai_score=ai_score,
        bias_score=bias_score,
        manipulation_score=manipulation_score,
        sarcasm_detected=sarcasm_detected,
        conspiracy_flag=conspiracy_flag,
        claim_verifiable=claim_verifiable,
        opinion_detected=opinion_detected,
    )

    verdict = primary_verdict.replace("_", " ").title()
    risk_level = "Medium"
    recommendation = "Cross-check with trusted sources."

    if primary_verdict in {"FALSE_FACT", "MANIPULATIVE_CONTENT", "CONSPIRACY_OR_EXTRAORDINARY_CLAIM"}:
        risk_level = "High"
        recommendation = "High-risk content. Avoid sharing without independent verification."
    elif primary_verdict in {"VERIFIED_FACT"}:
        risk_level = "Low"
        recommendation = "Content appears factually grounded."
    elif primary_verdict in {"UNVERIFIED_CLAIM", "OPINION", "SATIRE_OR_SARCASM"}:
        risk_level = "Medium"
        recommendation = "Interpret with context and verify before relying on it."

    if ai_score > 0.85:
        key_factors.append("Strong AI-generation indicators")
    if truth_score < 0.30 and claim_verifiable:
        key_factors.append("Low factual alignment on verifiable claim")
    if bias_score > 0.70:
        key_factors.append("High levels of manipulative or biased framing")
    if sarcasm_detected:
        key_factors.append("Sarcasm/satire marker triggered")
    if conspiracy_flag:
        key_factors.append("Conspiracy/extraordinary claim marker triggered")
    if manipulation_score > 0.75:
        key_factors.append("Manipulative language indicators exceeded threshold")

    if confidence < 0.5:
        recommendation = "Low confidence analysis. Manual review recommended."
        key_factors.append("Insufficient data for a high-confidence verdict")

    return {
        "primary_verdict": primary_verdict,
        "triggered_rule": triggered_rule,
        "verdict": verdict,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "key_factors": list(set(key_factors))
    }
