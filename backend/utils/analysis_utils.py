from typing import Any, Dict, Tuple

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


def to_bucket(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


def stage_route_primary_verdict(
    *,
    claim_type: str,
    claim_verifiable: bool,
    opinion_detected: bool,
    sarcasm_detected: bool,
    factual_claim: bool,
    trust_agent_confidence: str,
    retrieval_support_score: float,
    retrieval_contradiction_score: float,
    bias_score: float,
    manipulation_score: float,
    conspiracy_flag: bool,
    ai_generated_score: float,
) -> Dict[str, Any]:
    claim_type = claim_type or "MIXED"
    text_type_map = {
        "FACTUAL_CLAIM": "factual_claim",
        "OPINION": "opinion",
        "PERSUASIVE_COPY": "persuasive/manipulative",
        "SARCASTIC": "sarcastic/satirical",
        "SPECULATIVE": "speculative/unverifiable",
        "MIXED": "mixed",
    }
    text_type_detected = text_type_map.get(claim_type, "mixed")

    if sarcasm_detected or claim_type == "SARCASTIC":
        return {
            "primary_verdict": "SATIRE_OR_SARCASM",
            "triggered_rule": "STAGE_2_SARCASM_GATE",
            "verifiability_result": "non_factual_sarcasm",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": False,
            "why_verdict_chosen": "Sarcasm cues triggered before factual evaluation.",
        }

    if opinion_detected or claim_type == "OPINION":
        return {
            "primary_verdict": "OPINION",
            "triggered_rule": "STAGE_2_OPINION_GATE",
            "verifiability_result": "non_factual_opinion",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": False,
            "why_verdict_chosen": "Subjective/preference language gated this away from factual truth scoring.",
        }

    if claim_type == "SPECULATIVE" or not claim_verifiable:
        return {
            "primary_verdict": "UNVERIFIED_CLAIM",
            "triggered_rule": "STAGE_2_VERIFIABILITY_GATE",
            "verifiability_result": "non_verifiable_claim",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": False,
            "why_verdict_chosen": "Speculative or unverifiable claim cannot be resolved factually.",
        }

    if factual_claim:
        if trust_agent_confidence == "support" and retrieval_support_score >= 0.72 and retrieval_contradiction_score < 0.60:
            return {
                "primary_verdict": "VERIFIED_FACT",
                "triggered_rule": "STAGE_3_FACTUAL_SUPPORT",
                "verifiability_result": "verifiable_factual_claim",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": True,
                "why_verdict_chosen": "Strong support retrieval confidence on a verifiable factual claim.",
            }
        if trust_agent_confidence == "contradiction" and retrieval_contradiction_score >= 0.72:
            return {
                "primary_verdict": "FALSE_FACT",
                "triggered_rule": "STAGE_3_FACTUAL_CONTRADICTION",
                "verifiability_result": "verifiable_factual_claim",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": True,
                "why_verdict_chosen": "Strong contradictory evidence on a verifiable factual claim.",
            }
        return {
            "primary_verdict": "UNVERIFIED_CLAIM",
            "triggered_rule": "STAGE_3_FACTUAL_INCONCLUSIVE",
            "verifiability_result": "verifiable_but_inconclusive",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": True,
            "why_verdict_chosen": "Factual claim did not reach support/contradiction thresholds.",
        }

    if bias_score >= 0.70:
        return {
            "primary_verdict": "BIASED_CONTENT",
            "triggered_rule": "STAGE_4_BIAS_OVERRIDE",
            "verifiability_result": "non_factual_mixed",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": False,
            "why_verdict_chosen": "High slant/bias signal on non-factual text.",
        }
    if manipulation_score >= 0.70:
        return {
            "primary_verdict": "MANIPULATIVE_CONTENT",
            "triggered_rule": "STAGE_4_MANIPULATION_OVERRIDE",
            "verifiability_result": "non_factual_mixed",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": False,
            "why_verdict_chosen": "High coercive/manipulative signal on non-factual text.",
        }
    if conspiracy_flag:
        return {
            "primary_verdict": "CONSPIRACY_OR_EXTRAORDINARY_CLAIM",
            "triggered_rule": "STAGE_4_CONSPIRACY_OVERRIDE",
            "verifiability_result": "non_factual_mixed",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": False,
            "why_verdict_chosen": "Extraordinary/conspiracy markers dominated non-factual analysis.",
        }
    if ai_generated_score >= 0.95:
        return {
            "primary_verdict": "LIKELY_AI_GENERATED",
            "triggered_rule": "STAGE_4_AI_AUXILIARY",
            "verifiability_result": "non_factual_mixed",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": False,
            "why_verdict_chosen": "Very high AI-likelihood used only as secondary non-factual signal.",
        }
    return {
        "primary_verdict": "MIXED_ANALYSIS",
        "triggered_rule": "STAGE_4_MIXED_FALLBACK",
        "verifiability_result": "non_factual_mixed",
        "text_type_detected": text_type_detected,
        "factual_verdict_locked": False,
        "why_verdict_chosen": "No dominant non-factual override crossed threshold.",
    }


def aggregate_primary_verdict(
    *,
    factual_claim: bool,
    opinion_detected: bool,
    claim_is_unverifiable: bool,
    evidence_support: float,
    evidence_contradiction: float,
    bias_score: float,
    manipulation_score: float,
    sarcasm_detected: bool,
    extraordinary_claim_flag: bool,
    ai_score: float,
    claim_type: str = "MIXED",
) -> Tuple[str, str, str]:
    """
    Deterministic verdict mapping with strict priority.
    Returns (verdict, triggered_rule, why_verdict_chosen).
    """
    if sarcasm_detected:
        return "SATIRE_OR_SARCASM", "RULE_1_SARCASM", "Sarcasm/satire detector fired before factual scoring."
    if opinion_detected:
        return "OPINION", "RULE_2_OPINION", "Subjective or preference language dominates this text."
    if claim_is_unverifiable:
        return "UNVERIFIED_CLAIM", "RULE_3_UNVERIFIABLE", "Statement is currently unverifiable/speculative."
    if factual_claim and evidence_support >= 0.70:
        return "VERIFIED_FACT", "RULE_4_VERIFIED", "Factual claim supported by high evidence alignment."
    if factual_claim and evidence_contradiction >= 0.70:
        return "FALSE_FACT", "RULE_5_FALSE_FACT", "Factual claim contradicted by high-confidence evidence."
    if bias_score >= 0.70:
        return "BIASED_CONTENT", "RULE_6_BIAS", "Loaded or slanted framing is the dominant signal."
    if manipulation_score >= 0.70:
        return "MANIPULATIVE_CONTENT", "RULE_7_MANIPULATION", "Emotional pressure/coercive language is dominant."
    if extraordinary_claim_flag:
        return "CONSPIRACY_OR_EXTRAORDINARY_CLAIM", "RULE_8_EXTRAORDINARY", "Extraordinary claim marker triggered."
    if ai_score >= 0.90 and not factual_claim:
        return "LIKELY_AI_GENERATED", "RULE_9_AI_AUXILIARY", "Very high AI-likelihood on non-factual content."
    return "MIXED_ANALYSIS", "RULE_10_MIXED", "No single dominant rule exceeded decision thresholds."


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
    claim_type: str = "MIXED",
    factual_claim: bool = False,
) -> Dict[str, Any]:
    """Return deterministic verdict fields, compatibility risk summary, and debug info."""
    evidence_support = max(0.0, min(1.0, float(truth_score)))
    evidence_contradiction = max(0.0, min(1.0, 1.0 - float(truth_score)))
    claim_is_unverifiable = not claim_verifiable

    primary_verdict, triggered_rule, why_verdict = aggregate_primary_verdict(
        factual_claim=factual_claim,
        opinion_detected=opinion_detected,
        claim_is_unverifiable=claim_is_unverifiable,
        evidence_support=evidence_support,
        evidence_contradiction=evidence_contradiction,
        bias_score=bias_score,
        manipulation_score=manipulation_score,
        sarcasm_detected=sarcasm_detected,
        extraordinary_claim_flag=conspiracy_flag,
        ai_score=ai_score,
        claim_type=claim_type,
    )

    detector_fired_first = "none"
    for detector, fired in [
        ("sarcasm_detector", sarcasm_detected),
        ("opinion_detector", opinion_detected),
        ("verifiability_gate", claim_is_unverifiable),
        ("truth_support_gate", factual_claim and evidence_support >= 0.70),
        ("truth_contradiction_gate", factual_claim and evidence_contradiction >= 0.70),
        ("bias_detector", bias_score >= 0.70),
        ("manipulation_detector", manipulation_score >= 0.70),
        ("extraordinary_claim_detector", conspiracy_flag),
        ("ai_likelihood_detector", ai_score >= 0.90),
    ]:
        if fired:
            detector_fired_first = detector
            break

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

    key_factors = []
    if ai_score > 0.85:
        key_factors.append("Strong AI-generation indicators")
    if factual_claim and evidence_contradiction > 0.70:
        key_factors.append("High contradiction on a factual claim")
    if bias_score > 0.70:
        key_factors.append("High loaded or slanted framing")
    if manipulation_score > 0.70:
        key_factors.append("Manipulative pressure signals exceeded threshold")
    if sarcasm_detected:
        key_factors.append("Sarcasm/satire marker triggered")
    if claim_is_unverifiable:
        key_factors.append("Verifiability gate blocked factual verdict")
    if conspiracy_flag:
        key_factors.append("Conspiracy/extraordinary claim marker triggered")
    if confidence < 0.5:
        recommendation = "Low confidence analysis. Manual review recommended."
        key_factors.append("Insufficient data for a high-confidence verdict")

    debug = {
        "detector_fired_first": detector_fired_first,
        "why_verdict_chosen": why_verdict,
        "final_rule_triggered": triggered_rule,
        "raw_intermediate_scores": {
            "truth_score": round(float(truth_score), 4),
            "evidence_support": round(evidence_support, 4),
            "evidence_contradiction": round(evidence_contradiction, 4),
            "verifiability": "HIGH" if claim_verifiable else "LOW",
            "ai_likelihood_score": round(float(ai_score), 4),
            "ai_likelihood_bucket": to_bucket(float(ai_score)),
            "bias_score": round(float(bias_score), 4),
            "bias_bucket": to_bucket(float(bias_score)),
            "manipulation_score": round(float(manipulation_score), 4),
            "manipulation_bucket": to_bucket(float(manipulation_score)),
            "sarcasm_detected": bool(sarcasm_detected),
            "opinion_detected": bool(opinion_detected),
            "conspiracy_flag": bool(conspiracy_flag),
            "factual_claim": bool(factual_claim),
            "claim_type": claim_type,
            "claim_verifiable": bool(claim_verifiable),
        },
    }

    return {
        "primary_verdict": primary_verdict,
        "triggered_rule": triggered_rule,
        "verdict": verdict,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "key_factors": sorted(set(key_factors)),
        "debug": debug,
    }
