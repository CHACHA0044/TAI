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

# ---------------------------------------------------------------------------
# Calibration constants — tune here, not scattered through routing logic
# ---------------------------------------------------------------------------
# Support threshold: lowered from 0.72 → 0.60 to reduce false UNVERIFIED on
# well-known facts that get moderate (not perfect) retrieval confidence.
VERIFIED_SUPPORT_THRESHOLD = 0.60
# Contradiction threshold kept strict so FALSE_FACT requires strong evidence.
STRICT_CONTRADICTION_THRESHOLD = 0.72
# Bias / manipulation verdict thresholds
BIAS_VERDICT_THRESHOLD = 0.65
MANIPULATION_VERDICT_THRESHOLD = 0.65
# AI-likelihood threshold — ONLY route LIKELY_AI_GENERATED above this level
AI_VERDICT_THRESHOLD = 0.95


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
    # Optional extra signals for mixed-claim detection
    sarcasm_score: float = 0.0,
    opinion_score: float = 0.0,
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

    # Calibration margins — exposed in debug output
    support_margin = round(retrieval_support_score - VERIFIED_SUPPORT_THRESHOLD, 4)
    contradiction_margin = round(retrieval_contradiction_score - STRICT_CONTRADICTION_THRESHOLD, 4)

    threshold_values = {
        "verified_support_threshold": VERIFIED_SUPPORT_THRESHOLD,
        "strict_contradiction_threshold": STRICT_CONTRADICTION_THRESHOLD,
        "bias_verdict_threshold": BIAS_VERDICT_THRESHOLD,
        "manipulation_verdict_threshold": MANIPULATION_VERDICT_THRESHOLD,
        "ai_verdict_threshold": AI_VERDICT_THRESHOLD,
    }

    def _base(verdict: str, rule: str, verifiability: str, locked: bool, why: str) -> Dict[str, Any]:
        return {
            "primary_verdict": verdict,
            "triggered_rule": rule,
            "verifiability_result": verifiability,
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": locked,
            "why_verdict_chosen": why,
            "threshold_values_used": threshold_values,
            "trust_support_margin": support_margin,
            "contradiction_margin": contradiction_margin,
        }

    # Stage 1 — Sarcasm gate (highest priority)
    if sarcasm_detected or claim_type == "SARCASTIC":
        return _base(
            "SATIRE_OR_SARCASM",
            "STAGE_2_SARCASM_GATE",
            "non_factual_sarcasm",
            False,
            "Sarcasm cues triggered before factual evaluation.",
        )

    # Stage 2 — Opinion gate
    if opinion_detected or claim_type == "OPINION":
        return _base(
            "OPINION",
            "STAGE_2_OPINION_GATE",
            "non_factual_opinion",
            False,
            "Subjective/preference language gated this away from factual truth scoring.",
        )

    # Stage 3 — Verifiability gate (speculative / unverifiable content)
    if claim_type == "SPECULATIVE" or not claim_verifiable:
        return _base(
            "UNVERIFIED_CLAIM",
            "STAGE_2_VERIFIABILITY_GATE",
            "non_verifiable_claim",
            False,
            "Speculative or unverifiable claim cannot be resolved factually.",
        )

    # Stage 4 — Factual evaluation
    if factual_claim:
        # Verified: support is good AND contradiction is comfortably below the strict threshold
        if (
            trust_agent_confidence == "support"
            and retrieval_support_score >= VERIFIED_SUPPORT_THRESHOLD
            and retrieval_contradiction_score < STRICT_CONTRADICTION_THRESHOLD
        ):
            return _base(
                "VERIFIED_FACT",
                "STAGE_3_FACTUAL_SUPPORT",
                "verifiable_factual_claim",
                True,
                f"Support score {retrieval_support_score:.2f} ≥ threshold {VERIFIED_SUPPORT_THRESHOLD} with no strong contradiction.",
            )

        # False: requires STRONG contradictory evidence — keep gate strict
        if (
            trust_agent_confidence == "contradiction"
            and retrieval_contradiction_score >= STRICT_CONTRADICTION_THRESHOLD
        ):
            return _base(
                "FALSE_FACT",
                "STAGE_3_FACTUAL_CONTRADICTION",
                "verifiable_factual_claim",
                True,
                f"Contradiction score {retrieval_contradiction_score:.2f} ≥ strict threshold {STRICT_CONTRADICTION_THRESHOLD}.",
            )

        # Missing / inconclusive retrieval → UNVERIFIED, never FALSE
        return _base(
            "UNVERIFIED_CLAIM",
            "STAGE_3_FACTUAL_INCONCLUSIVE",
            "verifiable_but_inconclusive",
            True,
            "Factual claim did not reach support/contradiction thresholds — inconclusive retrieval.",
        )

    # Stage 5 — Mixed-claim guard: multiple medium-to-high signals with no clear winner
    # Prefer MIXED_ANALYSIS over forcing a single category on multi-clause / nuanced text
    signal_count = sum([
        bias_score >= 0.40,
        manipulation_score >= 0.40,
        sarcasm_score >= 0.35,
        opinion_score >= 0.40,
        claim_type == "MIXED",
    ])
    if signal_count >= 2 and bias_score < BIAS_VERDICT_THRESHOLD and manipulation_score < MANIPULATION_VERDICT_THRESHOLD:
        return _base(
            "MIXED_ANALYSIS",
            "STAGE_4_MIXED_MULTI_SIGNAL",
            "non_factual_mixed",
            False,
            f"Multiple medium-strength signals ({signal_count}) without a dominant category — prefer MIXED_ANALYSIS.",
        )

    # Stage 6 — Secondary single-category overrides
    if bias_score >= BIAS_VERDICT_THRESHOLD:
        return _base(
            "BIASED_CONTENT",
            "STAGE_4_BIAS_OVERRIDE",
            "non_factual_mixed",
            False,
            f"Bias score {bias_score:.2f} ≥ threshold {BIAS_VERDICT_THRESHOLD}.",
        )
    if manipulation_score >= MANIPULATION_VERDICT_THRESHOLD:
        return _base(
            "MANIPULATIVE_CONTENT",
            "STAGE_4_MANIPULATION_OVERRIDE",
            "non_factual_mixed",
            False,
            f"Manipulation score {manipulation_score:.2f} ≥ threshold {MANIPULATION_VERDICT_THRESHOLD}.",
        )
    if conspiracy_flag:
        return _base(
            "CONSPIRACY_OR_EXTRAORDINARY_CLAIM",
            "STAGE_4_CONSPIRACY_OVERRIDE",
            "non_factual_mixed",
            False,
            "Extraordinary/conspiracy markers dominated non-factual analysis.",
        )
    # AI verdict only at very high confidence — purely informational otherwise
    if ai_generated_score >= AI_VERDICT_THRESHOLD:
        return _base(
            "LIKELY_AI_GENERATED",
            "STAGE_4_AI_AUXILIARY",
            "non_factual_mixed",
            False,
            f"AI-likelihood {ai_generated_score:.2f} ≥ strict threshold {AI_VERDICT_THRESHOLD} — used only as secondary non-factual signal.",
        )

    return _base(
        "MIXED_ANALYSIS",
        "STAGE_4_MIXED_FALLBACK",
        "non_factual_mixed",
        False,
        "No dominant non-factual override crossed threshold.",
    )


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
    Uses the calibrated threshold constants for consistency.
    """
    if sarcasm_detected:
        return "SATIRE_OR_SARCASM", "RULE_1_SARCASM", "Sarcasm/satire detector fired before factual scoring."
    if opinion_detected:
        return "OPINION", "RULE_2_OPINION", "Subjective or preference language dominates this text."
    if claim_is_unverifiable:
        return "UNVERIFIED_CLAIM", "RULE_3_UNVERIFIABLE", "Statement is currently unverifiable/speculative."
    if factual_claim and evidence_support >= VERIFIED_SUPPORT_THRESHOLD:
        return "VERIFIED_FACT", "RULE_4_VERIFIED", f"Factual claim supported by evidence ≥ {VERIFIED_SUPPORT_THRESHOLD}."
    if factual_claim and evidence_contradiction >= STRICT_CONTRADICTION_THRESHOLD:
        return "FALSE_FACT", "RULE_5_FALSE_FACT", f"Factual claim contradicted by strong evidence ≥ {STRICT_CONTRADICTION_THRESHOLD}."
    if bias_score >= BIAS_VERDICT_THRESHOLD:
        return "BIASED_CONTENT", "RULE_6_BIAS", "Loaded or slanted framing is the dominant signal."
    if manipulation_score >= MANIPULATION_VERDICT_THRESHOLD:
        return "MANIPULATIVE_CONTENT", "RULE_7_MANIPULATION", "Emotional pressure/coercive language is dominant."
    if extraordinary_claim_flag:
        return "CONSPIRACY_OR_EXTRAORDINARY_CLAIM", "RULE_8_EXTRAORDINARY", "Extraordinary claim marker triggered."
    if ai_score >= AI_VERDICT_THRESHOLD and not factual_claim:
        return "LIKELY_AI_GENERATED", "RULE_9_AI_AUXILIARY", f"Very high AI-likelihood ({ai_score:.2f}) on non-factual content."
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
        ("truth_support_gate", factual_claim and evidence_support >= VERIFIED_SUPPORT_THRESHOLD),
        ("truth_contradiction_gate", factual_claim and evidence_contradiction >= STRICT_CONTRADICTION_THRESHOLD),
        ("bias_detector", bias_score >= BIAS_VERDICT_THRESHOLD),
        ("manipulation_detector", manipulation_score >= MANIPULATION_VERDICT_THRESHOLD),
        ("extraordinary_claim_detector", conspiracy_flag),
        ("ai_likelihood_detector", ai_score >= AI_VERDICT_THRESHOLD),
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
    if ai_score > AI_VERDICT_THRESHOLD:
        key_factors.append("Strong AI-generation indicators")
    if factual_claim and evidence_contradiction > STRICT_CONTRADICTION_THRESHOLD:
        key_factors.append("High contradiction on a factual claim")
    if bias_score > BIAS_VERDICT_THRESHOLD:
        key_factors.append("High loaded or slanted framing")
    if manipulation_score > MANIPULATION_VERDICT_THRESHOLD:
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
        "threshold_values_used": {
            "verified_support_threshold": VERIFIED_SUPPORT_THRESHOLD,
            "strict_contradiction_threshold": STRICT_CONTRADICTION_THRESHOLD,
            "bias_verdict_threshold": BIAS_VERDICT_THRESHOLD,
            "manipulation_verdict_threshold": MANIPULATION_VERDICT_THRESHOLD,
            "ai_verdict_threshold": AI_VERDICT_THRESHOLD,
        },
        "detector_confidences": {
            "evidence_support": round(evidence_support, 4),
            "evidence_contradiction": round(evidence_contradiction, 4),
            "bias_score": round(float(bias_score), 4),
            "manipulation_score": round(float(manipulation_score), 4),
            "ai_score": round(float(ai_score), 4),
        },
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
