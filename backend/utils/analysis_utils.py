from typing import List, Dict, Any

def get_verdict_and_risk(truth_score: float, ai_score: float, bias_score: float, confidence: float) -> Dict[str, Any]:
    """
    Maps analysis scores to a human-readable verdict, risk level, and recommendation.
    """
    verdict = "Inconclusive"
    risk_level = "Medium"
    recommendation = "Verify against trusted sources."
    key_factors = []

    # 1. AI Generation Flags
    if ai_score > 0.8:
        verdict = "Likely AI-Generated"
        risk_level = "High"
        key_factors.append("Strong synthetic artifact patterns detected")
        recommendation = "Be cautious; this content exhibits traits of machine-generated media."
    elif ai_score > 0.6:
        verdict = "Potentially AI-Generated"
        risk_level = "Medium"
        key_factors.append("Subtle AI-generated patterns found")

    # 2. Truth Verification Flags
    if truth_score < 0.3:
        verdict = "Likely False or Misleading"
        risk_level = "High"
        key_factors.append("Contradicts verified factual data")
        recommendation = "High risk of misinformation. Do not share without secondary verification."
    elif truth_score > 0.8:
        if verdict == "Inconclusive":
            verdict = "Verified Authentic"
            risk_level = "Low"
            recommendation = "Content aligns with known factual records."
        else:
            # Complex case: High truth but flagged as AI
            verdict = f"{verdict} (Factually Consistent)"
            key_factors.append("Claims appear factually accurate despite delivery method")

    # 3. Bias Flags
    if bias_score > 0.7:
        key_factors.append("High levels of manipulative or biased framing")
        if risk_level == "Low":
            risk_level = "Medium"
        recommendation += " Note: Content uses highly emotional or biased language."

    # 4. Confidence Calibration
    if confidence < 0.5:
        verdict = f"Uncertain: {verdict}"
        risk_level = "Medium"
        recommendation = "Low confidence analysis. Manual review strongly recommended."
        key_factors.append("Insufficient data for a high-confidence verdict")

    return {
        "verdict": verdict,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "key_factors": list(set(key_factors))
    }
