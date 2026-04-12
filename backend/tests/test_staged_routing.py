import unittest

from services.bias_detector import detect_bias
from services.claim_type_detector import classify_claim_type
from services.manipulation_detector import detect_manipulation
from services.sarcasm_detector import SarcasmDetector
from services.verifiability import assess_claim_verifiability
from utils.analysis_utils import stage_route_primary_verdict


class StagedRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sarcasm_detector = SarcasmDetector()

    def _route_text(self, text: str):
        claim_type = classify_claim_type(text).get("claim_type", "MIXED")
        verifiability = assess_claim_verifiability(text, claim_type=claim_type)
        sarcasm = self.sarcasm_detector.detect(text)
        bias_score = detect_bias(text).get("score", 0.0)
        manipulation_score = detect_manipulation(text, {"lexical_repetition": 0.0}).get("score", 0.0)
        normalized = text.lower()
        misinformation_score = 0.0
        if any(p in normalized for p in ["scientists confirmed", "scientists proved", "experts confirmed", "secret report says"]):
            misinformation_score = 0.72
        if any(p in normalized for p in ["doctors hate this", "cures all", "miracle cure"]):
            misinformation_score = max(misinformation_score, 0.84)

        trust_agent_confidence = "inconclusive"
        retrieval_support_score = 0.25
        retrieval_contradiction_score = 0.0
        model_truth_score = 0.5

        if "206 bones" in normalized:
            trust_agent_confidence = "support"
            retrieval_support_score = 0.9
            model_truth_score = 0.88
        elif "water boils at 100 degrees celsius" in normalized:
            trust_agent_confidence = "support"
            retrieval_support_score = 0.84
            model_truth_score = 0.83
        elif "eiffel tower is in berlin" in normalized:
            trust_agent_confidence = "contradiction"
            retrieval_contradiction_score = 0.93
            model_truth_score = 0.14
        elif "unverified forum thread" in normalized:
            trust_agent_confidence = "inconclusive"
            retrieval_support_score = 0.1
            model_truth_score = 0.45
        elif "ai style sample" in normalized:
            model_truth_score = 0.5

        return stage_route_primary_verdict(
            claim_type=claim_type,
            claim_verifiable=bool(verifiability.get("claim_verifiable", True)),
            opinion_detected=bool(verifiability.get("opinion_detected", False)),
            sarcasm_detected=bool(sarcasm.get("sarcasm", False)),
            factual_claim=claim_type == "FACTUAL_CLAIM",
            trust_agent_confidence=trust_agent_confidence,
            retrieval_support_score=retrieval_support_score,
            retrieval_contradiction_score=retrieval_contradiction_score,
            bias_score=float(bias_score),
            manipulation_score=float(manipulation_score),
            conspiracy_flag=False,
            ai_generated_score=0.92 if "ai style sample" in normalized else 0.2,
            misinformation_score=misinformation_score,
            model_truth_score=model_truth_score,
        )

    def test_required_benchmark_verdicts(self):
        cases = {
            "The human body has 206 bones.": "VERIFIED_FACT",
            "The Eiffel Tower is in Berlin.": "FALSE_FACT",
            "Honestly I tried that new cafe near campus and I loved the ambiance.": "OPINION",
            "Water boils at 100 degrees Celsius at sea level.": "VERIFIED_FACT",
            "Scientists confirmed coffee cures all forms of cancer.": "CONSPIRACY_OR_EXTRAORDINARY_CLAIM",
            "If you care about your family, buy this now.": "MANIPULATIVE_CONTENT",
            "The media is a propaganda machine and only true patriots can save the country.": "BIASED_CONTENT",
            "Vaccines prevented millions of deaths, but public communication around mandates was often poor.": "MIXED_ANALYSIS",
            "Yeah, because the Earth is flat.": "SATIRE_OR_SARCASM",
            "A researcher claims to have found a new species in an unverified forum thread.": "UNVERIFIED_CLAIM",
            "AI style sample: Therefore, in conclusion, this system demonstrates comprehensive optimization across all domains.": "LIKELY_AI_GENERATED",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                routing = self._route_text(text)
                self.assertEqual(routing["primary_verdict"], expected)

    def test_consistency_guard_prevents_mild_verdicts(self):
        routing = stage_route_primary_verdict(
            claim_type="MIXED",
            claim_verifiable=True,
            opinion_detected=False,
            sarcasm_detected=False,
            factual_claim=False,
            trust_agent_confidence="inconclusive",
            retrieval_support_score=0.05,
            retrieval_contradiction_score=0.22,
            bias_score=0.88,
            manipulation_score=0.32,
            conspiracy_flag=False,
            ai_generated_score=0.12,
            misinformation_score=0.0,
            model_truth_score=0.21,
        )
        self.assertEqual(routing["primary_verdict"], "BIASED_CONTENT")

    def test_mixed_fact_opinion_route_stays_mixed_when_not_clear_opinion(self):
        routing = stage_route_primary_verdict(
            claim_type="MIXED",
            claim_verifiable=True,
            opinion_detected=False,
            sarcasm_detected=False,
            factual_claim=False,
            trust_agent_confidence="inconclusive",
            retrieval_support_score=0.42,
            retrieval_contradiction_score=0.21,
            bias_score=0.39,
            manipulation_score=0.39,
            conspiracy_flag=False,
            ai_generated_score=0.23,
            misinformation_score=0.0,
            sarcasm_score=0.36,
            opinion_score=0.44,
            model_truth_score=0.58,
        )
        self.assertEqual(routing["primary_verdict"], "MIXED_ANALYSIS")


if __name__ == "__main__":
    unittest.main()
