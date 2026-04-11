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

        trust_agent_confidence = "inconclusive"
        retrieval_support_score = 0.25
        retrieval_contradiction_score = 0.0

        normalized = text.lower()
        if "206 bones" in normalized:
            trust_agent_confidence = "support"
            retrieval_support_score = 0.9
        elif "eiffel tower is in berlin" in normalized:
            trust_agent_confidence = "contradiction"
            retrieval_contradiction_score = 0.93

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
            ai_generated_score=0.2,
        )

    def test_required_benchmark_verdicts(self):
        cases = {
            "The human body has 206 bones.": "VERIFIED_FACT",
            "The Eiffel Tower is in Berlin.": "FALSE_FACT",
            "Dogs are better pets than cats.": "OPINION",
            "If you care about your family, buy this now.": "MANIPULATIVE_CONTENT",
            "Yeah, because the Earth is flat.": "SATIRE_OR_SARCASM",
            "A researcher claims to have found a new species.": "UNVERIFIED_CLAIM",
            "The media is a propaganda machine.": "BIASED_CONTENT",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                routing = self._route_text(text)
                self.assertEqual(routing["primary_verdict"], expected)


if __name__ == "__main__":
    unittest.main()
